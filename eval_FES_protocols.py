#!/usr/bin/env python
"""
Unified evaluation for FES face recognition — Protocols 1 and 2.

Identity  = outer folder (YYYYMMDD_sequenceID).
Session   = inner folder, selected numerically as int(name) < --max_session (default 16).
Frame     = .png file.

ONE embedding pass feeds ALL protocols, so results differ only in how frames are
grouped for scoring (not in model or preprocessing).

Protocol 1 (matched, aggregated 0..15):
    frame-level     : genuine=same subject (any 0..15 session), impostor=different subject
    template-level  : per-(subject,session) mean templates, cross-session verification
                      (genuine = two different 0..15 sessions of same subject)
    identification  : optional closed-set gallery (one pooled template per subject),
                      rank-1/5 via odd/even session split (reported separately)

Protocol 2 (session-wise, 0..15 independently):
    for each session s: frame-level genuine/impostor pairs WITHIN session s only.
    Reports per-session AP/AUC/EER + mean/std across sessions.

Quality control:
    Records per-frame backbone feature norm (||f(x)||) and logs its distribution
    overall and per session — a no-new-dependency proxy for frame quality / presence.
    Low-norm outliers flag noisy or no-person frames even inside 0..15.

Outputs (in --output_dir/eval_<ts>/):
    protocol1_metrics.json
    protocol2_per_session.csv, protocol2_summary.json
    feature_norms_per_session.csv
    summary.json

Run:
    python eval_protocols.py configs/fes_FT_bk.py \
        --weights output_ft/model_FT.pt \
        --test_db /path/to/FES_frames_fixed_duration_valid \
        --output_dir eval_protocols --max_session 16
"""

import os
import csv
import json
import random
import logging
import argparse
from datetime import datetime
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import average_precision_score, roc_curve, auc

from backbones import get_model
from utils.utils_config import get_config
from sklearn.model_selection import KFold

FAR_TARGETS = (0.1, 0.01, 0.001, 0.0001)   # set from args in main()
ACC_FOLDS = 10
VALID_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")


# --------------------------------------------------------------------------- #
# Model loading (identical logic to the training/eval scripts)
# --------------------------------------------------------------------------- #
def strip_known_prefixes(sd):
    out = {}
    for k, v in sd.items():
        nk = k
        changed = True
        while changed:
            changed = False
            for p in ("module.", "backbone.", "model."):
                if nk.startswith(p):
                    nk = nk[len(p):]; changed = True
        out[nk] = v
    return out


def extract_backbone_state_dict(ckpt):
    if not isinstance(ckpt, dict):
        raise TypeError("weights are not a dict/state_dict")
    for k in ("state_dict_backbone", "backbone_state_dict", "model_state_dict",
              "state_dict", "backbone", "model"):
        if k in ckpt and isinstance(ckpt[k], dict):
            return ckpt[k]
    if sum(torch.is_tensor(v) for v in ckpt.values()) > 0:
        return ckpt
    raise RuntimeError("no usable backbone state_dict found")


def load_model(cfg, weights, device):
    model = get_model(cfg.network, dropout=0.0, fp16=False,
                      num_features=cfg.embedding_size).to(device)
    sd = strip_known_prefixes(extract_backbone_state_dict(
        torch.load(weights, map_location=device)))
    inc = model.load_state_dict(sd, strict=False)
    if inc.missing_keys:
        logging.warning(f"missing keys: {len(inc.missing_keys)} (first {inc.missing_keys[:8]})")
    if inc.unexpected_keys:
        logging.warning(f"unexpected keys: {len(inc.unexpected_keys)} (first {inc.unexpected_keys[:8]})")
    if len(inc.missing_keys) > 20:
        raise RuntimeError("too many missing keys — config.network likely mismatches the checkpoint")
    model.eval()
    logging.info(f"Loaded weights: {weights}")
    return model


# --------------------------------------------------------------------------- #
# Dataset scan (numeric session selection)
# --------------------------------------------------------------------------- #
def scan_sessions(root_dir, max_session):
    subjects = sorted(d for d in os.listdir(root_dir)
                      if os.path.isdir(os.path.join(root_dir, d)))
    if not subjects:
        raise FileNotFoundError(f"No subject folders in {root_dir}")
    subj_to_label = {s: i for i, s in enumerate(subjects)}
    label_to_name = {i: s for s, i in subj_to_label.items()}

    samples = []           # (path, subj_label, session_int)
    skipped_sessions = 0
    for subj in subjects:
        subj_path = os.path.join(root_dir, subj)
        for sess in sorted(os.listdir(subj_path)):
            sess_path = os.path.join(subj_path, sess)
            if not os.path.isdir(sess_path):
                continue
            try:
                sid = int(sess)
            except ValueError:
                skipped_sessions += 1
                continue
            if max_session is not None and sid >= max_session:
                continue
            for f in sorted(os.listdir(sess_path)):
                if f.lower().endswith(VALID_IMAGE_EXTS):
                    samples.append((os.path.join(sess_path, f), subj_to_label[subj], sid))
    if not samples:
        raise FileNotFoundError(f"No frames selected under {root_dir} (max_session={max_session})")
    logging.info(f"Scanned {root_dir}: subjects={len(subjects)} frames={len(samples):,} "
                 f"sessions<{max_session}")
    return samples, label_to_name


# --------------------------------------------------------------------------- #
# Embedding pass (flip-TTA) + feature-norm QC
# --------------------------------------------------------------------------- #
@torch.no_grad()
def embed_all(model, samples, image_size, batch_size, device, flip_tta=True):
    embs, subj, sess, norms = [], [], [], []
    buf_i, buf_s, buf_se = [], [], []
    skipped = 0

    def flush():
        if not buf_i:
            return
        t = torch.stack(buf_i).to(device, non_blocking=True)
        f = model(t).float()
        norms.extend(torch.norm(f, dim=1).cpu().tolist())   # raw feature norm (QC)
        if flip_tta:
            f = f + model(torch.flip(t, dims=[3])).float()
        f = F.normalize(f).cpu()
        embs.append(f); subj.extend(buf_s); sess.extend(buf_se)
        buf_i.clear(); buf_s.clear(); buf_se.clear()

    for path, s, se in tqdm(samples, desc="Embedding"):
        img = cv2.imread(path)
        if img is None:
            skipped += 1
            continue
        img = cv2.resize(img, (image_size, image_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1)).astype(np.float32)
        img = (img - 127.5) / 127.5
        buf_i.append(torch.from_numpy(np.ascontiguousarray(img)))
        buf_s.append(s); buf_se.append(se)
        if len(buf_i) >= batch_size:
            flush()
    flush()
    if skipped:
        logging.warning(f"skipped unreadable frames: {skipped}")
    if not embs:
        raise RuntimeError("no embeddings extracted")
    return (torch.cat(embs, 0), np.array(subj), np.array(sess), np.array(norms, dtype=np.float32))


def feature_norm_report(norms, subj, sess, out_csv):
    q = np.percentile(norms, [1, 5, 50, 95, 99])
    logging.info(f"Feature-norm distribution: min={norms.min():.2f} "
                 f"p1={q[0]:.2f} p5={q[1]:.2f} median={q[2]:.2f} p95={q[3]:.2f} "
                 f"p99={q[4]:.2f} max={norms.max():.2f}")
    # per-session mean/min norm: low values flag noisy/no-person sessions
    rows = []
    for s in sorted(set(sess.tolist())):
        m = sess == s
        rows.append([int(s), int(m.sum()),
                     float(norms[m].mean()), float(norms[m].min()), float(norms[m].std())])
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["session", "n_frames", "mean_norm", "min_norm", "std_norm"])
        w.writerows(rows)
    return {"min": float(norms.min()), "p1": float(q[0]), "median": float(q[2]),
            "max": float(norms.max())}


# --------------------------------------------------------------------------- #
# Shared scoring core
# --------------------------------------------------------------------------- #
def _accuracy_best_threshold(y, sc, n_thr=400):
    yb = np.asarray(y).astype(bool)
    thr = np.linspace(sc.min(), sc.max(), n_thr)
    accs = [((sc >= t) == yb).mean() for t in thr]
    b = int(np.argmax(accs))
    return float(accs[b]), float(thr[b])


def _accuracy_kfold(y, sc, nfolds=10, seed=42, n_thr=400):
    """LFW-style: per fold, pick best threshold on train, report on held-out fold."""
    y = np.asarray(y); sc = np.asarray(sc)
    idx = np.arange(len(y))
    np.random.RandomState(seed).shuffle(idx)
    y, sc = y[idx], sc[idx]
    yb = y.astype(bool)
    thr = np.linspace(sc.min(), sc.max(), n_thr)
    accs = []
    for tr, te in KFold(n_splits=nfolds, shuffle=False).split(idx):
        tr_acc = [((sc[tr] >= t) == yb[tr]).mean() for t in thr]
        bt = thr[int(np.argmax(tr_acc))]
        accs.append(((sc[te] >= bt) == yb[te]).mean())
    return float(np.mean(accs)), float(np.std(accs))


def _tar_at_far(y, sc, far_targets):
    """TAR (=TPR) at fixed FAR operating points, computed from the score distributions."""
    yb = np.asarray(y).astype(bool); sc = np.asarray(sc)
    pos, neg = sc[yb], sc[~yb]
    n_neg = len(neg)
    out = {}
    if n_neg == 0 or len(pos) == 0:
        return {f"{f:.0e}": None for f in far_targets}
    neg_desc = np.sort(neg)[::-1]
    for far in far_targets:
        k = int(round(far * n_neg))
        if k < 1:                      # FAR finer than impostor-pair resolution
            out[f"{far:.0e}"] = None
            continue
        thr = neg_desc[k - 1]
        out[f"{far:.0e}"] = float((pos >= thr).mean())
    return out

def _metrics_from_pairs(y, sc):
    y, sc = np.asarray(y), np.asarray(sc)
    if len(np.unique(y)) < 2:
        return None
    fpr, tpr, _ = roc_curve(y, sc)
    fnr = 1 - tpr
    k = np.nanargmin(np.abs(fnr - fpr))
    acc, acc_thr = _accuracy_best_threshold(y, sc)
    acc_mean, acc_std = _accuracy_kfold(y, sc, nfolds=ACC_FOLDS)
    return {
        "ap": float(average_precision_score(y, sc)),
        "auc": float(auc(fpr, tpr)),
        "eer": float((fpr[k] + fnr[k]) / 2),
        "accuracy": acc,                          # best single threshold
        "accuracy_threshold": acc_thr,
        "accuracy_10fold_mean": acc_mean,         # AdaFace-style headline
        "accuracy_10fold_std": acc_std,
        "tar_at_far": _tar_at_far(y, sc, FAR_TARGETS),
        "num_pairs": int(len(y)),
        "num_genuine": int((y == 1).sum()),
        "num_impostor": int((y == 0).sum()),
    }


def score_frames(embs, subj, sess=None, cross_session=False, max_pairs=20000, seed=42):
    """Genuine = same subject, impostor = different subject. If cross_session,
    genuine pairs are forced to come from different sessions."""
    rng = random.Random(seed)
    by_subj = defaultdict(list)
    for i, s in enumerate(subj):
        by_subj[int(s)].append(i)
    all_s = list(by_subj.keys())
    pos_s = [s for s, ix in by_subj.items() if len(ix) >= 2]
    if len(all_s) < 2 or not pos_s:
        return None
    npos = max_pairs // 2
    y, sc = [], []
    for _ in range(npos):
        s = rng.choice(pos_s); ix = by_subj[s]
        i, j = rng.sample(ix, 2)
        if cross_session and sess is not None:
            t = 0
            while sess[i] == sess[j] and t < 10:
                j = rng.choice(ix); t += 1
        sc.append(float(torch.dot(embs[i], embs[j]))); y.append(1)
    for _ in range(max_pairs - npos):
        a, b = rng.sample(all_s, 2)
        i = rng.choice(by_subj[a]); j = rng.choice(by_subj[b])
        sc.append(float(torch.dot(embs[i], embs[j]))); y.append(0)
    return _metrics_from_pairs(y, sc)


def build_session_templates(embs, subj, sess):
    """One mean-normalized template per (subject, session)."""
    groups = defaultdict(list)
    for i in range(len(subj)):
        groups[(int(subj[i]), int(sess[i]))].append(i)
    tmpl, tsubj, tsess = [], [], []
    for (s, se), ix in groups.items():
        tmpl.append(F.normalize(embs[ix].mean(dim=0), dim=0))
        tsubj.append(s); tsess.append(se)
    return torch.stack(tmpl), np.array(tsubj), np.array(tsess)


def score_templates(embs, subj, sess, max_pairs=20000, seed=42):
    tmpl, tsubj, tsess = build_session_templates(embs, subj, sess)
    if len(tmpl) < 2:
        return None
    m = score_frames(tmpl, tsubj, tsess, cross_session=True, max_pairs=max_pairs, seed=seed)
    if m:
        m["num_templates"] = int(len(tmpl))
    return m


# --------------------------------------------------------------------------- #
# Optional closed-set identification (pooled subject gallery)
# --------------------------------------------------------------------------- #
def identification_oddeven(embs, subj, sess, ks=(1, 5)):
    """Gallery = mean template from EVEN sessions; probe = ODD-session templates.
    Reports rank-1/5. This is the 'one template per subject' pooling, used correctly."""
    gal = defaultdict(list); pr = []
    by = defaultdict(lambda: defaultdict(list))
    for i in range(len(subj)):
        by[int(subj[i])][int(sess[i])].append(i)
    gal_vecs, gal_lab = [], []
    probe_vecs, probe_lab = [], []
    for s, sessions in by.items():
        even_ix = [i for se, ix in sessions.items() if se % 2 == 0 for i in ix]
        odd_sessions = {se: ix for se, ix in sessions.items() if se % 2 == 1}
        if not even_ix or not odd_sessions:
            continue
        gal_vecs.append(F.normalize(embs[even_ix].mean(dim=0), dim=0)); gal_lab.append(s)
        for se, ix in odd_sessions.items():
            probe_vecs.append(F.normalize(embs[ix].mean(dim=0), dim=0)); probe_lab.append(s)
    if len(gal_vecs) < 2 or not probe_vecs:
        return None
    G = torch.stack(gal_vecs); gl = np.array(gal_lab)
    P = torch.stack(probe_vecs); pl = np.array(probe_lab)
    sims = P @ G.T
    order = torch.argsort(sims, dim=1, descending=True).numpy()
    res = {}
    for k in ks:
        hit = 0
        for r in range(len(pl)):
            topk = gl[order[r, :k]]
            hit += int(pl[r] in topk)
        res[f"rank_{k}"] = hit / len(pl)
    res["num_gallery"] = int(len(gl)); res["num_probe"] = int(len(pl))
    return res


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(args):
    setup_logging()
    global FAR_TARGETS, ACC_FOLDS
    FAR_TARGETS = tuple(float(x) for x in args.far_targets.split(","))
    ACC_FOLDS = args.acc_folds
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(args.output_dir, f"eval_{ts}")
    os.makedirs(out, exist_ok=True)
    logging.info(f"Output dir: {out}")

    cfg = get_config(args.config)
    device = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu")
    model = load_model(cfg, args.weights, device)

    samples, label_to_name = scan_sessions(args.test_db, args.max_session)
    embs, subj, sess, norms = embed_all(
        model, samples, args.image_size, args.batch_size, device, flip_tta=not args.no_flip_tta)

    norm_stats = feature_norm_report(
        norms, subj, sess, os.path.join(out, "feature_norms_per_session.csv"))

    # ---- Protocol 1 -------------------------------------------------------
    p1_frame = score_frames(embs, subj, sess, cross_session=False,
                            max_pairs=args.max_pairs, seed=args.seed)
    p1_tmpl = score_templates(embs, subj, sess, max_pairs=args.max_pairs, seed=args.seed)
    p1_ident = identification_oddeven(embs, subj, sess)

    protocol1 = {"frame": p1_frame, "template": p1_tmpl, "identification_oddeven": p1_ident,
                 "num_subjects": len(label_to_name), "num_frames": int(len(subj))}
    with open(os.path.join(out, "protocol1_metrics.json"), "w") as f:
        json.dump(protocol1, f, indent=4)

    logging.info("=" * 70)
    logging.info("PROTOCOL 1 (matched, aggregated 0..%d)" % (args.max_session - 1))
    if p1_frame:
        tar = " ".join(f"{k}={v:.3f}" for k, v in p1_frame["tar_at_far"].items() if v is not None)
        logging.info(f"  frame    : Acc={p1_frame['accuracy_10fold_mean']*100:.2f}%±{p1_frame['accuracy_10fold_std']*100:.2f} "
                     f"| TAR@FAR[{tar}]")
    if p1_tmpl:
        tar = " ".join(f"{k}={v:.3f}" for k, v in p1_tmpl["tar_at_far"].items() if v is not None)
        logging.info(f"  template : Acc={p1_tmpl['accuracy_10fold_mean']*100:.2f}%±{p1_tmpl['accuracy_10fold_std']*100:.2f} "
                     f"| TAR@FAR[{tar}]")
    if p1_ident:
        logging.info(f"  identif. : rank1={p1_ident['rank_1']*100:.2f}% rank5={p1_ident['rank_5']*100:.2f}% "
                     f"(gallery={p1_ident['num_gallery']} probe={p1_ident['num_probe']})")

    # ---- Protocol 2 (per-session, frame-level within each session) --------
    session_ids = sorted(set(sess.tolist()))
    rows, aps, aucs, eers = [], [], [], []
    for s in session_ids:
        m = sess == s
        e_s, subj_s = embs[m], subj[m]
        res = score_frames(e_s, subj_s, cross_session=False,
                           max_pairs=args.max_pairs, seed=args.seed)
        if res:
            rows.append([int(s), int(m.sum()), res["num_pairs"],
                         res["ap"], res["auc"], res["eer"], res["accuracy"]])
            aps.append(res["ap"]); aucs.append(res["auc"]); eers.append(res["eer"])
            logging.info(f"  [session {int(s):2d}] AP={res['ap']:.4f} AUC={res['auc']:.4f} "
                         f"EER={res['eer']:.4f} (frames={int(m.sum())})")
        else:
            rows.append([int(s), int(m.sum()), 0, None, None, None])

    with open(os.path.join(out, "protocol2_per_session.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["session", "n_frames", "num_pairs", "ap", "auc", "eer", "accuracy"])
        w.writerows(rows)

    p2_summary = {
        "num_sessions_scored": len(aps),
        "ap_mean": float(np.mean(aps)) if aps else None,
        "ap_std": float(np.std(aps)) if aps else None,
        "auc_mean": float(np.mean(aucs)) if aucs else None,
        "eer_mean": float(np.mean(eers)) if eers else None,
        "hardest_session_by_ap": int(session_ids[int(np.argmin(aps))]) if aps else None,
        "easiest_session_by_ap": int(session_ids[int(np.argmax(aps))]) if aps else None,
    }
    with open(os.path.join(out, "protocol2_summary.json"), "w") as f:
        json.dump(p2_summary, f, indent=4)

    logging.info("-" * 70)
    logging.info("PROTOCOL 2 (session-wise, frame-level)")
    if aps:
        logging.info(f"  mean AP={p2_summary['ap_mean']:.4f} +/- {p2_summary['ap_std']:.4f} | "
                     f"mean AUC={p2_summary['auc_mean']:.4f} | mean EER={p2_summary['eer_mean']:.4f}")
        logging.info(f"  hardest session (AP): {p2_summary['hardest_session_by_ap']} | "
                     f"easiest: {p2_summary['easiest_session_by_ap']}")

    summary = {"test_db": args.test_db, "weights": args.weights,
               "max_session": args.max_session, "flip_tta": not args.no_flip_tta,
               "feature_norm": norm_stats, "protocol1": protocol1, "protocol2": p2_summary}
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=4)
    logging.info("=" * 70)
    logging.info(f"Saved all outputs to {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="FES recognition — Protocol 1 & 2 evaluation")
    p.add_argument("config", type=str)
    p.add_argument("--far_targets", type=str, default="1e-1,1e-2,1e-3,1e-4")
    p.add_argument("--acc_folds", type=int, default=10)
    p.add_argument("--weights", type=str, required=True)
    p.add_argument("--test_db", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--max_session", type=int, default=16,
                   help="Select sessions with int(name) < this (default 16).")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--image_size", type=int, default=112)
    p.add_argument("--max_pairs", type=int, default=20000)
    p.add_argument("--no_flip_tta", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    main(p.parse_args())