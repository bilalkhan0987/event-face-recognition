#!/usr/bin/env python
"""
testEvent_rgbPretrained.py

Zero-shot BASELINE: take an RGB-face-pretrained ArcFace r50 (e.g. ms1mv3_arcface_r50)
and test it DIRECTLY on the FES event dataset — no event training, no fine-tuning.

This measures the raw RGB->event domain gap and is the floor against which the
Glint360K-event training and FES fine-tuning are compared. Same FES session structure,
same Protocol 1 / Protocol 2 scoring, and the same accuracy / TAR@FAR metrics as
eval_protocols.py, so results are directly comparable to the fine-tuned model.

Run:
    python testEvent_rgbPretrained.py configs/fes_rgb_pretrained.py \
        --weights models/ms1mv3_arcface_r50_fp16/backbone.pth \
        --test_db /path/to/FES_frames_fixed_duration_test \
        --output_dir eval_rgb_pretrained --max_session 16
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
from sklearn.model_selection import KFold

from backbones import get_model
from utils.utils_config import get_config

VALID_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
FAR_TARGETS = (0.1, 0.01, 0.001, 0.0001)     # keep identical across all eval scripts
ACC_FOLDS = 10


def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")


# --------------------------------------------------------------------------- #
# Model loading (handles a raw backbone.pth state_dict)
# --------------------------------------------------------------------------- #
def strip_known_prefixes(sd):
    out = {}
    for k, v in sd.items():
        nk = k; ch = True
        while ch:
            ch = False
            for p in ("module.", "backbone.", "model."):
                if nk.startswith(p):
                    nk = nk[len(p):]; ch = True
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
    sd = strip_known_prefixes(extract_backbone_state_dict(torch.load(weights, map_location=device)))
    inc = model.load_state_dict(sd, strict=False)
    if inc.missing_keys:
        logging.warning(f"missing keys: {len(inc.missing_keys)} (first {inc.missing_keys[:8]})")
    if inc.unexpected_keys:
        logging.warning(f"unexpected keys: {len(inc.unexpected_keys)} (first {inc.unexpected_keys[:8]})")
    if len(inc.missing_keys) > 20:
        raise RuntimeError("too many missing keys — config.network likely mismatches the checkpoint")
    model.eval()
    logging.info(f"Loaded RGB-pretrained weights: {weights}")
    return model


# --------------------------------------------------------------------------- #
# FES scan / embedding
# --------------------------------------------------------------------------- #
def scan_sessions(root_dir, max_session):
    subjects = sorted(d for d in os.listdir(root_dir)
                      if os.path.isdir(os.path.join(root_dir, d)))
    if not subjects:
        raise FileNotFoundError(f"No subject folders in {root_dir}")
    subj_to_label = {s: i for i, s in enumerate(subjects)}
    label_to_name = {i: s for s, i in subj_to_label.items()}
    samples = []
    for subj in subjects:
        sp = os.path.join(root_dir, subj)
        for sess in sorted(os.listdir(sp)):
            ssp = os.path.join(sp, sess)
            if not os.path.isdir(ssp):
                continue
            try:
                sid = int(sess)
            except ValueError:
                continue
            if max_session is not None and sid >= max_session:
                continue
            for f in sorted(os.listdir(ssp)):
                if f.lower().endswith(VALID_IMAGE_EXTS):
                    samples.append((os.path.join(ssp, f), subj_to_label[subj], sid))
    if not samples:
        raise FileNotFoundError(f"No frames selected under {root_dir} (max_session={max_session})")
    logging.info(f"Scanned {root_dir}: subjects={len(subjects)} frames={len(samples):,} sessions<{max_session}")
    return samples, label_to_name


@torch.no_grad()
def embed_all(model, samples, image_size, batch_size, device, flip_tta=True):
    embs, subj, sess, norms = [], [], [], []
    bi, bs, bse = [], [], []

    def flush():
        if not bi:
            return
        t = torch.stack(bi).to(device, non_blocking=True)
        f = model(t).float()
        norms.extend(torch.norm(f, dim=1).cpu().tolist())
        if flip_tta:
            f = f + model(torch.flip(t, dims=[3])).float()
        f = F.normalize(f).cpu()
        embs.append(f); subj.extend(bs); sess.extend(bse)
        bi.clear(); bs.clear(); bse.clear()

    for path, s, se in tqdm(samples, desc="Embedding"):
        img = cv2.imread(path)
        if img is None:
            continue
        img = cv2.resize(img, (image_size, image_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1)).astype(np.float32)
        img = (img - 127.5) / 127.5
        bi.append(torch.from_numpy(np.ascontiguousarray(img))); bs.append(s); bse.append(se)
        if len(bi) >= batch_size:
            flush()
    flush()
    if not embs:
        raise RuntimeError("no embeddings extracted")
    return (torch.cat(embs, 0), np.array(subj), np.array(sess), np.array(norms, dtype=np.float32))


# --------------------------------------------------------------------------- #
# Metrics (identical to eval_protocols.py / eval_feret.py)
# --------------------------------------------------------------------------- #
def _accuracy_best_threshold(y, sc, n_thr=400):
    yb = np.asarray(y).astype(bool)
    thr = np.linspace(sc.min(), sc.max(), n_thr)
    accs = [((sc >= t) == yb).mean() for t in thr]
    b = int(np.argmax(accs))
    return float(accs[b]), float(thr[b])


def _accuracy_kfold(y, sc, nfolds=10, seed=42, n_thr=400):
    y = np.asarray(y); sc = np.asarray(sc)
    idx = np.arange(len(y)); np.random.RandomState(seed).shuffle(idx)
    y, sc = y[idx], sc[idx]; yb = y.astype(bool)
    thr = np.linspace(sc.min(), sc.max(), n_thr); accs = []
    for tr, te in KFold(n_splits=nfolds, shuffle=False).split(idx):
        tr_acc = [((sc[tr] >= t) == yb[tr]).mean() for t in thr]
        bt = thr[int(np.argmax(tr_acc))]
        accs.append(((sc[te] >= bt) == yb[te]).mean())
    return float(np.mean(accs)), float(np.std(accs))


def _tar_at_far(y, sc, far_targets):
    yb = np.asarray(y).astype(bool); sc = np.asarray(sc)
    pos, neg = sc[yb], sc[~yb]
    if len(pos) == 0 or len(neg) == 0:
        return {f"{f:.0e}": None for f in far_targets}
    neg_desc = np.sort(neg)[::-1]; n_neg = len(neg); out = {}
    for far in far_targets:
        k = int(round(far * n_neg))
        out[f"{far:.0e}"] = float((pos >= neg_desc[k - 1]).mean()) if k >= 1 else None
    return out


def _metrics_from_pairs(y, sc):
    y, sc = np.asarray(y), np.asarray(sc)
    if len(np.unique(y)) < 2:
        return None
    fpr, tpr, _ = roc_curve(y, sc); fnr = 1 - tpr
    k = np.nanargmin(np.abs(fnr - fpr))
    acc, acc_thr = _accuracy_best_threshold(y, sc)
    acc_mean, acc_std = _accuracy_kfold(y, sc, nfolds=ACC_FOLDS)
    return {"ap": float(average_precision_score(y, sc)), "auc": float(auc(fpr, tpr)),
            "eer": float((fpr[k] + fnr[k]) / 2),
            "accuracy": acc, "accuracy_threshold": acc_thr,
            "accuracy_10fold_mean": acc_mean, "accuracy_10fold_std": acc_std,
            "tar_at_far": _tar_at_far(y, sc, FAR_TARGETS),
            "num_pairs": int(len(y)),
            "num_genuine": int((y == 1).sum()), "num_impostor": int((y == 0).sum())}


def score_frames(embs, subj, sess=None, cross_session=False, max_pairs=20000, seed=42):
    rng = random.Random(seed)
    by = defaultdict(list)
    for i, s in enumerate(subj):
        by[int(s)].append(i)
    alls = list(by.keys()); poss = [s for s, ix in by.items() if len(ix) >= 2]
    if len(alls) < 2 or not poss:
        return None
    npos = max_pairs // 2; y, sc = [], []
    for _ in range(npos):
        s = rng.choice(poss); ix = by[s]; i, j = rng.sample(ix, 2)
        if cross_session and sess is not None:
            t = 0
            while sess[i] == sess[j] and t < 10:
                j = rng.choice(ix); t += 1
        sc.append(float(torch.dot(embs[i], embs[j]))); y.append(1)
    for _ in range(max_pairs - npos):
        a, b = rng.sample(alls, 2)
        sc.append(float(torch.dot(embs[rng.choice(by[a])], embs[rng.choice(by[b])]))); y.append(0)
    return _metrics_from_pairs(y, sc)


def score_templates(embs, subj, sess, max_pairs=20000, seed=42):
    groups = defaultdict(list)
    for i in range(len(subj)):
        groups[(int(subj[i]), int(sess[i]))].append(i)
    T, ts, tse = [], [], []
    for (s, se), ix in groups.items():
        T.append(F.normalize(embs[ix].mean(0), dim=0)); ts.append(s); tse.append(se)
    if len(T) < 2:
        return None
    m = score_frames(torch.stack(T), np.array(ts), np.array(tse),
                     cross_session=True, max_pairs=max_pairs, seed=seed)
    if m:
        m["num_templates"] = int(len(T))
    return m


def identification_oddeven(embs, subj, sess, ks=(1, 5)):
    by = defaultdict(lambda: defaultdict(list))
    for i in range(len(subj)):
        by[int(subj[i])][int(sess[i])].append(i)
    gv, gl, pv, pl = [], [], [], []
    for s, sessions in by.items():
        even = [i for se, ix in sessions.items() if se % 2 == 0 for i in ix]
        odd = {se: ix for se, ix in sessions.items() if se % 2 == 1}
        if not even or not odd:
            continue
        gv.append(F.normalize(embs[even].mean(0), dim=0)); gl.append(s)
        for se, ix in odd.items():
            pv.append(F.normalize(embs[ix].mean(0), dim=0)); pl.append(s)
    if len(gv) < 2 or not pv:
        return None
    G = torch.stack(gv); gl = np.array(gl); P = torch.stack(pv); pl = np.array(pl)
    order = torch.argsort(P @ G.T, dim=1, descending=True).numpy()
    res = {}
    for k in ks:
        res[f"rank_{k}"] = sum(int(pl[r] in gl[order[r, :k]]) for r in range(len(pl))) / len(pl)
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
    out = os.path.join(args.output_dir, f"rgb_zeroshot_{ts}")
    os.makedirs(out, exist_ok=True)
    logging.info(f"Output dir: {out}")

    cfg = get_config(args.config)
    weights = args.weights or getattr(cfg, "pretrained", None)
    if not weights:
        raise ValueError("Provide --weights or set config.pretrained (the RGB backbone.pth).")

    device = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu")
    model = load_model(cfg, weights, device)

    samples, label_to_name = scan_sessions(args.test_db, args.max_session)
    embs, subj, sess, norms = embed_all(
        model, samples, args.image_size, args.batch_size, device, flip_tta=not args.no_flip_tta)

    q = np.percentile(norms, [1, 50, 99])
    logging.info(f"feature-norm: min={norms.min():.2f} p1={q[0]:.2f} median={q[1]:.2f} max={norms.max():.2f}")

    p1_frame = score_frames(embs, subj, sess, cross_session=False, max_pairs=args.max_pairs, seed=args.seed)
    p1_tmpl = score_templates(embs, subj, sess, max_pairs=args.max_pairs, seed=args.seed)
    p1_ident = identification_oddeven(embs, subj, sess)

    logging.info("=" * 70)
    logging.info("RGB-PRETRAINED ZERO-SHOT on FES (Protocol 1, sessions 0..%d)" % (args.max_session - 1))

    def _log(tag, m):
        if not m:
            return
        tar = " ".join(f"{k}={v:.3f}" for k, v in m["tar_at_far"].items() if v is not None)
        extra = f" templates={m['num_templates']}" if "num_templates" in m else ""
        logging.info(f"  {tag:8s}: AP={m['ap']:.4f} AUC={m['auc']:.4f} EER={m['eer']:.4f} "
                     f"Acc={m['accuracy_10fold_mean']*100:.2f}%±{m['accuracy_10fold_std']*100:.2f} "
                     f"| TAR@FAR[{tar}]{extra}")
    _log("frame", p1_frame)
    _log("template", p1_tmpl)
    if p1_ident:
        logging.info(f"  identif.: rank1={p1_ident['rank_1']*100:.2f}% rank5={p1_ident['rank_5']*100:.2f}% "
                     f"(gallery={p1_ident['num_gallery']} probe={p1_ident['num_probe']})")

    # Protocol 2 (per session)
    session_ids = sorted(set(sess.tolist()))
    rows, aps, accs, eers = [], [], [], []
    for s in session_ids:
        m = sess == s
        res = score_frames(embs[m], subj[m], cross_session=False, max_pairs=args.max_pairs, seed=args.seed)
        if res:
            rows.append([int(s), int(m.sum()), res["num_pairs"], res["ap"], res["auc"],
                         res["eer"], res["accuracy"], res["accuracy_10fold_mean"]])
            aps.append(res["ap"]); accs.append(res["accuracy_10fold_mean"]); eers.append(res["eer"])
            logging.info(f"  [session {int(s):2d}] AP={res['ap']:.4f} EER={res['eer']:.4f} "
                         f"Acc={res['accuracy_10fold_mean']*100:.2f}% (frames={int(m.sum())})")
    with open(os.path.join(out, "protocol2_per_session.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["session", "n_frames", "num_pairs", "ap", "auc", "eer", "accuracy", "accuracy_10fold"])
        w.writerows(rows)

    p2 = {"num_sessions": len(aps),
          "ap_mean": float(np.mean(aps)) if aps else None,
          "ap_std": float(np.std(aps)) if aps else None,
          "accuracy_mean": float(np.mean(accs)) if accs else None,
          "eer_mean": float(np.mean(eers)) if eers else None}

    with open(os.path.join(out, "rgb_zeroshot_metrics.json"), "w") as f:
        json.dump({"experiment": "rgb_pretrained_zeroshot_on_FES",
                   "weights": weights, "test_db": args.test_db, "max_session": args.max_session,
                   "num_subjects": len(label_to_name), "num_frames": int(len(subj)),
                   "protocol1": {"frame": p1_frame, "template": p1_tmpl,
                                 "identification_oddeven": p1_ident},
                   "protocol2": p2}, f, indent=4)

    if aps:
        logging.info("-" * 70)
        logging.info(f"PROTOCOL 2: mean AP={p2['ap_mean']:.4f}±{p2['ap_std']:.4f} | "
                     f"mean Acc={p2['accuracy_mean']*100:.2f}% | mean EER={p2['eer_mean']:.4f}")
    logging.info(f"Saved outputs to {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Zero-shot RGB-pretrained r50 tested on FES events.")
    p.add_argument("config", type=str)
    p.add_argument("--weights", type=str, default=None,
                   help="RGB backbone.pth (defaults to config.pretrained).")
    p.add_argument("--test_db", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--max_session", type=int, default=16)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--image_size", type=int, default=112)
    p.add_argument("--max_pairs", type=int, default=20000)
    p.add_argument("--far_targets", type=str, default="1e-1,1e-2,1e-3,1e-4")
    p.add_argument("--acc_folds", type=int, default=10)
    p.add_argument("--no_flip_tta", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    main(p.parse_args())