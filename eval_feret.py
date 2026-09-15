#!/usr/bin/env python
"""
Zero-shot evaluation of an ArcFace backbone on FERET converted to event frames (.npz).

Dataset layout (colorferet_events):
    <root>/
        00527/
            00527_940519_fa.npz      # SUBJECT_DATE_POSE[_a].npz
            00527_940519_fb.npz
            00527_940519_pl.npz
            ...

    Identity = subject id (leading digits of filename / folder name).
    Pose code = fa, fb (frontal); ql/qr, hl/hr, pl/pr (increasing yaw); ra/rb/rc (misc).
    '_a' = alternate/aligned capture of the same shot.

Each npz has final_hist (event histogram) — reconstructed to an image exactly like the
training/eval npz path, so preprocessing matches the model. Set --clip_val to whatever
the model was TRAINED with (default 5.0).

Two evaluation families (both reported):
  1. VERIFICATION  — sampled genuine/impostor cosine pairs; frame- and template-level
                     AP/AUC/EER. Comparable to the FES Protocol-1 numbers.
  2. IDENTIFICATION — FERET-native closed-set: gallery = one pose (default fa),
                     probes = other poses; rank-1/5, with a per-probe-pose breakdown.

Run:
    python eval_feret.py configs/glint360.py \
        --weights output/model.pt \
        --feret_db /data/facerecognition_events_paul/FERET/colorferet_events \
        --output_dir eval_feret --clip_val 5.0 \
        --gallery_pose fa
"""

import os
import re
import csv
import json
import glob
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


def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

from sklearn.model_selection import KFold

FAR_TARGETS = (0.1, 0.01, 0.001, 0.0001)   # set from args in main()
ACC_FOLDS = 10
# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #
def strip_known_prefixes(sd):
    out = {}
    for k, v in sd.items():
        nk = k
        ch = True
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
    logging.info(f"Loaded weights: {weights}")
    return model


# --------------------------------------------------------------------------- #
# npz -> image (must match training preprocessing)
# --------------------------------------------------------------------------- #
def hist_to_uint8(final_hist, clip_val):
    img = np.clip(final_hist.astype(np.float32), -float(clip_val), float(clip_val))
    img = (img + float(clip_val)) / (2.0 * float(clip_val))
    img = np.squeeze(img)
    return (img * 255.0).astype(np.uint8)


def preprocess_npz(path, image_size, hist_key, clip_val):
    try:
        with np.load(path, allow_pickle=True) as data:
            if hist_key not in data:
                raise KeyError(f"{hist_key} not in {list(data.keys())}")
            fh = data[hist_key]
        img = hist_to_uint8(fh, clip_val)
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        elif img.ndim == 3 and img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=-1)
        img = cv2.resize(img, (image_size, image_size))
        img = np.transpose(img, (2, 0, 1)).astype(np.float32)
        img = (img - 127.5) / 127.5
        return img
    except Exception as e:
        logging.warning(f"bad npz {path}: {repr(e)}")
        return None


# --------------------------------------------------------------------------- #
# Filename parsing
# --------------------------------------------------------------------------- #
FNAME_RE = re.compile(r"^(?P<subj>\d+)_(?P<date>\d+)_(?P<pose>[a-zA-Z]+)(?P<alt>_a)?$")


def parse_feret_name(path):
    """Return (subject_id, pose_code, is_alt) or None if unparseable."""
    stem = os.path.splitext(os.path.basename(path))[0]
    m = FNAME_RE.match(stem)
    if not m:
        return None
    return m.group("subj"), m.group("pose").lower(), bool(m.group("alt"))


def scan_feret(root, include_alt, subject_set=None):
    files = sorted(glob.glob(os.path.join(root, "**", "*.npz"), recursive=True))
    samples = []
    subj_names, pose_counts = set(), defaultdict(int)
    unparsed = 0
    for f in files:
        parsed = parse_feret_name(f)
        if parsed is None:
            unparsed += 1
            continue
        subj, pose, is_alt = parsed
        if is_alt and not include_alt:
            continue
        if subject_set is not None and subj not in subject_set:   # <-- add
            continue
        samples.append((f, subj, pose))
        subj_names.add(subj); pose_counts[pose] += 1
    if not samples:
        raise FileNotFoundError(f"No parseable .npz under {root}")
    subj_sorted = sorted(subj_names)
    subj_to_label = {s: i for i, s in enumerate(subj_sorted)}
    label_to_name = {i: s for s, i in subj_to_label.items()}
    samples = [(p, subj_to_label[s], pose) for p, s, pose in samples]
    logging.info(f"FERET scan: files={len(files)} used={len(samples)} subjects={len(subj_sorted)} "
                 f"unparsed={unparsed}")
    logging.info(f"pose counts: {dict(sorted(pose_counts.items(), key=lambda x:-x[1]))}")
    return samples, label_to_name, dict(pose_counts)


# --------------------------------------------------------------------------- #
# Embedding
# --------------------------------------------------------------------------- #
@torch.no_grad()
def embed_all(model, samples, image_size, batch_size, device, hist_key, clip_val, flip_tta=True):
    embs, subj, pose, norms = [], [], [], []
    buf_i, buf_s, buf_p = [], [], []

    def flush():
        if not buf_i:
            return
        t = torch.stack(buf_i).to(device, non_blocking=True)
        f = model(t).float()
        norms.extend(torch.norm(f, dim=1).cpu().tolist())
        if flip_tta:
            f = f + model(torch.flip(t, dims=[3])).float()
        f = F.normalize(f).cpu()
        embs.append(f); subj.extend(buf_s); pose.extend(buf_p)
        buf_i.clear(); buf_s.clear(); buf_p.clear()

    for path, s, p in tqdm(samples, desc="Embedding"):
        img = preprocess_npz(path, image_size, hist_key, clip_val)
        if img is None:
            continue
        buf_i.append(torch.from_numpy(np.ascontiguousarray(img)))
        buf_s.append(s); buf_p.append(p)
        if len(buf_i) >= batch_size:
            flush()
    flush()
    if not embs:
        raise RuntimeError("no embeddings extracted")
    return (torch.cat(embs, 0), np.array(subj), np.array(pose, dtype=object),
            np.array(norms, dtype=np.float32))


# --------------------------------------------------------------------------- #
# Scoring core
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


def _metrics(y, sc):
    y, sc = np.asarray(y), np.asarray(sc)
    if len(np.unique(y)) < 2:
        return None
    fpr, tpr, _ = roc_curve(y, sc)
    fnr = 1 - tpr
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


def verification(embs, subj, max_pairs=40000, seed=42):
    rng = random.Random(seed)
    by = defaultdict(list)
    for i, s in enumerate(subj):
        by[int(s)].append(i)
    alls = list(by.keys())
    poss = [s for s, ix in by.items() if len(ix) >= 2]
    if len(alls) < 2 or not poss:
        return None
    npos = max_pairs // 2
    y, sc = [], []
    for _ in range(npos):
        s = rng.choice(poss); i, j = rng.sample(by[s], 2)
        sc.append(float(torch.dot(embs[i], embs[j]))); y.append(1)
    for _ in range(max_pairs - npos):
        a, b = rng.sample(alls, 2)
        i, j = rng.choice(by[a]), rng.choice(by[b])
        sc.append(float(torch.dot(embs[i], embs[j]))); y.append(0)
    return _metrics(y, sc)


def subject_templates(embs, subj):
    """One mean template per subject (pool all poses). For template verification we
    need >=2 templates/subject, so this pools per (subject) — used for identification
    gallery, not verification. Kept separate on purpose."""
    by = defaultdict(list)
    for i, s in enumerate(subj):
        by[int(s)].append(i)
    T, lab = [], []
    for s, ix in by.items():
        T.append(F.normalize(embs[ix].mean(dim=0), dim=0)); lab.append(s)
    return torch.stack(T), np.array(lab)


def template_verification(embs, subj, pose, max_pairs=40000, seed=42):
    """Per-(subject,pose) templates; genuine = two DIFFERENT poses of same subject."""
    groups = defaultdict(list)
    for i in range(len(subj)):
        groups[(int(subj[i]), str(pose[i]))].append(i)
    T, tsubj, tpose = [], [], []
    for (s, p), ix in groups.items():
        T.append(F.normalize(embs[ix].mean(dim=0), dim=0)); tsubj.append(s); tpose.append(p)
    if len(T) < 2:
        return None
    T = torch.stack(T); tsubj = np.array(tsubj); tpose = np.array(tpose, dtype=object)
    rng = random.Random(seed)
    by = defaultdict(list)
    for i, s in enumerate(tsubj):
        by[int(s)].append(i)
    poss = [s for s, ix in by.items() if len(ix) >= 2]
    alls = list(by.keys())
    if not poss or len(alls) < 2:
        return None
    npos = max_pairs // 2
    y, sc = [], []
    for _ in range(npos):
        s = rng.choice(poss); i, j = rng.sample(by[s], 2)
        sc.append(float(torch.dot(T[i], T[j]))); y.append(1)
    for _ in range(max_pairs - npos):
        a, b = rng.sample(alls, 2)
        i, j = rng.choice(by[a]), rng.choice(by[b])
        sc.append(float(torch.dot(T[i], T[j]))); y.append(0)
    m = _metrics(y, sc)
    if m:
        m["num_templates"] = int(len(T))
    return m


# --------------------------------------------------------------------------- #
# FERET gallery/probe identification
# --------------------------------------------------------------------------- #
def identification(embs, subj, pose, gallery_pose="fa", ks=(1, 5), per_pose=True):
    """Gallery = mean embedding of gallery_pose per subject. Probes = every non-gallery
    frame. Rank-k over cosine to gallery. Also a per-probe-pose rank-1 breakdown."""
    pose = np.array([str(p) for p in pose], dtype=object)
    # gallery: subjects that have the gallery pose
    gal_idx = defaultdict(list)
    for i in range(len(subj)):
        if pose[i] == gallery_pose:
            gal_idx[int(subj[i])].append(i)
    if len(gal_idx) < 2:
        logging.warning(f"gallery pose '{gallery_pose}' has <2 subjects; identification skipped.")
        return None
    gal_vecs, gal_lab = [], []
    for s, ix in gal_idx.items():
        gal_vecs.append(F.normalize(embs[ix].mean(dim=0), dim=0)); gal_lab.append(s)
    G = torch.stack(gal_vecs); gl = np.array(gal_lab)
    gal_subjects = set(gl.tolist())

    # probes: all non-gallery-pose frames whose subject is in the gallery
    probe_ix = [i for i in range(len(subj))
                if pose[i] != gallery_pose and int(subj[i]) in gal_subjects]
    if not probe_ix:
        return None
    P = embs[probe_ix]
    P_lab = np.array([int(subj[i]) for i in probe_ix])
    P_pose = np.array([pose[i] for i in probe_ix], dtype=object)

    sims = P @ G.T
    order = torch.argsort(sims, dim=1, descending=True).numpy()

    def rank_hits(rows):
        out = {}
        for k in ks:
            hit = sum(int(P_lab[r] in gl[order[r, :k]]) for r in rows)
            out[f"rank_{k}"] = hit / len(rows) if rows else None
        return out

    overall = rank_hits(list(range(len(P_lab))))
    overall["num_gallery"] = int(len(gl))
    overall["num_probe"] = int(len(P_lab))
    overall["gallery_pose"] = gallery_pose

    per = {}
    if per_pose:
        for pz in sorted(set(P_pose.tolist())):
            rows = [r for r in range(len(P_lab)) if P_pose[r] == pz]
            hh = rank_hits(rows)
            hh["num_probe"] = len(rows)
            per[pz] = hh
    return {"overall": overall, "per_pose": per}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(args):
    setup_logging()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(args.output_dir, f"feret_{ts}")
    os.makedirs(out, exist_ok=True)
    logging.info(f"Output dir: {out}")

    cfg = get_config(args.config)
    device = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu")
    model = load_model(cfg, args.weights, device)

    subject_set = None
    if args.subject_list:
        with open(args.subject_list) as f:
            subject_set = set(l.strip() for l in f if l.strip())
        logging.info(f"Restricting to {len(subject_set)} subjects from {args.subject_list}")
    samples, label_to_name, pose_counts = scan_feret(
        args.feret_db, include_alt=args.include_alt, subject_set=subject_set)

    embs, subj, pose, norms = embed_all(
        model, samples, args.image_size, args.batch_size, device,
        hist_key=args.hist_key, clip_val=args.clip_val, flip_tta=not args.no_flip_tta)

    q = np.percentile(norms, [1, 50, 99])
    logging.info(f"feature-norm: min={norms.min():.2f} p1={q[0]:.2f} median={q[1]:.2f} "
                 f"p99={q[2]:.2f} max={norms.max():.2f}")

    # ---- verification ----
    ver_frame = verification(embs, subj, max_pairs=args.max_pairs, seed=args.seed)
    ver_tmpl = template_verification(embs, subj, pose, max_pairs=args.max_pairs, seed=args.seed)

    # ---- identification (FERET native) ----
    ident = identification(embs, subj, pose, gallery_pose=args.gallery_pose)

    logging.info("=" * 70)
    logging.info("FERET ZERO-SHOT")
    if ver_frame:
        logging.info(f"  verification frame    : AP={ver_frame['ap']:.4f} AUC={ver_frame['auc']:.4f} EER={ver_frame['eer']:.4f}")
    if ver_tmpl:
        logging.info(f"  verification template : AP={ver_tmpl['ap']:.4f} AUC={ver_tmpl['auc']:.4f} EER={ver_tmpl['eer']:.4f} "
                     f"(templates={ver_tmpl['num_templates']})")
    if ident:
        ov = ident["overall"]
        logging.info(f"  identification (gallery={ov['gallery_pose']}): "
                     f"rank1={ov['rank_1']*100:.2f}% rank5={ov['rank_5']*100:.2f}% "
                     f"(gallery={ov['num_gallery']} probes={ov['num_probe']})")
        for pz, hh in sorted(ident["per_pose"].items()):
            r1 = hh['rank_1']*100 if hh['rank_1'] is not None else float('nan')
            logging.info(f"      probe {pz:4s}: rank1={r1:6.2f}%  (n={hh['num_probe']})")

    # ---- save ----
    with open(os.path.join(out, "feret_metrics.json"), "w") as f:
        json.dump({"feret_db": args.feret_db, "weights": args.weights,
                   "clip_val": args.clip_val, "include_alt": args.include_alt,
                   "num_subjects": len(label_to_name), "num_frames": int(len(subj)),
                   "pose_counts": pose_counts,
                   "verification_frame": ver_frame, "verification_template": ver_tmpl,
                   "identification": ident,
                   "feature_norm": {"min": float(norms.min()), "median": float(q[1]),
                                    "max": float(norms.max())}}, f, indent=4)

    if ident:
        with open(os.path.join(out, "feret_identification_per_pose.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["probe_pose", "num_probe", "rank_1", "rank_5"])
            w.writerow(["OVERALL", ident["overall"]["num_probe"],
                        ident["overall"]["rank_1"], ident["overall"]["rank_5"]])
            for pz, hh in sorted(ident["per_pose"].items()):
                w.writerow([pz, hh["num_probe"], hh["rank_1"], hh["rank_5"]])

    logging.info(f"Saved outputs to {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Zero-shot FERET-events evaluation.")
    p.add_argument("config", type=str)
    p.add_argument("--weights", type=str, required=True)
    p.add_argument("--feret_db", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--clip_val", type=float, default=5.0,
                   help="MUST match training. Default 5.0.")
    p.add_argument("--gallery_pose", type=str, default="fa",
                   help="Pose code used as the identification gallery (FERET default: fa).")
    p.add_argument("--include_alt", action="store_true",
                   help="Include '_a' alternate captures (default: skip them).")
    p.add_argument("--hist_key", type=str, default="final_hist")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--image_size", type=int, default=112)
    p.add_argument("--max_pairs", type=int, default=40000)
    p.add_argument("--no_flip_tta", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--subject_list", type=str, default=None,
                   help="Restrict evaluation to subjects listed in this file (one per line). "
                        "Use test_subjects.txt for the held-out set.")
    main(p.parse_args())