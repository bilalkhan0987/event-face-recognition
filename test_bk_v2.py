# test_fes_eval_v2.py
"""
Evaluate an ArcFace/PartialFC backbone on either:

1) Event-camera Glint360K-style .npz data using an index file:
       /path/to/file.npz<TAB>label

   Example:
       python test_fes_eval_v2.py configs/fes_glint360.py \
           --weights output/model.pt \
           --test_db indexes_all_70_15_15/test_index.txt \
           --data_type npz_index \
           --output_dir eval_npz_test \
           --batch_size 128 \
           --max_pairs 200000

2) Nested image folders, matching the original FES structure:
       test_db/
           20210903/
               01/
                   img1.jpg
                   img2.jpg
               02/
                   img3.jpg

3) Generic image-folder data:
       test_db/
           identity_001/
               img1.jpg
           identity_002/
               img2.jpg

Outputs:
  - metrics.json
  - predictions.csv
  - per_class_accuracy.csv
  - pairwise_scores.csv
  - roc_curve.png / pr_curve.png / similarity_histogram.png when pairwise labels contain both classes
  - embeddings.npz unless --no_save_embeddings is used

Important:
  This evaluates the BACKBONE embeddings, not the PartialFC classifier.
  That is correct for identity-disjoint val/test identities.
"""

import os
import csv
import json
import argparse
import logging
import random
from collections import defaultdict
import re
import pickle
from sklearn.model_selection import KFold

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from datetime import datetime
from tqdm import tqdm
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)

from backbones import get_model
from utils.utils_config import get_config


VALID_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

VALID_NPZ_EXTS = (".npz",)

def load_issame(bin_path, expected_pairs=None):
    """
    Load boolean pair labels from issame.bin.
    Tries pickle (InsightFace convention), then .npy, then raw bytes.
    """
    issame = None
    try:
        with open(bin_path, "rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, tuple) and len(obj) == 2:
            # InsightFace-style (bins, issame): pick the element matching pair count
            a, b = obj
            _len = lambda x: len(x) if hasattr(x, "__len__") else -1
            if expected_pairs is not None and _len(b) == expected_pairs:
                candidate = b
            elif expected_pairs is not None and _len(a) == expected_pairs:
                candidate = a
            else:
                candidate = b
        else:
            candidate = obj
        issame = np.asarray(candidate).astype(bool).ravel()
    except Exception:
        issame = None

    if issame is None:
        try:
            issame = np.load(bin_path, allow_pickle=True).astype(bool).ravel()
        except Exception:
            issame = None

    if issame is None:
        for dtype in (np.bool_, np.uint8, np.int8, np.int32, np.int64):
            arr = np.fromfile(bin_path, dtype=dtype)
            if expected_pairs is not None and len(arr) == expected_pairs:
                issame = arr.astype(bool)
                break

    if issame is None:
        raise RuntimeError(f"Could not parse issame labels from {bin_path}")

    if expected_pairs is not None and len(issame) != expected_pairs:
        raise ValueError(
            f"issame length ({len(issame)}) != number of pairs ({expected_pairs}) "
            f"in {bin_path}. Check file ordering / issame format."
        )
    return issame


def load_event_verification_dataset(bench_dir):
    """
    InsightFace-style verification set: ordered .npz files + issame.bin.
    Files sorted numerically -> consecutive pairs (2k, 2k+1); issame[k] labels pair k.
    """
    npz_files = sorted(
        (os.path.join(bench_dir, f) for f in os.listdir(bench_dir)
         if f.lower().endswith(".npz")),
        key=lambda p: [int(t) if t.isdigit() else t
                       for t in re.split(r"(\d+)", os.path.basename(p))],
    )
    if not npz_files:
        raise FileNotFoundError(f"No .npz files in {bench_dir}")
    if len(npz_files) % 2 != 0:
        raise ValueError(
            f"{bench_dir} has an odd number of npz files ({len(npz_files)}); "
            "verification requires pairs (even count)."
        )
    num_pairs = len(npz_files) // 2

    issame_path = os.path.join(bench_dir, "issame.bin")
    if not os.path.isfile(issame_path):
        raise FileNotFoundError(f"issame.bin not found in {bench_dir}")
    issame = load_issame(issame_path, expected_pairs=num_pairs)
    return npz_files, issame

def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")


def l2_norm(x, axis=1):
    norm = torch.norm(x, 2, axis, True)
    return torch.div(x, norm.clamp_min(1e-12))


def strip_known_prefixes(state_dict):
    """
    Handle common prefixes created by DDP/checkpoint wrappers.
    Examples:
        module.conv1.weight
        backbone.conv1.weight
        model.conv1.weight
    """
    cleaned = {}

    for key, value in state_dict.items():
        new_key = key

        changed = True
        while changed:
            changed = False
            for prefix in ("module.", "backbone.", "model."):
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
                    changed = True

        cleaned[new_key] = value

    return cleaned


def extract_backbone_state_dict(checkpoint):
    """
    Supports:
      1) model.pt saved as pure backbone state_dict
      2) checkpoint_gpu_*.pt with key 'state_dict_backbone'
      3) common checkpoint keys: 'state_dict', 'model_state_dict', 'backbone'
    """
    if not isinstance(checkpoint, dict):
        raise TypeError("Loaded weights are not a dictionary/state_dict.")

    checkpoint_keys = set(checkpoint.keys())

    preferred_keys = [
        "state_dict_backbone",
        "backbone_state_dict",
        "model_state_dict",
        "state_dict",
        "backbone",
        "model",
    ]

    for key in preferred_keys:
        if key in checkpoint and isinstance(checkpoint[key], dict):
            logging.info(f"Detected checkpoint wrapper key: {key}")
            return checkpoint[key]

    # Pure state_dict case: all or most values are tensors.
    tensor_values = sum(torch.is_tensor(v) for v in checkpoint.values())
    if tensor_values > 0:
        logging.info("Detected plain backbone state_dict.")
        return checkpoint

    raise RuntimeError(
        "Could not find a usable backbone state_dict in the weights file. "
        f"Available top-level keys: {sorted(list(checkpoint_keys))[:30]}"
    )


def load_model(cfg, weights_path, gpu_id):
    """
    Where model weights are loaded:

        model = get_model(...)
        checkpoint = torch.load(weights_path, map_location=device)
        state_dict = extract_backbone_state_dict(checkpoint)
        model.load_state_dict(state_dict)

    This function supports both:
      - output/model.pt
      - output/checkpoint_gpu_0.pt
    """
    device = torch.device(
        f"cuda:{gpu_id}" if gpu_id >= 0 and torch.cuda.is_available() else "cpu"
    )

    logging.info(f"Creating backbone: network={cfg.network}, embedding_size={cfg.embedding_size}")
    model = get_model(
        cfg.network,
        dropout=0.0,
        fp16=False,
        num_features=cfg.embedding_size,
    ).to(device)

    logging.info(f"Loading weights from: {weights_path}")
    checkpoint = torch.load(weights_path, map_location=device)

    state_dict = extract_backbone_state_dict(checkpoint)
    state_dict = strip_known_prefixes(state_dict)

    try:
        model.load_state_dict(state_dict, strict=True)
        logging.info("Loaded weights with strict=True.")
    except RuntimeError as e:
        logging.warning("Strict weight loading failed. Trying strict=False.")
        logging.warning(str(e))

        incompatible = model.load_state_dict(state_dict, strict=False)
        missing = list(incompatible.missing_keys)
        unexpected = list(incompatible.unexpected_keys)

        logging.warning(f"Loaded with strict=False. Missing keys: {len(missing)}")
        logging.warning(f"Loaded with strict=False. Unexpected keys: {len(unexpected)}")

        if missing:
            logging.warning(f"First missing keys: {missing[:20]}")
        if unexpected:
            logging.warning(f"First unexpected keys: {unexpected[:20]}")

        if len(missing) > 20:
            raise RuntimeError(
                "Too many missing keys while loading model. "
                "This probably means the config network does not match the checkpoint."
            )

    model.eval()
    return model, device


def preprocess_image(img_path, image_size=112):
    img = cv2.imread(img_path)
    if img is None:
        return None

    img = cv2.resize(img, (image_size, image_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1)).astype(np.float32)
    img = (img - 127.5) / 127.5
    return img


def hist_to_uint8(final_hist, clip_val=5):
    img = np.clip(final_hist.astype(np.float32), -float(clip_val), float(clip_val))
    img = (img + float(clip_val)) / (2.0 * float(clip_val))
    img = np.squeeze(img)
    img = (img * 255.0).astype(np.uint8)
    return img


def preprocess_npz(npz_path, image_size=112, hist_key="final_hist", clip_val=5):
    try:
        with np.load(npz_path, allow_pickle=False) as data:
            if hist_key not in data:
                raise KeyError(f"{hist_key} not found. Available keys: {list(data.keys())}")
            final_hist = data[hist_key]

        img = hist_to_uint8(final_hist, clip_val=clip_val)

        # Convert HxW to HxWx3 to match the backbone input.
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        elif img.ndim == 3 and img.shape[-1] == 1:
            img = np.repeat(img, repeats=3, axis=-1)
        elif img.ndim == 3 and img.shape[-1] == 3:
            pass
        else:
            raise ValueError(f"Unsupported final_hist image shape after squeeze: {img.shape}")

        img = cv2.resize(img, (image_size, image_size))
        img = np.transpose(img, (2, 0, 1)).astype(np.float32)
        img = (img - 127.5) / 127.5
        return img

    except Exception as e:
        logging.warning(f"Could not read npz: {npz_path} -> {repr(e)}")
        return None


def maybe_limit_samples(samples, max_samples_per_identity=None, max_total_samples=None, seed=42):
    if max_samples_per_identity is not None and max_samples_per_identity > 0:
        grouped = defaultdict(list)
        for row in samples:
            grouped[row[1]].append(row)

        rng = random.Random(seed)
        limited = []
        for _label, rows in grouped.items():
            rows = list(rows)
            rng.shuffle(rows)
            limited.extend(rows[:max_samples_per_identity])

        samples = limited

    if max_total_samples is not None and max_total_samples > 0 and len(samples) > max_total_samples:
        rng = random.Random(seed)
        samples = rng.sample(samples, max_total_samples)

    return samples


def load_npz_index_dataset(index_file, max_samples_per_identity=None, max_total_samples=None):
    """
    Expected index format:
        /abs/path/to/file.npz<TAB>label

    Labels may be raw identities or contiguous labels.
    For embedding evaluation, either is fine.
    """
    samples = []

    with open(index_file, "r") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            if "\t" in line:
                path, label = line.rsplit("\t", 1)
            else:
                parts = line.rsplit(maxsplit=1)
                if len(parts) != 2:
                    raise ValueError(
                        f"Bad index format at line {line_no}. "
                        "Expected path<TAB>label or path label."
                    )
                path, label = parts

            label = int(label)
            identity_name = str(label)
            samples.append((path, label, identity_name, "npz"))

    samples = maybe_limit_samples(
        samples,
        max_samples_per_identity=max_samples_per_identity,
        max_total_samples=max_total_samples,
    )

    unique_labels = sorted({label for _path, label, _name, _kind in samples})
    label_to_name = {label: str(label) for label in unique_labels}

    return samples, label_to_name


def load_nested_image_dataset(root_dir, max_samples_per_identity=None, max_total_samples=None):
    """
    Expected structure:
        test_db/
            20210903/
                01/
                    img1.jpg
                    img2.jpg
                02/
                    img3.jpg

    Identity = outer_folder/inner_folder
    """
    identity_names = []

    outer_dirs = sorted(
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    )

    for outer in outer_dirs:
        outer_path = os.path.join(root_dir, outer)
        inner_dirs = sorted(
            d for d in os.listdir(outer_path)
            if os.path.isdir(os.path.join(outer_path, d))
        )

        for inner in inner_dirs:
            identity_names.append(f"{outer}")

    identity_names = sorted(set(identity_names))
    name_to_label = {name: idx for idx, name in enumerate(identity_names)}
    label_to_name = {idx: name for name, idx in name_to_label.items()}

    samples = []

    for outer in outer_dirs:
        outer_path = os.path.join(root_dir, outer)
        inner_dirs = sorted(
            d for d in os.listdir(outer_path)
            if os.path.isdir(os.path.join(outer_path, d))
        )

        for inner in inner_dirs:
            identity_name = f"{outer}"
            label = name_to_label[identity_name]
            inner_path = os.path.join(outer_path, inner)

            for fname in sorted(os.listdir(inner_path)):
                if fname.lower().endswith(VALID_IMAGE_EXTS):
                    samples.append(
                        (os.path.join(inner_path, fname), label, identity_name, "image")
                    )

    samples = maybe_limit_samples(
        samples,
        max_samples_per_identity=max_samples_per_identity,
        max_total_samples=max_total_samples,
    )

    return samples, label_to_name


def load_imagefolder_dataset(root_dir, max_samples_per_identity=None, max_total_samples=None):
    """
    Expected structure:
        test_db/
            identity_001/
                img1.jpg
            identity_002/
                img2.jpg

    Identity = direct parent folder.
    """
    identity_names = sorted(
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    )

    name_to_label = {name: idx for idx, name in enumerate(identity_names)}
    label_to_name = {idx: name for name, idx in name_to_label.items()}

    samples = []

    for identity_name in identity_names:
        label = name_to_label[identity_name]
        identity_dir = os.path.join(root_dir, identity_name)

        for fname in sorted(os.listdir(identity_dir)):
            if fname.lower().endswith(VALID_IMAGE_EXTS):
                samples.append(
                    (os.path.join(identity_dir, fname), label, identity_name, "image")
                )

    samples = maybe_limit_samples(
        samples,
        max_samples_per_identity=max_samples_per_identity,
        max_total_samples=max_total_samples,
    )

    return samples, label_to_name


def infer_data_type(test_db):
    if os.path.isfile(test_db):
        return "npz_index"

    if os.path.isdir(test_db):
        # If direct children contain image files, generic imagefolder may still be needed,
        # but nested_images is the original default for this script.
        return "nested_images"

    raise FileNotFoundError(f"test_db does not exist: {test_db}")


def load_eval_dataset(args):
    data_type = args.data_type
    if data_type == "auto":
        data_type = infer_data_type(args.test_db)

    logging.info(f"Evaluation data type: {data_type}")

    if data_type == "npz_index":
        return load_npz_index_dataset(
            args.test_db,
            max_samples_per_identity=args.max_samples_per_identity,
            max_total_samples=args.max_total_samples,
        )

    if data_type == "nested_images":
        return load_nested_image_dataset(
            args.test_db,
            max_samples_per_identity=args.max_samples_per_identity,
            max_total_samples=args.max_total_samples,
        )

    if data_type == "imagefolder":
        return load_imagefolder_dataset(
            args.test_db,
            max_samples_per_identity=args.max_samples_per_identity,
            max_total_samples=args.max_total_samples,
        )
    
    # if data_type == "event_benchmark":
    #     return load_event_benchmark_dataset(
    #         args.test_db,
    #         max_samples_per_identity=args.max_samples_per_identity,
    #         max_total_samples=args.max_total_samples,
    #     )

    raise ValueError(f"Unsupported data_type: {data_type}")


@torch.no_grad()
def extract_pair_embeddings(model, device, npz_files, batch_size, image_size, hist_key, clip_val):
    embs = [None] * len(npz_files)
    for start in tqdm(range(0, len(npz_files), batch_size), desc="Extracting embeddings"):
        batch_paths = npz_files[start:start + batch_size]
        imgs, idxs = [], []
        for j, path in enumerate(batch_paths):
            img = preprocess_npz(path, image_size=image_size, hist_key=hist_key, clip_val=clip_val)
            if img is None:
                continue
            imgs.append(img)
            idxs.append(start + j)
        if not imgs:
            continue
        t = torch.from_numpy(np.stack(imgs)).float().to(device, non_blocking=True)
        feats = l2_norm(model(t)).cpu()
        for k, global_i in enumerate(idxs):
            embs[global_i] = feats[k]
    return embs


@torch.no_grad()
def extract_embeddings(model, device, samples, batch_size, image_size, hist_key, clip_val):
    embeddings, labels, names, paths = [], [], [], []
    skipped = 0
    for start in tqdm(range(0, len(samples), batch_size), desc="Extracting embeddings"):
        batch = samples[start:start + batch_size]
        imgs, b_labels, b_names, b_paths = [], [], [], []
        for path, label, identity_name, kind in batch:
            if kind == "npz":
                img = preprocess_npz(path, image_size=image_size, hist_key=hist_key, clip_val=clip_val)
            else:
                img = preprocess_image(path, image_size=image_size)
            if img is None:
                skipped += 1
                continue
            imgs.append(img); b_labels.append(label); b_names.append(identity_name); b_paths.append(path)
        if not imgs:
            continue
        t = torch.from_numpy(np.stack(imgs)).float().to(device, non_blocking=True)
        feats = l2_norm(model(t)).cpu()
        embeddings.append(feats); labels.extend(b_labels); names.extend(b_names); paths.extend(b_paths)

    if not embeddings:
        raise RuntimeError("No embeddings extracted. Check test_db and preprocessing options.")
    embeddings = torch.cat(embeddings, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)
    if skipped:
        logging.warning(f"Skipped unreadable/malformed samples: {skipped}")
    return embeddings, labels, names, paths


def build_pairs(embs, issame):
    num_pairs = len(embs) // 2
    emb1, emb2, labels, dropped = [], [], [], 0
    for k in range(num_pairs):
        a, b = embs[2 * k], embs[2 * k + 1]
        if a is None or b is None:
            dropped += 1
            continue
        emb1.append(a); emb2.append(b); labels.append(bool(issame[k]))
    if not emb1:
        raise RuntimeError("No valid pairs after embedding extraction.")
    return torch.stack(emb1), torch.stack(emb2), np.array(labels, dtype=bool), dropped

def _acc_all_thresholds(scores, issame, thresholds):
    preds = scores[None, :] >= thresholds[:, None]      # (T, M)
    return np.mean(preds == issame[None, :], axis=1)    # (T,)

def tar_at_far(y_true, scores, far_targets=(1e-1, 1e-2, 1e-3, 1e-4)):
    """TAR (TPR) at fixed FAR points, from the fixed genuine/impostor scores."""
    y = np.asarray(y_true).astype(bool); sc = np.asarray(scores)
    pos, neg = sc[y], sc[~y]
    out = {}
    if len(pos) == 0 or len(neg) == 0:
        return {f"{f:.0e}": None for f in far_targets}
    neg_desc = np.sort(neg)[::-1]
    n_neg = len(neg)
    for far in far_targets:
        k = int(round(far * n_neg))
        out[f"{far:.0e}"] = float((pos >= neg_desc[k - 1]).mean()) if k >= 1 else None
    return out


def verification_10fold(scores, issame, nrof_folds=10, thresholds=None):
    """
    Standard InsightFace protocol on cosine scores: for each fold, pick the
    threshold that maximizes train accuracy, report it on the held-out fold.
    """
    if thresholds is None:
        thresholds = np.arange(-1.0, 1.0 + 1e-9, 0.001)
    issame = np.asarray(issame).astype(bool)
    scores = np.asarray(scores, dtype=np.float64)
    kf = KFold(n_splits=nrof_folds, shuffle=False)

    accuracies = np.zeros(nrof_folds)
    best_thresholds = np.zeros(nrof_folds)
    for fi, (train_idx, test_idx) in enumerate(kf.split(np.arange(len(scores)))):
        train_acc = _acc_all_thresholds(scores[train_idx], issame[train_idx], thresholds)
        best_t = thresholds[int(np.argmax(train_acc))]
        best_thresholds[fi] = best_t
        accuracies[fi] = np.mean((scores[test_idx] >= best_t) == issame[test_idx])
    return float(accuracies.mean()), float(accuracies.std()), float(best_thresholds.mean())

def compute_centroids(embeddings, labels):
    unique_labels = torch.unique(labels)
    centroids = []
    centroid_labels = []

    for c in tqdm(unique_labels.tolist(), desc="Computing centroids"):
        c = int(c)
        class_embs = embeddings[labels == c]
        centroid = F.normalize(class_embs.mean(dim=0, keepdim=True), dim=1)
        centroids.append(centroid)
        centroid_labels.append(c)

    centroids = torch.cat(centroids, dim=0)
    return centroids, centroid_labels


def nearest_centroid_eval_chunked(embeddings, labels, batch_size=4096, ks=(1, 5)):
    """
    Chunked nearest-centroid evaluation.

    Avoids building a huge [num_samples x num_identities] similarity matrix.
    """
    centroids, centroid_label_order = compute_centroids(embeddings, labels)

    pred_labels_all = []
    pred_scores_all = []

    rank_correct = {k: 0 for k in ks}
    total = labels.shape[0]

    for start in tqdm(range(0, total, batch_size), desc="Nearest-centroid eval"):
        end = min(start + batch_size, total)
        emb = embeddings[start:end]
        gt = labels[start:end]

        sims = emb @ centroids.T

        max_scores, max_idx = torch.max(sims, dim=1)
        batch_pred_labels = torch.tensor(
            [centroid_label_order[i] for i in max_idx.tolist()],
            dtype=torch.long,
        )

        pred_labels_all.append(batch_pred_labels)
        pred_scores_all.append(max_scores.cpu())

        for k in ks:
            topk = torch.topk(sims, k=min(k, sims.shape[1]), dim=1).indices

            for row_i in range(topk.shape[0]):
                top_labels = [centroid_label_order[j] for j in topk[row_i].tolist()]
                if int(gt[row_i].item()) in top_labels:
                    rank_correct[k] += 1

    pred_labels = torch.cat(pred_labels_all, dim=0)
    pred_scores = torch.cat(pred_scores_all, dim=0)

    closed_set_acc = (pred_labels == labels.cpu()).float().mean().item()
    rank_results = {f"rank_{k}": rank_correct[k] / total for k in ks}

    return closed_set_acc, pred_labels, pred_scores, rank_results


def save_predictions_csv(save_path, paths, labels, names, pred_labels, pred_scores, label_to_name):
    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "path",
            "gt_label",
            "gt_name",
            "pred_label",
            "pred_name",
            "score",
            "correct",
        ])

        for p, gt, gt_name, pred, score in zip(
            paths,
            labels.tolist(),
            names,
            pred_labels.tolist(),
            pred_scores.tolist(),
        ):
            pred_name = label_to_name.get(pred, str(pred))

            writer.writerow([
                p,
                gt,
                gt_name,
                pred,
                pred_name,
                score,
                int(gt == pred),
            ])


def save_per_class_accuracy(save_path, labels, pred_labels, label_to_name):
    rows = []

    for c in sorted(set(labels.tolist())):
        mask = labels == c
        total = int(mask.sum().item())
        correct = int((pred_labels[mask] == labels[mask]).sum().item())
        acc = correct / total if total > 0 else 0.0
        rows.append([c, label_to_name.get(c, str(c)), total, correct, acc])

    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "name", "total", "correct", "accuracy"])
        writer.writerows(rows)


def plot_confusion_matrix(save_path, labels, pred_labels, label_to_name, max_classes=100):
    unique_labels = sorted(set(labels.tolist()))

    if len(unique_labels) > max_classes:
        logging.info(
            f"Skipping confusion matrix: {len(unique_labels)} classes > max_classes={max_classes}"
        )
        return False

    cm = confusion_matrix(labels.numpy(), pred_labels.numpy(), labels=unique_labels)
    acc = accuracy_score(labels.numpy(), pred_labels.numpy())

    tick_names = [label_to_name.get(c, str(c)) for c in unique_labels]

    plt.figure(figsize=(12, 10))
    plt.imshow(cm)
    plt.title(f"Confusion Matrix | Accuracy = {acc * 100:.2f}%")
    plt.xlabel("Predicted")
    plt.ylabel("Ground Truth")
    plt.xticks(range(len(tick_names)), tick_names, rotation=90, fontsize=6)
    plt.yticks(range(len(tick_names)), tick_names, fontsize=6)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    return True


def build_label_to_indices(labels):
    label_to_indices = defaultdict(list)
    for idx, label in enumerate(labels.tolist()):
        label_to_indices[int(label)].append(idx)
    return label_to_indices


def compute_pairwise_scores(embeddings, labels, max_pairs=200000, seed=42):
    """
    Sample pairwise genuine/impostor cosine scores.

    This avoids enumerating all O(N^2) pairs, which is impossible for large
    Glint360K test_index files.
    """
    n = embeddings.shape[0]
    label_to_indices = build_label_to_indices(labels)
    labels_list = sorted(label_to_indices.keys())

    if len(labels_list) < 2:
        logging.warning("Pairwise metrics require at least two identities. Skipping.")
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

    rng = random.Random(seed)

    if max_pairs is None or max_pairs <= 0:
        if n > 10000:
            logging.warning(
                "max_pairs was unlimited but N is large. Using 200000 sampled pairs for safety."
            )
            max_pairs = 200000
        else:
            # Small dataset: generate all pairs.
            y_true = []
            scores = []
            for i in tqdm(range(n), desc="Computing all pairwise scores"):
                for j in range(i + 1, n):
                    sim = torch.dot(embeddings[i], embeddings[j]).item()
                    same = int(labels[i].item() == labels[j].item())
                    scores.append(sim)
                    y_true.append(same)
            return np.array(y_true), np.array(scores)

    num_positive = max_pairs // 2
    num_negative = max_pairs - num_positive

    positive_capable_labels = [lab for lab, idxs in label_to_indices.items() if len(idxs) >= 2]

    y_true = []
    scores = []

    # Positive/genuine pairs.
    if positive_capable_labels:
        for _ in tqdm(range(num_positive), desc="Sampling genuine pairs"):
            lab = rng.choice(positive_capable_labels)
            i, j = rng.sample(label_to_indices[lab], 2)
            sim = torch.dot(embeddings[i], embeddings[j]).item()
            scores.append(sim)
            y_true.append(1)
    else:
        logging.warning("No identities have >=2 samples; cannot sample genuine pairs.")

    # Negative/impostor pairs.
    for _ in tqdm(range(num_negative), desc="Sampling impostor pairs"):
        lab1, lab2 = rng.sample(labels_list, 2)
        i = rng.choice(label_to_indices[lab1])
        j = rng.choice(label_to_indices[lab2])
        sim = torch.dot(embeddings[i], embeddings[j]).item()
        scores.append(sim)
        y_true.append(0)

    return np.array(y_true), np.array(scores)


def save_pairwise_csv(save_path, y_true, scores):
    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["same_identity", "similarity"])
        for y, s in zip(y_true, scores):
            writer.writerow([int(y), float(s)])


def plot_roc(save_path, y_true, scores):
    if len(np.unique(y_true)) < 2:
        logging.warning("ROC requires both positive and negative pairs. Skipping ROC.")
        return None, None, None

    fpr, tpr, thresholds = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)

    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fnr - fpr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
    eer_threshold = thresholds[eer_idx]

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}, EER = {eer:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    return float(roc_auc), float(eer), float(eer_threshold)


def plot_pr(save_path, y_true, scores):
    if len(np.unique(y_true)) < 2:
        logging.warning("PR requires both positive and negative pairs. Skipping PR.")
        return None

    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    ap = average_precision_score(y_true, scores)

    plt.figure(figsize=(7, 6))
    plt.plot(recall, precision, label=f"AP = {ap:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    return float(ap)


def plot_similarity_histogram(save_path, y_true, scores):
    if len(y_true) == 0:
        logging.warning("No pairwise scores available. Skipping histogram.")
        return False

    genuine = scores[y_true == 1]
    impostor = scores[y_true == 0]

    plt.figure(figsize=(8, 6))

    if len(impostor) > 0:
        plt.hist(impostor, bins=60, alpha=0.6, label="Impostor / Different Identity")
    if len(genuine) > 0:
        plt.hist(genuine, bins=60, alpha=0.6, label="Genuine / Same Identity")

    plt.xlabel("Cosine Similarity")
    plt.ylabel("Frequency")
    plt.title("Genuine vs Impostor Similarity Distribution")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    return True

def evaluate_verification(model, device, args, bench_dir, output_dir, benchmark_name=None):
    os.makedirs(output_dir, exist_ok=True)
    tag = benchmark_name or os.path.basename(bench_dir.rstrip("/"))

    npz_files, issame = load_event_verification_dataset(bench_dir)
    num_pairs = len(npz_files) // 2
    logging.info(f"[{tag}] pairs={num_pairs:,} | files={len(npz_files):,} | "
                 f"same={int(np.sum(issame))} diff={int(np.sum(~issame))}")

    embs = extract_pair_embeddings(
        model, device, npz_files,
        batch_size=args.batch_size, image_size=args.image_size,
        hist_key=args.hist_key, clip_val=args.clip_val,
    )
    emb1, emb2, labels, dropped = build_pairs(embs, issame)
    if dropped:
        logging.warning(f"[{tag}] Dropped {dropped} pair(s) with unreadable npz files.")

    cos = torch.sum(emb1 * emb2, dim=1).numpy()   # both L2-normalized -> cosine similarity
    y_true = labels.astype(np.int64)

    acc_mean, acc_std, mean_threshold = verification_10fold(cos, labels, nrof_folds=args.nrof_folds)
    tar = tar_at_far(y_true, cos)

    save_pairwise_csv(os.path.join(output_dir, "pairwise_scores.csv"), y_true, cos)
    roc_auc, eer, eer_threshold = plot_roc(os.path.join(output_dir, "roc_curve.png"), y_true, cos)
    average_precision = plot_pr(os.path.join(output_dir, "pr_curve.png"), y_true, cos)
    plot_similarity_histogram(os.path.join(output_dir, "similarity_histogram.png"), y_true, cos)

    metrics = {
        "benchmark": benchmark_name,
        "protocol": "insightface_10fold_verification",
        "test_db": bench_dir,
        "weights": args.weights,
        "num_pairs": int(len(labels)),
        "num_same": int(labels.sum()),
        "num_diff": int((~labels).sum()),
        "tar_at_far": tar,
        "verification_accuracy_mean": acc_mean,
        "verification_accuracy_std": acc_std,
        "mean_best_threshold_cosine": mean_threshold,
        "roc_auc": roc_auc,
        "eer": eer,
        "eer_threshold_cosine": eer_threshold,
        "average_precision": average_precision,
        "dropped_pairs": int(dropped),
    }
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    logging.info("=" * 70)
    logging.info(f"[{tag}] Verification Acc: {acc_mean*100:.2f}% +/- {acc_std*100:.2f}%  "
                 f"(mean thr={mean_threshold:.4f})")
    if roc_auc is not None:
        logging.info(f"[{tag}] AUC: {roc_auc:.4f} | EER: {eer:.4f} | AP: {average_precision:.4f}")
    logging.info(f"[{tag}] " + " | ".join(
        f"TAR@FAR={k}:{v:.4f}" for k, v in tar.items() if v is not None))
    logging.info(f"[{tag}] Saved to: {output_dir}")
    logging.info("=" * 70)
    return metrics


def evaluate_identification(model, device, args, samples, label_to_name, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    if len(samples) == 0:
        raise RuntimeError("No samples found in evaluation dataset.")

    logging.info(f"Identities/classes: {len(label_to_name):,} | Samples: {len(samples):,}")

    embeddings, labels, names, paths = extract_embeddings(
        model=model, device=device, samples=samples,
        batch_size=args.batch_size, image_size=args.image_size,
        hist_key=args.hist_key, clip_val=args.clip_val,
    )

    if not args.no_save_embeddings:
        np.savez(os.path.join(output_dir, "embeddings.npz"),
                 embeddings=embeddings.numpy(), labels=labels.numpy(),
                 names=np.array(names), paths=np.array(paths))

    closed_set_acc, pred_labels, pred_scores, rank_results = nearest_centroid_eval_chunked(
        embeddings=embeddings, labels=labels, batch_size=args.centroid_batch_size, ks=(1, 5))

    save_predictions_csv(os.path.join(output_dir, "predictions.csv"),
                         paths, labels, names, pred_labels, pred_scores, label_to_name)
    save_per_class_accuracy(os.path.join(output_dir, "per_class_accuracy.csv"),
                            labels, pred_labels, label_to_name)
    confusion_saved = plot_confusion_matrix(os.path.join(output_dir, "confusion_matrix.png"),
                                            labels, pred_labels, label_to_name,
                                            max_classes=args.max_confusion_classes)

    y_true, pair_scores = compute_pairwise_scores(embeddings, labels,
                                                  max_pairs=args.max_pairs, seed=args.seed)
    if len(y_true) > 0:
        save_pairwise_csv(os.path.join(output_dir, "pairwise_scores.csv"), y_true, pair_scores)
    roc_auc, eer, eer_threshold = plot_roc(os.path.join(output_dir, "roc_curve.png"), y_true, pair_scores)
    average_precision = plot_pr(os.path.join(output_dir, "pr_curve.png"), y_true, pair_scores)
    plot_similarity_histogram(os.path.join(output_dir, "similarity_histogram.png"), y_true, pair_scores)

    metrics = {
        "data_type": args.data_type, "test_db": args.test_db, "weights": args.weights,
        "num_identities": len(label_to_name), "num_samples": int(labels.shape[0]),
        "closed_set_nearest_centroid_accuracy": float(closed_set_acc),
        "rank_1_accuracy": float(rank_results.get("rank_1", 0.0)),
        "rank_5_accuracy": float(rank_results.get("rank_5", 0.0)),
        "pairwise_num_pairs": int(len(y_true)),
        "roc_auc": roc_auc, "eer": eer, "eer_threshold": eer_threshold,
        "average_precision": average_precision,
    }
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    logging.info("=" * 70)
    logging.info(f"Nearest-centroid Acc: {closed_set_acc*100:.2f}% | "
                 f"Rank-1: {rank_results['rank_1']*100:.2f}% | Rank-5: {rank_results['rank_5']*100:.2f}%")
    if roc_auc is not None:
        logging.info(f"AUC: {roc_auc:.4f} | EER: {eer:.4f} | AP: {average_precision:.4f}")
    logging.info(f"Saved outputs to: {output_dir}")
    logging.info("=" * 70)
    return metrics


def main(args):
    setup_logging()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = os.path.join(args.output_dir, f"eval_{timestamp}")
    os.makedirs(base_output_dir, exist_ok=True)
    logging.info(f"Writing evaluation outputs to: {base_output_dir}")

    cfg = get_config(args.config)
    if args.gpu >= 0 and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)

    logging.info("Loading model...")
    model, device = load_model(cfg, args.weights, args.gpu)

    # Verification mode: ordered .npz + issame.bin (single folder or parent-of-folders).
    if args.data_type in ("event_suite", "event_verification"):
        if args.data_type == "event_verification":
            benchmark_dirs = [args.test_db]
        else:
            benchmark_dirs = [
                os.path.join(args.test_db, d) for d in sorted(os.listdir(args.test_db))
                if os.path.isdir(os.path.join(args.test_db, d))
            ]
        if not benchmark_dirs:
            raise FileNotFoundError(f"No benchmark folders found under: {args.test_db}")

        suite_summary = {}
        for bench_dir in benchmark_dirs:
            name = os.path.basename(bench_dir.rstrip("/"))
            logging.info(f"===== Verification benchmark: {name} =====")
            suite_summary[name] = evaluate_verification(
                model, device, args, bench_dir,
                os.path.join(base_output_dir, name), benchmark_name=name,
            )

        with open(os.path.join(base_output_dir, "summary.json"), "w") as f:
            json.dump(suite_summary, f, indent=4)

        logging.info("=" * 70)
        logging.info("SUITE SUMMARY")
        for name, m in suite_summary.items():
            auc_v = m["roc_auc"] if m["roc_auc"] is not None else float("nan")
            eer_v = m["eer"] if m["eer"] is not None else float("nan")
            logging.info(f"{name:16s} | acc={m['verification_accuracy_mean']*100:6.2f}% "
                         f"+/-{m['verification_accuracy_std']*100:4.2f} | AUC={auc_v:.4f} | EER={eer_v:.4f}")
        logging.info("=" * 70)
        return

    # Identification / embedding mode (nested_images / imagefolder / npz_index).
    logging.info("Loading evaluation dataset...")
    samples, label_to_name = load_eval_dataset(args)
    evaluate_identification(model, device, args, samples, label_to_name, base_output_dir)
    # # Single-dataset mode.
    # logging.info("Loading evaluation dataset...")
    # samples, label_to_name = load_eval_dataset(args)
    # evaluate_verification(model, device, cfg, args, samples, label_to_name, base_output_dir)

    
    # Single-dataset mode (npz_index / imagefolder / nested_images) was replaced by the
    # # verification pipeline and is no longer wired up.
    # raise NotImplementedError(
    #     f"data_type '{args.data_type}' is not supported in this build. "
    #     "Use --data_type event_verification (single benchmark folder) or "
    #     "event_suite (parent folder of benchmark subfolders)."
    # )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate ArcFace backbone on .npz index data or image-folder data."
    )

    parser.add_argument("config", type=str, help="Path to config file")

    parser.add_argument("--gpu", type=int, default=0)

    parser.add_argument("--nrof_folds", type=int, default=10,
                        help="Folds for verification cross-validation (default 10).")

    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to model.pt or checkpoint_gpu_*.pt",
    )

    parser.add_argument(
        "--data_type",
        type=str,
        default="auto",
        choices=["auto", "npz_index", "nested_images", "imagefolder",
                 "event_verification", "event_suite"],
        help="event_verification: a single benchmark folder (ordered .npz + issame.bin); "
             "event_suite: parent eval folder holding several such benchmark subfolders.",
    )

    parser.add_argument(
        "--test_db",
        type=str,
        required=True,
        help="Either test_index.txt for .npz data or a test image folder",
    )

    parser.add_argument("--batch_size", type=int, default=64)

    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--image_size", type=int, default=112)

    parser.add_argument("--hist_key", type=str, default="final_hist")

    parser.add_argument("--clip_val", type=float, default=5.0)

    parser.add_argument(
        "--max_pairs",
        type=int,
        default=200000,
        help="Number of sampled pairwise comparisons for ROC/PR. Use -1 for all pairs on small data.",
    )

    parser.add_argument(
        "--centroid_batch_size",
        type=int,
        default=4096,
        help="Chunk size for nearest-centroid similarity computation.",
    )

    parser.add_argument(
        "--max_confusion_classes",
        type=int,
        default=100,
        help="Skip confusion matrix when class count is larger than this.",
    )

    parser.add_argument(
        "--max_samples_per_identity",
        type=int,
        default=None,
        help="Optional cap per identity for faster debugging/evaluation.",
    )

    parser.add_argument(
        "--max_total_samples",
        type=int,
        default=None,
        help="Optional total sample cap for faster debugging/evaluation.",
    )

    parser.add_argument(
        "--no_save_embeddings",
        action="store_true",
        help="Do not save embeddings.npz. Useful for very large test sets.",
    )

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.max_pairs == -1:
        args.max_pairs = None

    main(args)
