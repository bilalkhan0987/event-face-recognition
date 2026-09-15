#!/usr/bin/env python
"""
Fine-tune an ArcFace / PartialFC backbone on real FES nested-jpg data, starting
from a pretrained model.pt, and save the fine-tuned backbone as model_FT.pt.

Based on the distributed PartialFC training script, with ONE change to the data
path: instead of reading event .npz via get_dataloader, it scans a nested image
folder and derives the class from the OUTER (date_format) folder.

Expected layout (--rec / cfg.rec):
    FES_frames_train/
        20210931_0/            <- CLASS (subject)
            0/  img_000.jpg ...   \
            1/  ...                > sessions 0..15 = same subject = same class
            15/ ...               /
        20211103_1/
            0/ ...

Class count and image count are discovered from the folder tree and override
cfg.num_classes / cfg.num_image automatically, so any placeholder values in the
config file are ignored.

Single GPU:
    python finetune_fes_partialfc.py configs/fes_finetune.py

Multi-GPU (torchrun), same as your training entrypoint:
    torchrun --nproc_per_node=4 finetune_fes_partialfc.py configs/fes_finetune.py

Re-test afterwards:
    python test_bk_v2.py configs/fes_finetune.py --weights <output>/model_FT.pt \
        --test_db /path/to/val --data_type nested_images --output_dir eval_FT
"""

import os
import argparse
import logging
from datetime import datetime

import csv
import random
from collections import defaultdict
from sklearn.metrics import average_precision_score, roc_curve, auc
import torch.nn.functional as F

import cv2
import numpy as np
import torch
from torch import distributed
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import fp16_compress_hook

from backbones import get_model
from losses import CombinedMarginLoss
from lr_scheduler import PolynomialLRWarmup
from partial_fc_v2 import PartialFC_V2
from utils.utils_callbacks import CallBackLogging
from utils.utils_config import get_config
from utils.utils_distributed_sampler import setup_seed
from utils.utils_logging import AverageMeter, init_logging


assert torch.__version__ >= "1.12.0"

VALID_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


# --------------------------------------------------------------------------- #
# Distributed helpers (unchanged from the training script)
# --------------------------------------------------------------------------- #
def setup_distributed():
    if "RANK" in os.environ and "LOCAL_RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        distributed.init_process_group(backend="nccl")
        use_ddp = True
    else:
        rank, local_rank, world_size, use_ddp = 0, 0, 1, False
    return rank, local_rank, world_size, use_ddp


def cleanup():
    if distributed.is_available() and distributed.is_initialized():
        try:
            distributed.destroy_process_group()
        except Exception:
            pass


def unwrap_model(model, use_ddp):
    return model.module if use_ddp else model


def get_cfg_value(cfg, key, default=None):
    if hasattr(cfg, key):
        return getattr(cfg, key)
    try:
        return cfg[key]
    except Exception:
        return default


def set_cfg_value(cfg, key, value):
    try:
        setattr(cfg, key, value)
    except Exception:
        cfg[key] = value


# --------------------------------------------------------------------------- #
# Pretrained backbone loading (matches test_bk_v2.py so weights load identically)
# --------------------------------------------------------------------------- #
def strip_known_prefixes(state_dict):
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
    if not isinstance(checkpoint, dict):
        raise TypeError("Loaded weights are not a dictionary/state_dict.")
    for key in ("state_dict_backbone", "backbone_state_dict", "model_state_dict",
                "state_dict", "backbone", "model"):
        if key in checkpoint and isinstance(checkpoint[key], dict):
            return checkpoint[key]
    if sum(torch.is_tensor(v) for v in checkpoint.values()) > 0:
        return checkpoint
    raise RuntimeError("Could not find a usable backbone state_dict in the weights file.")


def load_pretrained_into_backbone(backbone_module, pretrained_path, rank):
    if not pretrained_path or not os.path.isfile(pretrained_path):
        raise FileNotFoundError(
            f"cfg.pretrained must point to the starting model.pt (got: {pretrained_path}). "
            "Fine-tuning requires a pretrained backbone."
        )
    ckpt = torch.load(pretrained_path, map_location="cpu")
    sd = strip_known_prefixes(extract_backbone_state_dict(ckpt))
    incompatible = backbone_module.load_state_dict(sd, strict=False)
    if rank == 0:
        logging.info(f"Loaded pretrained backbone from {pretrained_path}")
        if incompatible.missing_keys:
            logging.warning(f"Missing keys: {len(incompatible.missing_keys)} "
                            f"(first: {incompatible.missing_keys[:8]})")
        if incompatible.unexpected_keys:
            logging.warning(f"Unexpected keys: {len(incompatible.unexpected_keys)} "
                            f"(first: {incompatible.unexpected_keys[:8]})")


def _augment(img, rng):
    h, w = img.shape[:2]
    if rng.random() < 0.5:                                   # random resized crop
        scale = rng.uniform(0.8, 1.0)
        nh, nw = int(h * scale), int(w * scale)
        top, left = rng.randint(0, h - nh), rng.randint(0, w - nw)
        img = cv2.resize(img[top:top + nh, left:left + nw], (w, h))
    if rng.random() < 0.5:                                   # small rotation
        M = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-10, 10), 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    if rng.random() < 0.5:                                   # brightness/contrast
        img = np.clip(img.astype(np.float32) * rng.uniform(0.8, 1.2)
                      + rng.uniform(-20, 20), 0, 255).astype(np.uint8)
    return img


def _worker_init_fn(worker_id):
    seed = (torch.initial_seed() + worker_id) % (2 ** 32)
    np.random.seed(seed); random.seed(seed)

# --------------------------------------------------------------------------- #
# FES nested-folder dataset  (class = outer date_format folder)
# --------------------------------------------------------------------------- #
def build_identity_name(outer, inner, identity_level):
    if identity_level == "outer":
        return outer
    if identity_level == "inner":
        return inner
    return f"{outer}/{inner}"


class FESNestedDataset(Dataset):
    """
    Walks root/<outer>/<inner>/*.jpg (or root/<outer>/*.jpg) and assigns a
    contiguous class label per identity. Exposes .labels and .label_to_name.
    Preprocessing matches the eval script: resize -> RGB -> (x-127.5)/127.5.
    """

    def __init__(self, root_dir, identity_level="outer", image_size=112,
                 train=True, flip=True, augment = False):
        self.image_size = image_size
        self.train = train
        self.flip = flip
        self.augment = augment

        outer_dirs = sorted(d for d in os.listdir(root_dir)
                            if os.path.isdir(os.path.join(root_dir, d)))
        if not outer_dirs:
            raise FileNotFoundError(f"No subject folders found in {root_dir}")

        raw = []  # (path, identity_name)
        for outer in outer_dirs:
            outer_path = os.path.join(root_dir, outer)
            inner_dirs = sorted(d for d in os.listdir(outer_path)
                                if os.path.isdir(os.path.join(outer_path, d)))
            if inner_dirs:
                for inner in inner_dirs:
                    inner_path = os.path.join(outer_path, inner)
                    name = build_identity_name(outer, inner, identity_level)
                    for f in sorted(os.listdir(inner_path)):
                        if f.lower().endswith(VALID_IMAGE_EXTS):
                            raw.append((os.path.join(inner_path, f), name))
            else:
                # images directly under the subject folder
                for f in sorted(os.listdir(outer_path)):
                    if f.lower().endswith(VALID_IMAGE_EXTS):
                        raw.append((os.path.join(outer_path, f), outer))

        if not raw:
            raise FileNotFoundError(f"No images found under {root_dir}")

        names = sorted({n for _p, n in raw})
        name_to_label = {n: i for i, n in enumerate(names)}
        self.label_to_name = {i: n for n, i in name_to_label.items()}
        self.samples = [(p, name_to_label[n]) for p, n in raw]
        self.labels = [lab for _p, lab in self.samples]

    @property
    def num_classes(self):
        return len(self.label_to_name)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv2.imread(path)
        if img is None:
            return None
        img = cv2.resize(img, (self.image_size, self.image_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.train and self.augment:
            img = _augment(img, random)
        if self.train and self.flip and random.random() < 0.5:
            img = img[:, ::-1, :]
        img = np.transpose(img, (2, 0, 1)).astype(np.float32)
        img = (img - 127.5) / 127.5
        return torch.from_numpy(np.ascontiguousarray(img)), label


def collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    imgs = torch.stack([b[0] for b in batch])
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return imgs, labels


def build_fes_dataloader(cfg, local_rank, world_size, use_ddp, identity_level, image_size):
    dataset = FESNestedDataset(
        root_dir=cfg.rec,
        identity_level=identity_level,
        image_size=image_size,
        train=True,
        flip=True,
        augment=get_cfg_value(cfg, "augment", True)
    )

    sampler = None
    if use_ddp and world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=local_rank,
                                     shuffle=True, drop_last=True)

    loader = DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=(sampler is None), sampler=sampler,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
        collate_fn=collate_skip_none, worker_init_fn=_worker_init_fn,
    )
    return loader, dataset


# --------------------------------------------------------------------------- #
# Helper
# ---------------------------------------------------------------------------- #

def scan_fes_with_sessions(root_dir, identity_level="outer"):
    """Returns samples=[(path, subj_label)], sessions_global=[int], aligned."""
    outer_dirs = sorted(d for d in os.listdir(root_dir)
                        if os.path.isdir(os.path.join(root_dir, d)))
    rows = []  # (path, subj_name, session_name)
    for outer in outer_dirs:
        op = os.path.join(root_dir, outer)
        inner_dirs = sorted(d for d in os.listdir(op) if os.path.isdir(os.path.join(op, d)))
        if inner_dirs:
            for inner in inner_dirs:
                ip = os.path.join(op, inner)
                subj = build_identity_name(outer, inner, identity_level)
                for f in sorted(os.listdir(ip)):
                    if f.lower().endswith(VALID_IMAGE_EXTS):
                        rows.append((os.path.join(ip, f), subj, f"{outer}/{inner}"))
        else:
            for f in sorted(os.listdir(op)):
                if f.lower().endswith(VALID_IMAGE_EXTS):
                    rows.append((os.path.join(op, f), outer, outer))
    if not rows:
        raise FileNotFoundError(f"No images under {root_dir}")
    subj_names = sorted({r[1] for r in rows})
    sess_names = sorted({r[2] for r in rows})
    subj_to_lab = {n: i for i, n in enumerate(subj_names)}
    sess_to_id = {n: i for i, n in enumerate(sess_names)}
    samples = [(p, subj_to_lab[s]) for p, s, _ in rows]
    sessions = [sess_to_id[se] for _, _, se in rows]
    return samples, sessions


def _select_subset(samples, sessions, max_n, seed=42):
    if max_n is None or max_n <= 0 or len(samples) <= max_n:
        return samples, sessions
    idx = list(range(len(samples)))
    random.Random(seed).shuffle(idx)
    idx = idx[:max_n]
    return [samples[i] for i in idx], [sessions[i] for i in idx]


@torch.no_grad()
def _embed(module, samples, sessions, image_size, batch_size, device, flip_tta=True):
    module.eval()
    embs, labs, sess = [], [], []
    buf_i, buf_l, buf_s = [], [], []

    def flush():
        if not buf_i:
            return
        t = torch.stack(buf_i).to(device, non_blocking=True)
        f = module(t).float()
        if flip_tta:
            f = f + module(torch.flip(t, dims=[3])).float()   # image + horizontal flip
        f = F.normalize(f).cpu()
        embs.append(f); labs.extend(buf_l); sess.extend(buf_s)
        buf_i.clear(); buf_l.clear(); buf_s.clear()

    for (path, label), sg in zip(samples, sessions):
        img = cv2.imread(path)
        if img is None:
            continue
        img = cv2.resize(img, (image_size, image_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1)).astype(np.float32)
        img = (img - 127.5) / 127.5
        buf_i.append(torch.from_numpy(np.ascontiguousarray(img)))
        buf_l.append(label); buf_s.append(sg)
        if len(buf_i) >= batch_size:
            flush()
    flush()
    if not embs:
        return None, None, None
    return torch.cat(embs, 0), np.array(labs), np.array(sess)


def pairwise_ap(embs, subj, sess, max_pairs=20000, seed=42, cross_session=True):
    rng = random.Random(seed)
    by_subj = defaultdict(list)
    for i, s in enumerate(subj):
        by_subj[int(s)].append(i)
    all_subj = list(by_subj.keys())
    pos_subj = [s for s, ix in by_subj.items() if len(ix) >= 2]
    if len(all_subj) < 2 or not pos_subj:
        return None
    npos = max_pairs // 2
    y, sc = [], []
    for _ in range(npos):
        s = rng.choice(pos_subj)
        ix = by_subj[s]
        i, j = rng.sample(ix, 2)
        if cross_session:
            tries = 0
            while sess[i] == sess[j] and tries < 10:
                j = rng.choice(ix); tries += 1
        sc.append(float(torch.dot(embs[i], embs[j]))); y.append(1)
    for _ in range(max_pairs - npos):
        a, b = rng.sample(all_subj, 2)
        i = rng.choice(by_subj[a]); j = rng.choice(by_subj[b])
        sc.append(float(torch.dot(embs[i], embs[j]))); y.append(0)
    y, sc = np.array(y), np.array(sc)
    if len(np.unique(y)) < 2:
        return None
    fpr, tpr, _ = roc_curve(y, sc)
    fnr = 1 - tpr
    k = np.nanargmin(np.abs(fnr - fpr))
    return {"ap": float(average_precision_score(y, sc)),
            "auc": float(auc(fpr, tpr)),
            "eer": float((fpr[k] + fnr[k]) / 2),
            "num_pairs": int(len(y))}



def template_ap(embs, subj, sess, max_pairs=20000, seed=42):
    """Per-(subject,session) mean-embedding templates; cross-session genuine pairs."""
    groups = defaultdict(list)
    for i in range(len(subj)):
        groups[(int(subj[i]), int(sess[i]))].append(i)

    tmpl, tsubj = [], []
    for (s, _se), idxs in groups.items():
        v = F.normalize(embs[idxs].mean(dim=0), dim=0)
        tmpl.append(v); tsubj.append(s)
    if len(tmpl) < 2:
        return None
    tmpl = torch.stack(tmpl)
    tsubj = np.array(tsubj)

    by_subj = defaultdict(list)
    for i, s in enumerate(tsubj):
        by_subj[int(s)].append(i)
    pos_subj = [s for s, ix in by_subj.items() if len(ix) >= 2]  # >=2 sessions
    all_subj = list(by_subj.keys())
    if len(all_subj) < 2 or not pos_subj:
        return None

    rng = random.Random(seed)
    npos = max_pairs // 2
    y, sc = [], []
    for _ in range(npos):
        s = rng.choice(pos_subj)
        i, j = rng.sample(by_subj[s], 2)
        sc.append(float(torch.dot(tmpl[i], tmpl[j]))); y.append(1)
    for _ in range(max_pairs - npos):
        a, b = rng.sample(all_subj, 2)
        i = rng.choice(by_subj[a]); j = rng.choice(by_subj[b])
        sc.append(float(torch.dot(tmpl[i], tmpl[j]))); y.append(0)

    y, sc = np.array(y), np.array(sc)
    if len(np.unique(y)) < 2:
        return None
    fpr, tpr, _ = roc_curve(y, sc)
    fnr = 1 - tpr
    k = np.nanargmin(np.abs(fnr - fpr))
    return {"ap": float(average_precision_score(y, sc)), "auc": float(auc(fpr, tpr)),
            "eer": float((fpr[k] + fnr[k]) / 2),
            "num_templates": int(len(tmpl)), "num_pairs": int(len(y))}


def run_ap(module, samples, sessions, cfg, image_size, device, tag, seed=42):
    emb, lab, ses = _embed(module, samples, sessions, image_size, cfg.batch_size, device,
                           flip_tta=get_cfg_value(cfg, "ap_flip_tta", True))
    if emb is None:
        return None
    mp = get_cfg_value(cfg, "ap_max_pairs", 20000)
    frame = pairwise_ap(emb, lab, ses, max_pairs=mp, seed=seed,
                        cross_session=get_cfg_value(cfg, "ap_cross_session", True))
    tmpl = template_ap(emb, lab, ses, max_pairs=mp, seed=seed)
    if frame:
        logging.info(f"[{tag}] frame    AP={frame['ap']:.4f} AUC={frame['auc']:.4f} EER={frame['eer']:.4f}")
    if tmpl:
        logging.info(f"[{tag}] template AP={tmpl['ap']:.4f} AUC={tmpl['auc']:.4f} EER={tmpl['eer']:.4f} "
                     f"(templates={tmpl['num_templates']})")
    return {"frame": frame, "template": tmpl}


def apply_freezing(backbone_module, freeze_until, rank):
    """Freeze early iresnet stages. freeze_until in {none,stem,layer1,layer2,layer3,layer4}."""
    if not freeze_until or freeze_until == "none":
        return []
    order = ["conv1", "bn1", "prelu", "layer1", "layer2", "layer3", "layer4"]
    stage_idx = {"stem": 2, "layer1": 3, "layer2": 4, "layer3": 5, "layer4": 6}
    if freeze_until not in stage_idx:
        if rank == 0:
            logging.warning(f"freeze_until='{freeze_until}' unrecognized; not freezing.")
        return []
    freeze_names = set(order[:stage_idx[freeze_until] + 1])
    frozen_modules, n_frozen = [], 0
    for name, module in backbone_module.named_children():
        if name in freeze_names:
            for p in module.parameters():
                p.requires_grad_(False); n_frozen += p.numel()
            module.eval(); frozen_modules.append(module)
    if rank == 0:
        n_train = sum(p.numel() for p in backbone_module.parameters() if p.requires_grad)
        logging.info(f"Freeze up to '{freeze_until}': frozen={n_frozen:,} trainable={n_train:,} | "
                     f"frozen stages present: {sorted(n for n in freeze_names)}")
    return frozen_modules

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(args, rank, local_rank, world_size, use_ddp):
    cfg = get_config(args.config)

    identity_level = args.identity_level or get_cfg_value(cfg, "identity_level", "outer")
    image_size = get_cfg_value(cfg, "image_size", 112)
    pretrained = args.pretrained or get_cfg_value(cfg, "pretrained", None)

    setup_seed(seed=cfg.seed, cuda_deterministic=False)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this training script.")
    torch.cuda.set_device(local_rank)

    os.makedirs(cfg.output, exist_ok=True)
    init_logging(rank, cfg.output)

    summary_writer = (SummaryWriter(log_dir=os.path.join(cfg.output, "tensorboard"))
                      if rank == 0 else None)

    # ---- data + auto count -------------------------------------------------
    train_loader, dataset = build_fes_dataloader(
        cfg, local_rank, world_size, use_ddp, identity_level, image_size)

    actual_num_classes = dataset.num_classes
    actual_num_images = len(dataset)

    if rank == 0:
        logging.info("=" * 90)
        logging.info("FES DATASET SCAN")
        logging.info(f"root:                 {cfg.rec}")
        logging.info(f"identity_level:       {identity_level}  (class = outer date_format folder)")
        logging.info(f"images found:         {actual_num_images:,}")
        logging.info(f"classes found:        {actual_num_classes:,}")
        logging.info(f"cfg.num_classes:      {int(get_cfg_value(cfg,'num_classes',-1)):,} -> {actual_num_classes:,}")
        logging.info(f"cfg.num_image:        {int(get_cfg_value(cfg,'num_image',-1)):,} -> {actual_num_images:,}")
        logging.info("=" * 90)

    set_cfg_value(cfg, "num_classes", actual_num_classes)
    set_cfg_value(cfg, "num_image", actual_num_images)

    if use_ddp and world_size > 1:
        distributed.barrier()

    # ---- backbone (init from pretrained model.pt) --------------------------
    backbone = get_model(cfg.network, dropout=0.0, fp16=cfg.fp16,
                         num_features=cfg.embedding_size).cuda(local_rank)
    load_pretrained_into_backbone(backbone, pretrained, rank)
    freeze_until = args.freeze_until or get_cfg_value(cfg, "freeze_until", "none")
    frozen_modules = apply_freezing(backbone, freeze_until, rank)
    backbone.train()

    if use_ddp:
        backbone = torch.nn.parallel.DistributedDataParallel(
            module=backbone, broadcast_buffers=False, device_ids=[local_rank],
            bucket_cap_mb=16, find_unused_parameters=True)
        backbone.register_comm_hook(None, fp16_compress_hook)
        try:
            backbone._set_static_graph()
        except Exception:
            pass

    # ---- margin + PartialFC (fresh classifier for FES classes) -------------
    margin_loss = CombinedMarginLoss(
        64,
        cfg.margin_list[0],
        cfg.margin_list[1],
        cfg.margin_list[2],
        cfg.interclass_filtering_threshold,
    )
    module_partial_fc = PartialFC_V2(
        margin_loss, cfg.embedding_size, cfg.num_classes, cfg.sample_rate, False)
    module_partial_fc.train().cuda(local_rank)

    # ---- optimizer ---------------------------------------------------------
    if cfg.optimizer == "sgd":
        opt = torch.optim.SGD(
            params=[{"params": backbone.parameters()},
                    {"params": module_partial_fc.parameters()}],
            lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == "adamw":
        opt = torch.optim.AdamW(
            params=[{"params": backbone.parameters()},
                    {"params": module_partial_fc.parameters()}],
            lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {cfg.optimizer}")

    opt.zero_grad(set_to_none=True)

    cfg.total_batch_size = cfg.batch_size * world_size
    cfg.warmup_step = cfg.num_image // cfg.total_batch_size * cfg.warmup_epoch
    cfg.total_step = cfg.num_image // cfg.total_batch_size * cfg.num_epoch
    if cfg.total_step <= 0:
        raise RuntimeError(
            f"cfg.total_step={cfg.total_step}. Check num_image={cfg.num_image}, "
            f"batch_size={cfg.batch_size}, world_size={world_size}, num_epoch={cfg.num_epoch}.")

    lr_scheduler = PolynomialLRWarmup(
        optimizer=opt, warmup_iters=cfg.warmup_step, total_iters=cfg.total_step)


    val_db = args.val_db or get_cfg_value(cfg, "val_db", None)
    ap_train_max = get_cfg_value(cfg, "ap_train_max_images", 6000)

    eval_train_samples = eval_train_sessions = None
    eval_val_samples = eval_val_sessions = None
    if rank == 0:
        tr_s, tr_se = scan_fes_with_sessions(cfg.rec, identity_level)
        eval_train_samples, eval_train_sessions = _select_subset(tr_s, tr_se, ap_train_max)
        if val_db:
            eval_val_samples, eval_val_sessions = scan_fes_with_sessions(val_db, identity_level)
            logging.info(f"AP val folder: {val_db} "
                         f"({len(eval_val_samples)} imgs)")
        else:
            logging.warning("No val_db set (cfg.val_db or --val_db); "
                            "reporting TRAIN AP only. Best model will be selected on train AP.")

    ap_csv = os.path.join(cfg.output, "epoch_ap.csv")

    if rank == 0:
        with open(ap_csv, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "train_frame_ap", "train_tmpl_ap",
                                    "val_frame_ap", "val_tmpl_ap", "val_tmpl_auc", "val_tmpl_eer"])
    best_ap = -1.0
    best_epoch = -1
    epochs_no_improve = 0
    patience = get_cfg_value(cfg, "early_stop_patience", 4)
    min_delta = get_cfg_value(cfg, "early_stop_min_delta", 0.0)
    device = torch.device(f"cuda:{local_rank}")
    start_epoch = 0
    global_step = 0

    if cfg.resume:
        checkpoint_path = os.path.join(cfg.output, f"checkpoint_gpu_{rank}.pt")
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_path}")
        dict_checkpoint = torch.load(checkpoint_path, map_location="cpu")
        start_epoch = dict_checkpoint["epoch"]
        global_step = dict_checkpoint["global_step"]
        unwrap_model(backbone, use_ddp).load_state_dict(dict_checkpoint["state_dict_backbone"])
        module_partial_fc.load_state_dict(dict_checkpoint["state_dict_softmax_fc"])
        opt.load_state_dict(dict_checkpoint["state_optimizer"])
        lr_scheduler.load_state_dict(dict_checkpoint["state_lr_scheduler"])
        del dict_checkpoint
        if rank == 0:
            logging.info(f"Resumed from {checkpoint_path}: start_epoch={start_epoch}, step={global_step}")

    for key, value in cfg.items():
        logging.info(": " + key + " " * max(1, 25 - len(key)) + str(value))

    callback_logging = CallBackLogging(
        frequent=cfg.frequent, total_step=cfg.total_step, batch_size=cfg.batch_size,
        start_step=global_step, writer=summary_writer)

    loss_am = AverageMeter()
    amp = torch.cuda.amp.GradScaler(enabled=bool(cfg.fp16), growth_interval=100)

    ft_model_path = os.path.join(cfg.output, "model_FT.pt")

    for epoch in range(start_epoch, cfg.num_epoch):
        if hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        backbone.train()
        module_partial_fc.train()
        for m in frozen_modules:
            m.eval()

        for batch in train_loader:
            if batch is None:
                continue
            img, local_labels = batch
            global_step += 1

            img = img.cuda(local_rank, non_blocking=True)
            local_labels = local_labels.cuda(local_rank, non_blocking=True)

            local_embeddings = backbone(img)
            loss = module_partial_fc(local_embeddings, local_labels)

            if cfg.fp16:
                amp.scale(loss).backward()
                if global_step % cfg.gradient_acc == 0:
                    amp.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
                    amp.step(opt)
                    amp.update()
                    opt.zero_grad(set_to_none=True)
            else:
                loss.backward()
                if global_step % cfg.gradient_acc == 0:
                    torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
                    opt.step()
                    opt.zero_grad(set_to_none=True)

            lr_scheduler.step()

            with torch.no_grad():
                loss_am.update(loss.item(), 1)
                callback_logging(global_step, loss_am, epoch, cfg.fp16,
                                 lr_scheduler.get_last_lr()[0], amp)

        if cfg.save_all_states:
            checkpoint = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "state_dict_backbone": unwrap_model(backbone, use_ddp).state_dict(),
                "state_dict_softmax_fc": module_partial_fc.state_dict(),
                "state_optimizer": opt.state_dict(),
                "state_lr_scheduler": lr_scheduler.state_dict(),
            }
            torch.save(checkpoint, os.path.join(cfg.output, f"checkpoint_gpu_{rank}.pt"))

        # if rank == 0:
        #     torch.save(unwrap_model(backbone, use_ddp).state_dict(), ft_model_path)
        #     logging.info(f"[epoch {epoch + 1}] saved fine-tuned backbone -> {ft_model_path}")

        if rank == 0:
            module = unwrap_model(backbone, use_ddp)
            was_training = module.training
            torch.save(module.state_dict(), os.path.join(cfg.output, "model_FT_last.pt"))

            tr = run_ap(module, eval_train_samples, eval_train_sessions,
                        cfg, image_size, device, tag=f"train E{epoch+1}")
            va = None
            if eval_val_samples is not None:
                va = run_ap(module, eval_val_samples, eval_val_sessions,
                            cfg, image_size, device, tag=f"val   E{epoch+1}")

            def _g(m, kind, key):
                return f"{m[kind][key]:.6f}" if (m and m.get(kind)) else ""
            with open(ap_csv, "a", newline="") as f:
                csv.writer(f).writerow([
                    epoch + 1,
                    _g(tr, "frame", "ap"), _g(tr, "template", "ap"),
                    _g(va, "frame", "ap"), _g(va, "template", "ap"),
                    _g(va, "template", "auc"), _g(va, "template", "eer"),
                ])

            def _sel(m):
                if m is None: return -1.0
                if m.get("template"): return m["template"]["ap"]
                return m["frame"]["ap"] if m.get("frame") else -1.0
            sel = _sel(va) if va is not None else _sel(tr)

            if sel > best_ap + min_delta:
                best_ap = sel
                best_epoch = epoch + 1
                epochs_no_improve = 0
                torch.save(module.state_dict(), ft_model_path)
                logging.info(f"[epoch {epoch+1}] NEW BEST "
                             f"{'val' if va else 'train'}_tmplAP={sel:.4f} -> {ft_model_path}")
            else:
                epochs_no_improve += 1
                logging.info(f"[epoch {epoch+1}] no improvement "
                             f"({epochs_no_improve}/{patience}); best={best_ap:.4f} @E{best_epoch}")

            if was_training:
                module.train()

        # broadcast stop decision so all ranks break together
        stop_flag = torch.zeros(1, device=device)
        if rank == 0 and epochs_no_improve >= patience:
            stop_flag[0] = 1.0
            logging.info(f"Early stopping at epoch {epoch+1}: "
                         f"no val-AP gain for {patience} epochs. Best={best_ap:.4f} @E{best_epoch}.")
        if use_ddp and world_size > 1:
            distributed.broadcast(stop_flag, src=0)
            distributed.barrier()
        if stop_flag.item() > 0.5:
            break



    if rank == 0:
        torch.save(unwrap_model(backbone, use_ddp).state_dict(), ft_model_path)
        logging.info("=" * 90)
        logging.info(f"Fine-tuning complete. Backbone: {ft_model_path}")
        logging.info(f"Classes: {cfg.num_classes:,} | Images: {cfg.num_image:,}")
        logging.info(f"Re-test: python test_bk_v2.py {args.config} --weights {ft_model_path} "
                     f"--test_db <val_folder> --data_type nested_images --output_dir eval_FT")
        logging.info("=" * 90)
        if summary_writer is not None:
            summary_writer.close()


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser(
        description="Fine-tune ArcFace/PartialFC backbone on FES nested-jpg data.")
    parser.add_argument("config", type=str, help="Python config file")
    parser.add_argument("--pretrained", type=str, default=None,
                        help="Override cfg.pretrained (starting model.pt).")
    parser.add_argument("--identity_level", type=str, default=None,
                        choices=["outer", "inner", "outer_inner"],
                        help="Class definition from the folder tree (default: outer).")
    parser.add_argument("--val_db", type=str, default=None,
                        help="Held-out FES folder for verification AP monitoring.")
    
    parser.add_argument("--freeze_until", type=str, default=None,
                        choices=["none", "stem", "layer1", "layer2", "layer3", "layer4"])
    parser.add_argument("--no_augment", action="store_true")

    args = parser.parse_args()

    rank, local_rank, world_size, use_ddp = setup_distributed()
    try:
        main(args, rank, local_rank, world_size, use_ddp)
    finally:
        if use_ddp:
            cleanup()