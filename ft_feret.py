#!/usr/bin/env python
"""
Fine-tune an ArcFace/PartialFC backbone on FERET event frames (.npz), identity-disjoint.

Trains ONLY on the subjects listed in --subject_list (the train split from
make_feret_split.py). A subject-disjoint val slice is carved from within those train
subjects (--val_subject_ratio) and used purely for AP monitoring + early stopping —
the held-out TEST subjects are never touched here. Saves model_FT_feret.pt.

Identity = subject id (folder name = filename prefix). "condition" for cross-condition
AP monitoring = FERET pose code (fa/fb/ql/qr/hl/hr/pl/pr/...).

Single GPU:
    torchrun --nproc_per_node=1 ft_feret.py configs/feret_ft.py \
        --feret_db /data/facerecognition_events_paul/FERET/colorferet_events \
        --subject_list feret_split/train_subjects.txt
"""
import os
import re
import csv
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
from torch import distributed
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import fp16_compress_hook
from sklearn.metrics import average_precision_score, roc_curve, auc

from backbones import get_model
from losses import CombinedMarginLoss
from lr_scheduler import PolynomialLRWarmup
from partial_fc_v2 import PartialFC_V2
from utils.utils_callbacks import CallBackLogging
from utils.utils_config import get_config
from utils.utils_distributed_sampler import setup_seed
from utils.utils_logging import AverageMeter, init_logging

FNAME_RE = re.compile(r"^(?P<subj>\d+)_(?P<date>\d+)_(?P<pose>[a-zA-Z]+)(?P<alt>_a)?$")


# ---------------- distributed helpers ----------------
def setup_distributed():
    if "RANK" in os.environ and "LOCAL_RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"]); local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        distributed.init_process_group(backend="nccl"); use_ddp = True
    else:
        rank, local_rank, world_size, use_ddp = 0, 0, 1, False
        if not distributed.is_initialized():
            os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
            os.environ.setdefault("MASTER_PORT", "12355")
            distributed.init_process_group(backend="nccl", rank=0, world_size=1)
    return rank, local_rank, world_size, use_ddp


def cleanup():
    if distributed.is_available() and distributed.is_initialized():
        try: distributed.destroy_process_group()
        except Exception: pass


def unwrap_model(m, use_ddp): return m.module if use_ddp else m
def get_cfg_value(cfg, k, d=None):
    if hasattr(cfg, k): return getattr(cfg, k)
    try: return cfg[k]
    except Exception: return d
def set_cfg_value(cfg, k, v):
    try: setattr(cfg, k, v)
    except Exception: cfg[k] = v


# ---------------- backbone loading ----------------
def strip_known_prefixes(sd):
    out = {}
    for k, v in sd.items():
        nk = k; ch = True
        while ch:
            ch = False
            for p in ("module.", "backbone.", "model."):
                if nk.startswith(p): nk = nk[len(p):]; ch = True
        out[nk] = v
    return out


def extract_backbone_state_dict(ckpt):
    if not isinstance(ckpt, dict): raise TypeError("weights not a dict")
    for k in ("state_dict_backbone", "backbone_state_dict", "model_state_dict",
              "state_dict", "backbone", "model"):
        if k in ckpt and isinstance(ckpt[k], dict): return ckpt[k]
    if sum(torch.is_tensor(v) for v in ckpt.values()) > 0: return ckpt
    raise RuntimeError("no usable backbone state_dict")


def load_pretrained(backbone, path, rank):
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"cfg.pretrained must point to the base model.pt (got {path})")
    sd = strip_known_prefixes(extract_backbone_state_dict(torch.load(path, map_location="cpu")))
    inc = backbone.load_state_dict(sd, strict=False)
    if rank == 0:
        logging.info(f"Loaded pretrained {path} | missing={len(inc.missing_keys)} "
                     f"unexpected={len(inc.unexpected_keys)}")


# ---------------- npz preprocessing + augmentation ----------------
def hist_to_uint8(fh, clip_val):
    img = np.clip(fh.astype(np.float32), -float(clip_val), float(clip_val))
    img = (img + float(clip_val)) / (2.0 * float(clip_val))
    return (np.squeeze(img) * 255.0).astype(np.uint8)


def npz_to_img_uint8(path, image_size, hist_key, clip_val):
    try:
        with np.load(path, allow_pickle=True) as d:
            if hist_key not in d: return None
            fh = d[hist_key]
        img = hist_to_uint8(fh, clip_val)
        if img.ndim == 2: img = np.stack([img, img, img], -1)
        elif img.ndim == 3 and img.shape[-1] == 1: img = np.repeat(img, 3, -1)
        return cv2.resize(img, (image_size, image_size))
    except Exception:
        return None


def augment(img, rng):
    h, w = img.shape[:2]
    if rng.random() < 0.5:
        s = rng.uniform(0.8, 1.0); nh, nw = int(h*s), int(w*s)
        top, left = rng.randint(0, h-nh), rng.randint(0, w-nw)
        img = cv2.resize(img[top:top+nh, left:left+nw], (w, h))
    if rng.random() < 0.5:
        M = cv2.getRotationMatrix2D((w/2, h/2), rng.uniform(-10, 10), 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    if rng.random() < 0.5:
        img = np.clip(img.astype(np.float32)*rng.uniform(0.8, 1.2)+rng.uniform(-20, 20),
                      0, 255).astype(np.uint8)
    return img


def worker_init_fn(wid):
    seed = (torch.initial_seed() + wid) % (2**32)
    np.random.seed(seed); random.seed(seed)


# ---------------- FERET scan ----------------
def parse_name(path):
    m = FNAME_RE.match(os.path.splitext(os.path.basename(path))[0])
    if not m: return None
    return m.group("subj"), m.group("pose").lower(), bool(m.group("alt"))


def read_subject_list(path):
    with open(path) as f:
        return set(l.strip() for l in f if l.strip())


def scan_feret(root, subject_set, include_alt):
    """Return [(path, subj_str, pose)] for files whose subject is in subject_set."""
    out = []
    for f in sorted(glob.glob(os.path.join(root, "**", "*.npz"), recursive=True)):
        parsed = parse_name(f)
        if parsed is None: continue
        subj, pose, is_alt = parsed
        if is_alt and not include_alt: continue
        if subj not in subject_set: continue
        out.append((f, subj, pose))
    if not out:
        raise FileNotFoundError(f"No FERET npz matched subject list under {root}")
    return out


def split_train_val_subjects(subjects, val_ratio, seed):
    subs = sorted(subjects)
    rng = random.Random(seed); rng.shuffle(subs)
    n_val = max(1, int(round(len(subs) * val_ratio)))
    return set(subs[n_val:]), set(subs[:n_val])   # fit, val


# ---------------- training Dataset ----------------
class FeretTrainDataset(Dataset):
    def __init__(self, samples, subj_to_label, image_size, hist_key, clip_val, augment_on):
        self.samples = samples          # (path, subj_str, pose)
        self.subj_to_label = subj_to_label
        self.image_size = image_size
        self.hist_key = hist_key
        self.clip_val = clip_val
        self.augment_on = augment_on
        self.labels = [subj_to_label[s] for _p, s, _po in samples]

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, subj, _pose = self.samples[idx]
        img = npz_to_img_uint8(path, self.image_size, self.hist_key, self.clip_val)
        if img is None: return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.augment_on: img = augment(img, random)
        if random.random() < 0.5: img = img[:, ::-1, :]
        img = np.transpose(img, (2, 0, 1)).astype(np.float32)
        img = (img - 127.5) / 127.5
        return torch.from_numpy(np.ascontiguousarray(img)), self.subj_to_label[subj]


def collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    if not batch: return None
    return torch.stack([b[0] for b in batch]), torch.tensor([b[1] for b in batch], dtype=torch.long)


# ---------------- AP monitoring (cross-pose template) ----------------
@torch.no_grad()
def embed(model, samples, image_size, hist_key, clip_val, batch_size, device, flip_tta=True):
    embs, subj, pose = [], [], []
    bi, bs, bp = [], [], []
    def flush():
        if not bi: return
        t = torch.stack(bi).to(device, non_blocking=True)
        f = model(t).float()
        if flip_tta: f = f + model(torch.flip(t, dims=[3])).float()
        f = F.normalize(f).cpu()
        embs.append(f); subj.extend(bs); pose.extend(bp)
        bi.clear(); bs.clear(); bp.clear()
    for path, s, po in samples:
        img = npz_to_img_uint8(path, image_size, hist_key, clip_val)
        if img is None: continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1)).astype(np.float32)
        img = (img - 127.5) / 127.5
        bi.append(torch.from_numpy(np.ascontiguousarray(img))); bs.append(s); bp.append(po)
        if len(bi) >= batch_size: flush()
    flush()
    if not embs: return None, None, None
    return torch.cat(embs, 0), np.array(subj, dtype=object), np.array(pose, dtype=object)


def _metrics(y, sc):
    y, sc = np.asarray(y), np.asarray(sc)
    if len(np.unique(y)) < 2: return None
    fpr, tpr, _ = roc_curve(y, sc); fnr = 1-tpr
    k = np.nanargmin(np.abs(fnr-fpr))
    return {"ap": float(average_precision_score(y, sc)), "auc": float(auc(fpr, tpr)),
            "eer": float((fpr[k]+fnr[k])/2)}


def template_ap(embs, subj, pose, max_pairs, seed):
    groups = defaultdict(list)
    for i in range(len(subj)):
        groups[(subj[i], pose[i])].append(i)
    T, ts = [], []
    for (s, _p), ix in groups.items():
        T.append(F.normalize(embs[ix].mean(0), dim=0)); ts.append(s)
    if len(T) < 2: return None
    T = torch.stack(T)
    by = defaultdict(list)
    for i, s in enumerate(ts): by[s].append(i)
    poss = [s for s, ix in by.items() if len(ix) >= 2]; alls = list(by.keys())
    if not poss or len(alls) < 2: return None
    rng = random.Random(seed); npos = max_pairs//2; y, sc = [], []
    for _ in range(npos):
        s = rng.choice(poss); i, j = rng.sample(by[s], 2)
        sc.append(float(torch.dot(T[i], T[j]))); y.append(1)
    for _ in range(max_pairs-npos):
        a, b = rng.sample(alls, 2)
        sc.append(float(torch.dot(T[rng.choice(by[a])], T[rng.choice(by[b])]))); y.append(0)
    m = _metrics(y, sc)
    if m: m["num_templates"] = len(T)
    return m


def run_val_ap(model, samples, cfg, image_size, device, tag):
    e, s, p = embed(model, samples, image_size, cfg.hist_key if hasattr(cfg, "hist_key") else "final_hist",
                    get_cfg_value(cfg, "clip_val", 5.0), cfg.batch_size, device,
                    flip_tta=get_cfg_value(cfg, "ap_flip_tta", True))
    if e is None: return None
    m = template_ap(e, s, p, get_cfg_value(cfg, "ap_max_pairs", 20000), 42)
    if m:
        logging.info(f"[{tag}] template AP={m['ap']:.4f} AUC={m['auc']:.4f} "
                     f"EER={m['eer']:.4f} (templates={m['num_templates']})")
    return m


# ---------------- freezing ----------------
def apply_freezing(backbone, freeze_until, rank):
    if not freeze_until or freeze_until == "none": return []
    order = ["conv1", "bn1", "prelu", "layer1", "layer2", "layer3", "layer4"]
    idx = {"stem": 2, "layer1": 3, "layer2": 4, "layer3": 5, "layer4": 6}
    if freeze_until not in idx: return []
    names = set(order[:idx[freeze_until]+1]); frozen = []
    for name, mod in backbone.named_children():
        if name in names:
            for pr in mod.parameters(): pr.requires_grad_(False)
            mod.eval(); frozen.append(mod)
    if rank == 0:
        nt = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
        logging.info(f"Freeze up to '{freeze_until}': trainable params={nt:,}")
    return frozen


# ---------------- main ----------------
def main(args, rank, local_rank, world_size, use_ddp):
    cfg = get_config(args.config)
    image_size = get_cfg_value(cfg, "image_size", 112)
    hist_key = get_cfg_value(cfg, "hist_key", "final_hist")
    clip_val = get_cfg_value(cfg, "clip_val", 5.0)
    pretrained = args.pretrained or get_cfg_value(cfg, "pretrained", None)

    setup_seed(seed=cfg.seed, cuda_deterministic=False)
    torch.cuda.set_device(local_rank)
    os.makedirs(cfg.output, exist_ok=True)
    init_logging(rank, cfg.output)
    device = torch.device(f"cuda:{local_rank}")

    # subjects: train_list -> (fit, val) subject-disjoint
    train_subjects = read_subject_list(args.subject_list)
    fit_subj, val_subj = split_train_val_subjects(
        train_subjects, get_cfg_value(cfg, "val_subject_ratio", 0.1), cfg.seed)

    all_samples = scan_feret(args.feret_db, train_subjects, include_alt=args.include_alt)
    fit_samples = [s for s in all_samples if s[1] in fit_subj]
    val_samples = [s for s in all_samples if s[1] in val_subj]

    subj_sorted = sorted({s for _p, s, _po in fit_samples})
    subj_to_label = {s: i for i, s in enumerate(subj_sorted)}
    num_classes = len(subj_sorted)

    if rank == 0:
        logging.info("=" * 80)
        logging.info(f"FERET FT | train_list subjects={len(train_subjects)} "
                     f"fit={len(fit_subj)} val={len(val_subj)}")
        logging.info(f"fit images={len(fit_samples):,} val images={len(val_samples):,} "
                     f"classes(fit)={num_classes}")
        logging.info("=" * 80)

    set_cfg_value(cfg, "num_classes", num_classes)
    set_cfg_value(cfg, "num_image", len(fit_samples))

    # data
    dataset = FeretTrainDataset(fit_samples, subj_to_label, image_size, hist_key, clip_val,
                                augment_on=get_cfg_value(cfg, "augment", True))
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=local_rank,
                                 shuffle=True, drop_last=True) if (use_ddp and world_size > 1) else None
    train_loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=(sampler is None),
                              sampler=sampler, num_workers=cfg.num_workers, pin_memory=True,
                              drop_last=True, collate_fn=collate_skip_none,
                              worker_init_fn=worker_init_fn)

    # backbone
    backbone = get_model(cfg.network, dropout=0.0, fp16=cfg.fp16,
                         num_features=cfg.embedding_size).cuda(local_rank)
    load_pretrained(backbone, pretrained, rank)
    frozen = apply_freezing(backbone, args.freeze_until or get_cfg_value(cfg, "freeze_until", "none"), rank)
    backbone.train()
    if use_ddp:
        backbone = torch.nn.parallel.DistributedDataParallel(
            backbone, broadcast_buffers=False, device_ids=[local_rank],
            bucket_cap_mb=16, find_unused_parameters=True)
        backbone.register_comm_hook(None, fp16_compress_hook)
        try: backbone._set_static_graph()
        except Exception: pass

    margin_loss = CombinedMarginLoss(64, cfg.margin_list[0], cfg.margin_list[1],
                                     cfg.margin_list[2], cfg.interclass_filtering_threshold)
    pfc = PartialFC_V2(margin_loss, cfg.embedding_size, cfg.num_classes, cfg.sample_rate, False)
    pfc.train().cuda(local_rank)

    if cfg.optimizer == "sgd":
        opt = torch.optim.SGD([{"params": backbone.parameters()}, {"params": pfc.parameters()}],
                              lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    else:
        opt = torch.optim.AdamW([{"params": backbone.parameters()}, {"params": pfc.parameters()}],
                                lr=cfg.lr, weight_decay=cfg.weight_decay)
    opt.zero_grad(set_to_none=True)

    cfg.total_batch_size = cfg.batch_size * world_size
    cfg.warmup_step = cfg.num_image // cfg.total_batch_size * cfg.warmup_epoch
    cfg.total_step = cfg.num_image // cfg.total_batch_size * cfg.num_epoch
    if cfg.total_step <= 0:
        raise RuntimeError(f"total_step={cfg.total_step}; check num_image/batch/epochs.")
    lr_sched = PolynomialLRWarmup(opt, warmup_iters=cfg.warmup_step, total_iters=cfg.total_step)

    for k, v in cfg.items():
        logging.info(": " + k + " " * max(1, 25-len(k)) + str(v))

    callback = CallBackLogging(frequent=cfg.frequent, total_step=cfg.total_step,
                               batch_size=cfg.batch_size, start_step=0, writer=None)
    loss_am = AverageMeter()
    amp = torch.cuda.amp.GradScaler(enabled=bool(cfg.fp16), growth_interval=100)

    ft_path = os.path.join(cfg.output, "model_FT_feret.pt")
    ap_csv = os.path.join(cfg.output, "feret_epoch_ap.csv")
    if rank == 0:
        with open(ap_csv, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "val_ap", "val_auc", "val_eer"])

    best_ap, best_epoch, no_improve = -1.0, -1, 0
    patience = get_cfg_value(cfg, "early_stop_patience", 4)
    global_step = 0

    for epoch in range(cfg.num_epoch):
        if sampler is not None: sampler.set_epoch(epoch)
        backbone.train(); pfc.train()
        for m in frozen: m.eval()

        for batch in train_loader:
            if batch is None: continue
            img, lbl = batch
            global_step += 1
            img = img.cuda(local_rank, non_blocking=True)
            lbl = lbl.cuda(local_rank, non_blocking=True)
            emb = backbone(img)
            loss = pfc(emb, lbl)
            if cfg.fp16:
                amp.scale(loss).backward()
                if global_step % cfg.gradient_acc == 0:
                    amp.unscale_(opt); torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
                    amp.step(opt); amp.update(); opt.zero_grad(set_to_none=True)
            else:
                loss.backward()
                if global_step % cfg.gradient_acc == 0:
                    torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
                    opt.step(); opt.zero_grad(set_to_none=True)
            lr_sched.step()
            with torch.no_grad():
                loss_am.update(loss.item(), 1)
                callback(global_step, loss_am, epoch, cfg.fp16, lr_sched.get_last_lr()[0], amp)

        # ---- val AP + early stopping (rank 0 decides, broadcast) ----
        if rank == 0:
            module = unwrap_model(backbone, use_ddp)
            was_tr = module.training
            torch.save(module.state_dict(), os.path.join(cfg.output, "model_FT_feret_last.pt"))
            va = run_val_ap(module, val_samples, cfg, image_size, device, tag=f"val E{epoch+1}")
            sel = va["ap"] if va else -1.0
            with open(ap_csv, "a", newline="") as f:
                csv.writer(f).writerow([epoch+1,
                                        f"{va['ap']:.6f}" if va else "",
                                        f"{va['auc']:.6f}" if va else "",
                                        f"{va['eer']:.6f}" if va else ""])
            if sel > best_ap:
                best_ap, best_epoch, no_improve = sel, epoch+1, 0
                torch.save(module.state_dict(), ft_path)
                logging.info(f"[epoch {epoch+1}] NEW BEST val_AP={sel:.4f} -> {ft_path}")
            else:
                no_improve += 1
                logging.info(f"[epoch {epoch+1}] no improvement ({no_improve}/{patience}); "
                             f"best={best_ap:.4f} @E{best_epoch}")
            if was_tr: module.train()

        stop = torch.zeros(1, device=device)
        if rank == 0 and no_improve >= patience:
            stop[0] = 1.0
            logging.info(f"Early stopping at epoch {epoch+1}. Best val_AP={best_ap:.4f} @E{best_epoch}")
        if use_ddp and world_size > 1:
            distributed.broadcast(stop, src=0); distributed.barrier()
        if stop.item() > 0.5: break

    if rank == 0:
        logging.info("=" * 80)
        logging.info(f"FERET fine-tuning done. Best backbone -> {ft_path} (val_AP={best_ap:.4f})")
        logging.info(f"Now evaluate on HELD-OUT TEST subjects with eval_feret.py --subject_list "
                     f"feret_split/test_subjects.txt --weights {ft_path}")
        logging.info("=" * 80)


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    p = argparse.ArgumentParser(description="Fine-tune ArcFace/PartialFC on FERET events (identity-disjoint).")
    p.add_argument("config", type=str)
    p.add_argument("--feret_db", type=str, required=True)
    p.add_argument("--subject_list", type=str, required=True, help="train_subjects.txt")
    p.add_argument("--pretrained", type=str, default=None, help="override cfg.pretrained (base model.pt)")
    p.add_argument("--freeze_until", type=str, default=None,
                   choices=["none", "stem", "layer1", "layer2", "layer3", "layer4"])
    p.add_argument("--include_alt", action="store_true")
    args = p.parse_args()
    rank, local_rank, world_size, use_ddp = setup_distributed()
    try:
        main(args, rank, local_rank, world_size, use_ddp)
    finally:
        if distributed.is_initialized(): cleanup()