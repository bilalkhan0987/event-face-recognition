import argparse
import logging
import os
from datetime import datetime

import torch
from torch import distributed
from torch.utils.tensorboard import SummaryWriter
from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import fp16_compress_hook

from backbones import get_model
from dataset_v2 import get_dataloader
from losses import CombinedMarginLoss
from lr_scheduler import PolynomialLRWarmup
from partial_fc_v2 import PartialFC_V2
from utils.utils_callbacks import CallBackLogging
from utils.utils_config import get_config
from utils.utils_distributed_sampler import setup_seed
from utils.utils_logging import AverageMeter, init_logging


assert torch.__version__ >= "1.12.0", (
    "In order to enjoy the features of the new torch, "
    "we have upgraded the torch to 1.12.0. torch before than 1.12.0 may not work in the future."
)


def setup_distributed():
    """
    Initialize distributed training only when launched with torchrun.
    """
    if "RANK" in os.environ and "LOCAL_RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

        distributed.init_process_group(backend="nccl")
        use_ddp = True
    else:
        rank = 0
        local_rank = 0
        world_size = 1
        use_ddp = False

    return rank, local_rank, world_size, use_ddp


def cleanup():
    """
    Cleanly destroy the process group if it was initialized.
    """
    if distributed.is_available() and distributed.is_initialized():
        try:
            distributed.destroy_process_group()
        except Exception:
            pass


def unwrap_model(model, use_ddp: bool):
    return model.module if use_ddp else model


def get_cfg_value(cfg, key, default=None):
    """
    Safe getter for config objects that may behave like dicts or objects.
    """
    if hasattr(cfg, key):
        return getattr(cfg, key)
    try:
        return cfg[key]
    except Exception:
        return default


def set_cfg_value(cfg, key, value):
    """
    Safe setter for config objects that may behave like dicts or objects.
    """
    try:
        setattr(cfg, key, value)
    except Exception:
        cfg[key] = value


def remap_train_dataset_labels_to_contiguous(train_loader, output_dir, rank=0):
    """
    The identity-level split index may contain raw Glint360K identity IDs, for example:

        4, 5, 7, ..., 360231

    But PartialFC / ArcFace training needs labels in a compact range:

        0, 1, 2, ..., num_train_classes - 1

    This function remaps train labels in memory only.

    It preserves the train split membership.
    It does not modify the original train_index.txt.
    It writes train_label_map.tsv to output_dir on rank 0.
    """

    dataset = train_loader.dataset

    if not hasattr(dataset, "labels"):
        raise RuntimeError(
            "The training dataset does not expose dataset.labels. "
            "Cannot remap raw identity labels to contiguous labels."
        )

    raw_labels = [int(x) for x in dataset.labels]

    if len(raw_labels) == 0:
        raise RuntimeError("Training dataset has zero labels.")

    unique_raw_labels = sorted(set(raw_labels))
    raw_to_contiguous = {raw_label: idx for idx, raw_label in enumerate(unique_raw_labels)}

    contiguous_labels = [raw_to_contiguous[x] for x in raw_labels]
    dataset.labels = contiguous_labels

    num_train_samples = len(contiguous_labels)
    num_train_classes = len(unique_raw_labels)
    min_raw_label = min(raw_labels)
    max_raw_label = max(raw_labels)
    min_new_label = min(contiguous_labels)
    max_new_label = max(contiguous_labels)

    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)

        map_path = os.path.join(output_dir, "train_label_map.tsv")
        with open(map_path, "w") as f:
            f.write("contiguous_label\traw_identity\n")
            for raw_label in unique_raw_labels:
                f.write(f"{raw_to_contiguous[raw_label]}\t{raw_label}\n")

        print("=" * 90)
        print("TRAIN LABEL REMAPPING")
        print("=" * 90)
        print(f"train samples/files:             {num_train_samples:,}")
        print(f"unique train raw identities:     {num_train_classes:,}")
        print(f"raw label min:                   {min_raw_label}")
        print(f"raw label max:                   {max_raw_label}")
        print(f"new contiguous label min:        {min_new_label}")
        print(f"new contiguous label max:        {max_new_label}")
        print(f"new required num_classes:        {num_train_classes:,}")
        print(f"wrote label map:                 {map_path}")
        print("=" * 90)

    if min_new_label != 0:
        raise RuntimeError(f"Contiguous label remap failed: min label is {min_new_label}, expected 0.")

    if max_new_label != num_train_classes - 1:
        raise RuntimeError(
            f"Contiguous label remap failed: max label is {max_new_label}, "
            f"expected {num_train_classes - 1}."
        )

    return {
        "num_train_samples": num_train_samples,
        "num_train_classes": num_train_classes,
        "min_raw_label": min_raw_label,
        "max_raw_label": max_raw_label,
        "min_new_label": min_new_label,
        "max_new_label": max_new_label,
    }


def build_train_loader(cfg, local_rank):
    """
    Build only the training dataloader.

    For identity-disjoint train/val/test splits, val/test identities should not be
    used as normal closed-set classification validation during PartialFC training.
    """

    train_index_file = os.environ.get("EVENT_TRAIN_INDEX", None)
    event_index_dir = os.environ.get("EVENT_INDEX_DIR", None)

    train_loader = get_dataloader(
        root_dir=cfg.rec,
        local_rank=local_rank,
        batch_size=cfg.batch_size,
        dali=cfg.dali,
        dali_aug=cfg.dali_aug,
        seed=cfg.seed,
        num_workers=cfg.num_workers,
        event_split="train",
        event_index_file=train_index_file,
        event_index_dir=event_index_dir,
    )

    return train_loader


def main(args, rank, local_rank, world_size, use_ddp):
    cfg = get_config(args.config)

    setup_seed(seed=cfg.seed, cuda_deterministic=False)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this training script.")

    torch.cuda.set_device(local_rank)

    os.makedirs(cfg.output, exist_ok=True)
    init_logging(rank, cfg.output)

    summary_writer = (
        SummaryWriter(log_dir=os.path.join(cfg.output, "tensorboard"))
        if rank == 0
        else None
    )

    wandb_logger = None
    run_name = None
    wandb = None

    if cfg.using_wandb:
        import wandb as wandb_module

        wandb = wandb_module

        try:
            wandb.login(key=cfg.wandb_key)
        except Exception as e:
            print("WandB key must be provided in config file.")
            print(f"Config Error: {e}")

        run_name = datetime.now().strftime("%y%m%d_%H%M") + f"_GPU{rank}"
        run_name = run_name if cfg.suffix_run_name is None else run_name + f"_{cfg.suffix_run_name}"

        try:
            wandb_logger = (
                wandb.init(
                    entity=cfg.wandb_entity,
                    project=cfg.wandb_project,
                    sync_tensorboard=True,
                    resume=cfg.wandb_resume,
                    name=run_name,
                    notes=cfg.notes,
                )
                if rank == 0 or cfg.wandb_log_all
                else None
            )

            if wandb_logger:
                wandb_logger.config.update(cfg)

        except Exception as e:
            print("WandB entity/project settings must be provided in config file.")
            print(f"Config Error: {e}")

    train_loader = build_train_loader(cfg, local_rank)

    remap_stats = remap_train_dataset_labels_to_contiguous(
        train_loader=train_loader,
        output_dir=cfg.output,
        rank=rank,
    )

    actual_num_classes = remap_stats["num_train_classes"]
    actual_num_images = remap_stats["num_train_samples"]

    old_num_classes = int(cfg.num_classes)
    old_num_image = int(cfg.num_image)

    set_cfg_value(cfg, "num_classes", actual_num_classes)
    set_cfg_value(cfg, "num_image", actual_num_images)

    if rank == 0:
        print("=" * 90)
        print("CONFIG OVERRIDE FROM TRAIN INDEX")
        print("=" * 90)
        print(f"old cfg.num_classes: {old_num_classes:,}")
        print(f"new cfg.num_classes: {cfg.num_classes:,}")
        print(f"old cfg.num_image:   {old_num_image:,}")
        print(f"new cfg.num_image:   {cfg.num_image:,}")
        print("=" * 90)

    if use_ddp and world_size > 1:
        distributed.barrier()

    backbone = get_model(
        cfg.network,
        dropout=0.0,
        fp16=cfg.fp16,
        num_features=cfg.embedding_size,
    ).cuda(local_rank)

    backbone.train()

    if use_ddp:
        backbone = torch.nn.parallel.DistributedDataParallel(
            module=backbone,
            broadcast_buffers=False,
            device_ids=[local_rank],
            bucket_cap_mb=16,
            find_unused_parameters=True,
        )

        backbone.register_comm_hook(None, fp16_compress_hook)

        try:
            backbone._set_static_graph()
        except Exception:
            pass

    margin_loss = CombinedMarginLoss(
        64,
        cfg.margin_list[0],
        cfg.margin_list[1],
        cfg.margin_list[2],
        cfg.interclass_filtering_threshold,
    )

    module_partial_fc = PartialFC_V2(
        margin_loss,
        cfg.embedding_size,
        cfg.num_classes,
        cfg.sample_rate,
        False,
    )

    module_partial_fc.train().cuda(local_rank)

    if cfg.optimizer == "sgd":
        opt = torch.optim.SGD(
            params=[
                {"params": backbone.parameters()},
                {"params": module_partial_fc.parameters()},
            ],
            lr=cfg.lr,
            momentum=0.9,
            weight_decay=cfg.weight_decay,
        )

    elif cfg.optimizer == "adamw":
        opt = torch.optim.AdamW(
            params=[
                {"params": backbone.parameters()},
                {"params": module_partial_fc.parameters()},
            ],
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )

    else:
        raise ValueError(f"Unsupported optimizer: {cfg.optimizer}")

    opt.zero_grad(set_to_none=True)

    cfg.total_batch_size = cfg.batch_size * world_size
    cfg.warmup_step = cfg.num_image // cfg.total_batch_size * cfg.warmup_epoch
    cfg.total_step = cfg.num_image // cfg.total_batch_size * cfg.num_epoch

    if cfg.total_step <= 0:
        raise RuntimeError(
            f"cfg.total_step became {cfg.total_step}. "
            f"Check cfg.num_image={cfg.num_image}, batch_size={cfg.batch_size}, "
            f"world_size={world_size}, num_epoch={cfg.num_epoch}."
        )

    lr_scheduler = PolynomialLRWarmup(
        optimizer=opt,
        warmup_iters=cfg.warmup_step,
        total_iters=cfg.total_step,
    )

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
            print(f"Resumed from checkpoint: {checkpoint_path}")
            print(f"start_epoch={start_epoch}, global_step={global_step}")

    for key, value in cfg.items():
        num_space = max(1, 25 - len(key))
        logging.info(": " + key + " " * num_space + str(value))

    callback_logging = CallBackLogging(
        frequent=cfg.frequent,
        total_step=cfg.total_step,
        batch_size=cfg.batch_size,
        start_step=global_step,
        writer=summary_writer,
    )

    loss_am = AverageMeter()
    amp = torch.cuda.amp.GradScaler(
        enabled=bool(cfg.fp16),
        growth_interval=100,
    )

    for epoch in range(start_epoch, cfg.num_epoch):
        if hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        backbone.train()
        module_partial_fc.train()

        for _batch_idx, (img, local_labels) in enumerate(train_loader):
            global_step += 1

            if not img.is_cuda:
                img = img.cuda(local_rank, non_blocking=True)

            if not local_labels.is_cuda:
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

                if wandb_logger:
                    wandb_logger.log(
                        {
                            "Loss/Step Loss": loss.item(),
                            "Loss/Train Loss": loss_am.avg,
                            "Process/Step": global_step,
                            "Process/Epoch": epoch,
                            "LearningRate": lr_scheduler.get_last_lr()[0],
                        }
                    )

                callback_logging(
                    global_step,
                    loss_am,
                    epoch,
                    cfg.fp16,
                    lr_scheduler.get_last_lr()[0],
                    amp,
                )

        if cfg.save_all_states:
            checkpoint = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "state_dict_backbone": unwrap_model(backbone, use_ddp).state_dict(),
                "state_dict_softmax_fc": module_partial_fc.state_dict(),
                "state_optimizer": opt.state_dict(),
                "state_lr_scheduler": lr_scheduler.state_dict(),
            }

            torch.save(
                checkpoint,
                os.path.join(cfg.output, f"checkpoint_gpu_{rank}.pt"),
            )

        if rank == 0:
            path_module = os.path.join(cfg.output, "model.pt")
            torch.save(unwrap_model(backbone, use_ddp).state_dict(), path_module)

            if wandb_logger and cfg.save_artifacts:
                artifact_name = f"{run_name}_E{epoch}"
                artifact = wandb.Artifact(artifact_name, type="model")
                artifact.add_file(path_module)
                wandb_logger.log_artifact(artifact)

        if hasattr(train_loader, "reset"):
            train_loader.reset()

    if rank == 0:
        path_module = os.path.join(cfg.output, "model.pt")
        torch.save(unwrap_model(backbone, use_ddp).state_dict(), path_module)

        if wandb_logger and cfg.save_artifacts:
            artifact_name = f"{run_name}_Final"
            artifact = wandb.Artifact(artifact_name, type="model")
            artifact.add_file(path_module)
            wandb_logger.log_artifact(artifact)

        if summary_writer is not None:
            summary_writer.close()


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser(
        description="Distributed ArcFace / PartialFC training for event .npz data"
    )

    parser.add_argument(
        "config",
        type=str,
        help="Python config file",
    )

    args = parser.parse_args()

    rank, local_rank, world_size, use_ddp = setup_distributed()

    try:
        main(args, rank, local_rank, world_size, use_ddp)
    finally:
        if use_ddp:
            cleanup()