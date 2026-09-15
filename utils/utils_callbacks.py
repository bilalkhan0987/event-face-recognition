# import logging
# import os
# import time
# from typing import List

# import torch

# from eval import verification
# from utils.utils_logging import AverageMeter
# from torch.utils.tensorboard import SummaryWriter
# from torch import distributed


# class CallBackVerification(object):
    
#     def __init__(self, val_targets, rec_prefix, summary_writer=None, image_size=(112, 112), wandb_logger=None):
#         self.rank: int = distributed.get_rank()
#         self.highest_acc: float = 0.0
#         self.highest_acc_list: List[float] = [0.0] * len(val_targets)
#         self.ver_list: List[object] = []
#         self.ver_name_list: List[str] = []
#         if self.rank is 0:
#             self.init_dataset(val_targets=val_targets, data_dir=rec_prefix, image_size=image_size)

#         self.summary_writer = summary_writer
#         self.wandb_logger = wandb_logger

#     def ver_test(self, backbone: torch.nn.Module, global_step: int):
#         results = []
#         for i in range(len(self.ver_list)):
#             acc1, std1, acc2, std2, xnorm, embeddings_list = verification.test(
#                 self.ver_list[i], backbone, 10, 10)
#             logging.info('[%s][%d]XNorm: %f' % (self.ver_name_list[i], global_step, xnorm))
#             logging.info('[%s][%d]Accuracy-Flip: %1.5f+-%1.5f' % (self.ver_name_list[i], global_step, acc2, std2))

#             self.summary_writer: SummaryWriter
#             self.summary_writer.add_scalar(tag=self.ver_name_list[i], scalar_value=acc2, global_step=global_step, )
#             if self.wandb_logger:
#                 import wandb
#                 self.wandb_logger.log({
#                     f'Acc/val-Acc1 {self.ver_name_list[i]}': acc1,
#                     f'Acc/val-Acc2 {self.ver_name_list[i]}': acc2,
#                     # f'Acc/val-std1 {self.ver_name_list[i]}': std1,
#                     # f'Acc/val-std2 {self.ver_name_list[i]}': acc2,
#                 })

#             if acc2 > self.highest_acc_list[i]:
#                 self.highest_acc_list[i] = acc2
#             logging.info(
#                 '[%s][%d]Accuracy-Highest: %1.5f' % (self.ver_name_list[i], global_step, self.highest_acc_list[i]))
#             results.append(acc2)

#     def init_dataset(self, val_targets, data_dir, image_size):
#         for name in val_targets:
#             path = os.path.join(data_dir, name + ".bin")
#             if os.path.exists(path):
#                 data_set = verification.load_bin(path, image_size)
#                 self.ver_list.append(data_set)
#                 self.ver_name_list.append(name)

#     def __call__(self, num_update, backbone: torch.nn.Module):
#         if self.rank is 0 and num_update > 0:
#             backbone.eval()
#             self.ver_test(backbone, num_update)
#             backbone.train()


# class CallBackLogging(object):
#     def __init__(self, frequent, total_step, batch_size, start_step=0,writer=None):
#         self.frequent: int = frequent
#         self.rank: int = distributed.get_rank()
#         self.world_size: int = distributed.get_world_size()
#         self.time_start = time.time()
#         self.total_step: int = total_step
#         self.start_step: int = start_step
#         self.batch_size: int = batch_size
#         self.writer = writer

#         self.init = False
#         self.tic = 0

#     def __call__(self,
#                  global_step: int,
#                  loss: AverageMeter,
#                  epoch: int,
#                  fp16: bool,
#                  learning_rate: float,
#                  grad_scaler: torch.cuda.amp.GradScaler):
#         if self.rank == 0 and global_step > 0 and global_step % self.frequent == 0:
#             if self.init:
#                 try:
#                     speed: float = self.frequent * self.batch_size / (time.time() - self.tic)
#                     speed_total = speed * self.world_size
#                 except ZeroDivisionError:
#                     speed_total = float('inf')

#                 #time_now = (time.time() - self.time_start) / 3600
#                 #time_total = time_now / ((global_step + 1) / self.total_step)
#                 #time_for_end = time_total - time_now
#                 time_now = time.time()
#                 time_sec = int(time_now - self.time_start)
#                 time_sec_avg = time_sec / (global_step - self.start_step + 1)
#                 eta_sec = time_sec_avg * (self.total_step - global_step - 1)
#                 time_for_end = eta_sec/3600
#                 if self.writer is not None:
#                     self.writer.add_scalar('time_for_end', time_for_end, global_step)
#                     self.writer.add_scalar('learning_rate', learning_rate, global_step)
#                     self.writer.add_scalar('loss', loss.avg, global_step)
#                 if fp16:
#                     msg = "Speed %.2f samples/sec   Loss %.4f   LearningRate %.6f   Epoch: %d   Global Step: %d   " \
#                           "Fp16 Grad Scale: %2.f   Required: %1.f hours" % (
#                               speed_total, loss.avg, learning_rate, epoch, global_step,
#                               grad_scaler.get_scale(), time_for_end
#                           )
#                 else:
#                     msg = "Speed %.2f samples/sec   Loss %.4f   LearningRate %.6f   Epoch: %d   Global Step: %d   " \
#                           "Required: %1.f hours" % (
#                               speed_total, loss.avg, learning_rate, epoch, global_step, time_for_end
#                           )
#                 logging.info(msg)
#                 loss.reset()
#                 self.tic = time.time()
#             else:
#                 self.init = True
#                 self.tic = time.time()


import logging
import os
import time
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import distributed
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from eval import verification
from utils.utils_logging import AverageMeter


def get_rank_safe():
    if distributed.is_available() and distributed.is_initialized():
        return distributed.get_rank()
    return 0


def get_world_size_safe():
    if distributed.is_available() and distributed.is_initialized():
        return distributed.get_world_size()
    return 1


def l2_norm(input_tensor, axis=1):
    norm = torch.norm(input_tensor, 2, axis, True)
    return torch.div(input_tensor, norm)


def preprocess_image(img_bgr, image_size=(112, 112)):
    img = cv2.resize(img_bgr, image_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1)).astype(np.float32)
    img = (img - 127.5) / 127.5
    return img


class FolderFaceValDataset:
    """
    Expected structure:
        root/
            subject_1/
                img1.jpg
                img2.jpg
            subject_2/
                img1.jpg
                img2.jpg
    """
    def __init__(self, root_dir, image_size=(112, 112)):
        self.root_dir = root_dir
        self.image_size = image_size
        self.samples = []

        if not os.path.isdir(root_dir):
            raise ValueError(f"Validation folder not found: {root_dir}")

        subject_names = sorted(
            [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        )
        class_to_idx = {name: idx for idx, name in enumerate(subject_names)}

        valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

        for subject in subject_names:
            subject_dir = os.path.join(root_dir, subject)
            for fname in sorted(os.listdir(subject_dir)):
                if fname.lower().endswith(valid_exts):
                    self.samples.append(
                        (os.path.join(subject_dir, fname), class_to_idx[subject])
                    )

        self.num_classes = len(subject_names)


class CallBackVerification(object):
    def __init__(
        self,
        val_targets,
        rec_prefix,
        summary_writer: Optional[SummaryWriter] = None,
        image_size=(112, 112),
        wandb_logger=None,
        batch_size=64,
    ):
        self.rank = get_rank_safe()
        self.summary_writer = summary_writer
        self.wandb_logger = wandb_logger
        self.image_size = image_size
        self.batch_size = batch_size
        self.highest_acc = 0.0

        if not val_targets or len(val_targets) != 1:
            raise ValueError(
                f"Expected exactly one validation target, got: {val_targets}"
            )

        self.val_path = val_targets[0]
        self.val_name = os.path.basename(os.path.normpath(self.val_path))

        if self.rank == 0:
            if os.path.isdir(self.val_path):
                self.val_type = "folder"
                self.val_data = FolderFaceValDataset(self.val_path, image_size=image_size)
                logging.info(
                    f"Loaded folder validation set: {self.val_path}, "
                    f"samples={len(self.val_data.samples)}, classes={self.val_data.num_classes}"
                )
            else:
                # support legacy .bin usage if val_targets contains a name like "lfw"
                bin_path = os.path.join(rec_prefix, self.val_path + ".bin")
                if os.path.exists(bin_path):
                    self.val_type = "bin"
                    self.val_data = verification.load_bin(bin_path, image_size)
                    logging.info(f"Loaded bin validation set: {bin_path}")
                else:
                    raise ValueError(
                        f"Validation target is neither a folder nor a valid .bin target: {self.val_path}"
                    )

    @torch.no_grad()
    def run_folder_test(self, backbone: torch.nn.Module, global_step: int):
        samples = self.val_data.samples
        image_size = self.val_data.image_size

        if len(samples) == 0:
            logging.warning(f"[{self.val_name}] Empty folder validation dataset.")
            return 0.0

        embeddings = []
        labels = []

        for start in range(0, len(samples), self.batch_size):
            batch_samples = samples[start:start + self.batch_size]
            batch_imgs = []
            batch_labels = []

            for img_path, label in batch_samples:
                img = cv2.imread(img_path)
                if img is None:
                    logging.warning(f"Could not read image: {img_path}")
                    continue
                img = preprocess_image(img, image_size=image_size)
                batch_imgs.append(img)
                batch_labels.append(label)

            if len(batch_imgs) == 0:
                continue

            batch_imgs = torch.from_numpy(np.stack(batch_imgs)).cuda(non_blocking=True)
            feats = backbone(batch_imgs)
            feats = l2_norm(feats).cpu()

            embeddings.append(feats)
            labels.extend(batch_labels)

        if len(embeddings) == 0:
            logging.warning(f"[{self.val_name}] No embeddings produced.")
            return 0.0

        embeddings = torch.cat(embeddings, dim=0)
        labels = torch.tensor(labels, dtype=torch.long)

        centroids = []
        present_classes = []

        for c in range(self.val_data.num_classes):
            class_mask = labels == c
            class_embs = embeddings[class_mask]
            if class_embs.shape[0] == 0:
                continue
            centroid = class_embs.mean(dim=0, keepdim=True)
            centroid = F.normalize(centroid, dim=1)
            centroids.append(centroid)
            present_classes.append(c)

        if len(centroids) == 0:
            logging.warning(f"[{self.val_name}] No class centroids created.")
            return 0.0

        centroids = torch.cat(centroids, dim=0)
        sims = embeddings @ centroids.T
        pred_idx = torch.argmax(sims, dim=1)
        pred_labels = torch.tensor([present_classes[i] for i in pred_idx.tolist()], dtype=torch.long)
        acc = (pred_labels == labels).float().mean().item()

        logging.info(f"[{self.val_name}][{global_step}]Folder-Val-Acc: {acc:.5f}")

        if self.summary_writer is not None:
            self.summary_writer.add_scalar(
                tag=f"val/{self.val_name}_acc",
                scalar_value=acc,
                global_step=global_step,
            )
            self.summary_writer.flush()

        if self.wandb_logger:
            self.wandb_logger.log({
                f"Acc/folder-val {self.val_name}": acc,
                "Process/Step": global_step,
            })

        return acc

    def run_bin_test(self, backbone: torch.nn.Module, global_step: int):
        acc1, std1, acc2, std2, xnorm, embeddings_list = verification.test(
            self.val_data, backbone, 10, 10
        )

        logging.info(f"[{self.val_name}][{global_step}]XNorm: {xnorm:.6f}")
        logging.info(f"[{self.val_name}][{global_step}]Accuracy-Flip: {acc2:.5f}+-{std2:.5f}")

        if self.summary_writer is not None:
            self.summary_writer.add_scalar(
                tag=f"val/{self.val_name}_acc",
                scalar_value=acc2,
                global_step=global_step,
            )
            self.summary_writer.flush()

        if self.wandb_logger:
            self.wandb_logger.log({
                f"Acc/val-Acc1 {self.val_name}": acc1,
                f"Acc/val-Acc2 {self.val_name}": acc2,
                "Process/Step": global_step,
            })

        return acc2

    def ver_test(self, backbone: torch.nn.Module, global_step: int):
        if self.val_type == "bin":
            metric = self.run_bin_test(backbone, global_step)
        elif self.val_type == "folder":
            metric = self.run_folder_test(backbone, global_step)
        else:
            raise ValueError(f"Unsupported validation type: {self.val_type}")

        if metric > self.highest_acc:
            self.highest_acc = metric

        logging.info(f"[{self.val_name}][{global_step}]Accuracy-Highest: {self.highest_acc:.5f}")

        if self.summary_writer is not None:
            self.summary_writer.add_scalar(
                tag=f"val/{self.val_name}_best_acc",
                scalar_value=self.highest_acc,
                global_step=global_step,
            )
            self.summary_writer.flush()

    def __call__(self, num_update, backbone: torch.nn.Module):
        if self.rank == 0 and num_update > 0:
            backbone.eval()
            self.ver_test(backbone, num_update)
            backbone.train()


class CallBackLogging(object):
    def __init__(self, frequent, total_step, batch_size, start_step=0, writer=None):
        self.frequent: int = frequent
        self.rank: int = get_rank_safe()
        self.world_size: int = get_world_size_safe()
        self.time_start = time.time()
        self.total_step: int = total_step
        self.start_step: int = start_step
        self.batch_size: int = batch_size
        self.writer = writer

        self.init = False
        self.tic = 0

    def __call__(
        self,
        global_step: int,
        loss: AverageMeter,
        epoch: int,
        fp16: bool,
        learning_rate: float,
        grad_scaler: torch.cuda.amp.GradScaler,
    ):
        if self.rank == 0 and global_step > 0 and global_step % self.frequent == 0:
            if self.init:
                try:
                    speed = self.frequent * self.batch_size / (time.time() - self.tic)
                    speed_total = speed * self.world_size
                except ZeroDivisionError:
                    speed_total = float("inf")

                time_now = time.time()
                time_sec = int(time_now - self.time_start)
                time_sec_avg = time_sec / (global_step - self.start_step + 1)
                eta_sec = time_sec_avg * (self.total_step - global_step - 1)
                time_for_end = eta_sec / 3600

                if self.writer is not None:
                    self.writer.add_scalar("time_for_end", time_for_end, global_step)
                    self.writer.add_scalar("learning_rate", learning_rate, global_step)
                    self.writer.add_scalar("loss", loss.avg, global_step)
                    self.writer.flush()

                if fp16:
                    msg = (
                        "Speed %.2f samples/sec   Loss %.4f   LearningRate %.6f   "
                        "Epoch: %d   Global Step: %d   Fp16 Grad Scale: %2.f   Required: %1.f hours"
                    ) % (
                        speed_total,
                        loss.avg,
                        learning_rate,
                        epoch,
                        global_step,
                        grad_scaler.get_scale(),
                        time_for_end,
                    )
                else:
                    msg = (
                        "Speed %.2f samples/sec   Loss %.4f   LearningRate %.6f   "
                        "Epoch: %d   Global Step: %d   Required: %1.f hours"
                    ) % (
                        speed_total,
                        loss.avg,
                        learning_rate,
                        epoch,
                        global_step,
                        time_for_end,
                    )

                logging.info(msg)
                loss.reset()
                self.tic = time.time()
            else:
                self.init = True
                self.tic = time.time()