import glob
import numbers
import os
import queue as Queue
import threading
from typing import Iterable

import mxnet as mx
import numpy as np
import torch
from functools import partial
from torch import distributed
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from utils.utils_distributed_sampler import DistributedSampler
from utils.utils_distributed_sampler import get_dist_info, worker_init_fn


def _contains_npz(root_dir: str) -> bool:
    """Return True as soon as a single .npz is found under root_dir.

    Uses os.walk so it short-circuits instead of enumerating the whole
    (potentially huge) tree just to detect the dataset type.
    """
    if not os.path.isdir(root_dir):
        return False
    for _root, _dirs, files in os.walk(root_dir):
        for f in files:
            if f.endswith(".npz"):
                return True
    return False



def _split_index_names(split: str):
    """Return accepted index filenames for a given split."""
    split = split.lower()
    if split in ("val", "valid", "validation"):
        return ["val_index.txt", "valid_index.txt", "validation_index.txt"]
    if split == "test":
        return ["test_index.txt"]
    return ["train_index.txt"]


def _normalize_split_name(split: str) -> str:
    """Normalize split aliases."""
    split = (split or "train").lower()
    if split in ("valid", "validation"):
        return "val"
    return split


def _resolve_event_index_file(
    root_dir: str,
    split: str = "train",
    index_file: str = None,
    index_dir: str = None,
) -> str:
    """Resolve train/val/test index files for EventNPZDataset.

    Search order:
      1) explicit index_file argument
      2) split-specific environment variable, e.g. EVENT_TRAIN_INDEX / EVENT_VAL_INDEX
      3) legacy EVENT_INDEX for the train split only
      4) explicit index_dir/<split>_index.txt
      5) EVENT_INDEX_DIR/<split>_index.txt
      6) <root_dir>/<split>_index.txt
      7) <cwd>/<split>_index.txt

    For validation, it also accepts valid_index.txt and validation_index.txt.
    """
    split = _normalize_split_name(split)

    if split not in ("train", "val", "test"):
        raise ValueError(f"split must be one of train, val, test. Got: {split}")

    candidates = []

    if index_file:
        candidates.append(index_file)

    env_name = f"EVENT_{split.upper()}_INDEX"
    if os.environ.get(env_name):
        candidates.append(os.environ[env_name])

    # Backward compatibility with the previous single-index setup.
    if split == "train" and os.environ.get("EVENT_INDEX"):
        candidates.append(os.environ["EVENT_INDEX"])

    # Directory-based index lookup.
    if index_dir:
        for name in _split_index_names(split):
            candidates.append(os.path.join(index_dir, name))

    if os.environ.get("EVENT_INDEX_DIR"):
        for name in _split_index_names(split):
            candidates.append(os.path.join(os.environ["EVENT_INDEX_DIR"], name))

    for name in _split_index_names(split):
        candidates.append(os.path.join(root_dir, name))
        candidates.append(os.path.join(os.getcwd(), name))

    resolved = next((c for c in candidates if c and os.path.isfile(c)), None)
    if resolved is None:
        raise RuntimeError(
            f"{split} index file not found. Looked in: "
            + ", ".join(c for c in candidates if c)
            + ". Build train_index.txt / val_index.txt / test_index.txt, "
            + "or pass event_index_file / train_index_file / val_index_file."
        )

    return resolved



def _build_torch_loader(
    dataset,
    local_rank,
    batch_size,
    num_workers=2,
    seed=2048,
    shuffle=True,
    drop_last=True,
):
    """Build the CUDA-prefetching DataLoaderX with a distributed sampler."""
    rank, world_size = get_dist_info()
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=shuffle,
        seed=seed,
    )

    if seed is None:
        init_fn = None
    else:
        init_fn = partial(worker_init_fn, num_workers=num_workers, rank=rank, seed=seed)

    return DataLoaderX(
        local_rank=local_rank,
        dataset=dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
        worker_init_fn=init_fn,
    )


def get_dataloader(
    root_dir,
    local_rank,
    batch_size,
    dali=False,
    dali_aug=False,
    seed=2048,
    num_workers=2,
    # NPZ split-index options. These keep compatibility with both train_v3/train_v4
    # and the older single-index loader.
    event_split="train",
    event_index_file=None,
    event_index_dir=None,
    return_val=False,
    val_batch_size=None,
    train_index_file=None,
    val_index_file=None,
    test_index_file=None,
    image_size=112,
    clip_val=5,
    to_three_channel=True,
    hist_key="final_hist",
    max_classes=None,
    max_images_per_class=None,
    max_images=None,
) -> Iterable:
    """Create dataloader(s).
        
    """

    rec = os.path.join(root_dir, 'train.rec')
    idx = os.path.join(root_dir, 'train.idx')
    train_set = None
    val_set = None

    # Synthetic
    if root_dir == "synthetic":
        train_set = SyntheticDataset()
        dali = False

    # Mxnet RecordIO
    elif os.path.exists(rec) and os.path.exists(idx):
        train_set = MXFaceDataset(root_dir=root_dir, local_rank=local_rank)

    # Event-camera .npz dataset (e.g. Glint360K event dump)
    elif _contains_npz(root_dir):
        event_split = _normalize_split_name(event_split)

        if return_val:
            # Train/validation pair mode.
            # Prefer explicit train_index_file / val_index_file, but also allow
            # event_index_file to serve as the train index for backward compatibility.
            resolved_train_index = train_index_file or event_index_file
            resolved_val_index = val_index_file

            train_set = EventNPZDataset(
                root_dir=root_dir,
                local_rank=local_rank,
                split="train",
                index_file=resolved_train_index,
                index_dir=event_index_dir,
                image_size=image_size,
                clip_val=clip_val,
                to_three_channel=to_three_channel,
                augment=True,
                hist_key=hist_key,
                max_classes=max_classes,
                max_images_per_class=max_images_per_class,
                max_images=max_images,
                subset_seed=seed if seed is not None else 2048,
            )

            val_set = EventNPZDataset(
                root_dir=root_dir,
                local_rank=local_rank,
                split="val",
                index_file=resolved_val_index,
                index_dir=event_index_dir,
                image_size=image_size,
                clip_val=clip_val,
                to_three_channel=to_three_channel,
                augment=False,
                hist_key=hist_key,
            )

        else:
            # Single-split mode. This is what your current training script should use.
            if event_split == "train":
                resolved_index = train_index_file or event_index_file
                augment = True
            elif event_split == "val":
                resolved_index = val_index_file or event_index_file
                augment = False
            elif event_split == "test":
                resolved_index = test_index_file or event_index_file
                augment = False
            else:
                raise ValueError(f"Unsupported event_split: {event_split}")

            train_set = EventNPZDataset(
                root_dir=root_dir,
                local_rank=local_rank,
                split=event_split,
                index_file=resolved_index,
                index_dir=event_index_dir,
                image_size=image_size,
                clip_val=clip_val,
                to_three_channel=to_three_channel,
                augment=augment,
                hist_key=hist_key,
                # Subset limits only ever apply to the training split.
                max_classes=max_classes if event_split == "train" else None,
                max_images_per_class=max_images_per_class if event_split == "train" else None,
                max_images=max_images if event_split == "train" else None,
                subset_seed=seed if seed is not None else 2048,
            )

        dali = False  # DALI's mxnet reader cannot read .npz event files

    # Image Folder
    else:
        transform = transforms.Compose([
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
             ])
        train_set = ImageFolder(root_dir, transform)

    if return_val and val_set is None:
        raise RuntimeError(
            "return_val=True is currently supported for the .npz EventNPZDataset path. "
            "For RecordIO/ImageFolder, provide a separate validation dataset loader."
        )

    # DALI
    if dali:
        if return_val:
            raise RuntimeError("return_val=True is not supported with DALI in this loader.")
        return dali_data_iter(
            batch_size=batch_size, rec_file=rec, idx_file=idx,
            num_threads=2, local_rank=local_rank, dali_aug=dali_aug)

    train_loader = _build_torch_loader(
        dataset=train_set,
        local_rank=local_rank,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
        shuffle=True,
        drop_last=True,
    )

    if return_val:
        if val_batch_size is None:
            val_batch_size = batch_size
        val_loader = _build_torch_loader(
            dataset=val_set,
            local_rank=local_rank,
            batch_size=val_batch_size,
            num_workers=num_workers,
            seed=seed,
            shuffle=False,
            drop_last=False,
        )
        return train_loader, val_loader

    return train_loader


class BackgroundGenerator(threading.Thread):
    def __init__(self, generator, local_rank, max_prefetch=6):
        super(BackgroundGenerator, self).__init__()
        self.queue = Queue.Queue(max_prefetch)
        self.generator = generator
        self.local_rank = local_rank
        self.daemon = True
        self.start()

    def run(self):
        torch.cuda.set_device(self.local_rank)
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def next(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __next__(self):
        return self.next()

    def __iter__(self):
        return self


class DataLoaderX(DataLoader):

    def __init__(self, local_rank, **kwargs):
        super(DataLoaderX, self).__init__(**kwargs)
        self.stream = torch.cuda.Stream(local_rank)
        self.local_rank = local_rank

    def __iter__(self):
        self.iter = super(DataLoaderX, self).__iter__()
        self.iter = BackgroundGenerator(self.iter, self.local_rank)
        self.preload()
        return self

    def preload(self):
        self.batch = next(self.iter, None)
        if self.batch is None:
            return None
        with torch.cuda.stream(self.stream):
            for k in range(len(self.batch)):
                self.batch[k] = self.batch[k].to(device=self.local_rank, non_blocking=True)

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        if batch is None:
            raise StopIteration
        self.preload()
        return batch


class MXFaceDataset(Dataset):
    def __init__(self, root_dir, local_rank):
        super(MXFaceDataset, self).__init__()
        self.transform = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
             ])
        self.root_dir = root_dir
        self.local_rank = local_rank
        path_imgrec = os.path.join(root_dir, 'train.rec')
        path_imgidx = os.path.join(root_dir, 'train.idx')
        self.imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
        s = self.imgrec.read_idx(0)
        header, _ = mx.recordio.unpack(s)
        if header.flag > 0:
            self.header0 = (int(header.label[0]), int(header.label[1]))
            self.imgidx = np.array(range(1, int(header.label[0])))
        else:
            self.imgidx = np.array(list(self.imgrec.keys))

    def __getitem__(self, index):
        idx = self.imgidx[index]
        s = self.imgrec.read_idx(idx)
        header, img = mx.recordio.unpack(s)
        label = header.label
        if not isinstance(label, numbers.Number):
            label = label[0]
        label = torch.tensor(label, dtype=torch.long)
        sample = mx.image.imdecode(img).asnumpy()
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label

    def __len__(self):
        return len(self.imgidx)


class EventNPZDataset(Dataset):
    """Event-camera .npz dataset driven by a prebuilt split index.

    Expected index format:
        abs_path<TAB>label
   
    """

    def __init__(
        self, root_dir, local_rank=0, image_size=112, 
        clip_val=5, to_three_channel=True, 
        augment=True, hist_key="final_hist", split="train", 
        index_file=None, index_dir=None, num_classes=None,
        max_classes=None, max_images_per_class=None, max_images=None,
        subset_seed=2048,
        ):
        super(EventNPZDataset, self).__init__()
        self.root_dir = root_dir
        self.local_rank = local_rank
        self.image_size = image_size
        self.clip_val = float(clip_val)
        self.to_three_channel = to_three_channel
        self.hist_key = hist_key
        self.split = _normalize_split_name(split)
        self.num_classes = num_classes      # <-- assign it

        # Restore index resolution (handles split/env-var/dir lookups)
        index_file = _resolve_event_index_file(
            root_dir=root_dir,
            split=self.split,
            index_file=index_file,
            index_dir=index_dir,
        )
        self.index_file = index_file        # <-- assign it (used in print)

        self.paths = []
        self.labels = []

        with open(index_file, "r") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                parts = ln.split("\t")
                self.paths.append(parts[0])
                if len(parts) >= 2 and parts[1] != "":
                    self.labels.append(int(parts[1]))   # fast path, no npz read
                else:
                    with np.load(parts[0], allow_pickle=False) as d:
                        self.labels.append(int(d["cls"].item()))

        if len(self.paths) == 0:
            raise RuntimeError(f"Index file is empty: {index_file}")

        self.num_index_classes = len(set(self.labels))
        self.num_index_images = len(self.paths)
        self._apply_subset(max_classes, max_images_per_class, max_images, subset_seed)

        # Always reflect what is actually in the (possibly subsetted) dataset.
        self.num_classes = len(set(self.labels))

        channels = 3 if to_three_channel else 1
        ops = [transforms.ToPILImage()]
        if image_size is not None:
            ops.append(transforms.Resize((image_size, image_size)))
        if augment:
            ops.append(transforms.RandomHorizontalFlip())
        ops += [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * channels, std=[0.5] * channels),
        ]
        self.transform = transforms.Compose(ops)

        print(
            f"[EventNPZDataset] split={self.split} | "
            f"samples={len(self.paths)}/{self.num_index_images} | "
            f"classes={self.num_classes}/{self.num_index_classes} | "
            f"index={self.index_file} | augment={augment}",
            flush=True,                     # <-- so you actually see it
        )

    def _apply_subset(self, max_classes, max_images_per_class, max_images, seed):
        """Deterministically restrict the dataset (same result on every rank).

        Applied in order:
          1. max_classes          - keep a random subset of identities
          2. max_images_per_class - cap images per kept identity
          3. max_images           - cap total images (may drop some identities entirely)
        Each limit is ignored when None or >= what is available.
        """
        if max_classes is None and max_images_per_class is None and max_images is None:
            return

        rng = np.random.default_rng(seed)

        by_class = {}
        for i, lab in enumerate(self.labels):
            by_class.setdefault(lab, []).append(i)

        classes = sorted(by_class)
        if max_classes is not None and max_classes < len(classes):
            classes = sorted(rng.choice(classes, size=max_classes, replace=False).tolist())

        keep = []
        for c in classes:
            idxs = by_class[c]
            if max_images_per_class is not None and len(idxs) > max_images_per_class:
                idxs = sorted(rng.choice(idxs, size=max_images_per_class, replace=False).tolist())
            keep.extend(idxs)

        if max_images is not None and len(keep) > max_images:
            keep = sorted(rng.choice(keep, size=max_images, replace=False).tolist())

        self.paths = [self.paths[i] for i in keep]
        self.labels = [self.labels[i] for i in keep]

        if len(self.paths) == 0:
            raise RuntimeError("Dataset subset is empty; check max_classes / max_images settings.")

    def _hist_to_uint8(self, final_hist):
        clip_val = self.clip_val
        img = np.clip(final_hist.astype(np.float32), -clip_val, clip_val)
        img = (img + clip_val) / (clip_val * 2.0)   # -> 0.0 .. 1.0
        img = np.squeeze(img)                       # drop singleton dims
        img = (img * 255.0).astype(np.uint8)        # -> 0 .. 255
        return img

    def __getitem__(self, index):
        path = self.paths[index]
        label = self.labels[index]          # use remapped contiguous label

        with np.load(path, allow_pickle=False) as data:
            final_hist = data[self.hist_key]

        img = self._hist_to_uint8(final_hist)

        if self.to_three_channel:
            if img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)
            elif img.ndim == 3 and img.shape[-1] == 1:
                img = np.repeat(img, 3, axis=-1)

        sample = self.transform(img)
        return sample, torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return len(self.paths)


class SyntheticDataset(Dataset):
    def __init__(self):
        super(SyntheticDataset, self).__init__()
        img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.int32)
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).squeeze(0).float()
        img = ((img / 255) - 0.5) / 0.5
        self.img = img
        self.label = 1

    def __getitem__(self, index):
        return self.img, self.label

    def __len__(self):
        return 1000000


def dali_data_iter(
    batch_size: int, rec_file: str, idx_file: str, num_threads: int,
    initial_fill=32768, random_shuffle=True,
    prefetch_queue_depth=1, local_rank=0, name="reader",
    mean=(127.5, 127.5, 127.5), 
    std=(127.5, 127.5, 127.5),
    dali_aug=False
    ):
    """
    Parameters:
    ----------
    initial_fill: int
        Size of the buffer that is used for shuffling. If random_shuffle is False, this parameter is ignored.

    """
    rank: int = distributed.get_rank()
    world_size: int = distributed.get_world_size()
    import nvidia.dali.fn as fn
    import nvidia.dali.types as types
    from nvidia.dali.pipeline import Pipeline
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator

    def dali_random_resize(img, resize_size, image_size=112):
        img = fn.resize(img, resize_x=resize_size, resize_y=resize_size)
        img = fn.resize(img, size=(image_size, image_size))
        return img
    def dali_random_gaussian_blur(img, window_size):
        img = fn.gaussian_blur(img, window_size=window_size * 2 + 1)
        return img
    def dali_random_gray(img, prob_gray):
        saturate = fn.random.coin_flip(probability=1 - prob_gray)
        saturate = fn.cast(saturate, dtype=types.FLOAT)
        img = fn.hsv(img, saturation=saturate)
        return img
    def dali_random_hsv(img, hue, saturation):
        img = fn.hsv(img, hue=hue, saturation=saturation)
        return img
    def multiplexing(condition, true_case, false_case):
        neg_condition = condition ^ True
        return condition * true_case + neg_condition * false_case

    condition_resize = fn.random.coin_flip(probability=0.1)
    size_resize = fn.random.uniform(range=(int(112 * 0.5), int(112 * 0.8)), dtype=types.FLOAT)
    condition_blur = fn.random.coin_flip(probability=0.2)
    window_size_blur = fn.random.uniform(range=(1, 2), dtype=types.INT32)
    condition_flip = fn.random.coin_flip(probability=0.5)
    condition_hsv = fn.random.coin_flip(probability=0.2)
    hsv_hue = fn.random.uniform(range=(0., 20.), dtype=types.FLOAT)
    hsv_saturation = fn.random.uniform(range=(1., 1.2), dtype=types.FLOAT)

    pipe = Pipeline(
        batch_size=batch_size, num_threads=num_threads,
        device_id=local_rank, prefetch_queue_depth=prefetch_queue_depth, )
    condition_flip = fn.random.coin_flip(probability=0.5)
    with pipe:
        jpegs, labels = fn.readers.mxnet(
            path=rec_file, index_path=idx_file, initial_fill=initial_fill, 
            num_shards=world_size, shard_id=rank,
            random_shuffle=random_shuffle, pad_last_batch=False, name=name)
        images = fn.decoders.image(jpegs, device="mixed", output_type=types.RGB)
        if dali_aug:
            images = fn.cast(images, dtype=types.UINT8)
            images = multiplexing(condition_resize, dali_random_resize(images, size_resize, image_size=112), images)
            images = multiplexing(condition_blur, dali_random_gaussian_blur(images, window_size_blur), images)
            images = multiplexing(condition_hsv, dali_random_hsv(images, hsv_hue, hsv_saturation), images)
            images = dali_random_gray(images, 0.1)

        images = fn.crop_mirror_normalize(
            images, dtype=types.FLOAT, mean=mean, std=std, mirror=condition_flip)
        pipe.set_outputs(images, labels)
    pipe.build()
    return DALIWarper(DALIClassificationIterator(pipelines=[pipe], reader_name=name, ))


class DALIWarper(object):
    def __init__(self, dali_iter):
        self.iter = dali_iter

    @torch.no_grad()
    def __next__(self):
        data_dict = self.iter.__next__()[0]
        tensor_data = data_dict['data'].cuda()
        tensor_label: torch.Tensor = data_dict['label'].cuda().long()
        tensor_label.squeeze_()
        return tensor_data, tensor_label

    def __iter__(self):
        return self

    def reset(self):
        self.iter.reset()