#!/usr/bin/env python3
"""
build_label_index.py

Build train/val/test index files for EventNPZDataset.

The generated index files contain ONE ABSOLUTE PATH PER LINE:

    /data/Glint360k-paul/id000001/00001.npz
    /data/Glint360k-paul/id000001/00002.npz
    ...

Labels are NOT written because EventNPZDataset reads
the identity directly from data["cls"].

Usage
-----
python build_label_index.py \
    --root /data/Glint360k-paul \
    --output ./indexes \
    --train_ratio 0.7 \
    --val_ratio 0.15 \
    --test_ratio 0.15
"""

import argparse
import glob
import os
import random

import numpy as np
from tqdm import tqdm


def main(args):

    random.seed(args.seed)

    pattern = os.path.join(args.root, "*", "*.npz")
    files = sorted(glob.glob(pattern))

    if len(files) == 0:
        raise RuntimeError(f"No npz files found under {args.root}")

    print(f"Found {len(files):,} npz files.")

    labels = []

    valid_files = []

    print("Checking dataset...")

    for file in tqdm(files):

        try:
            with np.load(file, allow_pickle=False) as data:

                if "cls" not in data:
                    print(f"Missing cls: {file}")
                    continue

                label = int(data["cls"].item())

            valid_files.append(file)
            labels.append(label)

        except Exception as e:
            print(f"Skipping {file}")
            print(e)

    print(f"\nValid files : {len(valid_files):,}")

    unique_labels = len(set(labels))

    print(f"Unique identities : {unique_labels:,}")

    indices = list(range(len(valid_files)))
    random.shuffle(indices)

    n = len(indices)

    n_train = int(n * args.train_ratio)
    n_val = int(n * args.val_ratio)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    os.makedirs(args.output, exist_ok=True)

    train_file = os.path.join(args.output, "train_index.txt")
    val_file = os.path.join(args.output, "val_index.txt")
    test_file = os.path.join(args.output, "test_index.txt")

    def write_index(filename, idxs):

        with open(filename, "w") as f:

            for idx in idxs:
                f.write(valid_files[idx] + "\n")

    write_index(train_file, train_idx)
    write_index(val_file, val_idx)
    write_index(test_file, test_idx)

    print("\n================ Dataset Summary ================")
    print(f"Total images      : {n:,}")
    print(f"Train images      : {len(train_idx):,}")
    print(f"Validation images : {len(val_idx):,}")
    print(f"Test images       : {len(test_idx):,}")
    print(f"Unique IDs        : {unique_labels:,}")
    print("=================================================")

    print("\nSaved:")
    print(train_file)
    print(val_file)
    print(test_file)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--root", required=True,
                        help="Root directory of Glint360 NPZ dataset")

    parser.add_argument("--output", default="./indexes")

    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-6:
        raise ValueError("Ratios must sum to 1.")

    main(args)