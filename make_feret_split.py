#!/usr/bin/env python
"""
Create a deterministic, subject-level (identity-disjoint) train/test split for FERET.

Run ONCE before anything else. Writes two files that every downstream script reuses,
so zero-shot eval, fine-tuning, and fine-tuned eval all operate on the SAME held-out
identities — a controlled comparison.

    python make_feret_split.py \
        --feret_db /data/facerecognition_events_paul/FERET/colorferet_events \
        --output_dir feret_split --train_ratio 0.7 --seed 42

Outputs:
    feret_split/train_subjects.txt   (~70% of subjects, for fine-tuning)
    feret_split/test_subjects.txt    (~30% of subjects, held out for evaluation)
    feret_split/split_info.json
"""
import os
import json
import random
import argparse


def main(args):
    subjects = sorted(d for d in os.listdir(args.feret_db)
                      if os.path.isdir(os.path.join(args.feret_db, d)))
    if not subjects:
        raise FileNotFoundError(f"No subject folders in {args.feret_db}")

    rng = random.Random(args.seed)
    shuffled = subjects[:]
    rng.shuffle(shuffled)

    n_train = int(round(len(shuffled) * args.train_ratio))
    train = sorted(shuffled[:n_train])
    test = sorted(shuffled[n_train:])

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "train_subjects.txt"), "w") as f:
        f.write("\n".join(train) + "\n")
    with open(os.path.join(args.output_dir, "test_subjects.txt"), "w") as f:
        f.write("\n".join(test) + "\n")
    with open(os.path.join(args.output_dir, "split_info.json"), "w") as f:
        json.dump({"feret_db": args.feret_db, "seed": args.seed,
                   "train_ratio": args.train_ratio,
                   "num_subjects_total": len(subjects),
                   "num_train": len(train), "num_test": len(test)}, f, indent=2)

    # overlap sanity check
    assert not (set(train) & set(test)), "train/test subject overlap — split is broken"

    print(f"Total subjects : {len(subjects)}")
    print(f"Train subjects : {len(train)}  -> {os.path.join(args.output_dir,'train_subjects.txt')}")
    print(f"Test subjects  : {len(test)}  -> {os.path.join(args.output_dir,'test_subjects.txt')}")
    print(f"Overlap        : 0 (identity-disjoint)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--feret_db", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="feret_split")
    p.add_argument("--train_ratio", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=42)
    main(p.parse_args())