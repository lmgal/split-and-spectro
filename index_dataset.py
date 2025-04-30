#!/usr/bin/env python3
"""
Split dataset directories into train and test sets and create index for training set.
Usage: python index_dataset.py SPLIT OUTPUT_NAME DIR [DIR ...]
  SPLIT: float in (0,1) fraction for training split.
  OUTPUT_NAME: base name for output directories; creates OUTPUT_NAME_train and OUTPUT_NAME_test.
  DIR [DIR ...]: one or more input directories for each category.
Generates OUTPUT_NAME_train/index.csv mapping file paths to category IDs.
"""
import os
import sys
import csv
import shutil  # for file copying and directory operations
import random

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Split dataset and create training index.")
    parser.add_argument("split", type=float,
                        help="Fraction of data to use for training (0 < split < 1)")
    parser.add_argument("output_name", type=str,
                        help="Base name for output directories (train/test)")
    parser.add_argument("dirs", nargs="+",
                        help="One or more input directories for each category")
    args = parser.parse_args()

    split = args.split
    if not (0.0 < split < 1.0):
        parser.error("split must be between 0 and 1")

    output_name = args.output_name
    cwd = os.getcwd()
    train_dir = os.path.join(cwd, f"{output_name}_train")
    test_dir = os.path.join(cwd, f"{output_name}_test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Create data subdirectories for train and test sets
    train_data_dir = os.path.join(train_dir, 'data')
    test_data_dir = os.path.join(test_dir, 'data')
    os.makedirs(train_data_dir, exist_ok=True)
    os.makedirs(test_data_dir, exist_ok=True)

    items = [(d, os.path.basename(os.path.normpath(d))) for d in args.dirs]
    items = [(d, cat) for d, cat in items if os.path.isdir(d)]
    if not items:
        sys.exit("No valid input directories provided.")
    items.sort(key=lambda x: x[1])
    category_to_id = {cat: idx for idx, (_, cat) in enumerate(items)}

    for dir_path, cat in items:
        files = []
        for root, _, fs in os.walk(dir_path):
            for fname in sorted(fs):
                files.append(os.path.join(root, fname))
        if not files:
            sys.stderr.write(f"Warning: no files found in {dir_path}, skipping\n")
            continue
        random.shuffle(files)
        n_train = int(len(files) * split)
        train_files = files[:n_train]
        test_files = files[n_train:]

        # Copy files into data/<category> subdirectories
        train_cat = os.path.join(train_data_dir, cat)
        test_cat = os.path.join(test_data_dir, cat)
        os.makedirs(train_cat, exist_ok=True)
        os.makedirs(test_cat, exist_ok=True)

        for src in train_files:
            shutil.copy2(src, os.path.join(train_cat, os.path.basename(src)))
        for src in test_files:
            shutil.copy2(src, os.path.join(test_cat, os.path.basename(src)))

    rows = []
    for root, _, files in os.walk(train_dir):
        for fname in sorted(files):
            if fname == 'index.csv':
                continue
            file_path = os.path.join(root, fname)
            rel_path = os.path.relpath(file_path, train_dir).replace(os.sep, '/')
            if not rel_path.startswith('.'): 
                rel_path = './' + rel_path
            cat = os.path.basename(os.path.dirname(file_path))
            cid = category_to_id.get(cat)
            rows.append((rel_path, cid))

    index_csv = os.path.join(train_dir, 'index.csv')
    with open(index_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['x', 'y'])
        for path, cid in rows:
            writer.writerow([path, cid])

    print(f"Wrote {len(rows)} entries to {index_csv}")

    rows = []
    for root, _, files in os.walk(test_dir):
        for fname in sorted(files):
            if fname == 'index.csv':
                continue
            file_path = os.path.join(root, fname)
            rel_path = os.path.relpath(file_path, test_dir).replace(os.sep, '/')
            if not rel_path.startswith('.'): 
                rel_path = './' + rel_path
            cat = os.path.basename(os.path.dirname(file_path))
            cid = category_to_id.get(cat)
            rows.append((rel_path, cid))

    index_csv = os.path.join(test_dir, 'index.csv')
    with open(index_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['x', 'y'])
        for path, cid in rows:
            writer.writerow([path, cid])

    print(f"Wrote {len(rows)} entries to {index_csv}")

if __name__ == '__main__':
    main()
