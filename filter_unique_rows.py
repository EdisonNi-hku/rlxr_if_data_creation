#!/usr/bin/env python3
"""
Filter rows from dataset1 that do not appear in dataset2 based on a key column.

Usage:
    python filter_unique_rows.py \
        --dataset1 username/dataset1 \
        --dataset2 username/dataset2 \
        --key_column instruction \
        --output_repo username/unique_rows \
        [--split1 train] \
        [--split2 train] \
        [--private]

Examples:
    # Using HuggingFace Hub datasets
    python filter_unique_rows.py \
        --dataset1 org/full_dataset \
        --dataset2 org/subset_dataset \
        --key_column id \
        --output_repo org/unique_entries

    # Using local datasets (paths)
    python filter_unique_rows.py \
        --dataset1 ./local_dataset1 \
        --dataset2 ./local_dataset2 \
        --key_column text \
        --output_repo myuser/filtered_dataset
"""

from __future__ import annotations

import argparse
import hashlib
import os
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from tqdm import tqdm


def load_dataset_smart(
    name_or_path: str,
    split: str | None = None,
) -> Dataset:
    """Load a dataset from HuggingFace Hub or local path."""
    # Check if it's a local directory (saved dataset)
    if os.path.isdir(name_or_path):
        dataset = load_from_disk(name_or_path)
        if isinstance(dataset, DatasetDict):
            if split and split in dataset:
                return dataset[split]
            # Return first available split
            return dataset[list(dataset.keys())[0]]
        return dataset

    # Check if it's a local file (JSONL/JSON)
    if name_or_path.endswith(".jsonl") or name_or_path.endswith(".json"):
        return load_dataset("json", data_files=name_or_path, split="train")

    # Load from HuggingFace Hub
    try:
        if split:
            return load_dataset(name_or_path, split=split)
        else:
            dataset = load_dataset(name_or_path)
            if isinstance(dataset, DatasetDict):
                # Prefer 'train' split, otherwise take the first one
                if "train" in dataset:
                    return dataset["train"]
                return dataset[list(dataset.keys())[0]]
            return dataset
    except Exception as e:
        raise ValueError(f"Failed to load dataset '{name_or_path}': {e}")


def normalize_value(value: Any) -> str:
    """Normalize a value for comparison (handles whitespace, None, etc.)."""
    if value is None:
        return ""
    if isinstance(value, str):
        return " ".join(value.strip().split())
    return str(value)


def compute_hash(value: Any) -> str:
    """Compute a hash for a value (useful for long strings)."""
    normalized = normalize_value(value)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter rows from dataset1 that don't appear in dataset2 based on a key column.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset1",
        required=True,
        help="First dataset (HuggingFace Hub name or local path). Rows from this dataset will be filtered.",
    )
    parser.add_argument(
        "--dataset2",
        required=True,
        help="Second dataset (HuggingFace Hub name or local path). Used as reference for filtering.",
    )
    parser.add_argument(
        "--key_column",
        required=True,
        help="Column name to use as the key for comparison.",
    )
    parser.add_argument(
        "--output_repo",
        required=True,
        help="HuggingFace Hub repo ID to push the filtered dataset (e.g., username/dataset_name).",
    )
    parser.add_argument(
        "--split1",
        default=None,
        help="Split to use from dataset1 (default: auto-detect, prefers 'train').",
    )
    parser.add_argument(
        "--split2",
        default=None,
        help="Split to use from dataset2 (default: auto-detect, prefers 'train').",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        default=False,
        help="Create the output repository as private.",
    )
    parser.add_argument(
        "--save_local",
        default=None,
        help="Optional: also save the filtered dataset locally to this path.",
    )
    parser.add_argument(
        "--use_hash",
        action="store_true",
        default=False,
        help="Use hashing for comparison (recommended for long text fields).",
    )

    args = parser.parse_args()

    # Load datasets
    print(f"Loading dataset1: {args.dataset1}...")
    dataset1 = load_dataset_smart(args.dataset1, args.split1)
    print(f"  Loaded {len(dataset1)} rows")

    print(f"Loading dataset2: {args.dataset2}...")
    dataset2 = load_dataset_smart(args.dataset2, args.split2)
    print(f"  Loaded {len(dataset2)} rows")

    # Validate key column exists in both datasets
    if args.key_column not in dataset1.column_names:
        raise ValueError(
            f"Key column '{args.key_column}' not found in dataset1. "
            f"Available columns: {dataset1.column_names}"
        )
    if args.key_column not in dataset2.column_names:
        raise ValueError(
            f"Key column '{args.key_column}' not found in dataset2. "
            f"Available columns: {dataset2.column_names}"
        )

    # Build set of keys from dataset2
    print(f"Building key set from dataset2 (column: '{args.key_column}')...")
    dataset2_keys = set()
    for row in tqdm(dataset2, desc="Processing dataset2"):
        value = row.get(args.key_column)
        if args.use_hash:
            key = compute_hash(value)
        else:
            key = normalize_value(value)
        dataset2_keys.add(key)
    print(f"  Found {len(dataset2_keys)} unique keys in dataset2")

    # Filter dataset1 to keep only rows with keys not in dataset2
    print("Filtering dataset1 for unique rows...")
    keep_indices = []
    duplicates = 0

    for idx, row in tqdm(enumerate(dataset1), total=len(dataset1), desc="Filtering"):
        value = row.get(args.key_column)
        if args.use_hash:
            key = compute_hash(value)
        else:
            key = normalize_value(value)

        if key not in dataset2_keys:
            keep_indices.append(idx)
        else:
            duplicates += 1

    # Create filtered dataset
    filtered_dataset = dataset1.select(keep_indices)

    print(f"\n{'='*50}")
    print(f"Results:")
    print(f"  Dataset1 original size: {len(dataset1)}")
    print(f"  Dataset2 reference size: {len(dataset2)}")
    print(f"  Duplicates found (removed): {duplicates}")
    print(f"  Unique rows (kept): {len(filtered_dataset)}")
    print(f"{'='*50}\n")

    # Save locally if requested
    if args.save_local:
        print(f"Saving filtered dataset locally to: {args.save_local}")
        filtered_dataset.save_to_disk(args.save_local)

    # Push to Hub
    print(f"Pushing filtered dataset to Hub: {args.output_repo}")
    filtered_dataset.push_to_hub(args.output_repo, private=args.private)
    print(f"\nSuccess! Dataset pushed to: https://huggingface.co/datasets/{args.output_repo}")


if __name__ == "__main__":
    main()
