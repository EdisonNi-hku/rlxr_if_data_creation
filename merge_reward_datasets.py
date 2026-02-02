#!/usr/bin/env python3
"""
Merge the outputs of analyze_checklist_reward_variance.py and analyze_rollouts.py.

Combines checklist soft/hard rewards with verifiable soft/hard rewards into a single dataset.

Example usage:
    python merge_reward_datasets.py \
        --checklist_dataset dataset_with_checklist_rewards \
        --verifiable_dataset dataset_with_verifiable_rewards \
        --output merged_dataset

    # Or merge into an existing base dataset
    python merge_reward_datasets.py \
        --base_dataset original_dataset \
        --checklist_dataset dataset_with_checklist_rewards \
        --verifiable_dataset dataset_with_verifiable_rewards \
        --output merged_dataset
"""

from __future__ import annotations

import argparse
import os

from datasets import load_dataset, load_from_disk
from tqdm import tqdm


def build_dataset(input_dataset: str, split: str):
    """Load dataset from various sources."""
    if os.path.isfile(input_dataset):
        return load_dataset(
            "json",
            data_files=input_dataset,
            split=split,
        )
    if os.path.isdir(input_dataset):
        dataset = load_from_disk(input_dataset)
        if hasattr(dataset, "keys"):
            return dataset[split]
        return dataset
    try:
        return load_dataset(input_dataset, split=split)
    except Exception:
        dataset = load_dataset(input_dataset)
        if isinstance(dataset, dict):
            if split in dataset:
                return dataset[split]
            return dataset[next(iter(dataset.keys()))]
        return dataset


def get_example_key(example: dict, index: int, dataset_name: str) -> str:
    """Generate a unique key for the example."""
    for field in ["id", "uuid", "key", "idx", "index"]:
        if field in example and example[field]:
            return str(example[field])
    return f"{dataset_name}_{index}"


def has_no_variance(values: list) -> bool:
    """Check if a list has no variance (all values are the same or list is empty/single element)."""
    if not values or len(values) <= 1:
        return True
    return len(set(values)) == 1


def build_key_map(dataset, dataset_name: str, columns: list[str]) -> dict:
    """Build a mapping from key to column values."""
    key_map = {}
    for index, example in enumerate(tqdm(dataset, desc=f"Indexing {dataset_name}")):
        key = get_example_key(example, index, dataset_name)
        key_map[key] = {col: example.get(col) for col in columns if col in example}
    return key_map


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge checklist and verifiable reward datasets."
    )
    parser.add_argument(
        "--base_dataset",
        type=str,
        default=None,
        help="Optional base dataset to merge into. If not provided, uses checklist_dataset as base.",
    )
    parser.add_argument(
        "--checklist_dataset",
        type=str,
        default=None,
        help="Dataset with checklist soft/hard rewards (from analyze_checklist_reward_variance.py).",
    )
    parser.add_argument(
        "--verifiable_dataset",
        type=str,
        default=None,
        help="Dataset with verifiable soft/hard rewards (from analyze_rollouts.py --grade_verifiable).",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path to save the merged dataset.",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to use (default: train).",
    )
    # Checklist column names
    parser.add_argument(
        "--checklist_soft_column",
        type=str,
        default="checklist_soft",
        help="Column name for checklist soft rewards in input (default: soft_rewards).",
    )
    parser.add_argument(
        "--checklist_hard_column",
        type=str,
        default="checklist_hard",
        help="Column name for checklist hard rewards in input (default: hard_rewards).",
    )
    parser.add_argument(
        "--output_checklist_soft_column",
        type=str,
        default="checklist_soft",
        help="Column name for checklist soft rewards in output (default: checklist_soft).",
    )
    parser.add_argument(
        "--output_checklist_hard_column",
        type=str,
        default="checklist_hard",
        help="Column name for checklist hard rewards in output (default: checklist_hard).",
    )
    # Verifiable column names
    parser.add_argument(
        "--verifiable_soft_column",
        type=str,
        default="verifiable_soft",
        help="Column name for verifiable soft rewards in input (default: soft_rewards).",
    )
    parser.add_argument(
        "--verifiable_hard_column",
        type=str,
        default="verifiable_hard",
        help="Column name for verifiable hard rewards in input (default: hard_rewards).",
    )
    parser.add_argument(
        "--output_verifiable_soft_column",
        type=str,
        default="verifiable_soft",
        help="Column name for verifiable soft rewards in output (default: verifiable_soft).",
    )
    parser.add_argument(
        "--output_verifiable_hard_column",
        type=str,
        default="verifiable_hard",
        help="Column name for verifiable hard rewards in output (default: verifiable_hard).",
    )
    parser.add_argument(
        "--no_variance_column",
        type=str,
        default="no_variance",
        help="Column name for no variance indicator (default: no_variance).",
    )
    args = parser.parse_args()

    if not args.checklist_dataset and not args.verifiable_dataset:
        parser.error("At least one of --checklist_dataset or --verifiable_dataset is required.")

    # Determine base dataset
    if args.base_dataset:
        base_path = args.base_dataset
    elif args.checklist_dataset:
        base_path = args.checklist_dataset
    elif args.verifiable_dataset:
        base_path = args.verifiable_dataset
    else:
        parser.error("No dataset provided.")

    print(f"Loading base dataset: {base_path}")
    base_dataset = build_dataset(base_path, args.split)
    base_name = os.path.basename(base_path).replace("/", "_")
    print(f"Base dataset has {len(base_dataset)} examples.")

    # Build key maps for checklist and verifiable datasets
    checklist_map = {}
    verifiable_map = {}

    if args.checklist_dataset:
        print(f"\nLoading checklist dataset: {args.checklist_dataset}")
        checklist_ds = build_dataset(args.checklist_dataset, args.split)
        checklist_name = os.path.basename(args.checklist_dataset).replace("/", "_")
        checklist_columns = [args.checklist_soft_column, args.checklist_hard_column]
        checklist_map = build_key_map(checklist_ds, checklist_name, checklist_columns)
        print(f"Indexed {len(checklist_map)} examples with checklist rewards.")

    if args.verifiable_dataset:
        print(f"\nLoading verifiable dataset: {args.verifiable_dataset}")
        verifiable_ds = build_dataset(args.verifiable_dataset, args.split)
        verifiable_name = os.path.basename(args.verifiable_dataset).replace("/", "_")
        verifiable_columns = [args.verifiable_soft_column, args.verifiable_hard_column]
        verifiable_map = build_key_map(verifiable_ds, verifiable_name, verifiable_columns)
        print(f"Indexed {len(verifiable_map)} examples with verifiable rewards.")

    # Merge columns
    checklist_soft_data = []
    checklist_hard_data = []
    verifiable_soft_data = []
    verifiable_hard_data = []
    no_variance_data = []

    missing_checklist = 0
    missing_verifiable = 0
    no_variance_count = 0

    for index, example in enumerate(tqdm(base_dataset, desc="Merging datasets")):
        key = get_example_key(example, index, base_name)

        # Get checklist rewards
        if args.checklist_dataset:
            checklist_data = checklist_map.get(key, {})
            checklist_soft = checklist_data.get(args.checklist_soft_column, [])
            checklist_hard = checklist_data.get(args.checklist_hard_column, [])
            if not checklist_soft and not checklist_hard:
                missing_checklist += 1
        else:
            # Try to get from base dataset if columns exist
            checklist_soft = example.get(args.checklist_soft_column, [])
            checklist_hard = example.get(args.checklist_hard_column, [])

        checklist_soft_data.append(checklist_soft if checklist_soft else [])
        checklist_hard_data.append(checklist_hard if checklist_hard else [])

        # Get verifiable rewards
        if args.verifiable_dataset:
            verifiable_data = verifiable_map.get(key, {})
            verifiable_soft = verifiable_data.get(args.verifiable_soft_column, [])
            verifiable_hard = verifiable_data.get(args.verifiable_hard_column, [])
            if not verifiable_soft and not verifiable_hard:
                missing_verifiable += 1
        else:
            # Try to get from base dataset if columns exist
            verifiable_soft = example.get(args.verifiable_soft_column, [])
            verifiable_hard = example.get(args.verifiable_hard_column, [])

        verifiable_soft_data.append(verifiable_soft if verifiable_soft else [])
        verifiable_hard_data.append(verifiable_hard if verifiable_hard else [])

        # Compute no_variance: True if either checklist_hard or verifiable_soft has no variance
        checklist_hard_no_var = has_no_variance(checklist_hard) if checklist_hard else True
        verifiable_soft_no_var = has_no_variance(verifiable_soft) if verifiable_soft else True
        no_var = checklist_hard_no_var or verifiable_soft_no_var
        no_variance_data.append(no_var)
        if no_var:
            no_variance_count += 1

    # Print merge statistics
    total = len(base_dataset)
    print("\n" + "=" * 60)
    print("MERGE STATISTICS")
    print("=" * 60)
    print(f"Total examples in base dataset: {total}")
    if args.checklist_dataset:
        print(f"Missing checklist rewards: {missing_checklist}")
    if args.verifiable_dataset:
        print(f"Missing verifiable rewards: {missing_verifiable}")
    print(f"No variance (checklist_hard OR verifiable_soft): {no_variance_count} ({no_variance_count / total:.2%})")
    print("=" * 60)

    # Add columns to dataset
    # First, remove existing columns if they exist (to avoid duplicates)
    columns_to_add = []
    
    if args.checklist_dataset or args.checklist_soft_column in base_dataset.column_names:
        columns_to_add.append((args.output_checklist_soft_column, checklist_soft_data))
        columns_to_add.append((args.output_checklist_hard_column, checklist_hard_data))
    
    if args.verifiable_dataset or args.verifiable_soft_column in base_dataset.column_names:
        columns_to_add.append((args.output_verifiable_soft_column, verifiable_soft_data))
        columns_to_add.append((args.output_verifiable_hard_column, verifiable_hard_data))

    # Add no_variance column
    columns_to_add.append((args.no_variance_column, no_variance_data))

    output_dataset = base_dataset
    for col_name, col_data in columns_to_add:
        # Remove column if it already exists
        if col_name in output_dataset.column_names:
            output_dataset = output_dataset.remove_columns([col_name])
        output_dataset = output_dataset.add_column(col_name, col_data)

    # Save merged dataset
    output_dataset.save_to_disk(args.output)
    print(f"\nMerged dataset saved to: {args.output}")
    print(f"Columns added:")
    for col_name, _ in columns_to_add:
        print(f"  - {col_name}")


if __name__ == "__main__":
    main()
