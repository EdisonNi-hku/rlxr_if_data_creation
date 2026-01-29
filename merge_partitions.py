#!/usr/bin/env python3
"""Merge partitioned datasets into a single dataset."""

from __future__ import annotations

import argparse
import os

from datasets import Dataset, concatenate_datasets, load_dataset, load_from_disk


def load_partition(path: str, split: str = "train") -> Dataset:
    """Load a dataset from HuggingFace Hub or local disk."""
    if os.path.isdir(path):
        return load_from_disk(path)
    try:
        return load_dataset(path, split=split)
    except Exception:
        dataset = load_dataset(path)
        if isinstance(dataset, dict):
            if split in dataset:
                return dataset[split]
            return dataset[next(iter(dataset.keys()))]
        return dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge partitioned datasets into a single dataset."
    )
    parser.add_argument(
        "--base_name",
        required=True,
        help=(
            "Base name of the partitioned datasets (without partition suffix). "
            "Example: 'JingweiNi/magpie_creative_dedup_filtered_checklist_train_v1'"
        ),
    )
    parser.add_argument(
        "--num_partitions",
        type=int,
        required=True,
        help="Total number of partitions to merge.",
    )
    parser.add_argument(
        "--save_to_disk",
        default=None,
        help="Path to save the merged dataset locally.",
    )
    parser.add_argument(
        "--push_to_hub",
        default=None,
        help="HuggingFace Hub repo_id to push the merged dataset.",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to use when loading from HuggingFace Hub (default: train).",
    )
    parser.add_argument(
        "--partition_format",
        default="{base}_{idx}_of_{total}",
        help=(
            "Format string for partition names. "
            "Available placeholders: {base}, {idx}, {total}. "
            "Default: '{base}_{idx}_of_{total}'"
        ),
    )
    args = parser.parse_args()

    if not args.save_to_disk and not args.push_to_hub:
        parser.error("Either --save_to_disk or --push_to_hub must be specified.")

    # Load all partitions
    partitions = []
    for idx in range(args.num_partitions):
        partition_name = args.partition_format.format(
            base=args.base_name,
            idx=idx,
            total=args.num_partitions,
        )
        print(f"[{idx + 1}/{args.num_partitions}] Loading: {partition_name}")
        try:
            partition = load_partition(partition_name, args.split)
            partitions.append(partition)
            print(f"  -> Loaded {len(partition)} samples")
        except Exception as e:
            print(f"  -> ERROR: Failed to load partition: {e}")
            raise

    # Merge partitions
    print(f"\n[MERGE] Concatenating {len(partitions)} partitions...")
    merged = concatenate_datasets(partitions)
    print(f"[MERGE] Total samples: {len(merged)}")

    # Save merged dataset
    if args.save_to_disk:
        print(f"\n[SAVE] Saving to disk: {args.save_to_disk}")
        merged.save_to_disk(args.save_to_disk)
        print("[SAVE] Done!")

    if args.push_to_hub:
        print(f"\n[PUSH] Pushing to HuggingFace Hub: {args.push_to_hub}")
        merged.push_to_hub(args.push_to_hub)
        print("[PUSH] Done!")

    print(f"\n[SUCCESS] Merged {args.num_partitions} partitions ({len(merged)} total samples)")


if __name__ == "__main__":
    main()
