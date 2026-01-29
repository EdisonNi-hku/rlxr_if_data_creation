#!/usr/bin/env python3
"""Load a HuggingFace dataset and create a test split if it doesn't exist."""

from __future__ import annotations

import argparse
import os

from datasets import DatasetDict, load_dataset, load_from_disk


def load_dataset_auto(path: str, split: str = None) -> DatasetDict:
    """Load a dataset from HuggingFace Hub or local disk."""
    if os.path.isdir(path):
        dataset = load_from_disk(path)
        # If load_from_disk returns a Dataset (not DatasetDict), wrap it
        if not isinstance(dataset, DatasetDict):
            return DatasetDict({"train": dataset})
        return dataset
    
    # Load from HuggingFace Hub
    if split:
        dataset = load_dataset(path, split=split)
        return DatasetDict({"train": dataset})
    return load_dataset(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load a HuggingFace dataset and create a test split if it doesn't exist."
    )
    parser.add_argument(
        "--input_dataset",
        required=True,
        help="Path or HuggingFace repo for the input dataset.",
    )
    parser.add_argument(
        "--save_to_disk",
        default=None,
        help="Path to save the dataset locally.",
    )
    parser.add_argument(
        "--push_to_hub",
        default=None,
        help="HuggingFace Hub repo_id to push the dataset.",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.15,
        help="Fraction of train to use for test split (default: 0.15).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splitting (default: 42).",
    )
    parser.add_argument(
        "--force_split",
        action="store_true",
        help="Force creating a new test split even if one already exists.",
    )
    args = parser.parse_args()

    if not args.save_to_disk and not args.push_to_hub:
        parser.error("Either --save_to_disk or --push_to_hub must be specified.")

    # Load dataset
    print(f"[LOAD] Loading dataset: {args.input_dataset}")
    dataset = load_dataset_auto(args.input_dataset)
    
    print(f"[INFO] Available splits: {list(dataset.keys())}")
    for split_name, split_data in dataset.items():
        print(f"  - {split_name}: {len(split_data)} samples")

    # Check if test split exists
    has_test = "test" in dataset
    
    if has_test and not args.force_split:
        print(f"\n[INFO] Test split already exists with {len(dataset['test'])} samples.")
        print("[INFO] Use --force_split to recreate the test split.")
    else:
        if has_test:
            print(f"\n[WARN] Overwriting existing test split (--force_split enabled)")
        
        if "train" not in dataset:
            print("[ERROR] No 'train' split found. Cannot create test split.")
            return
        
        # Create test split from train
        print(f"\n[SPLIT] Creating test split ({args.test_size * 100:.0f}% of train, seed={args.seed})")
        
        train_data = dataset["train"]
        split_result = train_data.train_test_split(
            test_size=args.test_size,
            seed=args.seed,
        )
        
        # Update dataset with new splits
        dataset["train"] = split_result["train"]
        dataset["test"] = split_result["test"]
        
        print(f"[SPLIT] New train size: {len(dataset['train'])} samples")
        print(f"[SPLIT] New test size: {len(dataset['test'])} samples")

    # Save dataset
    if args.save_to_disk:
        print(f"\n[SAVE] Saving to disk: {args.save_to_disk}")
        dataset.save_to_disk(args.save_to_disk)
        print("[SAVE] Done!")

    if args.push_to_hub:
        print(f"\n[PUSH] Pushing to HuggingFace Hub: {args.push_to_hub}")
        dataset.push_to_hub(args.push_to_hub)
        print("[PUSH] Done!")

    print("\n[SUCCESS] Dataset split complete!")
    print(f"  - Train: {len(dataset['train'])} samples")
    print(f"  - Test: {len(dataset['test'])} samples")


if __name__ == "__main__":
    main()
