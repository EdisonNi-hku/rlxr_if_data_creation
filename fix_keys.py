#!/usr/bin/env python3
"""Remove UUID prefixes from the key column of a HuggingFace dataset."""

from __future__ import annotations

import argparse
import os
import re

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from tqdm import tqdm


def extract_core_key(key: str) -> str:
    """
    Extract the core key (typically UUID) from a prefixed key.
    
    Examples:
        'magpie_creative_dedup_filtered_train_v1_magpie_creative_dedup_augmented_train_v1_magpie_creative_dedup_fde65e91-6663-504b-8241-203e97e03a45'
        -> 'fde65e91-6663-504b-8241-203e97e03a45'
        
        'dataset_name_12345'
        -> '12345'
    """
    # Try to find UUID pattern (8-4-4-4-12 hex format)
    uuid_pattern = r'[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}'
    uuid_match = re.search(uuid_pattern, key, re.IGNORECASE)
    if uuid_match:
        return uuid_match.group(0)
    
    # Fallback: take the last underscore-separated part
    parts = key.rsplit('_', 1)
    if len(parts) > 1:
        return parts[-1]
    
    return key


def load_dataset_auto(path: str, split: str = None) -> Dataset | DatasetDict:
    """Load a dataset from HuggingFace Hub or local disk."""
    if os.path.isdir(path):
        return load_from_disk(path)
    
    # Load from HuggingFace Hub
    if split:
        return load_dataset(path, split=split)
    return load_dataset(path)


def fix_keys_in_dataset(dataset: Dataset, key_field: str = "key") -> Dataset:
    """Remove prefixes from keys in a dataset."""
    if key_field not in dataset.column_names:
        print(f"[WARN] Column '{key_field}' not found in dataset. Skipping.")
        return dataset
    
    def fix_key(example):
        if key_field in example and example[key_field]:
            example[key_field] = extract_core_key(str(example[key_field]))
        return example
    
    return dataset.map(fix_key, desc="Fixing keys")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove UUID prefixes from the key column of a HuggingFace dataset."
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
        "--split",
        default=None,
        help="Dataset split to process. If not specified, processes all splits.",
    )
    parser.add_argument(
        "--key_field",
        default="key",
        help="Name of the key column (default: 'key').",
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=5,
        help="Number of examples to preview before/after (default: 5, set to 0 to disable).",
    )
    args = parser.parse_args()

    if not args.save_to_disk and not args.push_to_hub:
        parser.error("Either --save_to_disk or --push_to_hub must be specified.")

    # Load dataset
    print(f"[LOAD] Loading dataset: {args.input_dataset}")
    dataset = load_dataset_auto(args.input_dataset, args.split)

    # Handle both Dataset and DatasetDict
    if isinstance(dataset, DatasetDict):
        print(f"[INFO] Found splits: {list(dataset.keys())}")
        
        # Preview before
        if args.preview > 0:
            first_split = list(dataset.keys())[0]
            print(f"\n[PREVIEW] Before (first {args.preview} keys from '{first_split}'):")
            for i, example in enumerate(dataset[first_split]):
                if i >= args.preview:
                    break
                print(f"  {example.get(args.key_field, 'N/A')}")
        
        # Fix keys in all splits
        for split_name in dataset.keys():
            print(f"\n[FIX] Processing split: {split_name}")
            dataset[split_name] = fix_keys_in_dataset(dataset[split_name], args.key_field)
        
        # Preview after
        if args.preview > 0:
            print(f"\n[PREVIEW] After (first {args.preview} keys from '{first_split}'):")
            for i, example in enumerate(dataset[first_split]):
                if i >= args.preview:
                    break
                print(f"  {example.get(args.key_field, 'N/A')}")
    else:
        print(f"[INFO] Dataset has {len(dataset)} samples")
        
        # Preview before
        if args.preview > 0:
            print(f"\n[PREVIEW] Before (first {args.preview} keys):")
            for i, example in enumerate(dataset):
                if i >= args.preview:
                    break
                print(f"  {example.get(args.key_field, 'N/A')}")
        
        # Fix keys
        print("\n[FIX] Processing dataset...")
        dataset = fix_keys_in_dataset(dataset, args.key_field)
        
        # Preview after
        if args.preview > 0:
            print(f"\n[PREVIEW] After (first {args.preview} keys):")
            for i, example in enumerate(dataset):
                if i >= args.preview:
                    break
                print(f"  {example.get(args.key_field, 'N/A')}")

    # Save dataset
    if args.save_to_disk:
        print(f"\n[SAVE] Saving to disk: {args.save_to_disk}")
        dataset.save_to_disk(args.save_to_disk)
        print("[SAVE] Done!")

    if args.push_to_hub:
        print(f"\n[PUSH] Pushing to HuggingFace Hub: {args.push_to_hub}")
        dataset.push_to_hub(args.push_to_hub)
        print("[PUSH] Done!")

    print("\n[SUCCESS] Key fixing complete!")


if __name__ == "__main__":
    main()
