#!/usr/bin/env python3
"""Merge datasets from pipeline steps (augmented, filtered, checklist) by their keys."""

from __future__ import annotations

import argparse
import os
import re
from collections import defaultdict

from datasets import Dataset, load_dataset, load_from_disk
from tqdm import tqdm


def load_dataset_auto(path: str, split: str = "train") -> Dataset:
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


def build_key_map(dataset: Dataset, key_field: str = "key") -> dict[str, dict]:
    """Build a mapping from core key to row data."""
    key_map = {}
    for row in dataset:
        if key_field not in row:
            continue
        core_key = extract_core_key(row[key_field])
        key_map[core_key] = row
    return key_map


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge datasets from 3 pipeline steps by their keys."
    )
    parser.add_argument(
        "--augmented",
        required=True,
        help="Path or HuggingFace repo for the augmented dataset (base).",
    )
    parser.add_argument(
        "--filtered",
        default=None,
        help="Path or HuggingFace repo for the filtered dataset (contradiction check results).",
    )
    parser.add_argument(
        "--checklist",
        required=True,
        help="Path or HuggingFace repo for the checklist dataset.",
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
        "--key_field",
        default="key",
        help="Field name containing the key for matching (default: 'key').",
    )
    parser.add_argument(
        "--filtered_fields",
        default="contradiction_label,justification",
        help="Comma-separated fields to merge from filtered dataset (default: 'contradiction_label,justification').",
    )
    parser.add_argument(
        "--checklist_fields",
        default="checklist",
        help="Comma-separated fields to merge from checklist dataset (default: 'checklist').",
    )
    parser.add_argument(
        "--filter_contradictory",
        action="store_true",
        help="If set, exclude rows where contradiction_label is 'Contradictory'.",
    )
    parser.add_argument(
        "--split_by_raw_instruction",
        action="store_true",
        help="If set, split merged output into empty-raw_instruction and non-empty partitions.",
    )
    parser.add_argument(
        "--empty_raw_instruction_value",
        default="<EMPTY>",
        help="Raw instruction value treated as empty for splitting (default: '<EMPTY>').",
    )
    parser.add_argument(
        "--empty_suffix",
        default="",
        help="Suffix for the empty-raw_instruction output (default: '').",
    )
    args = parser.parse_args()

    if not args.save_to_disk and not args.push_to_hub:
        parser.error("Either --save_to_disk or --push_to_hub must be specified.")

    filtered_fields = [f.strip() for f in args.filtered_fields.split(",") if f.strip()]
    checklist_fields = [f.strip() for f in args.checklist_fields.split(",") if f.strip()]
    conflict_mode = args.filtered is None
    if conflict_mode:
        filtered_fields = []
        if args.filter_contradictory:
            parser.error("--filter_contradictory requires --filtered.")
        if not args.split_by_raw_instruction:
            args.split_by_raw_instruction = True
            print("[INFO] No --filtered provided; enabling --split_by_raw_instruction.")

    # Load datasets
    print(f"[LOAD] Loading augmented dataset: {args.augmented}")
    augmented_ds = load_dataset_auto(args.augmented, args.split)
    print(f"  -> {len(augmented_ds)} samples")

    filtered_ds = None
    if args.filtered:
        print(f"[LOAD] Loading filtered dataset: {args.filtered}")
        filtered_ds = load_dataset_auto(args.filtered, args.split)
        print(f"  -> {len(filtered_ds)} samples")

    print(f"[LOAD] Loading checklist dataset: {args.checklist}")
    checklist_ds = load_dataset_auto(args.checklist, args.split)
    print(f"  -> {len(checklist_ds)} samples")

    # Build key maps
    print("\n[INDEX] Building key maps...")
    filtered_map = {}
    if filtered_ds is not None:
        filtered_map = build_key_map(filtered_ds, args.key_field)
        print(f"  -> Filtered: {len(filtered_map)} unique keys")
    
    checklist_map = build_key_map(checklist_ds, args.key_field)
    print(f"  -> Checklist: {len(checklist_map)} unique keys")

    # Merge datasets
    print("\n[MERGE] Merging datasets...")
    merged_rows = []
    stats = defaultdict(int)

    for row in tqdm(augmented_ds, desc="Merging"):
        if args.key_field not in row:
            stats["missing_key"] += 1
            continue

        core_key = extract_core_key(row[args.key_field])
        merged_row = dict(row)
        merged_row["core_key"] = core_key

        # Merge filtered fields
        if filtered_fields:
            if core_key in filtered_map:
                filtered_row = filtered_map[core_key]
                for field in filtered_fields:
                    if field in filtered_row:
                        merged_row[field] = filtered_row[field]
                stats["filtered_matched"] += 1
            else:
                stats["filtered_missing"] += 1
                # Add None for missing filtered fields
                for field in filtered_fields:
                    merged_row[field] = None

        # Merge checklist fields
        raw_instruction_for_split = None
        if core_key in checklist_map:
            checklist_row = checklist_map[core_key]
            if args.split_by_raw_instruction:
                raw_instruction_for_split = checklist_row.get("raw_instruction")
            for field in checklist_fields:
                if field in checklist_row:
                    merged_row[field] = checklist_row[field]
            stats["checklist_matched"] += 1
        else:
            stats["checklist_missing"] += 1
            # Add None for missing checklist fields
            for field in checklist_fields:
                merged_row[field] = None
        if args.split_by_raw_instruction:
            merged_row["_raw_instruction_for_split"] = raw_instruction_for_split

        # Optionally filter out contradictory samples
        if args.filter_contradictory:
            label = merged_row.get("contradiction_label", "")
            if label and "contradictory" in str(label).lower() and "not" not in str(label).lower():
                stats["filtered_out_contradictory"] += 1
                continue

        merged_rows.append(merged_row)

    # Print statistics
    print("\n[STATS] Merge statistics:")
    print(f"  - Total augmented samples: {len(augmented_ds)}")
    if filtered_fields:
        print(f"  - Filtered matched: {stats['filtered_matched']}")
        print(f"  - Filtered missing: {stats['filtered_missing']}")
    print(f"  - Checklist matched: {stats['checklist_matched']}")
    print(f"  - Checklist missing: {stats['checklist_missing']}")
    if args.filter_contradictory:
        print(f"  - Filtered out (contradictory): {stats['filtered_out_contradictory']}")
    print(f"  - Final merged samples: {len(merged_rows)}")

    if args.split_by_raw_instruction:
        empty_token = args.empty_raw_instruction_value
        empty_rows = []
        non_empty_rows = []
        missing_augmented = 0
        for row in merged_rows:
            raw_instruction = row.get("raw_instruction")
            if raw_instruction is None:
                raw_instruction = row.get("_raw_instruction_for_split")
            if isinstance(raw_instruction, str) and raw_instruction.strip() == empty_token:
                empty_rows.append(row)
            else:
                non_empty_rows.append(row)
                augmented_prompt = row.get("augmented_prompt")
                if not isinstance(augmented_prompt, str) or not augmented_prompt.strip():
                    missing_augmented += 1

        for row in merged_rows:
            if "_raw_instruction_for_split" in row:
                del row["_raw_instruction_for_split"]

        print("\n[SPLIT] Split by raw_instruction:")
        print(f"  - Empty raw_instruction: {len(empty_rows)}")
        print(f"  - Non-empty augmented: {len(non_empty_rows)}")
        if missing_augmented:
            print(f"  - Non-empty missing/blank augmented_prompt: {missing_augmented}")

        empty_ds = Dataset.from_list(empty_rows)
        non_empty_ds = Dataset.from_list(non_empty_rows)

        if args.save_to_disk:
            empty_path = f"{args.save_to_disk}{args.empty_suffix}"
            non_empty_path = f"{args.save_to_disk}_non_empty_augmented"
            print(f"\n[SAVE] Saving empty raw_instruction to: {empty_path}")
            empty_ds.save_to_disk(empty_path)
            print("[SAVE] Done!")
            print(f"\n[SAVE] Saving non-empty augmented to: {non_empty_path}")
            non_empty_ds.save_to_disk(non_empty_path)
            print("[SAVE] Done!")

        if args.push_to_hub:
            empty_repo = f"{args.push_to_hub}{args.empty_suffix}"
            non_empty_repo = f"{args.push_to_hub}_non_empty_augmented"
            print(f"\n[PUSH] Pushing empty raw_instruction to: {empty_repo}")
            empty_ds.push_to_hub(empty_repo)
            print("[PUSH] Done!")
            print(f"\n[PUSH] Pushing non-empty augmented to: {non_empty_repo}")
            non_empty_ds.push_to_hub(non_empty_repo)
            print("[PUSH] Done!")

        print(
            f"\n[SUCCESS] Merged pipeline steps and split outputs "
            f"({len(merged_rows)} total samples)"
        )
        return

    # Create merged dataset
    merged_ds = Dataset.from_list(merged_rows)

    # Save merged dataset
    if args.save_to_disk:
        print(f"\n[SAVE] Saving to disk: {args.save_to_disk}")
        merged_ds.save_to_disk(args.save_to_disk)
        print("[SAVE] Done!")

    if args.push_to_hub:
        print(f"\n[PUSH] Pushing to HuggingFace Hub: {args.push_to_hub}")
        merged_ds.push_to_hub(args.push_to_hub)
        print("[PUSH] Done!")

    print(f"\n[SUCCESS] Merged pipeline steps ({len(merged_rows)} total samples)")


if __name__ == "__main__":
    main()
