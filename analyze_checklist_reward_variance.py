#!/usr/bin/env python3
"""
Compute soft/hard reward variance from analyze_checklist.py output.

Soft reward for a response = average of checklist item scores.
Hard reward for a response = 1 if all checklist items are 1, else 0.

Prints the proportion of dataset rows with zero soft/hard reward variance.

Supports merging multiple partitions before analysis.
"""

from __future__ import annotations

import argparse
import json
import math
import os

from datasets import load_dataset, load_from_disk
from tqdm import tqdm


def build_dataset(input_dataset: str, split: str, streaming: bool):
    """Load dataset from various sources."""
    if os.path.isfile(input_dataset):
        return load_dataset(
            "json",
            data_files=input_dataset,
            split=split,
            streaming=streaming,
        )
    if os.path.isdir(input_dataset):
        dataset = load_from_disk(input_dataset)
        if hasattr(dataset, "keys"):
            return dataset[split]
        return dataset
    try:
        return load_dataset(input_dataset, split=split, streaming=streaming)
    except Exception:
        dataset = load_dataset(input_dataset, streaming=streaming)
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


def load_results(input_path: str, format: str):
    """Load results from JSONL file or HuggingFace dataset."""
    if format == "jsonl":
        results = []
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                results.append(json.loads(line))
        return results
    if format == "disk":
        dataset = load_from_disk(input_path)
        return list(dataset)
    raise ValueError(f"Unknown format: {format}")


def merge_partitions(base_name: str, partition_num: int, format: str) -> list[dict]:
    """Merge results from multiple partition files.
    
    Args:
        base_name: Base name of the partition files (e.g., 'rollouts_checklist_test')
        partition_num: Total number of partitions
        format: Format of the partition files ('jsonl' or 'disk')
    
    Returns:
        Merged list of all results from all partitions
    """
    all_results = []
    
    for i in range(partition_num):
        if format == "jsonl":
            partition_path = f"{base_name}_p{i}_of_{partition_num}_graded.jsonl"
        else:
            partition_path = f"{base_name}_p{i}_of_{partition_num}_graded"
        
        if not os.path.exists(partition_path):
            print(f"Warning: Partition file not found: {partition_path}")
            continue
        
        print(f"Loading partition {i + 1}/{partition_num}: {partition_path}")
        partition_results = load_results(partition_path, format)
        all_results.extend(partition_results)
        print(f"  Loaded {len(partition_results)} results from partition {i}")
    
    return all_results


def variance(values: list[float]) -> float:
    """Population variance; returns 0 for empty or single-element lists."""
    n = len(values)
    if n <= 1:
        return 0.0
    mean = sum(values) / n
    return sum((v - mean) ** 2 for v in values) / n


def compute_soft_rewards(result: dict) -> list[float]:
    scores = result.get("scores")
    if scores:
        return [float(s) for s in scores]
    individual_scores = result.get("individual_scores") or []
    soft = []
    for item_scores in individual_scores:
        if not item_scores:
            soft.append(0.0)
        else:
            soft.append(sum(item_scores) / len(item_scores))
    return soft


def compute_hard_rewards(result: dict) -> list[int]:
    individual_scores = result.get("individual_scores") or []
    if individual_scores:
        return [1 if all(score == 1 for score in item_scores) else 0 for item_scores in individual_scores]
    scores = result.get("scores") or []
    return [1 if float(score) >= 1.0 else 0 for score in scores]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute soft/hard reward variance from analyze_checklist.py output."
    )
    parser.add_argument(
        "--base_name",
        required=True,
        help="Base name for partition files (e.g., 'rollouts_checklist_test').",
    )
    parser.add_argument(
        "--partition_num",
        type=int,
        required=True,
        help="Total number of partitions to merge.",
    )
    parser.add_argument(
        "--format",
        choices=["jsonl", "disk"],
        default="jsonl",
        help="Input format for analyze_checklist output.",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset used as input to analyze_checklist.py (for keys/row count).",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to use (default: train).",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-12,
        help="Tolerance for treating variance as zero.",
    )
    parser.add_argument(
        "--save_dataset",
        type=str,
        default=None,
        help="Optional path to save the dataset with soft/hard rewards appended.",
    )
    parser.add_argument(
        "--soft_rewards_column",
        type=str,
        default="soft_rewards",
        help="Column name for soft rewards (default: soft_rewards).",
    )
    parser.add_argument(
        "--hard_rewards_column",
        type=str,
        default="hard_rewards",
        help="Column name for hard rewards (default: hard_rewards).",
    )
    args = parser.parse_args()

    print(f"Merging {args.partition_num} partitions with base name: {args.base_name}")
    results = merge_partitions(args.base_name, args.partition_num, args.format)
    print(f"Loaded {len(results)} total results after merging.")

    print(f"Loading dataset: {args.dataset}")
    dataset = build_dataset(args.dataset, args.split, streaming=False)
    dataset_name = os.path.basename(args.dataset).replace("/", "_")

    result_map = {}
    for result in results:
        key = result.get("key")
        if key is None:
            continue
        result_map[key] = result

    total_rows = 0
    rows_with_results = 0
    missing_results = 0
    soft_zero_variance = 0
    hard_zero_variance = 0
    # Distribution counters
    soft_all_zero = 0
    soft_all_one = 0
    hard_all_zero = 0
    hard_all_one = 0
    soft_rewards_column_data = []
    hard_rewards_column_data = []

    for index, example in enumerate(tqdm(dataset, desc="Computing variance")):
        total_rows += 1
        key = get_example_key(example, index, dataset_name)
        result = result_map.get(key)
        if result is None:
            missing_results += 1
            soft_rewards_column_data.append([])
            hard_rewards_column_data.append([])
            continue

        soft_rewards = compute_soft_rewards(result)
        hard_rewards = compute_hard_rewards(result)
        if not soft_rewards and not hard_rewards:
            missing_results += 1
            soft_rewards_column_data.append([])
            hard_rewards_column_data.append([])
            continue

        rows_with_results += 1
        soft_rewards_column_data.append(soft_rewards)
        hard_rewards_column_data.append(hard_rewards)

        soft_var = variance(soft_rewards)
        hard_var = variance([float(v) for v in hard_rewards])

        if soft_var <= args.epsilon:
            soft_zero_variance += 1
        if hard_var <= args.epsilon:
            hard_zero_variance += 1

        # Track distribution: all-0 and all-1 rewards
        if soft_rewards and all(s == 0.0 for s in soft_rewards):
            soft_all_zero += 1
        if soft_rewards and all(s == 1.0 for s in soft_rewards):
            soft_all_one += 1
        if hard_rewards and all(h == 0 for h in hard_rewards):
            hard_all_zero += 1
        if hard_rewards and all(h == 1 for h in hard_rewards):
            hard_all_one += 1

    denom = rows_with_results if rows_with_results > 0 else 1
    print("\n" + "=" * 60)
    print("CHECKLIST REWARD VARIANCE")
    print("=" * 60)
    print(f"Total dataset rows: {total_rows}")
    print(f"Rows with results: {rows_with_results}")
    print(f"Rows missing results: {missing_results}")
    print(
        f"Soft reward variance == 0: {soft_zero_variance} ({soft_zero_variance / denom:.2%})"
    )
    print(
        f"Hard reward variance == 0: {hard_zero_variance} ({hard_zero_variance / denom:.2%})"
    )
    print("=" * 60)

    print("\n" + "=" * 60)
    print("REWARD DISTRIBUTION")
    print("=" * 60)
    print(f"Soft rewards - all 0: {soft_all_zero} ({soft_all_zero / denom:.2%})")
    print(f"Soft rewards - all 1: {soft_all_one} ({soft_all_one / denom:.2%})")
    print(f"Hard rewards - all 0: {hard_all_zero} ({hard_all_zero / denom:.2%})")
    print(f"Hard rewards - all 1: {hard_all_one} ({hard_all_one / denom:.2%})")
    print("=" * 60)

    if args.save_dataset:
        dataset = dataset.add_column(args.soft_rewards_column, soft_rewards_column_data)
        dataset = dataset.add_column(args.hard_rewards_column, hard_rewards_column_data)
        dataset.save_to_disk(args.save_dataset)
        print(f"\nDataset with rewards saved to: {args.save_dataset}")
        print(f"  Soft rewards column: {args.soft_rewards_column}")
        print(f"  Hard rewards column: {args.hard_rewards_column}")


if __name__ == "__main__":
    main()
