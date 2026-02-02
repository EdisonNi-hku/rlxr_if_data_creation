#!/usr/bin/env python3
"""
Transform analyze_checklist.py output JSONL into a human-review format.

Produces one row per response with checklist items aligned to LLM scores.

Supports filtering by soft reward distribution (all 0 or all 1) for targeted inspection.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from typing import Any

from datasets import load_from_disk


def split_checklist(checklist: str) -> list[str]:
    return [line.strip() for line in checklist.splitlines() if line.strip()]


def build_item_rows(items: list[str], scores: list[Any]) -> tuple[list[dict], bool]:
    max_len = max(len(items), len(scores))
    rows = []
    for i in range(max_len):
        rows.append({
            "item": items[i] if i < len(items) else "",
            "llm_score": scores[i] if i < len(scores) else None,
        })
    return rows, len(items) != len(scores)


def is_all_zero(rewards: list) -> bool:
    """Check if all rewards are 0."""
    if not rewards:
        return False
    return all(r == 0 or r == 0.0 for r in rewards)


def is_all_one(rewards: list) -> bool:
    """Check if all rewards are 1."""
    if not rewards:
        return False
    return all(r == 1 or r == 1.0 for r in rewards)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transform checklist grading JSONL into a human-reviewable format."
    )
    parser.add_argument("--input", required=True, help="Input JSONL from analyze_checklist.py")
    parser.add_argument("--output", required=True, help="Output path (.jsonl or .csv)")
    parser.add_argument(
        "--max_rows",
        type=int,
        default=None,
        help="Optional cap on number of output rows.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Randomly sample this many response rows.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling.",
    )
    # Graded dataset options
    parser.add_argument(
        "--graded_dataset",
        type=str,
        default=None,
        help="Path to graded dataset (from analyze_checklist_reward_variance.py --save_dataset).",
    )
    parser.add_argument(
        "--soft_rewards_column",
        type=str,
        default="soft_rewards",
        help="Column name for soft rewards in graded dataset.",
    )
    parser.add_argument(
        "--hard_rewards_column",
        type=str,
        default="hard_rewards",
        help="Column name for hard rewards in graded dataset.",
    )
    parser.add_argument(
        "--filter_soft_rewards",
        choices=["all_zero", "all_one", "all_zero_or_one"],
        default=None,
        help="Filter to only include rows where soft rewards match criteria.",
    )
    args = parser.parse_args()

    if args.sample is not None and args.sample < 0:
        raise SystemExit("--sample must be >= 0")
    if args.max_rows is not None and args.max_rows < 0:
        raise SystemExit("--max_rows must be >= 0")

    # Load graded dataset if provided (for filtering by soft/hard rewards)
    graded_rewards = {}
    if args.graded_dataset:
        print(f"Loading graded dataset from: {args.graded_dataset}")
        graded_ds = load_from_disk(args.graded_dataset)
        for example in graded_ds:
            key = None
            for field in ["id", "uuid", "key", "idx", "index"]:
                if field in example and example[field]:
                    key = str(example[field])
                    break
            if key:
                graded_rewards[key] = {
                    "soft_rewards": example.get(args.soft_rewards_column, []),
                    "hard_rewards": example.get(args.hard_rewards_column, []),
                }
        print(f"Loaded {len(graded_rewards)} graded examples.")

    as_csv = args.output.lower().endswith(".csv")
    rows = []
    filtered_count = 0
    total_count = 0

    with open(args.input, "r", encoding="utf-8") as f_in:
        for line in f_in:
            record = json.loads(line)
            key = record.get("key")
            instruction = record.get("instruction", "")
            checklist = record.get("checklist", "")
            responses = record.get("responses", [])
            scores = record.get("scores", [])
            individual_scores = record.get("individual_scores", [])
            grader_outputs = record.get("grader_outputs", [])

            # Get soft/hard rewards from graded dataset if available
            soft_rewards = []
            hard_rewards = []
            if key and key in graded_rewards:
                soft_rewards = graded_rewards[key].get("soft_rewards", [])
                hard_rewards = graded_rewards[key].get("hard_rewards", [])

            # Apply filter if specified
            total_count += 1
            if args.filter_soft_rewards:
                if args.filter_soft_rewards == "all_zero" and not is_all_zero(soft_rewards):
                    filtered_count += 1
                    continue
                elif args.filter_soft_rewards == "all_one" and not is_all_one(soft_rewards):
                    filtered_count += 1
                    continue
                elif args.filter_soft_rewards == "all_zero_or_one":
                    if not (is_all_zero(soft_rewards) or is_all_one(soft_rewards)):
                        filtered_count += 1
                        continue

            checklist_items = split_checklist(checklist)

            for idx, response in enumerate(responses):
                item_scores = individual_scores[idx] if idx < len(individual_scores) else []
                item_rows, mismatch = build_item_rows(checklist_items, item_scores)
                llm_score = scores[idx] if idx < len(scores) else None
                llm_raw = grader_outputs[idx] if idx < len(grader_outputs) else ""

                output = {
                    "key": key,
                    "response_index": idx,
                    "instruction": instruction,
                    "response": response,
                    "checklist": checklist,
                    "checklist_items": checklist_items,
                    "items": item_rows,
                    "items_mismatch": mismatch,
                    "llm_individual_scores": item_scores,
                    "llm_score": llm_score,
                    "llm_pass": True if llm_score is not None and llm_score >= 1.0 else False,
                    "llm_raw_output": llm_raw,
                    "num_items": len(checklist_items),
                    "soft_rewards": soft_rewards,
                    "hard_rewards": hard_rewards,
                    "soft_rewards_all_zero": is_all_zero(soft_rewards),
                    "soft_rewards_all_one": is_all_one(soft_rewards),
                }
                rows.append(output)

    if args.filter_soft_rewards:
        print(f"Filtered {filtered_count}/{total_count} records (kept {total_count - filtered_count})")

    if args.sample is not None and args.sample < len(rows):
        rng = random.Random(args.seed)
        rows = rng.sample(rows, args.sample)

    if args.max_rows is not None:
        rows = rows[: args.max_rows]

    if as_csv:
        with open(args.output, "w", encoding="utf-8", newline="") as f_out:
            writer = csv.DictWriter(
                f_out,
                fieldnames=[
                    "key",
                    "response_index",
                    "instruction",
                    "response",
                    "checklist",
                    "checklist_items",
                    "items",
                    "items_mismatch",
                    "llm_individual_scores",
                    "llm_score",
                    "llm_pass",
                    "llm_raw_output",
                    "num_items",
                    "soft_rewards",
                    "hard_rewards",
                    "soft_rewards_all_zero",
                    "soft_rewards_all_one",
                ],
            )
            writer.writeheader()
            for output in rows:
                output = dict(output)
                output["checklist_items"] = json.dumps(output["checklist_items"], ensure_ascii=False)
                output["items"] = json.dumps(output["items"], ensure_ascii=False)
                output["llm_individual_scores"] = json.dumps(
                    output["llm_individual_scores"], ensure_ascii=False
                )
                output["soft_rewards"] = json.dumps(output["soft_rewards"], ensure_ascii=False)
                output["hard_rewards"] = json.dumps(output["hard_rewards"], ensure_ascii=False)
                writer.writerow(output)
    else:
        with open(args.output, "w", encoding="utf-8") as f_out:
            for output in rows:
                f_out.write(json.dumps(output, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
