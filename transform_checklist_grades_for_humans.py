#!/usr/bin/env python3
"""
Transform analyze_checklist.py output JSONL into a human-review format.

Produces one row per response with checklist items aligned to LLM scores.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from typing import Any


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
    args = parser.parse_args()

    if args.sample is not None and args.sample < 0:
        raise SystemExit("--sample must be >= 0")
    if args.max_rows is not None and args.max_rows < 0:
        raise SystemExit("--max_rows must be >= 0")

    as_csv = args.output.lower().endswith(".csv")
    rows = []

    with open(args.input, "r", encoding="utf-8") as f_in:
        if as_csv:
            pass
        else:
            pass

        for line in f_in:
            record = json.loads(line)
            key = record.get("key")
            instruction = record.get("instruction", "")
            checklist = record.get("checklist", "")
            responses = record.get("responses", [])
            scores = record.get("scores", [])
            individual_scores = record.get("individual_scores", [])
            grader_outputs = record.get("grader_outputs", [])

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
                }
                rows.append(output)

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
                writer.writerow(output)
    else:
        with open(args.output, "w", encoding="utf-8") as f_out:
            for output in rows:
                f_out.write(json.dumps(output, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
