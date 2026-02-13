#!/usr/bin/env python3
"""Compute summary stats for analyze_quality_vllm.py output JSONL."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from typing import Iterable


QUALITY_DIMS = [
    "incoherent_expression",
    "logical_inconsistency",
    "inappropriate_word_choice",
    "repetitive_expression",
    "language_inconsistency",
]


def load_jsonl(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def update_counts(analysis: dict | None, counts: Counter, label: str) -> None:
    if not isinstance(analysis, dict):
        counts[f"{label}.missing"] += 1
        return
    counts[f"{label}.present"] += 1
    any_issue = False
    for dim in QUALITY_DIMS:
        val = analysis.get(dim, 0)
        if val == 1:
            counts[f"{label}.{dim}"] += 1
            any_issue = True
    if any_issue:
        counts[f"{label}.any_issue"] += 1
    notes = analysis.get("notes", "")
    if isinstance(notes, str) and notes.strip():
        counts[f"{label}.notes_nonempty"] += 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute summary stats from analyze_quality_vllm.py JSONL output."
    )
    parser.add_argument(
        "input",
        help="Path to quality_analysis.jsonl (output of analyze_quality_vllm.py).",
    )
    parser.add_argument(
        "--models",
        choices=["m1", "m2", "both"],
        default="both",
        help="Which model analysis to summarize. Default: both.",
    )
    args = parser.parse_args()

    counts = Counter()
    total_rows = 0

    for row in load_jsonl(args.input):
        total_rows += 1
        if args.models in ("m1", "both"):
            update_counts(row.get("m1_analysis"), counts, "m1")
        if args.models in ("m2", "both"):
            update_counts(row.get("m2_analysis"), counts, "m2")

    def report(label: str) -> None:
        present = counts.get(f"{label}.present", 0)
        missing = counts.get(f"{label}.missing", 0)
        any_issue = counts.get(f"{label}.any_issue", 0)
        notes = counts.get(f"{label}.notes_nonempty", 0)
        print(f"[{label}]")
        print(f"  present: {present}")
        print(f"  missing: {missing}")
        if present > 0:
            print(f"  any_issue: {any_issue} ({any_issue / present:.2%})")
            print(f"  notes_nonempty: {notes} ({notes / present:.2%})")
            for dim in QUALITY_DIMS:
                c = counts.get(f"{label}.{dim}", 0)
                print(f"  {dim}: {c} ({c / present:.2%})")
        print("")

    print(f"Total rows: {total_rows}")
    if args.models in ("m1", "both"):
        report("m1")
    if args.models in ("m2", "both"):
        report("m2")


if __name__ == "__main__":
    main()
