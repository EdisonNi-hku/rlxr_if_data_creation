#!/usr/bin/env python3
"""Deduplicate datasets by instruction text and save a new dataset."""

from __future__ import annotations

import argparse
import hashlib
from typing import Iterable

from datasets import DatasetDict, concatenate_datasets, load_from_disk
from tqdm import tqdm


def normalize_instruction(
    text: object,
    lower: bool,
    strip: bool,
    collapse_ws: bool,
) -> str:
    if text is None:
        value = ""
    elif isinstance(text, str):
        value = text
    else:
        value = str(text)

    if strip:
        value = value.strip()
    if collapse_ws:
        # Normalize all whitespace runs into single spaces.
        value = " ".join(value.split())
    if lower:
        value = value.lower()
    return value


def build_dataset(paths: Iterable[str], source_field: str):
    datasets = []
    for path in paths:
        dataset = load_from_disk(path)
        if isinstance(dataset, DatasetDict):
            if "train" not in dataset:
                raise ValueError(
                    f"Dataset at {path} is a DatasetDict without a 'train' split."
                )
            dataset = dataset["train"]
        dataset = dataset.add_column(source_field, [path] * len(dataset))
        datasets.append(dataset)
    if not datasets:
        raise ValueError("At least one --input_path is required.")
    if len(datasets) == 1:
        return datasets[0]
    return concatenate_datasets(datasets)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deduplicate instructions from one or more datasets."
    )
    parser.add_argument(
        "--input_path",
        action="append",
        required=True,
        help="Dataset path saved by save_to_disk (repeatable).",
    )
    parser.add_argument(
        "--output_path",
        required=True,
        help="Path to save the deduplicated dataset.",
    )
    parser.add_argument(
        "--instruction_field",
        default="instruction",
        help="Field name to deduplicate by.",
    )
    parser.add_argument(
        "--lower",
        action="store_true",
        help="Lowercase instructions before deduplication.",
    )
    parser.add_argument(
        "--no_strip",
        action="store_true",
        help="Disable stripping leading/trailing whitespace.",
    )
    parser.add_argument(
        "--no_collapse_ws",
        action="store_true",
        help="Disable collapsing repeated whitespace.",
    )
    parser.add_argument(
        "--drop_empty",
        action="store_true",
        help="Drop rows with empty normalized instructions.",
    )
    args = parser.parse_args()

    dataset = build_dataset(args.input_path, source_field="source_path")
    total = len(dataset)

    seen = set()
    keep_indices = []
    strip = not args.no_strip
    collapse_ws = not args.no_collapse_ws

    for idx, row in tqdm(
        enumerate(dataset),
        total=total,
        unit="row",
        desc="Deduplicating",
    ):
        raw_value = row.get(args.instruction_field)
        normalized = normalize_instruction(
            raw_value,
            lower=args.lower,
            strip=strip,
            collapse_ws=collapse_ws,
        )
        if args.drop_empty and not normalized:
            continue
        digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        keep_indices.append(idx)

    deduped = dataset.select(keep_indices)
    deduped.save_to_disk(args.output_path)

    print(
        "Saved deduplicated dataset: "
        f"{len(deduped)} rows (from {total})."
    )


if __name__ == "__main__":
    main()
