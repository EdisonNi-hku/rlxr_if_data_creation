#!/usr/bin/env python3
"""Load Magpie-Align/Magpie-Qwen2.5-Pro-300K-Filtered creative writing samples."""

from __future__ import annotations

import argparse
import json
from typing import Iterable, Optional

from datasets import Dataset, load_dataset
from tqdm import tqdm


def iter_creative_writing(
    dataset_name: str,
    split: str,
    streaming: bool,
    category: str,
    num_proc: Optional[int],
) -> Iterable[dict]:
    dataset = load_dataset(dataset_name, split=split, streaming=streaming)
    if streaming:
        # Filter to requested category.
        return dataset.filter(lambda ex: ex.get("task_category") == category)
    return dataset.filter(
        lambda batch: [c == category for c in batch["task_category"]],
        batched=True,
        num_proc=num_proc,
        desc="Filtering",
    )


def progress_iter(
    samples: Iterable[dict],
    total: Optional[int],
    max_samples: Optional[int],
) -> Iterable[dict]:
    limit = max_samples if max_samples is not None else total
    return tqdm(samples, total=limit, unit="sample", desc="Processing")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Load Magpie-Align/Magpie-Qwen2.5-Pro-300K-Filtered "
            "and select task_category == 'creative writing'."
        )
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Hugging Face dataset name",
    )
    parser.add_argument("--split", default="train", help="Dataset split")
    parser.add_argument(
        "--category",
        default="creative writing",
        help="task_category value to select",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Use streaming mode",
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=None,
        help="Number of processes for non-streaming filter",
    )
    parser.add_argument(
        "--output_path",
        help="Optional JSONL path to save filtered samples",
    )
    parser.add_argument(
        "--save_to_disk",
        help="Optional path to save filtered Dataset for load_from_disk",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional maximum number of samples to process",
    )
    args = parser.parse_args()

    filtered = iter_creative_writing(
        args.dataset,
        args.split,
        args.streaming,
        args.category,
        args.num_proc,
    )
    if not args.streaming and args.max_samples is not None:
        filtered = filtered.select(range(min(args.max_samples, filtered.num_rows)))

    count = 0
    if args.save_to_disk:
        # save_to_disk expects a non-streaming Dataset; materialize if streaming.
        if args.streaming:
            rows = []
            for sample in progress_iter(filtered, None, args.max_samples):
                rows.append(sample)
                count += 1
                if args.max_samples is not None and count >= args.max_samples:
                    break
            Dataset.from_list(rows).save_to_disk(args.save_to_disk)
        else:
            filtered.save_to_disk(args.save_to_disk)
            count = filtered.num_rows
    elif args.output_path:
        total = None if args.streaming else filtered.num_rows
        with open(args.output_path, "w", encoding="utf-8") as f:
            for sample in progress_iter(filtered, total, args.max_samples):
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                count += 1
                if args.max_samples is not None and count >= args.max_samples:
                    break
    else:
        if args.streaming:
            total = None
            for sample in progress_iter(filtered, total, args.max_samples):
                count += 1
                if args.max_samples is not None and count >= args.max_samples:
                    break
        else:
            count = filtered.num_rows

    print(f"Loaded {count} samples with task_category='{args.category}'.")


if __name__ == "__main__":
    main()
