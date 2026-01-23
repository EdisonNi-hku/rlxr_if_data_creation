#!/usr/bin/env python3
"""Upload a local Hugging Face dataset (saved via save_to_disk) to the Hub."""

from __future__ import annotations

import argparse
import os

from datasets import Dataset, DatasetDict, load_from_disk


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload a saved Hugging Face dataset directory to the Hub."
    )
    parser.add_argument(
        "--dataset_path",
        required=True,
        help="Local dataset directory created by save_to_disk.",
    )
    parser.add_argument(
        "--repo_id",
        required=True,
        help="Hub repo id, e.g. username/dataset_name.",
    )
    parser.add_argument(
        "--private",
        default=False,
        action="store_true",
        help="Create the Hub repo as private if it doesn't exist.",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="Split name when the saved dataset is a DatasetDict.",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.dataset_path):
        raise FileNotFoundError(f"Dataset directory not found: {args.dataset_path}")

    dataset = load_from_disk(args.dataset_path)

    # Handle Dataset vs DatasetDict
    if isinstance(dataset, DatasetDict):
        if args.split is not None:
            if args.split not in dataset:
                raise ValueError(
                    f"Split '{args.split}' not found. Available: {list(dataset.keys())}"
                )
            dataset_to_push = dataset[args.split]
        else:
            dataset_to_push = dataset
    else:
        # It's a plain Dataset
        dataset_to_push = dataset

    dataset_to_push.push_to_hub(args.repo_id, private=args.private)
    print(f"Uploaded dataset to {args.repo_id}")


if __name__ == "__main__":
    main()
