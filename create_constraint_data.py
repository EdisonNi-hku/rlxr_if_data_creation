#!/usr/bin/env python3
# Copyright 2025
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Create instruction-following data with verifiable constraints.

This script takes existing instruction-following data and adds constraints
from IFEvalG to create training data for instruction-following with
verifiable constraints.

Output format is similar to allenai/IF_multi_constraints_upto5:
- key: unique identifier
- messages: list with role/content
- ground_truth: JSON with instruction_id and kwargs
- dataset: source dataset identifier
- constraint_type: 'single' or 'multi'
- constraint: text description of constraint(s)

Example usage:
    python create_constraint_data.py \
        --input_dataset allenai/tulu-3-sft-mixture \
        --output_path output_data.jsonl \
        --num_samples 1000 \
        --min_constraints 1 \
        --max_constraints 5
"""

import argparse
import json
import random
from typing import Any

from datasets import Dataset, load_dataset, load_from_disk
from tqdm import tqdm

from IFEvalG import instructions_registry


def _get_all_instruction_ids() -> list[str]:
    if hasattr(instructions_registry, "get_all_instruction_ids"):
        return list(instructions_registry.get_all_instruction_ids())
    if hasattr(instructions_registry, "INSTRUCTION_DICT"):
        return list(instructions_registry.INSTRUCTION_DICT.keys())
    if hasattr(instructions_registry, "FUNCTION_DICT"):
        return list(instructions_registry.FUNCTION_DICT.keys())
    return []


def _get_instruction_class(instruction_id: str):
    if hasattr(instructions_registry, "get_instruction_class"):
        return instructions_registry.get_instruction_class(instruction_id)
    if hasattr(instructions_registry, "INSTRUCTION_DICT"):
        return instructions_registry.INSTRUCTION_DICT[instruction_id]
    if hasattr(instructions_registry, "FUNCTION_DICT"):
        return instructions_registry.FUNCTION_DICT[instruction_id]
    raise KeyError(f"Unknown instruction id: {instruction_id}")


def _get_simple_instruction_ids() -> list[str]:
    if hasattr(instructions_registry, "get_simple_instruction_ids"):
        return list(instructions_registry.get_simple_instruction_ids())
    if hasattr(instructions_registry, "SIMPLE_CONSTRAINTS"):
        return list(instructions_registry.SIMPLE_CONSTRAINTS)
    return _get_all_instruction_ids()


def _get_non_conflicting_instructions(selected_ids: list[str]) -> list[str]:
    if hasattr(instructions_registry, "get_non_conflicting_instructions"):
        return list(instructions_registry.get_non_conflicting_instructions(selected_ids))
    all_ids = set(_get_all_instruction_ids())
    conflicts = _get_instruction_conflicts()
    blocked = set(selected_ids)
    for constraint_id in selected_ids:
        blocked.update(conflicts.get(constraint_id, set()))
    return list(all_ids - blocked)


def _get_instruction_conflicts() -> dict[str, set[str]]:
    conflicts = getattr(instructions_registry, "INSTRUCTION_CONFLICTS", {})
    if not conflicts:
        return {}
    normalized: dict[str, set[str]] = {k: set(v) for k, v in conflicts.items()}
    # Ensure all referenced keys exist before calling conflict_make.
    for vals in list(normalized.values()):
        for constraint_id in vals:
            normalized.setdefault(constraint_id, set())
    if hasattr(instructions_registry, "conflict_make"):
        return instructions_registry.conflict_make(normalized)
    # Fallback: make conflicts symmetric and self-conflicting.
    for key, vals in list(normalized.items()):
        for conflict_id in vals:
            normalized.setdefault(conflict_id, set()).add(key)
        vals.add(key)
    return normalized


def get_original_instruction(example: dict) -> str | None:
    """Extract the original instruction from various dataset formats.

    Supports common formats:
    - messages: list of {role, content}
    - prompt: string
    - instruction: string
    - text: string
    - conversations: list of {from/role, value/content}

    Args:
        example: A single example from the dataset.

    Returns:
        The extracted instruction string, or None if not found.
    """
    # Format 1: messages list (common in chat datasets)
    if "messages" in example and isinstance(example["messages"], list):
        for msg in example["messages"]:
            if isinstance(msg, dict):
                role = msg.get("role", msg.get("from", ""))
                content = msg.get("content", msg.get("value", ""))
                if role in ("user", "human") and content:
                    return content

    # Format 2: conversations list (ShareGPT format)
    if "conversations" in example and isinstance(example["conversations"], list):
        for conv in example["conversations"]:
            if isinstance(conv, dict):
                role = conv.get("from", conv.get("role", ""))
                content = conv.get("value", conv.get("content", ""))
                if role in ("user", "human") and content:
                    return content

    # Format 3: Direct fields
    for field in ["prompt", "instruction", "text", "input", "question"]:
        if field in example and example[field]:
            return example[field]

    return None


def get_example_key(example: dict, index: int, dataset_name: str) -> str:
    """Generate a unique key for the example.

    Args:
        example: A single example from the dataset.
        index: Index of the example in the dataset.
        dataset_name: Name of the source dataset.

    Returns:
        A unique key string.
    """
    # Try to use existing ID fields
    for field in ["id", "key", "idx", "index"]:
        if field in example and example[field]:
            return f"{dataset_name}_{example[field]}"

    # Fall back to index-based key
    return f"{dataset_name}_{index}"


def sample_constraints(
    num_constraints: int,
    instruction: str | None = None,
) -> list[tuple[str, dict | None, str]]:
    """Sample a set of non-conflicting constraints.

    Args:
        num_constraints: Number of constraints to sample.
        instruction: Optional instruction text (for instruction-based constraints).

    Returns:
        List of tuples (instruction_id, kwargs, description).
    """
    selected = []
    selected_ids = []

    # Start with simple constraints that work well
    simple_ids = set(_get_simple_instruction_ids())
    available = list(simple_ids)
    random.shuffle(available)
    conflicts = _get_instruction_conflicts()

    for _ in range(num_constraints):
        if not available:
            # Get more non-conflicting constraints
            available = _get_non_conflicting_instructions(selected_ids)
            # Filter to simple constraints
            available = [c for c in available if c in simple_ids]
            random.shuffle(available)

        if not available:
            break

        constraint_id = available.pop()
        selected_ids.append(constraint_id)

        # Build the constraint and get its kwargs
        constraint_class = _get_instruction_class(constraint_id)
        constraint_instance = constraint_class(constraint_id)

        # Build with default/random parameters
        # Try different parameter combinations based on what the constraint accepts
        try:
            try:
                # First try with instruction parameter (some constraints use it)
                description = constraint_instance.build_description(instruction=instruction)
            except TypeError:
                try:
                    # Then try with prompt_to_repeat (copy/repeat constraints need this)
                    description = constraint_instance.build_description(prompt_to_repeat=instruction)
                except TypeError:
                    # Finally try with no parameters
                    description = constraint_instance.build_description()
            kwargs = constraint_instance.get_instruction_args()
            selected.append((constraint_id, kwargs, description))
        except (ValueError, TypeError, AttributeError):
            # Some constraints require specific parameters or fail on certain inputs
            # Skip this constraint and try another
            continue

        # Update available list to exclude conflicts
        available = [
            c for c in available
            if c not in conflicts.get(constraint_id, set())
        ]

    return selected


def build_constraint_descriptions(
    constraints: list[tuple[str, dict | None, str]]
) -> tuple[str, list[dict]]:
    """Build constraint description text and ground truth.

    Args:
        constraints: List of (instruction_id, kwargs, description) tuples.

    Returns:
        Tuple of (constraint_text, ground_truth_list).
    """
    descriptions = []
    ground_truth = []

    for constraint_id, kwargs, description in constraints:
        descriptions.append(description)
        ground_truth.append({
            "instruction_id": [constraint_id],
            "kwargs": [kwargs],
        })

    constraint_text = " ".join(descriptions)
    return constraint_text, ground_truth


def create_constrained_example(
    example: dict,
    index: int,
    dataset_name: str,
    min_constraints: int,
    max_constraints: int,
) -> dict | None:
    """Create a constrained version of an example.

    Args:
        example: Original example from the dataset.
        index: Index of the example.
        dataset_name: Name of the source dataset.
        min_constraints: Minimum number of constraints to add.
        max_constraints: Maximum number of constraints to add.

    Returns:
        New example with constraints, or None if instruction couldn't be extracted.
    """
    instruction = get_original_instruction(example)
    if not instruction:
        return None

    # Sample number of constraints
    num_constraints = random.randint(min_constraints, max_constraints)

    # Sample constraints
    constraints = sample_constraints(num_constraints, instruction)
    if not constraints:
        return None

    # Build constraint descriptions
    constraint_text, ground_truth = build_constraint_descriptions(constraints)

    # Create the constrained instruction
    constrained_instruction = f"{instruction} {constraint_text}"

    # Determine constraint type
    constraint_type = "single" if len(constraints) == 1 else "multi"

    # Build the output example
    return {
        "key": get_example_key(example, index, dataset_name),
        "messages": [
            {
                "role": "user",
                "content": constrained_instruction,
            }
        ],
        "ground_truth": json.dumps(ground_truth),
        "dataset": dataset_name,
        "constraint_type": constraint_type,
        "constraint": constraint_text,
    }


def process_dataset(
    input_dataset: str,
    output_path: str | None = None,
    save_to_disk_path: str | None = None,
    push_to_hub: str | None = None,
    num_samples: int | None = None,
    min_constraints: int = 1,
    max_constraints: int = 5,
    split: str = "train",
    seed: int = 42,
    streaming: bool = False,
) -> None:
    """Process a dataset and create constrained examples.

    Args:
        input_dataset: HuggingFace dataset name or path.
        output_path: Path to save the output JSONL file.
        save_to_disk_path: Path to save as HuggingFace dataset locally.
        push_to_hub: HuggingFace Hub repo name to push the dataset to.
        num_samples: Number of samples to process (None for all).
        min_constraints: Minimum number of constraints per example.
        max_constraints: Maximum number of constraints per example.
        split: Dataset split to use.
        seed: Random seed for reproducibility.
        streaming: Whether to use streaming mode for large datasets.
    """
    random.seed(seed)

    print(f"Loading dataset: {input_dataset}")
    import os
    if os.path.isdir(input_dataset):
        print("Detected local dataset directory, using load_from_disk...")
        dataset = load_from_disk(input_dataset)
    else:
        try:
            dataset = load_dataset(input_dataset, split=split, streaming=streaming)
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Trying to load without split specification...")
            dataset = load_dataset(input_dataset, streaming=streaming)
            if isinstance(dataset, dict):
                available_splits = list(dataset.keys())
                print(f"Available splits: {available_splits}")
                if split in available_splits:
                    dataset = dataset[split]
                else:
                    dataset = dataset[available_splits[0]]
                    print(f"Using split: {available_splits[0]}")

    # Get dataset name for keys
    dataset_name = input_dataset.replace("/", "_")

    # Process examples
    processed_count = 0
    skipped_count = 0
    results = []

    if streaming:
        iterator = enumerate(dataset)
    else:
        iterator = enumerate(tqdm(dataset, desc="Processing"))

    for index, example in iterator:
        if num_samples is not None and processed_count >= num_samples:
            break

        constrained_example = create_constrained_example(
            example,
            index,
            dataset_name,
            min_constraints,
            max_constraints,
        )

        if constrained_example:
            results.append(constrained_example)
            processed_count += 1
        else:
            skipped_count += 1

        if streaming and processed_count % 1000 == 0:
            print(f"Processed: {processed_count}, Skipped: {skipped_count}")

    print(f"\nDone! Processed {processed_count} examples, skipped {skipped_count}")

    # Save output
    if save_to_disk_path:
        print(f"Saving as HuggingFace dataset to: {save_to_disk_path}")
        hf_dataset = Dataset.from_list(results)
        hf_dataset.save_to_disk(save_to_disk_path)
        print(f"Dataset saved! Use `load_from_disk('{save_to_disk_path}')` to load.")
    if push_to_hub:
        print(f"Pushing dataset to HuggingFace Hub: {push_to_hub}")
        hf_dataset = Dataset.from_list(results)
        hf_dataset.push_to_hub(push_to_hub)
        print(f"Dataset pushed to: https://huggingface.co/datasets/{push_to_hub}")
    if output_path:
        print(f"Saving as JSONL to: {output_path}")
        with open(output_path, "w", encoding="utf-8") as f:
            for example in results:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
        print(f"Output saved to: {output_path}")
    if not save_to_disk_path and not push_to_hub and not output_path:
        print("No output path specified, results not saved.")


def main():
    parser = argparse.ArgumentParser(
        description="Create instruction-following data with verifiable constraints."
    )
    parser.add_argument(
        "--input_dataset",
        type=str,
        required=True,
        help="HuggingFace dataset name or path (e.g., allenai/tulu-3-sft-mixture)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to save the output JSONL file",
    )
    parser.add_argument(
        "--save_to_disk",
        type=str,
        default=None,
        help="Path to save as HuggingFace dataset locally",
    )
    parser.add_argument(
        "--push_to_hub",
        type=str,
        default=None,
        help="HuggingFace Hub repo name to push the dataset to (e.g., username/dataset-name)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to process (default: all)",
    )
    parser.add_argument(
        "--min_constraints",
        type=int,
        default=1,
        help="Minimum number of constraints per example (default: 1)",
    )
    parser.add_argument(
        "--max_constraints",
        type=int,
        default=5,
        help="Maximum number of constraints per example (default: 5)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to use (default: train)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Use streaming mode for large datasets",
    )

    args = parser.parse_args()

    if not args.output_path and not args.save_to_disk and not args.push_to_hub:
        parser.error("At least one of --output_path, --save_to_disk, or --push_to_hub must be specified")

    process_dataset(
        input_dataset=args.input_dataset,
        output_path=args.output_path,
        save_to_disk_path=args.save_to_disk,
        push_to_hub=args.push_to_hub,
        num_samples=args.num_samples,
        min_constraints=args.min_constraints,
        max_constraints=args.max_constraints,
        split=args.split,
        seed=args.seed,
        streaming=args.streaming,
    )


if __name__ == "__main__":
    main()
