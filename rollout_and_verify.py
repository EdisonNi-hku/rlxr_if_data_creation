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
Generate multiple rollouts for verifiable instructions and verify constraint compliance.

This script takes a dataset with verifiable instructions (e.g., created by create_constraint_data.py),
generates multiple rollout responses per prompt using vLLM's native LLM class with n=16 sampling,
and verifies whether each response follows the specified constraints.

Output includes:
- All generated responses per prompt
- Verification results for each response
- Pass@k statistics (e.g., pass@1, pass@8, pass@16)

Example usage:
    python rollout_and_verify.py \
        --input_dataset verifiable_data.jsonl \
        --output_path rollouts_verified.jsonl \
        --model "Qwen/Qwen3-8B" \
        --num_rollouts 16 \
        --tensor_parallel_size 1
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Optional

from datasets import Dataset, load_dataset, load_from_disk
from tqdm import tqdm
from vllm import LLM, SamplingParams

from verify_constraints import verify_multiple_constraints


def build_dataset(
    input_dataset: str,
    split: str,
    streaming: bool,
):
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
        # Handle DatasetDict (has splits like 'train', 'test')
        if hasattr(dataset, 'keys'):
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


def get_instruction_from_messages(example: dict, instruction_field: str) -> str | None:
    """Extract the instruction/prompt from the example.

    Args:
        example: A single example from the dataset.
        instruction_field: Field name containing the instruction (required).

    Returns:
        The instruction string, or None if not found.
    """
    value = example.get(instruction_field)
    if value is None:
        return None

    # If it's a string, return directly
    if isinstance(value, str):
        return value

    # If it's a list (messages format), extract user content
    if isinstance(value, list):
        for msg in value:
            if isinstance(msg, dict):
                role = msg.get("role", msg.get("from", ""))
                content = msg.get("content", msg.get("value", ""))
                if role in ("user", "human") and content:
                    return content

    return None


def get_ground_truth(example: dict) -> list[dict] | None:
    """Extract ground truth constraints from the example.

    Supports both standard 'ground_truth' and append_mode 'ground_truth_verifiable'.
    """
    # Check for append_mode format first
    gt = example.get("ground_truth_verifiable") or example.get("ground_truth")

    if gt is None:
        return None

    if isinstance(gt, str):
        try:
            return json.loads(gt)
        except json.JSONDecodeError:
            return None

    return gt


def best_of_k(scores: list[float], k: int) -> float:
    """Compute best score among the first k rollouts.

    Args:
        scores: List of scores for each rollout.
        k: Number of rollouts to consider.

    Returns:
        Best score among the first k rollouts.
    """
    if not scores:
        return 0.0
    return max(scores[:k])


def compute_advantages(scores: list[float], epsilon: float = 1e-8) -> list[float]:
    """Compute GRPO-style group-level advantages.

    Formula: A_i = (r_i - mu_g) / (sigma_g + epsilon)

    Args:
        scores: List of reward scores for each rollout in the group.
        epsilon: Small constant for numerical stability.

    Returns:
        List of advantage values for each rollout.
    """
    if not scores:
        return []

    n = len(scores)
    mu = sum(scores) / n
    variance = sum((s - mu) ** 2 for s in scores) / n
    sigma = math.sqrt(variance)

    return [(s - mu) / (sigma + epsilon) for s in scores]


def build_chat_prompt(
    tokenizer,
    instruction: str,
    system_prompt: str | None = None,
) -> str:
    """Build a chat prompt using the tokenizer's chat template."""
    if system_prompt:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": instruction},
        ]
    else:
        messages = [{"role": "user", "content": instruction}]

    # Use the tokenizer's chat template if available
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        # Fallback: simple concatenation
        if system_prompt:
            return f"{system_prompt}\n\nUser: {instruction}\nAssistant:"
        return f"User: {instruction}\nAssistant:"


def verify_responses(
    responses: list[str],
    ground_truth: list[dict],
) -> tuple[list[dict], int, list[bool]]:
    """Verify each response against the ground truth constraints.

    Returns:
        Tuple of (rollout_results, num_passed, all_constraint_results)
    """
    rollout_results = []
    num_passed = 0
    all_constraint_results = []

    for response in responses:
        if not response:
            rollout_results.append({
                "response": "",
                "passed": False,
                "score": 0.0,
                "constraint_results": [],
            })
            continue

        passed, individual_results = verify_multiple_constraints(response, ground_truth)

        # Score is the fraction of constraints passed
        score = sum(individual_results) / len(individual_results) if individual_results else 0.0

        rollout_results.append({
            "response": response,
            "passed": passed,
            "score": score,
            "constraint_results": individual_results,
        })

        if passed:
            num_passed += 1
        all_constraint_results.extend(individual_results)

    return rollout_results, num_passed, all_constraint_results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate rollouts and verify constraint compliance using vLLM."
    )

    # Dataset arguments
    parser.add_argument(
        "--input_dataset",
        required=True,
        help="Dataset name, path, or JSON/JSONL file with verifiable instructions.",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to use (default: train).",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Use streaming mode for large datasets.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to process.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Only process this many rows (for debugging).",
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=0,
        help="Skip samples before this index.",
    )
    parser.add_argument(
        "--instruction_field",
        default="messages",
        help="Field name containing the instruction (default: messages).",
    )

    # Output arguments
    parser.add_argument(
        "--output_path",
        default=None,
        help="Path to save output JSONL file.",
    )
    parser.add_argument(
        "--save_to_disk",
        default=None,
        help="Path to save as HuggingFace dataset.",
    )

    # vLLM model arguments
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-8B",
        help="Model name or path for vLLM.",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism (default: 1).",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization (default: 0.9).",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        help="Data type for model weights (auto, float16, bfloat16, float32).",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Trust remote code from HuggingFace.",
    )

    # Sampling arguments
    parser.add_argument(
        "--num_rollouts",
        type=int,
        default=16,
        help="Number of rollouts to generate per prompt (default: 16).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7).",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Top-p sampling parameter (default: 0.95).",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=-1,
        help="Top-k sampling parameter (default: -1, disabled).",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=4096,
        help="Maximum tokens to generate (default: 4096).",
    )
    parser.add_argument(
        "--presence_penalty",
        type=float,
        default=0.0,
        help="Presence penalty (default: 0.0).",
    )
    parser.add_argument(
        "--frequency_penalty",
        type=float,
        default=0.0,
        help="Frequency penalty (default: 0.0).",
    )

    # Prompt arguments
    parser.add_argument(
        "--system_prompt",
        type=str,
        default=None,
        help="Optional system prompt to prepend.",
    )
    parser.add_argument(
        "--system_prompt_path",
        type=str,
        default=None,
        help="Path to file containing system prompt.",
    )


    # Partitioning arguments
    parser.add_argument(
        "--partition_num",
        type=int,
        default=1,
        help="Total number of partitions for distributed processing.",
    )
    parser.add_argument(
        "--partition_index",
        type=int,
        default=0,
        help="Index of the current partition (0-based).",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.partition_num < 1:
        parser.error("--partition_num must be >= 1")
    if args.partition_index < 0 or args.partition_index >= args.partition_num:
        parser.error(f"--partition_index must be in range [0, {args.partition_num - 1}]")

    if not args.output_path and not args.save_to_disk:
        parser.error("At least one of --output_path or --save_to_disk must be specified")

    # Check for existing raw outputs
    raw_outputs_path = args.output_path.replace(".jsonl", "_raw.jsonl") if args.output_path else "raw_outputs.jsonl"
    dataset_name = os.path.basename(args.input_dataset).replace("/", "_")

    # Load dataset
    dataset = build_dataset(args.input_dataset, args.split, args.streaming)

    # Apply partitioning if specified
    if args.partition_num > 1:
        if args.streaming:
            raise ValueError("Partitioning is not supported with streaming mode.")
        total_size = len(dataset)
        indices = list(range(args.partition_index, total_size, args.partition_num))
        dataset = dataset.select(indices)
        print(f"[PARTITION] Processing partition {args.partition_index} of {args.partition_num}: "
              f"{len(indices)} samples (round-robin from {total_size} total)")

    # Determine sample limit (--sample overrides --max_samples for debugging)
    sample_limit = args.sample if args.sample is not None else args.max_samples

    # Prepare examples (always needed to get instruction and ground_truth)
    examples_to_process = []
    skipped_prep = 0

    print("Preparing examples...")
    for index, example in enumerate(tqdm(dataset, desc="Preparing")):
        if index < args.start_index:
            continue
        if sample_limit is not None and len(examples_to_process) >= sample_limit:
            break

        instruction = get_instruction_from_messages(example, args.instruction_field)
        ground_truth = get_ground_truth(example)

        if not instruction or ground_truth is None:
            skipped_prep += 1
            continue

        key = get_example_key(example, index, dataset_name)

        examples_to_process.append({
            "key": key,
            "index": index,
            "example": example,
            "instruction": instruction,
            "ground_truth": ground_truth,
        })

    print(f"Prepared {len(examples_to_process)} examples, skipped {skipped_prep}")

    if not examples_to_process:
        print("No samples to process. Exiting.")
        return

    if os.path.exists(raw_outputs_path):
        print(f"Found existing raw outputs: {raw_outputs_path}")
        print("Loading from cache (skipping vLLM initialization and generation)...")
        cached_responses = {}
        with open(raw_outputs_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                cached_responses[data["key"]] = data["responses"]
        raw_responses = [cached_responses[ex["key"]] for ex in examples_to_process]
        print(f"Loaded {len(raw_responses)} cached outputs.")
    else:
        # Load system prompt if specified
        system_prompt = args.system_prompt
        if args.system_prompt_path:
            with open(args.system_prompt_path, "r", encoding="utf-8") as f:
                system_prompt = f.read().strip()

        # Initialize vLLM
        print(f"Loading model: {args.model}")
        llm = LLM(
            model=args.model,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
        )
        tokenizer = llm.get_tokenizer()

        # Create sampling params with n=num_rollouts for efficient batch generation
        sampling_params = SamplingParams(
            n=args.num_rollouts,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k if args.top_k > 0 else -1,
            max_tokens=args.max_tokens,
            presence_penalty=args.presence_penalty,
            frequency_penalty=args.frequency_penalty,
        )

        print(f"Sampling params: n={args.num_rollouts}, temp={args.temperature}, top_p={args.top_p}")

        # Build prompts using chat template
        print("Building prompts...")
        prompts = []
        for ex in examples_to_process:
            prompt = build_chat_prompt(tokenizer, ex["instruction"], system_prompt)
            prompts.append(prompt)

        # Generate with vLLM - n rollouts per prompt in a single call
        print(f"Generating rollouts for {len(prompts)} prompts...")
        outputs = llm.generate(prompts, sampling_params)

        # Save raw outputs before processing
        print(f"Saving raw outputs to: {raw_outputs_path}")
        raw_responses = []
        with open(raw_outputs_path, "w", encoding="utf-8") as f:
            for ex, output in zip(examples_to_process, outputs):
                responses = [out.text for out in output.outputs]
                raw_responses.append(responses)
                raw_result = {
                    "key": ex["key"],
                    "instruction": ex["instruction"],
                    "responses": responses,
                }
                f.write(json.dumps(raw_result, ensure_ascii=False) + "\n")
        print(f"Raw outputs saved.")

    if not examples_to_process:
        print("No samples to process. Exiting.")
        return

    # Process all prompts at once
    results = []
    total_best_at_1 = 0.0
    total_best_at_8 = 0.0
    total_best_at_16 = 0.0
    total_passed = 0
    total_rollouts = 0
    total_constraints = 0
    total_constraints_passed = 0

    # Process outputs
    for ex, responses in tqdm(zip(examples_to_process, raw_responses), total=len(examples_to_process), desc="Processing outputs"):

            # Handle thinking tokens if present
            clean_responses = []
            for resp in responses:
                if '</think>' in resp:
                    resp = resp.split('</think>')[-1].strip()
                clean_responses.append(resp)

            # Verify responses
            rollout_results, num_passed, all_constraint_results = verify_responses(
                clean_responses, ex["ground_truth"]
            )

            n = len(clean_responses)

            # Extract responses and scores as separate lists
            responses = [r["response"] for r in rollout_results]
            scores = [r["score"] for r in rollout_results]

            # Compute best@k metrics (best score among first k rollouts)
            best_at_1 = best_of_k(scores, 1)
            best_at_8 = best_of_k(scores, min(8, n))
            best_at_16 = best_of_k(scores, n)

            # Compute GRPO-style group advantages
            advantages = compute_advantages(scores)

            result = {
                "key": ex["key"],
                "instruction": ex["instruction"],
                "ground_truth": ex["ground_truth"],
                "responses": responses,
                "scores": scores,
                "advantages": advantages,
                "num_rollouts": n,
                "num_passed": num_passed,
                "pass_rate": num_passed / n if n > 0 else 0.0,
                "best_at_1": best_at_1,
                "best_at_8": best_at_8,
                "best_at_16": best_at_16,
                "rollouts": rollout_results,
                "dataset": dataset_name,
                "constraint_accuracy": sum(all_constraint_results) / len(all_constraint_results) if all_constraint_results else 0.0,
            }
            results.append(result)

            # Update aggregated stats
            total_best_at_1 += best_at_1
            total_best_at_8 += best_at_8
            total_best_at_16 += best_at_16
            total_passed += num_passed
            total_rollouts += n
            total_constraints += len(all_constraint_results)
            total_constraints_passed += sum(all_constraint_results)

    processed = len(results)

    # Print final statistics
    print("\n" + "=" * 60)
    print("ROLLOUT AND VERIFICATION RESULTS")
    print("=" * 60)
    print(f"Total examples processed: {processed}")
    print(f"Total examples skipped: {skipped_prep}")
    print(f"Total rollouts generated: {total_rollouts}")
    print(f"Total rollouts passed: {total_passed}")
    print(f"Overall pass rate: {total_passed / total_rollouts:.2%}" if total_rollouts > 0 else "N/A")
    print(f"Total constraints checked: {total_constraints}")
    print(f"Total constraints passed: {total_constraints_passed}")
    print(f"Constraint accuracy: {total_constraints_passed / total_constraints:.2%}" if total_constraints > 0 else "N/A")
    print("-" * 60)
    print(f"Average best@1: {total_best_at_1 / processed:.2%}" if processed > 0 else "N/A")
    print(f"Average best@8: {total_best_at_8 / processed:.2%}" if processed > 0 else "N/A")
    print(f"Average best@16: {total_best_at_16 / processed:.2%}" if processed > 0 else "N/A")
    print("=" * 60)

    # Save results
    if args.output_path:
        with open(args.output_path, "w", encoding="utf-8") as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
        print(f"\nResults saved to: {args.output_path}")

    if args.save_to_disk:
        # For HuggingFace dataset, include list columns for responses, scores, and advantages
        flattened_results = []
        for result in results:
            flat = {
                "key": result["key"],
                "instruction": result["instruction"],
                "ground_truth": json.dumps(result["ground_truth"]),
                "responses": result["responses"],
                "scores": result["scores"],
                "advantages": result["advantages"],
                "num_rollouts": result["num_rollouts"],
                "num_passed": result["num_passed"],
                "pass_rate": result["pass_rate"],
                "best_at_1": result["best_at_1"],
                "best_at_8": result["best_at_8"],
                "best_at_16": result["best_at_16"],
                "dataset": result["dataset"],
                "constraint_accuracy": result["constraint_accuracy"],
            }
            flattened_results.append(flat)

        dataset_out = Dataset.from_list(flattened_results)
        dataset_out.save_to_disk(args.save_to_disk)
        print(f"Dataset saved to: {args.save_to_disk}")

    print("\nDone!")


if __name__ == "__main__":
    main()
