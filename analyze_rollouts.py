#!/usr/bin/env python3
"""
Analyze and optionally grade rollout results.

Three modes:
1. Analyze pre-graded results (from rollout_and_verify.py)
2. Grade and analyze raw rollouts (from rollout_only.py) with a dataset
3. Grade verifiable constraints from a checklist-graded dataset

Computes core metrics:
- Best soft reward among the first k rollouts
- Any non-zero hard reward among the first k rollouts
- Overall pass rate and constraint accuracy
- Count of rows with all scores = 1 (perfect)
- Count of rows with all scores = 0 (all failed)
- Count of rows with identical partial scores (no variance)

Example usage:
    # Analyze pre-graded results
    python analyze_rollouts.py --input graded_results.jsonl

    # Grade and analyze raw rollouts
    python analyze_rollouts.py --input rollouts.jsonl --dataset data.jsonl --grade

    # Grade verifiable constraints from checklist-graded dataset
    python analyze_rollouts.py --grade_verifiable --dataset graded_dataset
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import defaultdict

from datasets import load_dataset, load_from_disk
from tqdm import tqdm

from verify_constraints import verify_multiple_constraints


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


def get_ground_truth(example: dict) -> list[dict] | None:
    """Extract ground truth constraints from the example."""
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
    """Compute best score among the first k rollouts."""
    if not scores:
        return 0.0
    return max(scores[:k])


def nonzero_hard_reward_at_k(hard_rewards: list[int], k: int) -> bool:
    """Check if any of the first k rollouts has non-zero hard reward."""
    if not hard_rewards:
        return False
    return any(r > 0 for r in hard_rewards[:k])


def compute_advantages(scores: list[float], epsilon: float = 1e-8) -> list[float]:
    """Compute GRPO-style group-level advantages."""
    if not scores:
        return []

    n = len(scores)
    mu = sum(scores) / n
    variance = sum((s - mu) ** 2 for s in scores) / n
    sigma = math.sqrt(variance)

    return [(s - mu) / (sigma + epsilon) for s in scores]


def verify_responses(
    responses: list[str],
    ground_truth: list[dict],
) -> tuple[list[dict], int, list[bool]]:
    """Verify each response against the ground truth constraints."""
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


def load_results(input_paths: list[str], format: str):
    """Load results from JSONL file(s) or HuggingFace dataset."""
    if format == "jsonl":
        results = []
        for input_path in input_paths:
            with open(input_path, "r", encoding="utf-8") as f:
                for line in f:
                    results.append(json.loads(line))
        return results
    elif format == "disk":
        if len(input_paths) != 1:
            raise ValueError("HuggingFace disk format does not support partitioned inputs.")
        dataset = load_from_disk(input_paths[0])
        return list(dataset)
    else:
        raise ValueError(f"Unknown format: {format}")


def strip_partition_suffix(path: str) -> str:
    """Strip _p{idx}_of_{num} suffix if present."""
    base, ext = os.path.splitext(path)
    base = re.sub(r"_p\d+_of_\d+$", "", base)
    return f"{base}{ext}"


def build_partition_paths(base_path: str, partition_num: int) -> list[str]:
    """Build partitioned file paths from a base path."""
    base_path = strip_partition_suffix(base_path)
    base, ext = os.path.splitext(base_path)
    return [f"{base}_p{idx}_of_{partition_num}{ext}" for idx in range(partition_num)]


def grade_rollouts(
    rollouts: list[dict],
    dataset_path: str,
    split: str,
    strip_thinking: bool,
) -> list[dict]:
    """Grade raw rollouts against ground truth constraints."""
    # Load dataset and build key -> ground_truth mapping
    print(f"Loading dataset: {dataset_path}")
    dataset = build_dataset(dataset_path, split, streaming=False)
    dataset_name = os.path.basename(dataset_path).replace("/", "_")

    key_to_ground_truth = {}
    for index, example in enumerate(tqdm(dataset, desc="Building ground truth index")):
        key = get_example_key(example, index, dataset_name)
        gt = get_ground_truth(example)
        if gt is not None:
            key_to_ground_truth[key] = gt

    print(f"Indexed {len(key_to_ground_truth)} examples with ground truth.")

    # Grade rollouts
    results = []
    skipped = 0

    for rollout in tqdm(rollouts, desc="Grading"):
        key = rollout["key"]
        instruction = rollout["instruction"]
        responses = rollout["responses"]

        # Get ground truth
        ground_truth = key_to_ground_truth.get(key)
        if ground_truth is None:
            skipped += 1
            continue

        # Optionally strip thinking tokens
        if strip_thinking:
            clean_responses = []
            for resp in responses:
                if '</think>' in resp:
                    resp = resp.split('</think>')[-1].strip()
                clean_responses.append(resp)
            responses = clean_responses

        # Verify responses
        rollout_results, num_passed, all_constraint_results = verify_responses(
            responses, ground_truth
        )

        n = len(responses)
        scores = [r["score"] for r in rollout_results]

        # Compute best soft reward metrics
        best_soft_reward_at_1 = best_of_k(scores, 1)
        best_soft_reward_at_8 = best_of_k(scores, min(8, n))
        best_soft_reward_at_16 = best_of_k(scores, n)

        # Compute advantages
        advantages = compute_advantages(scores)

        result = {
            "key": key,
            "instruction": instruction,
            "ground_truth": ground_truth,
            "responses": [r["response"] for r in rollout_results],
        "scores": scores,
        "advantages": advantages,
        "num_rollouts": n,
            "num_passed": num_passed,
            "pass_rate": num_passed / n if n > 0 else 0.0,
        "best_soft_reward_at_1": best_soft_reward_at_1,
        "best_soft_reward_at_8": best_soft_reward_at_8,
        "best_soft_reward_at_16": best_soft_reward_at_16,
        "rollouts": rollout_results,
        "constraint_accuracy": sum(all_constraint_results) / len(all_constraint_results) if all_constraint_results else 0.0,
        }
        results.append(result)

    if skipped > 0:
        print(f"Skipped {skipped} rollouts (no ground truth found).")

    return results


def analyze(results: list[dict], k_values: list[int] = None):
    """Analyze rollout results and compute metrics."""
    if k_values is None:
        k_values = [1, 2, 4, 8, 16]

    total = len(results)
    if total == 0:
        print("No results to analyze.")
        return

    # Accumulators
    total_rollouts = 0
    total_passed = 0
    total_constraints = 0
    total_constraints_passed = 0

    best_soft_reward_by_k = defaultdict(float)
    nonzero_hard_reward_by_k_count = defaultdict(int)

    all_perfect = 0  # All scores = 1
    all_failed = 0   # All scores = 0
    all_identical_partial = 0  # All scores identical, but not 0 or 1

    for result in results:
        scores = result.get("scores", [])
        n = len(scores)

        if n == 0:
            continue

        total_rollouts += n
        total_passed += sum(1 for s in scores if s >= 1.0)

        # Constraint accuracy from rollouts if available
        if "rollouts" in result:
            for rollout in result["rollouts"]:
                constraint_results = rollout.get("constraint_results", [])
                total_constraints += len(constraint_results)
                total_constraints_passed += sum(constraint_results)

        hard_rewards = []
        if "rollouts" in result:
            for rollout in result["rollouts"]:
                hard_rewards.append(1 if rollout.get("passed") else 0)
        else:
            hard_rewards = [1 if s >= 1.0 else 0 for s in scores]

        # Best soft reward and non-zero hard reward among first k rollouts
        for k in k_values:
            actual_k = min(k, n)
            best_soft_reward_by_k[k] += best_of_k(scores, actual_k)
            if nonzero_hard_reward_at_k(hard_rewards, actual_k):
                nonzero_hard_reward_by_k_count[k] += 1

        # Count perfect and all-failed
        if all(s >= 1.0 for s in scores):
            all_perfect += 1
        elif all(s == 0.0 for s in scores):
            all_failed += 1
        elif len(set(scores)) == 1:
            # All scores identical (advantages all 0), but not 0 or 1
            all_identical_partial += 1

    # Print results
    print("\n" + "=" * 60)
    print("ROLLOUT ANALYSIS RESULTS")
    print("=" * 60)
    print(f"Total examples: {total}")
    print(f"Total rollouts: {total_rollouts}")
    print(f"Total rollouts passed (score=1): {total_passed}")
    print(f"Overall pass rate: {total_passed / total_rollouts:.2%}" if total_rollouts > 0 else "N/A")

    if total_constraints > 0:
        print(f"Total constraints checked: {total_constraints}")
        print(f"Total constraints passed: {total_constraints_passed}")
        print(f"Constraint accuracy: {total_constraints_passed / total_constraints:.2%}")

    print("-" * 60)
    print("Best soft reward (average best score among first k rollouts):")
    for k in k_values:
        avg = best_soft_reward_by_k[k] / total if total > 0 else 0
        print(f"  k={k}: {avg:.2%}")

    print("-" * 60)
    print("Non-zero hard reward (fraction with any hard reward > 0 in first k):")
    for k in k_values:
        rate = nonzero_hard_reward_by_k_count[k] / total if total > 0 else 0
        print(f"  k={k}: {rate:.2%} ({nonzero_hard_reward_by_k_count[k]}/{total})")

    print("-" * 60)
    print("Score distribution:")
    print(f"  All perfect (all scores = 1): {all_perfect} ({all_perfect / total:.2%})")
    print(f"  All failed (all scores = 0): {all_failed} ({all_failed / total:.2%})")
    print(f"  All identical partial (0 < score < 1, no variance): {all_identical_partial} ({all_identical_partial / total:.2%})")
    mixed = total - all_perfect - all_failed - all_identical_partial
    print(f"  Mixed (has variance): {mixed} ({mixed / total:.2%})")
    print("=" * 60)

    return {
        "total": total,
        "total_rollouts": total_rollouts,
        "total_passed": total_passed,
        "overall_pass_rate": total_passed / total_rollouts if total_rollouts > 0 else 0,
        "best_soft_reward_by_k": {k: best_soft_reward_by_k[k] / total for k in k_values},
        "nonzero_hard_reward_by_k": {k: nonzero_hard_reward_by_k_count[k] / total for k in k_values},
        "all_perfect": all_perfect,
        "all_failed": all_failed,
        "all_identical_partial": all_identical_partial,
    }


def extract_responses_from_messages(messages: list[dict]) -> list[str]:
    """Extract assistant responses from messages."""
    responses = []
    for msg in messages:
        if msg.get("role") == "assistant":
            responses.append(msg.get("content", ""))
    return responses


def grade_verifiable_from_dataset(
    dataset_path: str,
    split: str,
    strip_thinking: bool,
    responses_column: str | None = None,
    messages_column: str = "messages_verifiable",
) -> list[dict]:
    """Grade verifiable constraints directly from a dataset.
    
    The dataset should have:
    - ground_truth_verifiable: the verifiable constraints to check against
    - Either a responses column or messages column to extract responses from
    """
    print(f"Loading dataset: {dataset_path}")
    dataset = build_dataset(dataset_path, split, streaming=False)
    dataset_name = os.path.basename(dataset_path).replace("/", "_")

    results = []
    skipped = 0

    for index, example in enumerate(tqdm(dataset, desc="Grading verifiable constraints")):
        key = get_example_key(example, index, dataset_name)
        
        # Get ground truth verifiable constraints
        ground_truth = get_ground_truth(example)
        if ground_truth is None:
            skipped += 1
            continue

        # Get responses - either from a responses column or from messages
        responses = []
        if responses_column and responses_column in example:
            responses = example[responses_column]
            if isinstance(responses, str):
                responses = [responses]
        elif messages_column and messages_column in example:
            messages = example[messages_column]
            if messages:
                responses = extract_responses_from_messages(messages)
        
        if not responses:
            skipped += 1
            continue

        # Optionally strip thinking tokens
        if strip_thinking:
            clean_responses = []
            for resp in responses:
                if '</think>' in resp:
                    resp = resp.split('</think>')[-1].strip()
                clean_responses.append(resp)
            responses = clean_responses

        # Verify responses
        rollout_results, num_passed, all_constraint_results = verify_responses(
            responses, ground_truth
        )

        n = len(responses)
        scores = [r["score"] for r in rollout_results]

        # Compute best soft reward metrics
        best_soft_reward_at_1 = best_of_k(scores, 1)
        best_soft_reward_at_8 = best_of_k(scores, min(8, n))
        best_soft_reward_at_16 = best_of_k(scores, n)

        # Compute advantages
        advantages = compute_advantages(scores)

        # Get instruction
        instruction = ""
        if "augmented_prompt" in example:
            instruction = example["augmented_prompt"]
        elif "instruction" in example:
            instruction = example["instruction"]
        elif messages_column in example and example[messages_column]:
            for msg in example[messages_column]:
                if msg.get("role") == "user":
                    instruction = msg.get("content", "")
                    break

        result = {
            "key": key,
            "instruction": instruction,
            "ground_truth": ground_truth,
            "responses": [r["response"] for r in rollout_results],
            "scores": scores,
            "advantages": advantages,
            "num_rollouts": n,
            "num_passed": num_passed,
            "pass_rate": num_passed / n if n > 0 else 0.0,
            "best_soft_reward_at_1": best_soft_reward_at_1,
            "best_soft_reward_at_8": best_soft_reward_at_8,
            "best_soft_reward_at_16": best_soft_reward_at_16,
            "rollouts": rollout_results,
            "constraint_accuracy": sum(all_constraint_results) / len(all_constraint_results) if all_constraint_results else 0.0,
        }
        results.append(result)

    if skipped > 0:
        print(f"Skipped {skipped} examples (no ground truth or responses found).")

    return results


def save_dataset_with_scores(
    dataset_path: str,
    split: str,
    results: list[dict],
    output_path: str,
    scores_col: str = "scores",
    soft_rewards_col: str = "soft_rewards",
    hard_rewards_col: str = "hard_rewards",
    variance_col: str = "verifiable_has_variance",
) -> None:
    """Append rewards and variance to the dataset and save to disk."""
    dataset = build_dataset(dataset_path, split, streaming=False)
    dataset_name = os.path.basename(dataset_path).replace("/", "_")

    score_map = {}
    variance_map = {}
    soft_rewards_map = {}
    hard_rewards_map = {}
    for result in results:
        scores = result.get("scores", [])
        score_map[result["key"]] = scores
        variance_map[result["key"]] = 1 if len(set(scores)) > 1 else 0
        soft_rewards_map[result["key"]] = scores
        if "rollouts" in result:
            hard_rewards_map[result["key"]] = [
                1 if rollout.get("passed") else 0 for rollout in result["rollouts"]
            ]
        else:
            hard_rewards_map[result["key"]] = [1 if s >= 1.0 else 0 for s in scores]

    scores_column_data = []
    variance_column_data = []
    soft_rewards_column_data = []
    hard_rewards_column_data = []
    for index, example in enumerate(tqdm(dataset, desc="Appending scores")):
        key = get_example_key(example, index, dataset_name)
        scores = score_map.get(key, [])
        scores_column_data.append(scores)
        variance_column_data.append(variance_map.get(key, 0))
        soft_rewards_column_data.append(soft_rewards_map.get(key, []))
        hard_rewards_column_data.append(hard_rewards_map.get(key, []))

    dataset = dataset.add_column(scores_col, scores_column_data)
    dataset = dataset.add_column(soft_rewards_col, soft_rewards_column_data)
    dataset = dataset.add_column(hard_rewards_col, hard_rewards_column_data)
    dataset = dataset.add_column(variance_col, variance_column_data)
    dataset.save_to_disk(output_path)
    print(f"  Columns: {scores_col}, {soft_rewards_col}, {hard_rewards_col}, {variance_col}")


def main():
    parser = argparse.ArgumentParser(description="Analyze and optionally grade rollout results.")
    parser.add_argument(
        "--input",
        required=False,
        default=None,
        help="Path to input JSONL file or HuggingFace dataset directory (not required for --grade_verifiable).",
    )
    parser.add_argument(
        "--partition_num",
        type=int,
        default=1,
        help="Total number of partitions for partitioned JSONL inputs.",
    )
    parser.add_argument(
        "--format",
        choices=["jsonl", "disk"],
        default="jsonl",
        help="Input format: jsonl or disk (HuggingFace dataset).",
    )
    parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16],
        help="Values of k for best soft reward and non-zero hard reward metrics.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save metrics as JSON.",
    )

    # Grading options
    parser.add_argument(
        "--grade",
        action="store_true",
        help="Grade raw rollouts (requires --dataset).",
    )
    parser.add_argument(
        "--grade_verifiable",
        action="store_true",
        help="Grade verifiable constraints from a checklist-graded dataset (requires --dataset).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset with ground_truth constraints (required for --grade or --grade_verifiable).",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to use (default: train).",
    )
    parser.add_argument(
        "--strip_thinking",
        action="store_true",
        default=True,
        help="Strip thinking tokens (</think>) from responses before grading (default: True).",
    )
    parser.add_argument(
        "--no_strip_thinking",
        action="store_true",
        help="Disable stripping thinking tokens from responses.",
    )
    parser.add_argument(
        "--responses_column",
        type=str,
        default=None,
        help="Column name containing responses (for --grade_verifiable).",
    )
    parser.add_argument(
        "--messages_column",
        type=str,
        default="messages_verifiable",
        help="Column name containing messages to extract responses from (default: messages_verifiable).",
    )
    parser.add_argument(
        "--save_graded",
        type=str,
        default=None,
        help="Optional path to save graded results as JSONL.",
    )
    parser.add_argument(
        "--save_dataset",
        type=str,
        default=None,
        help="Path to save the dataset with appended scores/variance.",
    )

    # Column name customization
    parser.add_argument(
        "--scores_column",
        type=str,
        default="scores",
        help="Column name for scores (default: scores).",
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
    parser.add_argument(
        "--variance_column",
        type=str,
        default="verifiable_has_variance",
        help="Column name for variance indicator (default: verifiable_has_variance).",
    )

    args = parser.parse_args()

    # Handle --no_strip_thinking flag
    if args.no_strip_thinking:
        args.strip_thinking = False

    if args.grade and not args.dataset:
        parser.error("--dataset is required when using --grade")

    if args.grade_verifiable and not args.dataset:
        parser.error("--dataset is required when using --grade_verifiable")

    if args.grade and args.grade_verifiable:
        parser.error("Cannot use both --grade and --grade_verifiable")

    if not args.grade_verifiable and not args.input:
        parser.error("--input is required unless using --grade_verifiable")

    if args.partition_num < 1:
        parser.error("--partition_num must be >= 1")

    if args.partition_num > 1 and args.format != "jsonl":
        parser.error("--partition_num is only supported with --format jsonl")

    # Mode 3: Grade verifiable constraints from a checklist-graded dataset
    if args.grade_verifiable:
        results = grade_verifiable_from_dataset(
            args.dataset,
            args.split,
            args.strip_thinking,
            responses_column=args.responses_column,
            messages_column=args.messages_column,
        )
        print(f"Graded {len(results)} results for verifiable constraints.")

        if args.save_graded:
            with open(args.save_graded, "w", encoding="utf-8") as f:
                for result in results:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
            print(f"Graded results saved to: {args.save_graded}")

        output_path = args.save_dataset
        if output_path is None:
            if os.path.isdir(args.dataset):
                output_path = f"{args.dataset}_with_verifiable_scores"
            else:
                output_path = f"{os.path.basename(args.dataset)}_with_verifiable_scores"

        save_dataset_with_scores(
            args.dataset,
            args.split,
            results,
            output_path,
            scores_col=args.scores_column,
            soft_rewards_col=args.soft_rewards_column,
            hard_rewards_col=args.hard_rewards_column,
            variance_col=args.variance_column,
        )
        print(f"Dataset with verifiable scores saved to: {output_path}")

        metrics = analyze(results, args.k)

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)
            print(f"\nMetrics saved to: {args.output}")

        return

    # Mode 1 & 2: Load results from JSONL or disk
    if args.partition_num > 1:
        base_path = args.input
        input_paths = build_partition_paths(base_path, args.partition_num)
        for path in input_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing partition file: {path}")
        print(f"Loading results from {len(input_paths)} partitions (base: {base_path})")
    else:
        input_paths = [args.input]
        print(f"Loading results from: {args.input}")

    results = load_results(input_paths, args.format)
    print(f"Loaded {len(results)} results.")

    # Grade if requested (Mode 2)
    if args.grade:
        results = grade_rollouts(
            results,
            args.dataset,
            args.split,
            args.strip_thinking,
        )
        print(f"Graded {len(results)} results.")

        if args.save_graded:
            with open(args.save_graded, "w", encoding="utf-8") as f:
                for result in results:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
            print(f"Graded results saved to: {args.save_graded}")

        output_path = args.save_dataset
        if output_path is None:
            if os.path.isdir(args.dataset):
                output_path = f"{args.dataset}_with_rollout_scores"
            else:
                output_path = f"{os.path.basename(args.dataset)}_with_rollout_scores"

        save_dataset_with_scores(
            args.dataset,
            args.split,
            results,
            output_path,
            scores_col=args.scores_column,
            soft_rewards_col=args.soft_rewards_column,
            hard_rewards_col=args.hard_rewards_column,
            variance_col=args.variance_column,
        )
        print(f"Dataset with scores saved to: {output_path}")

    metrics = analyze(results, args.k)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nMetrics saved to: {args.output}")


if __name__ == "__main__":
    main()
