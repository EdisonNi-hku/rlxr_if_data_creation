#!/usr/bin/env python3
"""
Analyze and optionally grade rollout results.

Two modes:
1. Analyze pre-graded results (from rollout_and_verify.py)
2. Grade and analyze raw rollouts (from rollout_only.py) with a dataset

Computes core metrics:
- best@k (best score among first k rollouts)
- pass@k (at least one pass among first k rollouts)
- Overall pass rate and constraint accuracy
- Count of rows with all scores = 1 (perfect)
- Count of rows with all scores = 0 (all failed)
- Count of rows with identical partial scores (no variance)

Example usage:
    # Analyze pre-graded results
    python analyze_rollouts.py --input graded_results.jsonl

    # Grade and analyze raw rollouts
    python analyze_rollouts.py --input rollouts.jsonl --dataset data.jsonl --grade
"""

from __future__ import annotations

import argparse
import json
import math
import os
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


def pass_at_k(scores: list[float], k: int, threshold: float = 1.0) -> bool:
    """Check if at least one of the first k rollouts passes (score >= threshold)."""
    if not scores:
        return False
    return any(s >= threshold for s in scores[:k])


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


def load_results(input_path: str, format: str):
    """Load results from JSONL file or HuggingFace dataset."""
    if format == "jsonl":
        results = []
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                results.append(json.loads(line))
        return results
    elif format == "disk":
        dataset = load_from_disk(input_path)
        return list(dataset)
    else:
        raise ValueError(f"Unknown format: {format}")


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

        # Compute best@k metrics
        best_at_1 = best_of_k(scores, 1)
        best_at_8 = best_of_k(scores, min(8, n))
        best_at_16 = best_of_k(scores, n)

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
            "best_at_1": best_at_1,
            "best_at_8": best_at_8,
            "best_at_16": best_at_16,
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

    best_at_k = defaultdict(float)
    pass_at_k_count = defaultdict(int)

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

        # best@k and pass@k
        for k in k_values:
            actual_k = min(k, n)
            best_at_k[k] += best_of_k(scores, actual_k)
            if pass_at_k(scores, actual_k):
                pass_at_k_count[k] += 1

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
    print("Best@k (average best score among first k rollouts):")
    for k in k_values:
        avg = best_at_k[k] / total if total > 0 else 0
        print(f"  best@{k}: {avg:.2%}")

    print("-" * 60)
    print("Pass@k (fraction with at least one perfect score in first k):")
    for k in k_values:
        rate = pass_at_k_count[k] / total if total > 0 else 0
        print(f"  pass@{k}: {rate:.2%} ({pass_at_k_count[k]}/{total})")

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
        "best_at_k": {k: best_at_k[k] / total for k in k_values},
        "pass_at_k": {k: pass_at_k_count[k] / total for k in k_values},
        "all_perfect": all_perfect,
        "all_failed": all_failed,
        "all_identical_partial": all_identical_partial,
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze and optionally grade rollout results.")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input JSONL file or HuggingFace dataset directory.",
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
        help="Values of k for best@k and pass@k metrics.",
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
        "--dataset",
        type=str,
        default=None,
        help="Dataset with ground_truth constraints (required for --grade).",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to use (default: train).",
    )
    parser.add_argument(
        "--strip_thinking",
        action="store_true",
        help="Strip thinking tokens (</think>) from responses before grading.",
    )
    parser.add_argument(
        "--save_graded",
        type=str,
        default=None,
        help="Optional path to save graded results as JSONL.",
    )

    args = parser.parse_args()

    if args.grade and not args.dataset:
        parser.error("--dataset is required when using --grade")

    print(f"Loading results from: {args.input}")
    results = load_results(args.input, args.format)
    print(f"Loaded {len(results)} results.")

    # Grade if requested
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

    metrics = analyze(results, args.k)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nMetrics saved to: {args.output}")


if __name__ == "__main__":
    main()
