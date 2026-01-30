#!/usr/bin/env python3
"""
Analyze rollout results from rollout_and_verify.py.

Computes core metrics:
- best@k (best score among first k rollouts)
- pass@k (at least one pass among first k rollouts)
- Overall pass rate and constraint accuracy
- Count of rows with all scores = 1 (perfect)
- Count of rows with all scores = 0 (all failed)

Example usage:
    python analyze_rollouts.py --input results.jsonl
    python analyze_rollouts.py --input results_dataset --format disk
"""

import argparse
import json
from collections import defaultdict

from datasets import load_from_disk


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
        elif "constraint_accuracy" in result:
            # Estimate from constraint_accuracy and num constraints
            pass

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
    parser = argparse.ArgumentParser(description="Analyze rollout results.")
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

    args = parser.parse_args()

    print(f"Loading results from: {args.input}")
    results = load_results(args.input, args.format)
    print(f"Loaded {len(results)} results.")

    metrics = analyze(results, args.k)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nMetrics saved to: {args.output}")


if __name__ == "__main__":
    main()
