#!/usr/bin/env python3
"""
Analyze and grade rollouts using LLM-based checklist evaluation.

Takes rollout outputs and a dataset with checklists (from checklist_extraction_vllm.py),
uses an LLM to grade each response against its checklist, and computes metrics.

Example usage:
    # Grade and analyze rollouts
    python analyze_checklist.py \
        --input rollouts.jsonl \
        --checklist_dataset checklist_data \
        --model "Qwen/Qwen3-8B" \
        --base_url http://localhost:8000/v1

    # Analyze pre-graded results
    python analyze_checklist.py --input graded_checklist.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from datasets import load_dataset, load_from_disk
from tqdm import tqdm

from chat import GENERATION_CONFIGS, LocalChat


def load_prompt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


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


def parse_eval_output(output: str, num_checklist_items: int) -> list[int]:
    """Parse the LLM evaluation output to extract scores.

    Expected format:
    checklist_item_1\t1
    checklist_item_2\t0
    ...

    Returns list of scores (0 or 1).
    """
    scores = []
    lines = output.strip().split('\n')

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Try to extract score from end of line (after tab or last number)
        if '\t' in line:
            parts = line.rsplit('\t', 1)
            if len(parts) == 2:
                try:
                    score = int(parts[1].strip())
                    if score in (0, 1):
                        scores.append(score)
                        continue
                except ValueError:
                    pass

        # Fallback: look for 0 or 1 at end of line
        match = re.search(r'[01]\s*$', line)
        if match:
            scores.append(int(match.group().strip()))

    # If we got fewer scores than expected, pad with 0
    while len(scores) < num_checklist_items:
        scores.append(0)

    # If we got more, truncate
    scores = scores[:num_checklist_items]

    return scores


def best_of_k(scores: list[float], k: int) -> float:
    """Compute best score among the first k rollouts."""
    if not scores:
        return 0.0
    return max(scores[:k])


def pass_at_k(scores: list[float], k: int, threshold: float = 1.0) -> bool:
    """Check if at least one of the first k rollouts passes."""
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


def count_checklist_items(checklist: str) -> int:
    """Count the number of items in a checklist."""
    lines = [l.strip() for l in checklist.strip().split('\n') if l.strip()]
    return len(lines)


def grade_single_response(
    chat: LocalChat,
    system_prompt: str,
    user_prompt_template: str,
    instruction: str,
    response: str,
    checklist: str,
    no_system: bool,
) -> tuple[float, list[int]]:
    """Grade a single response against a checklist using LLM.

    Returns (score, individual_scores).
    """
    user_prompt = user_prompt_template.format(
        user_instruction=instruction,
        model_answer=response,
        checklist=checklist,
    )

    if no_system:
        messages = [{"role": "user", "content": f"{system_prompt}\n{user_prompt}"}]
    else:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    reply, _ = chat.ask(messages)

    if not reply:
        num_items = count_checklist_items(checklist)
        return 0.0, [0] * num_items

    num_items = count_checklist_items(checklist)
    individual_scores = parse_eval_output(reply, num_items)

    score = sum(individual_scores) / len(individual_scores) if individual_scores else 0.0

    return score, individual_scores


def grade_rollouts(
    rollouts: list[dict],
    checklist_dataset_path: str,
    split: str,
    chat: LocalChat,
    system_prompt: str,
    user_prompt_template: str,
    no_system: bool,
    strip_thinking: bool,
    num_workers: int,
    max_inflight: int,
) -> list[dict]:
    """Grade raw rollouts against checklists using LLM evaluation."""
    # Load checklist dataset and build key -> checklist mapping
    print(f"Loading checklist dataset: {checklist_dataset_path}")
    checklist_dataset = build_dataset(checklist_dataset_path, split, streaming=False)
    dataset_name = os.path.basename(checklist_dataset_path).replace("/", "_")

    key_to_checklist = {}
    key_to_instruction = {}
    for index, example in enumerate(tqdm(checklist_dataset, desc="Building checklist index")):
        key = get_example_key(example, index, dataset_name)
        checklist = example.get("checklist")
        instruction = example.get("raw_instruction") or example.get("instruction")
        if checklist:
            key_to_checklist[key] = checklist
            if instruction:
                key_to_instruction[key] = instruction

    print(f"Indexed {len(key_to_checklist)} examples with checklists.")

    # Grade rollouts
    results = []
    skipped = 0

    # Prepare grading tasks
    grading_tasks = []
    for rollout_idx, rollout in enumerate(rollouts):
        key = rollout["key"]
        instruction = rollout.get("instruction") or key_to_instruction.get(key, "")
        responses = rollout["responses"]

        checklist = key_to_checklist.get(key)
        if checklist is None:
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

        grading_tasks.append({
            "rollout_idx": rollout_idx,
            "key": key,
            "instruction": instruction,
            "responses": responses,
            "checklist": checklist,
        })

    if skipped > 0:
        print(f"Skipped {skipped} rollouts (no checklist found).")

    print(f"Grading {len(grading_tasks)} rollouts...")

    # Grade all responses
    results_map = {}

    def grade_task(task):
        scores = []
        all_individual_scores = []

        for response in task["responses"]:
            if not response:
                num_items = count_checklist_items(task["checklist"])
                scores.append(0.0)
                all_individual_scores.append([0] * num_items)
                continue

            score, individual_scores = grade_single_response(
                chat,
                system_prompt,
                user_prompt_template,
                task["instruction"],
                response,
                task["checklist"],
                no_system,
            )
            scores.append(score)
            all_individual_scores.append(individual_scores)

        return task["rollout_idx"], {
            "key": task["key"],
            "instruction": task["instruction"],
            "checklist": task["checklist"],
            "responses": task["responses"],
            "scores": scores,
            "individual_scores": all_individual_scores,
        }

    pbar = tqdm(total=len(grading_tasks), desc="Grading responses")

    with ThreadPoolExecutor(max_workers=max(1, num_workers)) as executor:
        futures = {}
        task_iter = iter(grading_tasks)

        # Submit initial batch
        for task in grading_tasks[:max_inflight]:
            future = executor.submit(grade_task, task)
            futures[future] = task["rollout_idx"]

        remaining_tasks = grading_tasks[max_inflight:]

        for future in as_completed(futures):
            try:
                rollout_idx, result = future.result()
                results_map[rollout_idx] = result
            except Exception as e:
                tqdm.write(f"Error grading: {e}")

            pbar.update(1)

            # Submit next task if available
            if remaining_tasks:
                next_task = remaining_tasks.pop(0)
                new_future = executor.submit(grade_task, next_task)
                futures[new_future] = next_task["rollout_idx"]

    pbar.close()

    # Build final results with metrics
    for rollout_idx in sorted(results_map.keys()):
        result = results_map[rollout_idx]
        scores = result["scores"]
        n = len(scores)

        # Compute metrics
        num_passed = sum(1 for s in scores if s >= 1.0)
        best_at_1 = best_of_k(scores, 1)
        best_at_8 = best_of_k(scores, min(8, n))
        best_at_16 = best_of_k(scores, n)
        advantages = compute_advantages(scores)

        # Flatten all individual scores for constraint accuracy
        all_constraint_results = []
        for ind_scores in result["individual_scores"]:
            all_constraint_results.extend(ind_scores)

        final_result = {
            "key": result["key"],
            "instruction": result["instruction"],
            "checklist": result["checklist"],
            "responses": result["responses"],
            "scores": scores,
            "advantages": advantages,
            "num_rollouts": n,
            "num_passed": num_passed,
            "pass_rate": num_passed / n if n > 0 else 0.0,
            "best_at_1": best_at_1,
            "best_at_8": best_at_8,
            "best_at_16": best_at_16,
            "individual_scores": result["individual_scores"],
            "constraint_accuracy": sum(all_constraint_results) / len(all_constraint_results) if all_constraint_results else 0.0,
        }
        results.append(final_result)

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

        # Constraint accuracy from individual_scores if available
        if "individual_scores" in result:
            for ind_scores in result["individual_scores"]:
                total_constraints += len(ind_scores)
                total_constraints_passed += sum(ind_scores)

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
            all_identical_partial += 1

    # Print results
    print("\n" + "=" * 60)
    print("CHECKLIST EVALUATION RESULTS")
    print("=" * 60)
    print(f"Total examples: {total}")
    print(f"Total rollouts: {total_rollouts}")
    print(f"Total rollouts passed (score=1): {total_passed}")
    print(f"Overall pass rate: {total_passed / total_rollouts:.2%}" if total_rollouts > 0 else "N/A")

    if total_constraints > 0:
        print(f"Total checklist items checked: {total_constraints}")
        print(f"Total checklist items passed: {total_constraints_passed}")
        print(f"Checklist item accuracy: {total_constraints_passed / total_constraints:.2%}")

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
    parser = argparse.ArgumentParser(
        description="Analyze and grade rollouts using LLM-based checklist evaluation."
    )
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
        help="Grade raw rollouts using LLM (requires --checklist_dataset).",
    )
    parser.add_argument(
        "--checklist_dataset",
        type=str,
        default=None,
        help="Dataset with checklists (from checklist_extraction_vllm.py).",
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

    # LLM options
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-8B",
        help="Model name served by vLLM.",
    )
    parser.add_argument(
        "--base_url",
        default="http://localhost:8000/v1",
        help="Base URL for the OpenAI-compatible API.",
    )
    parser.add_argument(
        "--cache_path",
        default=os.path.expanduser("~") + "/.cache",
        help="Disk cache directory for LocalChat.",
    )
    parser.add_argument(
        "--system_prompt_path",
        default="prompt/checklist_eval.txt",
        help="Path to the system prompt file.",
    )
    parser.add_argument(
        "--user_prompt_path",
        default="prompt/checklist_eval_user.txt",
        help="Path to the user prompt template file.",
    )
    parser.add_argument(
        "--no_system",
        action="store_true",
        help="Prepend system prompt to user prompt as single message.",
    )
    parser.add_argument(
        "--generation_config",
        type=str,
        default=None,
        help="Optional JSON string to override generation settings.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of threads for LLM calls.",
    )
    parser.add_argument(
        "--max_inflight",
        type=int,
        default=32,
        help="Maximum number of in-flight requests.",
    )

    args = parser.parse_args()

    if args.grade and not args.checklist_dataset:
        parser.error("--checklist_dataset is required when using --grade")

    print(f"Loading results from: {args.input}")
    results = load_results(args.input, args.format)
    print(f"Loaded {len(results)} results.")

    # Grade if requested
    if args.grade:
        # Load prompts
        system_prompt = load_prompt(args.system_prompt_path)
        user_prompt_template = load_prompt(args.user_prompt_path)

        # Initialize chat client
        generation_config = None
        if args.generation_config:
            generation_config = json.loads(args.generation_config)
        elif args.model not in GENERATION_CONFIGS:
            generation_config = {}

        chat = LocalChat(
            model=args.model,
            base_url=args.base_url,
            cache_path=args.cache_path,
            generation_config=generation_config,
        )

        results = grade_rollouts(
            results,
            args.checklist_dataset,
            args.split,
            chat,
            system_prompt,
            user_prompt_template,
            args.no_system,
            args.strip_thinking,
            args.num_workers,
            args.max_inflight,
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
