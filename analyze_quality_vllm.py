#!/usr/bin/env python3
"""Analyze response quality issues using a vLLM OpenAI-compatible API."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from tqdm import tqdm

from chat import GENERATION_CONFIGS, LocalChat, ApiChat


def load_prompt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def extract_response(rollout_row: dict) -> str:
    """Extract the response text from a rollout row."""
    if rollout_row.get("responses"):
        return rollout_row["responses"][0]
    return ""


def parse_json_response(reply: str) -> Optional[dict]:
    """Extract JSON from the LLM reply, handling markdown code fences."""
    # Try direct parse first
    try:
        return json.loads(reply)
    except (json.JSONDecodeError, TypeError):
        pass

    # Try extracting from markdown code fences
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", reply, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except (json.JSONDecodeError, TypeError):
            pass

    # Try finding the first { ... } block
    brace_match = re.search(r"\{.*\}", reply, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except (json.JSONDecodeError, TypeError):
            pass

    return None


QUALITY_DIMS = [
    "incoherent_expression",
    "logical_inconsistency",
    "inappropriate_word_choice",
    "repetitive_expression",
    "language_inconsistency",
]


def validate_analysis(data: dict) -> Optional[dict]:
    """Validate and normalize a single-response analysis JSON."""
    if not isinstance(data, dict):
        return None

    normalized = {}
    for dim in QUALITY_DIMS:
        val = data.get(dim)
        if val in (0, 1):
            normalized[dim] = val
        elif val in ("0", "1"):
            normalized[dim] = int(val)
        else:
            normalized[dim] = 0
    normalized["notes"] = str(data.get("notes", ""))
    return normalized


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze response quality using a vLLM model."
    )
    parser.add_argument(
        "--model1-rollout",
        default="checkpoint_eval/eval_Qwen3-30B-GDPO-150.jsonl",
        help="Path to model 1 rollout JSONL (with responses).",
    )
    parser.add_argument(
        "--model1-graded",
        default="checkpoint_eval/eval_Qwen3-30B-GDPO-150_graded.jsonl",
        help="Path to model 1 graded JSONL (with scores).",
    )
    parser.add_argument(
        "--model2-rollout",
        default="checkpoint_eval/eval_Qwen3-30B-PPO-Norm-150.jsonl",
        help="Path to model 2 rollout JSONL (with responses).",
    )
    parser.add_argument(
        "--model2-graded",
        default="checkpoint_eval/eval_Qwen3-30B-PPO-Norm-150_graded.jsonl",
        help="Path to model 2 graded JSONL (with scores).",
    )
    parser.add_argument(
        "--output", "-o",
        default="quality_analysis.jsonl",
        help="Output JSONL file path.",
    )
    parser.add_argument(
        "--model",
        default="openai/gpt-oss-120b",
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
        default="prompt/quality_analysis.txt",
        help="Path to the system prompt file.",
    )
    parser.add_argument(
        "--user_prompt_path",
        default="prompt/quality_analysis_user.txt",
        help="Path to the user prompt template file.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional maximum number of samples to process (takes first N).",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Randomly sample this many datapoints for analysis.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for --sample. Default: 42.",
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=0,
        help="Skip samples before this index.",
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
        help="Number of threads to call the vLLM backend.",
    )
    parser.add_argument(
        "--max_inflight",
        type=int,
        default=32,
        help="Maximum number of in-flight requests.",
    )
    parser.add_argument(
        "--no_system",
        action="store_true",
        help="Prepend system prompt to user prompt as a single user message.",
    )
    parser.add_argument(
        "--save_prompts_jsonl",
        default=None,
        help="Save all prompts to this JSONL file instead of calling vLLM.",
    )
    parser.add_argument(
        "--partition_num",
        type=int,
        default=1,
        help="Total number of partitions. Default: 1 (no partitioning).",
    )
    parser.add_argument(
        "--partition_index",
        type=int,
        default=0,
        help="Index of the current partition (0-based). Default: 0.",
    )
    parser.add_argument(
        "--api_config",
        default=None,
        help="Path to ApiChat config JSON. When set, uses ApiChat instead of LocalChat.",
    )
    args = parser.parse_args()

    if args.partition_num < 1:
        parser.error("--partition_num must be >= 1")
    if args.partition_index < 0 or args.partition_index >= args.partition_num:
        parser.error(f"--partition_index must be in range [0, {args.partition_num - 1}]")

    system_prompt = load_prompt(args.system_prompt_path)
    user_prompt_template = load_prompt(args.user_prompt_path)
    for placeholder in ("{instruction}", "{response}"):
        if placeholder not in user_prompt_template:
            raise ValueError(f"User prompt template must contain {placeholder}.")

    generation_config = None
    if args.generation_config:
        generation_config = json.loads(args.generation_config)
    elif args.model not in GENERATION_CONFIGS:
        generation_config = {}

    # Load the 4 input files
    print(f"Loading model1 rollout: {args.model1_rollout}")
    m1_rollout = load_jsonl(args.model1_rollout)
    print(f"Loading model1 graded: {args.model1_graded}")
    m1_graded = load_jsonl(args.model1_graded)
    print(f"Loading model2 rollout: {args.model2_rollout}")
    m2_rollout = load_jsonl(args.model2_rollout)
    print(f"Loading model2 graded: {args.model2_graded}")
    m2_graded = load_jsonl(args.model2_graded)

    assert len(m1_rollout) == len(m2_rollout), (
        f"Row count mismatch: {len(m1_rollout)} vs {len(m2_rollout)}"
    )
    total_rows = len(m1_rollout)

    # Build index list with partitioning
    indices = list(range(total_rows))
    if args.partition_num > 1:
        indices = list(range(args.partition_index, total_rows, args.partition_num))
        print(f"[PARTITION] Processing partition {args.partition_index} of {args.partition_num}: "
              f"{len(indices)} samples (round-robin from {total_rows} total)")

    # Apply start_index, sampling, and max_samples
    indices = [i for i in indices if i >= args.start_index]
    if args.sample is not None:
        rng = random.Random(args.seed)
        if args.sample < len(indices):
            indices = sorted(rng.sample(indices, args.sample))
            print(f"[SAMPLE] Randomly sampled {len(indices)} datapoints (seed={args.seed})")
        else:
            print(f"[SAMPLE] Requested {args.sample} but only {len(indices)} available, using all")
    if args.max_samples is not None:
        indices = indices[:args.max_samples]

    # Only initialize chat client if not just saving prompts
    chat = None
    if not args.save_prompts_jsonl:
        if args.api_config:
            chat = ApiChat(
                config_path=args.api_config,
                model=args.model,
                cache_path=args.cache_path,
                generation_config=generation_config,
            )
        else:
            chat = LocalChat(
                model=args.model,
                base_url=args.base_url,
                cache_path=args.cache_path,
                generation_config=generation_config,
            )

    def build_messages(idx: int, response: str) -> Optional[list[dict]]:
        """Build the messages for a given row index and a single response."""
        instruction = m1_rollout[idx].get("instruction", "")
        if not instruction or not response:
            return None

        user_prompt = (
            user_prompt_template
            .replace("{instruction}", instruction)
            .replace("{response}", response)
        )

        if args.no_system:
            return [{"role": "user", "content": f"{system_prompt}\n{user_prompt}"}]
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def call_and_parse(idx: int, messages: list[dict], label: str) -> Optional[dict]:
        """Call the LLM for a single response and parse the result."""
        reply, _ = chat.ask(messages)
        if not reply:
            return None

        # Debug: print the first prompt/response pair
        try:
            if idx == indices[0] and label == "m1":
                print("--- DEBUG: first prompt/response ---")
                print("MESSAGES:\n" + json.dumps(messages[:1], indent=2, ensure_ascii=False)[:500])
                print("MODEL REPLY:\n" + (reply.strip()[:500] if reply else "<empty>"))
                print("--- END DEBUG ---")
        except Exception:
            pass

        parsed = parse_json_response(reply.strip())
        if parsed is None:
            tqdm.write(f"[WARN] Failed to parse JSON for idx={idx} {label}: {reply[:200]}")
            return None

        validated = validate_analysis(parsed)
        if validated is None:
            tqdm.write(f"[WARN] Invalid analysis structure for idx={idx} {label}")
            return None

        return validated

    def build_result(idx: int) -> Optional[dict]:
        """Make two pointwise LLM calls (one per response) for a given row."""
        response_m1 = extract_response(m1_rollout[idx])
        response_m2 = extract_response(m2_rollout[idx])

        msgs_m1 = build_messages(idx, response_m1)
        msgs_m2 = build_messages(idx, response_m2)

        if msgs_m1 is None and msgs_m2 is None:
            return None

        m1_analysis = call_and_parse(idx, msgs_m1, "m1") if msgs_m1 else None
        m2_analysis = call_and_parse(idx, msgs_m2, "m2") if msgs_m2 else None

        if m1_analysis is None and m2_analysis is None:
            return None

        empty = {dim: 0 for dim in QUALITY_DIMS}
        empty["notes"] = ""

        return {
            "idx": idx,
            "m1_analysis": m1_analysis if m1_analysis else empty,
            "m2_analysis": m2_analysis if m2_analysis else empty,
        }

    processed = 0
    skipped = 0
    max_inflight = max(1, args.max_inflight)
    limit = len(indices)

    # Mode 1: Save prompts only (2 prompts per row: m1 and m2)
    if args.save_prompts_jsonl:
        prompts_list = []
        pbar = tqdm(total=limit, unit="sample", desc="Building prompts")
        for idx in indices:
            response_m1 = extract_response(m1_rollout[idx])
            response_m2 = extract_response(m2_rollout[idx])
            msgs_m1 = build_messages(idx, response_m1)
            msgs_m2 = build_messages(idx, response_m2)
            if msgs_m1 is None and msgs_m2 is None:
                skipped += 1
            else:
                if msgs_m1:
                    prompts_list.append({"idx": idx, "model": "m1", "prompt": msgs_m1})
                if msgs_m2:
                    prompts_list.append({"idx": idx, "model": "m2", "prompt": msgs_m2})
                processed += 1
            pbar.update(1)
            pbar.set_postfix({"ok": processed, "skip": skipped})
        pbar.close()

        if not prompts_list:
            print("No prompts generated. Nothing to save.")
            return

        with open(args.save_prompts_jsonl, "w", encoding="utf-8") as f:
            for row in prompts_list:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Done! Saved {len(prompts_list)} prompts ({processed} rows) to {args.save_prompts_jsonl}, skipped {skipped}.")
        return

    # Mode 2: Call vLLM and save results
    results_map: dict[int, dict] = {}
    pbar = tqdm(total=limit, unit="sample", desc="Quality analysis")

    def handle_future(future, idx: int):
        nonlocal processed, skipped
        try:
            result = future.result()
            if result is None:
                skipped += 1
            else:
                results_map[idx] = result
                processed += 1
        except Exception as exc:
            skipped += 1
            tqdm.write(f"Error processing sample {idx}: {exc}")
        pbar.update(1)
        pbar.set_postfix({"ok": processed, "skip": skipped})

    with ThreadPoolExecutor(max_workers=max(1, args.num_workers)) as executor:
        futures: dict = {}
        for idx in indices:
            future = executor.submit(build_result, idx)
            futures[future] = idx

            if len(futures) >= max_inflight:
                for done in as_completed(futures):
                    done_idx = futures.pop(done)
                    handle_future(done, done_idx)
                    if len(futures) < max_inflight:
                        break

        for done in as_completed(futures):
            done_idx = futures.pop(done)
            handle_future(done, done_idx)

    pbar.close()

    results = [results_map[idx] for idx in sorted(results_map)]
    if not results:
        print("No samples processed. Nothing to save.")
        return

    with open(args.output, "w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Done! Analyzed {processed} samples, skipped {skipped}. Output: {args.output}")


if __name__ == "__main__":
    main()
