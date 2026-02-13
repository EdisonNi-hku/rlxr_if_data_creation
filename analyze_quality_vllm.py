#!/usr/bin/env python3
"""Analyze response quality issues using a vLLM OpenAI-compatible API."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import threading
import traceback
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
        default=None,
        help="Path to model 2 rollout JSONL. Optional — omit to grade only model 1.",
    )
    parser.add_argument(
        "--model2-graded",
        default=None,
        help="Path to model 2 graded JSONL. Optional — omit to grade only model 1.",
    )
    parser.add_argument(
        "--output", "-o",
        default="quality_analysis.jsonl",
        help="Output JSONL file path.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name. Overrides config file model when using --api_config.",
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
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose debug output (prints full LLM responses, payloads, etc.).",
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
    elif args.model and args.model not in GENERATION_CONFIGS:
        generation_config = {}

    # Load input files
    print(f"Loading model1 rollout: {args.model1_rollout}")
    m1_rollout = load_jsonl(args.model1_rollout)
    print(f"Loading model1 graded: {args.model1_graded}")
    m1_graded = load_jsonl(args.model1_graded)

    two_models = args.model2_rollout is not None
    m2_rollout = None
    m2_graded = None
    if two_models:
        print(f"Loading model2 rollout: {args.model2_rollout}")
        m2_rollout = load_jsonl(args.model2_rollout)
        print(f"Loading model2 graded: {args.model2_graded}")
        m2_graded = load_jsonl(args.model2_graded)
        assert len(m1_rollout) == len(m2_rollout), (
            f"Row count mismatch: {len(m1_rollout)} vs {len(m2_rollout)}"
        )
    else:
        print("[INFO] Single-model mode (no model2 provided)")

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
            # For ApiChat: only pass model/generation_config if explicitly provided,
            # otherwise let ApiChat use values from the config file.
            chat = ApiChat(
                config_path=args.api_config,
                model=args.model,          # None unless user passed --model explicitly
                cache_path=args.cache_path,
                generation_config=generation_config,  # None unless user passed --generation_config
                debug=args.debug,
            )
        else:
            model = args.model or "openai/gpt-oss-120b"
            chat = LocalChat(
                model=model,
                base_url=args.base_url,
                cache_path=args.cache_path,
                generation_config=generation_config,
            )

    # Track failure reasons for the final summary
    fail_counts = {
        "empty_instruction": 0,
        "empty_response": 0,
        "empty_reply": 0,
        "json_parse_fail": 0,
        "validation_fail": 0,
        "exception": 0,
        "all_none": 0,
    }
    fail_lock = threading.Lock()

    def build_messages(idx: int, response: str) -> Optional[list[dict]]:
        """Build the messages for a given row index and a single response."""
        instruction = m1_rollout[idx].get("instruction", "")
        if not instruction:
            with fail_lock:
                fail_counts["empty_instruction"] += 1
            return None
        if not response:
            with fail_lock:
                fail_counts["empty_response"] += 1
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
        reply, reasoning = chat.ask(messages)

        if args.debug:
            tqdm.write(
                f"\n[DEBUG] idx={idx} {label}:\n"
                f"  USER PROMPT:\n{messages[-1]['content']}\n"
                f"  RAW REPLY:\n{reply}\n"
                f"  REASONING:\n{reasoning}\n"
            )

        if not reply:
            with fail_lock:
                fail_counts["empty_reply"] += 1
            tqdm.write(f"[SKIP] idx={idx} {label}: LLM returned empty reply")
            return None

        parsed = parse_json_response(reply.strip())
        if parsed is None:
            with fail_lock:
                fail_counts["json_parse_fail"] += 1
            tqdm.write(
                f"[SKIP] idx={idx} {label}: JSON parse failed\n"
                f"  Raw reply (first 300 chars): {reply.strip()[:300]}"
            )
            return None

        validated = validate_analysis(parsed)
        if validated is None:
            with fail_lock:
                fail_counts["validation_fail"] += 1
            tqdm.write(
                f"[SKIP] idx={idx} {label}: validation failed\n"
                f"  Parsed keys: {list(parsed.keys()) if isinstance(parsed, dict) else type(parsed).__name__}\n"
                f"  Parsed value: {json.dumps(parsed, ensure_ascii=False)[:300]}"
            )
            return None

        return validated

    def build_result(idx: int) -> Optional[dict]:
        """Make pointwise LLM calls for a given row."""
        response_m1 = extract_response(m1_rollout[idx])
        msgs_m1 = build_messages(idx, response_m1)

        if two_models:
            response_m2 = extract_response(m2_rollout[idx])
            msgs_m2 = build_messages(idx, response_m2)
        else:
            msgs_m2 = None

        if msgs_m1 is None and msgs_m2 is None:
            with fail_lock:
                fail_counts["all_none"] += 1
            tqdm.write(f"[SKIP] idx={idx}: could not build messages (empty instruction or response)")
            return None

        empty = {dim: 0 for dim in QUALITY_DIMS}
        empty["notes"] = ""

        m1_analysis = call_and_parse(idx, msgs_m1, "m1") if msgs_m1 else None
        result = {"idx": idx, "m1_analysis": m1_analysis if m1_analysis else empty}

        if two_models:
            m2_analysis = call_and_parse(idx, msgs_m2, "m2") if msgs_m2 else None
            result["m2_analysis"] = m2_analysis if m2_analysis else empty

        if m1_analysis is None and (not two_models or result.get("m2_analysis") == empty):
            return None

        return result

    processed = 0
    skipped = 0
    max_inflight = max(1, args.max_inflight)
    limit = len(indices)

    # Mode 1: Save prompts only
    if args.save_prompts_jsonl:
        prompts_list = []
        pbar = tqdm(total=limit, unit="sample", desc="Building prompts")
        for idx in indices:
            response_m1 = extract_response(m1_rollout[idx])
            msgs_m1 = build_messages(idx, response_m1)
            msgs_m2 = None
            if two_models:
                response_m2 = extract_response(m2_rollout[idx])
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
            with fail_lock:
                fail_counts["exception"] += 1
            tqdm.write(
                f"[ERROR] idx={idx}: exception in build_result\n"
                f"  {type(exc).__name__}: {exc}\n"
                f"  {''.join(traceback.format_tb(exc.__traceback__))}"
            )
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

    # Print failure breakdown
    total_failures = sum(fail_counts.values())
    if total_failures > 0 or skipped > 0:
        print(f"\n--- Failure breakdown ({skipped} skipped out of {limit}) ---")
        for reason, count in fail_counts.items():
            if count > 0:
                print(f"  {reason}: {count}")
        print("---")

    results = [results_map[idx] for idx in sorted(results_map)]
    if not results:
        print(f"\nNo samples processed. Nothing to save.")
        print(f"  Total attempted: {limit}")
        print(f"  Processed: {processed}")
        print(f"  Skipped: {skipped}")
        print(f"\nTroubleshooting:")
        if fail_counts["empty_instruction"] > 0:
            print(f"  - {fail_counts['empty_instruction']} rows had empty 'instruction' field. "
                  f"Check your rollout JSONL has an 'instruction' key.")
        if fail_counts["empty_response"] > 0:
            print(f"  - {fail_counts['empty_response']} rows had empty responses. "
                  f"Check your rollout JSONL has a 'responses' key with non-empty content.")
        if fail_counts["empty_reply"] > 0:
            print(f"  - {fail_counts['empty_reply']} LLM calls returned empty replies. "
                  f"Check your API endpoint/config (--base_url or --api_config).")
        if fail_counts["json_parse_fail"] > 0:
            print(f"  - {fail_counts['json_parse_fail']} LLM replies could not be parsed as JSON. "
                  f"The model may not be following the output format.")
        if fail_counts["validation_fail"] > 0:
            print(f"  - {fail_counts['validation_fail']} parsed JSONs failed validation. "
                  f"Expected keys: {QUALITY_DIMS}")
        if fail_counts["exception"] > 0:
            print(f"  - {fail_counts['exception']} rows raised exceptions (see tracebacks above).")
        if fail_counts["all_none"] > 0:
            print(f"  - {fail_counts['all_none']} rows had no valid instruction+response pair.")
        return

    with open(args.output, "w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nDone! Analyzed {processed} samples, skipped {skipped}. Output: {args.output}")


if __name__ == "__main__":
    main()
