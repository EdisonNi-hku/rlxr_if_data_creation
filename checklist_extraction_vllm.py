#!/usr/bin/env python3
"""Extract checklist constraints from instructions using a vLLM OpenAI-compatible API."""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from datasets import Dataset, load_dataset, load_from_disk
from tqdm import tqdm

from chat import GENERATION_CONFIGS, LocalChat


def load_prompt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def build_dataset(
    input_dataset: str,
    split: str,
    streaming: bool,
):
    if os.path.isfile(input_dataset):
        return load_dataset(
            "json",
            data_files=input_dataset,
            split=split,
            streaming=streaming,
        )
    if os.path.isdir(input_dataset):
        return load_from_disk(input_dataset)
    try:
        return load_dataset(input_dataset, split=split, streaming=streaming)
    except Exception:
        dataset = load_dataset(input_dataset, streaming=streaming)
        if isinstance(dataset, dict):
            if split in dataset:
                return dataset[split]
            return dataset[next(iter(dataset.keys()))]
        return dataset


def get_original_instruction(example: dict, instruction_field: str) -> Optional[str]:
    if instruction_field and example.get(instruction_field):
        value = example[instruction_field]
        return value if isinstance(value, str) else str(value)

    if "messages" in example and isinstance(example["messages"], list):
        for msg in example["messages"]:
            if isinstance(msg, dict):
                role = msg.get("role", msg.get("from", ""))
                content = msg.get("content", msg.get("value", ""))
                if role in ("user", "human") and content:
                    return content

    if "conversations" in example and isinstance(example["conversations"], list):
        for conv in example["conversations"]:
            if isinstance(conv, dict):
                role = conv.get("from", conv.get("role", ""))
                content = conv.get("value", conv.get("content", ""))
                if role in ("user", "human") and content:
                    return content

    for field in ["prompt", "instruction", "text", "input", "question"]:
        if field in example and example[field]:
            value = example[field]
            return value if isinstance(value, str) else str(value)

    return None


def get_example_key(example: dict, index: int, dataset_name: str) -> str:
    for field in ["id", "uuid", "key", "idx", "index"]:
        if field in example and example[field]:
            return f"{dataset_name}_{example[field]}"
    return f"{dataset_name}_{index}"


def build_user_prompt(template: str, raw_instruction: str) -> str:
    return template.replace("{instruction}", raw_instruction)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract checklist constraints using a vLLM model."
    )
    parser.add_argument(
        "--input_dataset",
        required=True,
        help="Dataset name, path, or JSON/JSONL file.",
    )
    parser.add_argument(
        "--save_to_disk",
        default=None,
        help="Path to save a Hugging Face dataset with save_to_disk.",
    )
    parser.add_argument(
        "--push_to_hub",
        default=None,
        help="Optional Hub repo_id to push after saving to disk.",
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
        default="prompt/checklist_extraction.txt",
        help="Path to the system prompt file.",
    )
    parser.add_argument(
        "--user_prompt_path",
        default="prompt/checklist_extraction_user.txt",
        help="Path to the user prompt template file.",
    )
    parser.add_argument(
        "--reference_constraints_path",
        default="prompt/reference_constraints_v2.txt",
        help="Path to the reference constraints file.",
    )
    parser.add_argument(
        "--instruction_field",
        default="instruction",
        help="Primary field name for the raw prompt.",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to use (for HF datasets).",
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
        help="Optional maximum number of samples to process.",
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
        help=(
            "If set, prepend the system prompt to the user prompt and send as "
            "a single user message."
        ),
    )
    parser.add_argument(
        "--save_prompts_jsonl",
        default=None,
        help=(
            "If set, save all prompts to this JSONL file instead of calling vLLM. "
            "Each line will be {\"prompt\": [{\"role\": \"system\", ...}, {\"role\": \"user\", ...}]}."
        ),
    )
    parser.add_argument(
        "--partition_num",
        type=int,
        default=1,
        help="Total number of partitions (for distributed processing). Default: 1 (no partitioning).",
    )
    parser.add_argument(
        "--partition_index",
        type=int,
        default=0,
        help="Index of the current partition (0-based). Default: 0.",
    )
    args = parser.parse_args()

    # Validate partition arguments
    if args.partition_num < 1:
        parser.error("--partition_num must be >= 1")
    if args.partition_index < 0 or args.partition_index >= args.partition_num:
        parser.error(f"--partition_index must be in range [0, {args.partition_num - 1}]")

    system_prompt = load_prompt(args.system_prompt_path)
    reference_constraints = load_prompt(args.reference_constraints_path)
    system_prompt = system_prompt.replace("{reference_constraints}", reference_constraints)
    user_prompt_template = load_prompt(args.user_prompt_path)
    if "{instruction}" not in user_prompt_template:
        raise ValueError("User prompt template must contain {instruction}.")

    generation_config = None
    if args.generation_config:
        generation_config = json.loads(args.generation_config)
    elif args.model not in GENERATION_CONFIGS:
        generation_config = {}

    # Only initialize chat client if we're not just saving prompts
    chat = None
    if not args.save_prompts_jsonl:
        chat = LocalChat(
            model=args.model,
            base_url=args.base_url,
            cache_path=args.cache_path,
            generation_config=generation_config,
        )

    dataset = build_dataset(args.input_dataset, args.split, args.streaming)
    dataset_name = os.path.basename(args.input_dataset).replace("/", "_")

    # Apply partitioning if specified (round-robin to distribute remainder evenly)
    if args.partition_num > 1:
        if args.streaming:
            raise ValueError("Partitioning is not supported with streaming mode.")
        total_size = len(dataset)
        # Round-robin: node i gets indices i, i+n, i+2n, ... where n = partition_num
        indices = list(range(args.partition_index, total_size, args.partition_num))
        dataset = dataset.select(indices)
        print(f"[PARTITION] Processing partition {args.partition_index} of {args.partition_num}: "
              f"{len(indices)} samples (round-robin from {total_size} total)")

    processed = 0
    skipped = 0
    submitted = 0
    total = None if args.streaming else len(dataset)

    results_map: dict[int, dict] = {}
    prompts_list: list[dict] = []
    iterator = enumerate(dataset)

    def build_messages(index: int, example: dict) -> Optional[tuple[str, list[dict]]]:
        """Build the messages for an example. Returns (raw_instruction, messages) or None."""
        raw_instruction = get_original_instruction(
            example,
            args.instruction_field,
        )
        if not raw_instruction:
            return None
        user_prompt = build_user_prompt(user_prompt_template, raw_instruction)
        if args.no_system:
            messages = [{"role": "user", "content": f"{system_prompt}\n{user_prompt}"}]
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        return raw_instruction, messages

    def build_result(index: int, example: dict) -> Optional[dict]:
        result = build_messages(index, example)
        if result is None:
            return None
        raw_instruction, messages = result
        
        reply, _ = chat.ask(messages)
        if not reply:
            return None

        # Debug: print the first prompt -> response pair for troubleshooting
        try:
            if index == args.start_index:
                print("--- DEBUG: first prompt/response ---")
                print("MESSAGES:\n" + json.dumps(messages, indent=2))
                print("MODEL REPLY:\n" + (reply.strip() if reply else "<empty>"))
                print("--- END DEBUG ---")
        except Exception:
            pass
        extracted = reply.strip()
        return {
            "key": get_example_key(example, index, dataset_name),
            "raw_instruction": raw_instruction,
            "checklist": extracted,
            "dataset": dataset_name,
            "model": args.model,
        }

    max_inflight = max(1, args.max_inflight)
    limit = args.max_samples if args.max_samples is not None else total

    # Mode 1: Save prompts only (no vLLM calls)
    if args.save_prompts_jsonl:
        pbar = tqdm(total=limit, unit="sample", desc="Building prompts")
        for index, example in iterator:
            if index < args.start_index:
                continue
            if args.max_samples is not None and submitted >= args.max_samples:
                break
            submitted += 1
            result = build_messages(index, example)
            if result is None:
                skipped += 1
            else:
                _, messages = result
                prompts_list.append({"prompt": messages})
                processed += 1
            pbar.update(1)
            pbar.set_postfix({"✓": processed, "✗": skipped})
        pbar.close()

        if not prompts_list:
            print("No prompts generated. Nothing to save.")
            return

        with open(args.save_prompts_jsonl, "w", encoding="utf-8") as f:
            for row in prompts_list:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Done! Saved {processed} prompts to {args.save_prompts_jsonl}, skipped {skipped}.")
        return

    # Mode 2: Call vLLM and save results
    pbar = tqdm(total=limit, unit="sample", desc="Checklist extraction")

    def handle_future(future, index: int):
        nonlocal processed, skipped
        try:
            result = future.result()
            if result is None:
                skipped += 1
            else:
                results_map[index] = result
                processed += 1
        except Exception as exc:
            skipped += 1
            tqdm.write(f"Error generating sample: {exc}")
        pbar.update(1)
        pbar.set_postfix({"✓": processed, "✗": skipped})

    with ThreadPoolExecutor(max_workers=max(1, args.num_workers)) as executor:
        futures: dict = {}
        for index, example in iterator:
            if index < args.start_index:
                continue
            if args.max_samples is not None and submitted >= args.max_samples:
                break
            future = executor.submit(build_result, index, example)
            futures[future] = index
            submitted += 1

            if len(futures) >= max_inflight:
                for future in as_completed(futures):
                    index = futures.pop(future)
                    handle_future(future, index)
                    if len(futures) < max_inflight:
                        break

        for future in as_completed(futures):
            index = futures.pop(future)
            handle_future(future, index)

    pbar.close()
    results = [results_map[idx] for idx in sorted(results_map)]

    if not results:
        print("No samples processed. Nothing to save.")
        return

    dataset_out = Dataset.from_list(results)
    dataset_out.save_to_disk(args.save_to_disk)
    if args.push_to_hub:
        dataset_out.push_to_hub(args.push_to_hub)

    print(f"Done! Extracted {processed} samples, skipped {skipped}.")


if __name__ == "__main__":
    main()
