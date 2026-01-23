#!/usr/bin/env python3
"""Check prompt constraints for self-contradiction using a vLLM OpenAI-compatible API."""

from __future__ import annotations

import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Tuple

from datasets import Dataset, load_dataset, load_from_disk
from tqdm import tqdm

from chat import GENERATION_CONFIGS, LocalChat

LABEL_RE = re.compile(r"label\s*:\s*(contradictory|not contradictory)", re.IGNORECASE)
JUSTIFICATION_RE = re.compile(r"justification\s*:\s*(.*)", re.IGNORECASE)


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


def build_user_prompt(template: str, raw_prompt: str) -> str:
    return template.replace("{target_prompt}", raw_prompt)


def parse_response(text: str) -> Tuple[Optional[str], str]:
    label = None
    justification = None
    for line in text.splitlines():
        if label is None:
            match = LABEL_RE.search(line)
            if match:
                value = match.group(1).strip().lower()
                label = "Contradictory" if value.startswith("contradictory") else "Not Contradictory"
        if justification is None:
            match = JUSTIFICATION_RE.search(line)
            if match:
                justification = match.group(1).strip()

    if label is None:
        lowered = text.lower()
        if "not contradictory" in lowered:
            label = "Not Contradictory"
        elif "contradictory" in lowered:
            label = "Contradictory"

    if not justification:
        justification = text.strip()

    return label, justification


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check for self-contradictory constraints using a vLLM model."
    )
    parser.add_argument(
        "--input_dataset",
        required=True,
        help="Dataset name, path, or JSON/JSONL file.",
    )
    parser.add_argument(
        "--save_to_disk",
        required=True,
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
        default="prompt/contradiction_check.txt",
        help="Path to the system prompt file.",
    )
    parser.add_argument(
        "--user_prompt_path",
        default="prompt/constradiction_check_user.txt",
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
    args = parser.parse_args()

    system_prompt = load_prompt(args.system_prompt_path)
    reference_constraints = load_prompt(args.reference_constraints_path)
    system_prompt = system_prompt.replace("{reference_constraints}", reference_constraints)
    user_prompt_template = load_prompt(args.user_prompt_path)
    if "{target_prompt}" not in user_prompt_template:
        raise ValueError("User prompt template must contain {target_prompt}.")

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

    dataset = build_dataset(args.input_dataset, args.split, args.streaming)
    dataset_name = os.path.basename(args.input_dataset).replace("/", "_")

    processed = 0
    skipped = 0
    submitted = 0
    total = None if args.streaming else len(dataset)

    results_map: dict[int, dict] = {}
    iterator = enumerate(dataset)

    def build_result(index: int, example: dict) -> Optional[dict]:
        raw_prompt = get_original_instruction(
            example,
            args.instruction_field,
        )
        if not raw_prompt:
            return None
        user_prompt = build_user_prompt(user_prompt_template, raw_prompt)
        if args.no_system:
            messages = [{"role": "user", "content": f"{system_prompt}\n{user_prompt}"}]
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        reply, _ = chat.ask(messages)
        if not reply:
            return None
        label, justification = parse_response(reply)
        return {
            "key": get_example_key(example, index, dataset_name),
            "raw_prompt": raw_prompt,
            "contradiction_label": label,
            "justification": justification,
            "raw_response": reply.strip(),
            "dataset": dataset_name,
            "model": args.model,
        }

    max_inflight = max(1, args.max_inflight)
    limit = args.max_samples if args.max_samples is not None else total
    pbar = tqdm(total=limit, unit="sample", desc="Contradiction check")

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

    print(f"Done! Labeled {processed} samples, skipped {skipped}.")


if __name__ == "__main__":
    main()
