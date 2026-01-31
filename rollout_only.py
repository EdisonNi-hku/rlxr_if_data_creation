#!/usr/bin/env python3
"""
Generate multiple rollouts for instructions without grading.

This script takes a dataset with instructions, generates multiple rollout
responses per prompt using vLLM, and saves the raw outputs.

Example usage:
    python rollout_only.py \
        --input_dataset data.jsonl \
        --output_path rollouts.jsonl \
        --model "Qwen/Qwen3-8B" \
        --num_rollouts 16
"""

from __future__ import annotations

import argparse
import json
import os

from datasets import load_dataset, load_from_disk
from tqdm import tqdm
from vllm import LLM, SamplingParams


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


def get_instruction_from_messages(example: dict, instruction_field: str) -> str | None:
    """Extract the instruction/prompt from the example."""
    value = example.get(instruction_field)
    if value is None:
        return None

    if isinstance(value, str):
        return value

    if isinstance(value, list):
        for msg in value:
            if isinstance(msg, dict):
                role = msg.get("role", msg.get("from", ""))
                content = msg.get("content", msg.get("value", ""))
                if role in ("user", "human") and content:
                    return content

    return None


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

    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        if system_prompt:
            return f"{system_prompt}\n\nUser: {instruction}\nAssistant:"
        return f"User: {instruction}\nAssistant:"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate rollouts using vLLM (no grading)."
    )

    # Dataset arguments
    parser.add_argument(
        "--input_dataset",
        required=True,
        help="Dataset name, path, or JSON/JSONL file with instructions.",
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
        required=True,
        help="Path to save output JSONL file.",
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

    # Processing arguments
    parser.add_argument(
        "--strip_thinking",
        action="store_true",
        help="Strip thinking tokens (</think>) from responses.",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.partition_num < 1:
        parser.error("--partition_num must be >= 1")
    if args.partition_index < 0 or args.partition_index >= args.partition_num:
        parser.error(f"--partition_index must be in range [0, {args.partition_num - 1}]")

    # Add partition suffix to output path if partitioning is enabled
    if args.partition_num > 1:
        base, ext = os.path.splitext(args.output_path)
        args.output_path = f"{base}_p{args.partition_index}_of_{args.partition_num}{ext}"
        print(f"[PARTITION] Output path: {args.output_path}")

    # Check for existing output (resume support)
    if os.path.exists(args.output_path):
        print(f"Output file already exists: {args.output_path}")
        print("Delete it to regenerate, or use a different output path.")
        return

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

    # Create sampling params
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

    # Load dataset
    dataset = build_dataset(args.input_dataset, args.split, args.streaming)
    dataset_name = os.path.basename(args.input_dataset).replace("/", "_")

    # Apply partitioning if specified
    if args.partition_num > 1:
        if args.streaming:
            raise ValueError("Partitioning is not supported with streaming mode.")
        total_size = len(dataset)
        indices = list(range(args.partition_index, total_size, args.partition_num))
        dataset = dataset.select(indices)
        print(f"[PARTITION] Processing partition {args.partition_index} of {args.partition_num}: "
              f"{len(indices)} samples (round-robin from {total_size} total)")

    # Prepare examples
    examples_to_process = []
    skipped = 0

    print("Preparing examples...")
    for index, example in enumerate(tqdm(dataset, desc="Preparing")):
        if index < args.start_index:
            continue
        if args.max_samples is not None and len(examples_to_process) >= args.max_samples:
            break

        instruction = get_instruction_from_messages(example, args.instruction_field)

        if not instruction:
            skipped += 1
            continue

        key = get_example_key(example, index, dataset_name)

        examples_to_process.append({
            "key": key,
            "index": index,
            "example": example,
            "instruction": instruction,
        })

    print(f"Prepared {len(examples_to_process)} examples, skipped {skipped}")

    if not examples_to_process:
        print("No samples to process. Exiting.")
        return

    # Build prompts
    print("Building prompts...")
    prompts = []
    for ex in examples_to_process:
        prompt = build_chat_prompt(tokenizer, ex["instruction"], system_prompt)
        prompts.append(prompt)

    # Generate with vLLM
    print(f"Generating rollouts for {len(prompts)} prompts...")
    outputs = llm.generate(prompts, sampling_params)

    # Save outputs
    print(f"Saving outputs to: {args.output_path}")
    with open(args.output_path, "w", encoding="utf-8") as f:
        for ex, output in zip(examples_to_process, outputs):
            responses = [out.text for out in output.outputs]

            # Optionally strip thinking tokens
            if args.strip_thinking:
                clean_responses = []
                for resp in responses:
                    if '</think>' in resp:
                        resp = resp.split('</think>')[-1].strip()
                    clean_responses.append(resp)
                responses = clean_responses

            result = {
                "key": ex["key"],
                "instruction": ex["instruction"],
                "responses": responses,
            }
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"\nDone! Generated {len(outputs)} x {args.num_rollouts} = {len(outputs) * args.num_rollouts} rollouts.")
    print(f"Output saved to: {args.output_path}")


if __name__ == "__main__":
    main()
