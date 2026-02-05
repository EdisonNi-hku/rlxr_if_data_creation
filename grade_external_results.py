#!/usr/bin/env python3
"""
Grade external model outputs and create a HuggingFace dataset with pass-rate results.

Supports two input formats:
  - "external" (default): flat JSONL with input/output/gts per row (e.g. from RLHF training)
      {"input": "user\\n<msg>\\nassistant\\n", "output": "...", "gts": "[...]", ...}
  - "rollout": JSONL from rollout_only.py with grouped responses per prompt
      {"key": "...", "instruction": "...", "responses": ["r1", "r2", ...]}
    Requires --dataset to supply ground truth.

Example usage:
    # External format (self-contained ground truth)
    python grade_external_results.py \\
        --input results/100.jsonl \\
        --column_name gdpo_100_pass_rate

    # Rollout format (ground truth from dataset)
    python grade_external_results.py \\
        --input rollouts.jsonl \\
        --input_format rollout \\
        --dataset JingweiNi/magpie_creative_dedup_verifiable_test_1_5 \\
        --column_name qwen3_pass_rate

    # With reward model grading
    python grade_external_results.py \\
        --input rollouts.jsonl \\
        --input_format rollout \\
        --dataset JingweiNi/magpie_creative_dedup_verifiable_test_1_5 \\
        --column_name qwen3_pass_rate \\
        --reward_model Skywork/Skywork-Reward-V2-Llama-3.1-8B
"""

from __future__ import annotations

import argparse
import json
import os
from collections import OrderedDict

from datasets import Dataset, load_dataset, load_from_disk
from tqdm import tqdm

from verify_constraints import verify_multiple_constraints


def extract_user_message(input_field: str) -> str:
    """Extract the user message content from the input field.

    The input field format is:
        user\\n<message>\\nassistant\\n
    """
    text = input_field.strip()
    if text.startswith("user\n"):
        text = text[len("user\n"):]
    if text.endswith("\nassistant\n"):
        text = text[: -len("\nassistant\n")]
    elif text.endswith("\nassistant"):
        text = text[: -len("\nassistant")]
    return text.strip()


def parse_ground_truth(gts) -> list[dict] | None:
    """Parse ground truth from various formats."""
    if gts is None:
        return None
    if isinstance(gts, str):
        try:
            return json.loads(gts)
        except json.JSONDecodeError:
            return None
    return gts


def build_dataset(dataset_path: str, split: str):
    """Load dataset from various sources."""
    if os.path.isfile(dataset_path):
        return load_dataset("json", data_files=dataset_path, split=split)
    if os.path.isdir(dataset_path):
        dataset = load_from_disk(dataset_path)
        if hasattr(dataset, "keys"):
            return dataset[split]
        return dataset
    try:
        return load_dataset(dataset_path, split=split)
    except Exception:
        dataset = load_dataset(dataset_path)
        if isinstance(dataset, dict):
            if split in dataset:
                return dataset[split]
            return dataset[next(iter(dataset.keys()))]
        return dataset


def load_external_format(input_path: str) -> OrderedDict[str, dict]:
    """Load the flat external format (input/output/gts per row), grouped by prompt."""
    prompt_groups: OrderedDict[str, dict] = OrderedDict()
    with open(input_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading"):
            if not line.strip():
                continue
            row = json.loads(line)
            user_msg = extract_user_message(row["input"])
            if user_msg not in prompt_groups:
                prompt_groups[user_msg] = {
                    "user_message": user_msg,
                    "gts": row["gts"],
                    "outputs": [],
                }
            prompt_groups[user_msg]["outputs"].append(row["output"])
    return prompt_groups


def load_rollout_format(
    input_path: str,
    dataset_path: str,
    split: str,
) -> OrderedDict[str, dict]:
    """Load rollout_only.py format and look up ground truth from a dataset."""
    # Load ground truth from dataset, keyed by both 'key' and instruction text
    print(f"Loading ground truth dataset: {dataset_path}")
    dataset = build_dataset(dataset_path, split)
    dataset_name = os.path.basename(dataset_path).replace("/", "_")

    key_to_gt: dict[str, str] = {}
    instruction_to_gt: dict[str, str] = {}
    for idx, example in enumerate(dataset):
        gt = example.get("ground_truth_verifiable") or example.get("ground_truth")
        if gt is None:
            continue
        # Index by key
        for field in ("id", "uuid", "key", "idx", "index"):
            if field in example and example[field]:
                key_to_gt[str(example[field])] = gt
                break
        else:
            key_to_gt[f"{dataset_name}_{idx}"] = gt
        # Also index by instruction text for fallback matching
        messages = example.get("messages")
        if messages:
            for msg in messages:
                if msg.get("role") == "user":
                    instruction_to_gt[msg["content"].strip()] = gt
                    break

    print(f"  Indexed {len(key_to_gt)} keys, {len(instruction_to_gt)} instructions.")

    # Load rollout JSONL
    prompt_groups: OrderedDict[str, dict] = OrderedDict()
    unmatched = 0
    with open(input_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading"):
            if not line.strip():
                continue
            row = json.loads(line)
            key = row.get("key", "")
            instruction = row.get("instruction", "")
            responses = row.get("responses", [])

            # Look up ground truth: try key first, then instruction text
            gt = key_to_gt.get(key) or instruction_to_gt.get(instruction.strip())
            if gt is None:
                unmatched += 1
                continue

            if instruction not in prompt_groups:
                prompt_groups[instruction] = {
                    "user_message": instruction,
                    "gts": gt,
                    "outputs": [],
                }
            prompt_groups[instruction]["outputs"].extend(responses)

    if unmatched > 0:
        print(f"  WARNING: {unmatched} rollout rows had no matching ground truth.")

    return prompt_groups


def score_with_reward_model(
    prompt_groups: OrderedDict[str, dict],
    model_name: str,
    batch_size: int,
    device: str,
) -> dict[str, list[float]]:
    """Score all (prompt, response) pairs with a reward model.

    Returns a dict mapping user_message -> list of reward scores (one per rollout).
    """
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    def parse_reward_devices(device_arg: str) -> tuple[str, list[int]]:
        if device_arg == "auto":
            return "auto", []
        parts = [p.strip() for p in device_arg.split(",") if p.strip()]
        if len(parts) <= 1:
            return device_arg, []
        device_ids: list[int] = []
        for p in parts:
            if not p.startswith("cuda:"):
                raise ValueError(
                    f"Invalid reward_device entry '{p}'. Use 'cuda:N' or 'auto'."
                )
            device_ids.append(int(p.split("cuda:")[1]))
        return "data_parallel", device_ids

    device_mode, device_ids = parse_reward_devices(device)

    print(f"\nLoading reward model: {model_name}")
    # tp_plan=None avoids OSError when device_map="auto" would trigger tensor
    # parallelism (which requires torch.distributed to be initialized).
    load_kwargs = dict(
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        num_labels=1,
        tp_plan=None,
    )
    auto_fallback = False
    if device_mode == "auto":
        load_kwargs["device_map"] = "auto"
        try:
            rm = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                **load_kwargs,
            )
            input_device = next(rm.parameters()).device
        except OSError as exc:
            if "torch.distributed" not in str(exc):
                raise
            print(
                "  WARNING: device_map='auto' requires torch.distributed for tensor "
                "parallelism. Falling back to data parallel on available GPUs."
            )
            auto_fallback = True

    if device_mode != "auto" or auto_fallback:
        load_kwargs["device_map"] = None
        rm = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            **load_kwargs,
        )
        if device_mode == "auto":
            if torch.cuda.is_available():
                device_ids = list(range(torch.cuda.device_count()))
                if len(device_ids) >= 2:
                    device_mode = "data_parallel"
                else:
                    device_mode = "single"
            else:
                device_mode = "single"
                device = "cpu"
        if device_mode == "data_parallel":
            if not device_ids:
                raise ValueError("No valid CUDA devices provided for data parallel.")
            primary_device = f"cuda:{device_ids[0]}"
            rm.to(primary_device)
            rm = torch.nn.DataParallel(rm, device_ids=device_ids)
            input_device = torch.device(primary_device)
        else:
            if device_mode == "single" and device == "auto":
                device = "cuda:0" if torch.cuda.is_available() else "cpu"
            rm.to(device)
            input_device = torch.device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    rm.eval()

    # Reward models typically need left padding for batched inference
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Flatten all (prompt, response) pairs while tracking which group they belong to
    all_convs = []
    group_keys = []
    for user_msg, group in prompt_groups.items():
        for output in group["outputs"]:
            conv = [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": output},
            ]
            formatted = tokenizer.apply_chat_template(conv, tokenize=False)
            # Remove potential duplicate bos token
            if tokenizer.bos_token is not None and formatted.startswith(
                tokenizer.bos_token
            ):
                formatted = formatted[len(tokenizer.bos_token) :]
            all_convs.append(formatted)
            group_keys.append(user_msg)

    print(f"  Scoring {len(all_convs)} responses in batches of {batch_size}...")

    # Score in batches
    all_scores = []
    for i in tqdm(range(0, len(all_convs), batch_size), desc="Reward model"):
        batch_texts = all_convs[i : i + batch_size]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=tokenizer.model_max_length,
        ).to(input_device)
        with torch.no_grad():
            logits = rm(**inputs).logits
        batch_scores = logits[:, 0].float().cpu().tolist()
        all_scores.extend(batch_scores)

    # Restore original padding side
    tokenizer.padding_side = original_padding_side

    # Group scores back by user message
    reward_scores: dict[str, list[float]] = {}
    for key, score in zip(group_keys, all_scores):
        if key not in reward_scores:
            reward_scores[key] = []
        reward_scores[key].append(score)

    # Free GPU memory
    del rm
    torch.cuda.empty_cache()

    return reward_scores


def main():
    parser = argparse.ArgumentParser(
        description="Grade external model outputs and create a HuggingFace dataset with pass-rate results."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the input JSONL file with model outputs.",
    )
    parser.add_argument(
        "--input_format",
        choices=["external", "rollout"],
        default="external",
        help="Input format: 'external' (input/output/gts per row) or 'rollout' (key/instruction/responses from rollout_only.py). Default: external.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="HuggingFace dataset for ground truth lookup (required for --input_format rollout).",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to use (default: train).",
    )
    parser.add_argument(
        "--column_name",
        required=True,
        help="Name of the new column for pass rate of constraints.",
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
        "--reward_model",
        type=str,
        default=None,
        help="Reward model name to score responses (e.g. Skywork/Skywork-Reward-V2-Llama-3.1-8B).",
    )
    parser.add_argument(
        "--reward_batch_size",
        type=int,
        default=8,
        help="Batch size for reward model inference (default: 8).",
    )
    parser.add_argument(
        "--reward_device",
        type=str,
        default="cuda:0",
        help=(
            "Device(s) for reward model. Use a single device like 'cuda:0', "
            "'cpu', 'auto' to shard across GPUs, or a comma-separated list "
            "like 'cuda:0,cuda:1' for data parallel (default: cuda:0)."
        ),
    )
    parser.add_argument(
        "--push_to_hub",
        type=str,
        default=None,
        help="Push the resulting dataset to HuggingFace Hub under this name (e.g. JingweiNi/my_dataset).",
    )
    parser.add_argument(
        "--save_local",
        type=str,
        default=None,
        help="Local path to save the dataset (default: <input_basename>_graded).",
    )
    parser.add_argument(
        "--save_graded_jsonl",
        type=str,
        default=None,
        help="Optional path to save detailed graded results as JSONL.",
    )

    args = parser.parse_args()

    if args.no_strip_thinking:
        args.strip_thinking = False

    if args.input_format == "rollout" and not args.dataset:
        parser.error("--dataset is required when using --input_format rollout")

    # ----------------------------------------------------------------
    # 1. Load model outputs and group by prompt
    # ----------------------------------------------------------------
    print(f"Loading model outputs from: {args.input} (format: {args.input_format})")

    if args.input_format == "external":
        prompt_groups = load_external_format(args.input)
    else:
        prompt_groups = load_rollout_format(args.input, args.dataset, args.split)

    total_lines = sum(len(g["outputs"]) for g in prompt_groups.values())
    print(f"  Loaded {total_lines} responses, {len(prompt_groups)} unique prompts.")

    # ----------------------------------------------------------------
    # 2. Reward model scoring (if enabled)
    # ----------------------------------------------------------------
    reward_scores_by_prompt: dict[str, list[float]] | None = None
    if args.reward_model:
        reward_scores_by_prompt = score_with_reward_model(
            prompt_groups,
            model_name=args.reward_model,
            batch_size=args.reward_batch_size,
            device=args.reward_device,
        )

    # ----------------------------------------------------------------
    # 3. Grade each prompt's rollouts (constraint verification)
    # ----------------------------------------------------------------
    dataset_rows = []
    total_constraints = 0
    total_constraints_passed = 0
    total_rollouts = 0
    total_rollouts_passed = 0
    skipped = 0

    for user_msg, group in tqdm(prompt_groups.items(), desc="Grading"):
        ground_truth = parse_ground_truth(group["gts"])
        if ground_truth is None:
            skipped += 1
            continue

        outputs = group["outputs"]
        scores = []
        num_passed = 0

        for output in outputs:
            response = output
            if args.strip_thinking and "</think>" in response:
                response = response.split("</think>")[-1].strip()

            if not response:
                scores.append(0.0)
                continue

            passed, individual_results = verify_multiple_constraints(
                response, ground_truth
            )
            score = (
                sum(individual_results) / len(individual_results)
                if individual_results
                else 0.0
            )
            scores.append(score)

            if passed:
                num_passed += 1

            total_constraints += len(individual_results)
            total_constraints_passed += sum(individual_results)

        n = len(outputs)
        total_rollouts += n
        total_rollouts_passed += num_passed
        pass_rate = num_passed / n if n > 0 else 0.0
        avg_score = sum(scores) / len(scores) if scores else 0.0

        row_data = {
            "messages": [{"role": "user", "content": user_msg}],
            "ground_truth": json.dumps(ground_truth, ensure_ascii=False)
            if not isinstance(group["gts"], str)
            else group["gts"],
            "num_rollouts": n,
            "scores": scores,
            "avg_score": avg_score,
            args.column_name: pass_rate,
        }

        # Attach reward model scores if available
        if reward_scores_by_prompt is not None:
            rm_scores = reward_scores_by_prompt.get(user_msg, [])
            row_data["reward_scores"] = rm_scores
            row_data["avg_reward"] = (
                sum(rm_scores) / len(rm_scores) if rm_scores else 0.0
            )

        dataset_rows.append(row_data)

    if skipped > 0:
        print(f"  WARNING: Skipped {skipped} prompts with unparseable ground truth.")

    # ----------------------------------------------------------------
    # 4. Report metrics
    # ----------------------------------------------------------------
    print()
    print("=" * 60)
    print("GRADING RESULTS")
    print("=" * 60)
    print(f"  Prompts graded: {len(dataset_rows)}")
    print(f"  Total rollouts: {total_rollouts}")
    print(
        f"  Overall pass rate (hard, rollout-level): "
        f"{total_rollouts_passed / total_rollouts:.4f}"
        if total_rollouts > 0
        else "  N/A"
    )
    if total_constraints > 0:
        print(
            f"  Constraint accuracy (soft): "
            f"{total_constraints_passed / total_constraints:.4f}"
        )
    all_pass_rates = [r[args.column_name] for r in dataset_rows]
    if all_pass_rates:
        print(
            f"  Average pass rate per prompt: "
            f"{sum(all_pass_rates) / len(all_pass_rates):.4f}"
        )
    all_avg_scores = [r["avg_score"] for r in dataset_rows]
    if all_avg_scores:
        print(
            f"  Average constraint score per prompt: "
            f"{sum(all_avg_scores) / len(all_avg_scores):.4f}"
        )

    if reward_scores_by_prompt is not None:
        all_avg_rewards = [r["avg_reward"] for r in dataset_rows]
        if all_avg_rewards:
            print(
                f"  Average reward per prompt: "
                f"{sum(all_avg_rewards) / len(all_avg_rewards):.4f}"
            )
            all_flat = [s for r in dataset_rows for s in r.get("reward_scores", [])]
            if all_flat:
                print(f"  Average reward per rollout: {sum(all_flat) / len(all_flat):.4f}")

    print("=" * 60)

    # ----------------------------------------------------------------
    # 5. Create HuggingFace Dataset and save
    # ----------------------------------------------------------------
    hf_dataset = Dataset.from_list(dataset_rows)
    print(f"\nCreated dataset with {len(hf_dataset)} rows.")
    print(f"  Columns: {hf_dataset.column_names}")

    # Save locally
    save_local = args.save_local
    if save_local is None:
        base = os.path.splitext(os.path.basename(args.input))[0]
        save_local = f"{base}_graded"

    hf_dataset.save_to_disk(save_local)
    print(f"  Dataset saved locally to: {save_local}")

    # Push to hub
    if args.push_to_hub:
        print(f"  Pushing dataset to HuggingFace Hub: {args.push_to_hub}")
        hf_dataset.push_to_hub(args.push_to_hub)
        print(
            f"  Done. Dataset available at: "
            f"https://huggingface.co/datasets/{args.push_to_hub}"
        )

    # Save detailed graded JSONL
    if args.save_graded_jsonl:
        with open(args.save_graded_jsonl, "w", encoding="utf-8") as f:
            for row in dataset_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"  Detailed graded results saved to: {args.save_graded_jsonl}")


if __name__ == "__main__":
    main()
