#!/usr/bin/env python3
"""
Data format conversion script
Convert HuggingFace datasets to training format with prompt/reward_model structure
"""

from __future__ import annotations

import argparse
import json
import os

import pandas as pd
from datasets import Dataset, load_dataset, load_from_disk


def load_dataset_auto(path: str, split: str = "train"):
    """Load a dataset from HuggingFace Hub or local disk."""
    if os.path.isdir(path):
        dataset = load_from_disk(path)
        if hasattr(dataset, "keys"):
            return dataset[split] if split in dataset else dataset[list(dataset.keys())[0]]
        return dataset
    return load_dataset(path, split=split)


def convert_dataset(
    dataset,
    data_source_name: str,
    ground_truth_field: str = "ground_truth_verifiable",
    messages_field: str = "messages_verifiable",
) -> list[dict]:
    """Convert dataset to training format.
    
    Args:
        dataset: HuggingFace dataset to convert
        data_source_name: Name to use for data_source field
        ground_truth_field: Field name containing ground truth
        messages_field: Field name containing messages
        
    Returns:
        List of converted examples
    """
    print(f'=== Converting dataset ===')
    print(f'Dataset size: {len(dataset)}')
    print(f'Columns: {dataset.column_names}')
    print()
    
    # Show first example
    if len(dataset) > 0:
        print('First example preview:')
        for col in dataset.column_names[:10]:  # Show first 10 columns
            val = dataset[0][col]
            print(f'  {col}: {type(val).__name__}')
        print()
    
    print('=== Starting conversion ===')
    
    converted_data = []
    
    for idx, example in enumerate(dataset):
        # 1. prompt: huggingface chat_template format
        if messages_field in example:
            prompt = example[messages_field]
        elif 'messages' in example:
            prompt = example['messages']
        elif 'conversations' in example:
            prompt = example['conversations']
        elif 'instruction' in example or 'input' in example:
            # Convert instruction format to chat format
            messages = []
            if 'system' in example and example.get('system'):
                messages.append({'role': 'system', 'content': example['system']})
            
            instruction = example.get('instruction', '')
            input_text = example.get('input', '')
            user_content = f"{instruction}\n{input_text}".strip() if input_text else instruction
            messages.append({'role': 'user', 'content': user_content})
            prompt = messages
        else:
            prompt = []
        
        # 2. data_source: dataset name for reward function indexing
        if 'dataset' in example:
            data_source = example['dataset']
        else:
            data_source = data_source_name
        
        # 3. ability: task category
        if 'ability' in example:
            ability = example['ability']
        elif 'creative' in str(data_source).lower():
            ability = 'creative_writing'
        else:
            ability = 'general'
        
        # 4. reward_model: contains ground_truth and other info
        reward_model_dict = {}
        if ground_truth_field in example and example.get(ground_truth_field):
            reward_model_dict['ground_truth'] = example[ground_truth_field]
        elif 'ground_truth' in example and example.get('ground_truth'):
            reward_model_dict['ground_truth'] = example['ground_truth']
        else:
            reward_model_dict['ground_truth'] = 'no_answer'
        
        # Use 'rule' as style
        reward_model_dict['style'] = 'rule'
        
        # 5. extra_info: preserve other information
        extra_info = {}
        excluded_fields = {
            messages_field, 'messages', 'conversations', 'response', 'output', 'answer',
            'label', 'dataset', 'ability', ground_truth_field, 'ground_truth',
            'instruction', 'input', 'system'
        }
        
        for col in example.keys():
            if col not in excluded_fields:
                val = example[col]
                if val is not None:
                    extra_info[col] = val
        
        # Always preserve key field
        if 'key' in example:
            extra_info['key'] = example['key']
        
        converted_row = {
            'prompt': prompt,
            'data_source': data_source,
            'ability': ability,
            'reward_model': reward_model_dict,
            'extra_info': extra_info
        }
        
        converted_data.append(converted_row)
    
    print(f'Conversion complete! Converted {len(converted_data)} examples')
    return converted_data


def main():
    parser = argparse.ArgumentParser(
        description="Convert HuggingFace dataset to training format."
    )
    parser.add_argument(
        "--input_dataset",
        required=True,
        help="HuggingFace dataset name or local path (e.g., 'JingweiNi/magpie_creative_dedup_verifiable_train_1_5')",
    )
    parser.add_argument(
        "--output_path",
        default=None,
        help="Path to save output parquet file",
    )
    parser.add_argument(
        "--save_to_disk",
        default=None,
        help="Path to save as HuggingFace dataset locally",
    )
    parser.add_argument(
        "--push_to_hub",
        default=None,
        help="HuggingFace Hub repo to push the converted dataset",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to use (default: train)",
    )
    parser.add_argument(
        "--data_source_name",
        default=None,
        help="Name to use for data_source field (default: input dataset name)",
    )
    parser.add_argument(
        "--ground_truth_field",
        default="ground_truth_verifiable",
        help="Field name containing ground truth (default: ground_truth_verifiable)",
    )
    parser.add_argument(
        "--messages_field",
        default="messages_verifiable",
        help="Field name containing messages (default: messages_verifiable)",
    )
    
    args = parser.parse_args()
    
    if not args.output_path and not args.save_to_disk and not args.push_to_hub:
        parser.error("At least one of --output_path, --save_to_disk, or --push_to_hub must be specified")
    
    # Load dataset
    print(f"[LOAD] Loading dataset: {args.input_dataset}")
    dataset = load_dataset_auto(args.input_dataset, args.split)
    print(f"[LOAD] Loaded {len(dataset)} examples")
    
    # Determine data source name
    data_source_name = args.data_source_name or args.input_dataset.replace("/", "_")
    
    # Convert dataset
    converted_data = convert_dataset(
        dataset,
        data_source_name,
        args.ground_truth_field,
        args.messages_field,
    )
    
    # Save output
    if args.output_path:
        print(f"\n[SAVE] Saving to parquet: {args.output_path}")
        df_converted = pd.DataFrame(converted_data)
        df_converted.to_parquet(args.output_path, index=False)
        print(f"[SAVE] Saved {len(df_converted)} examples")
        
        # Show preview
        print('\nConverted data preview (first 3 rows):')
        print(df_converted.head(3))
    
    if args.save_to_disk:
        print(f"\n[SAVE] Saving as HuggingFace dataset: {args.save_to_disk}")
        hf_dataset = Dataset.from_list(converted_data)
        hf_dataset.save_to_disk(args.save_to_disk)
        print("[SAVE] Done!")
    
    if args.push_to_hub:
        print(f"\n[PUSH] Pushing to HuggingFace Hub: {args.push_to_hub}")
        hf_dataset = Dataset.from_list(converted_data)
        hf_dataset.push_to_hub(args.push_to_hub)
        print("[PUSH] Done!")
    
    print("\n[SUCCESS] Conversion complete!")

if __name__ == '__main__':
    main()
