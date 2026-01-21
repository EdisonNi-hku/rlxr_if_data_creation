# Verifiable Constraint Data Creation

This repository provides tools for adding verifiable constraints to instruction-following data. It uses the IFEvalG constraint library (from [AllenAI's open-instruct](https://github.com/allenai/open-instruct)) to augment existing instruction datasets with verifiable constraints.

## Overview

The goal is to create training data where:
1. Each instruction has one or more verifiable constraints added
2. Constraints can be programmatically verified (e.g., word count, specific keywords, formatting)
3. Output format matches [allenai/IF_multi_constraints_upto5](https://huggingface.co/datasets/allenai/IF_multi_constraints_upto5)

## Installation

```bash
pip install -r requirements.txt
```

NLTK data will be automatically downloaded on first use.

## Usage

### Script Utilities

- `create_constraint_data.py`: Samples non-conflicting IFEvalG constraints and appends them to instructions, producing JSONL training data with verifiable ground truth.
- `verify_constraints.py`: Checks responses against stored constraint ground truth, supports single checks, batch verification, and accuracy stats.

### Basic Usage

```bash
python create_constraint_data.py \
    --input_dataset allenai/tulu-3-sft-mixture \
    --output_path output_data.jsonl \
    --num_samples 1000
```

### Full Options

```bash
python create_constraint_data.py \
    --input_dataset allenai/tulu-3-sft-mixture \
    --output_path output_data.jsonl \
    --num_samples 10000 \
    --min_constraints 1 \
    --max_constraints 5 \
    --split train \
    --seed 42 \
    --streaming  # Use for large datasets
```

### Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--input_dataset` | HuggingFace dataset name or path | Required |
| `--output_path` | Path to save output JSONL | Required |
| `--num_samples` | Number of samples to process | All |
| `--min_constraints` | Minimum constraints per example | 1 |
| `--max_constraints` | Maximum constraints per example | 5 |
| `--split` | Dataset split to use | train |
| `--seed` | Random seed | 42 |
| `--streaming` | Use streaming mode for large datasets | False |

## Output Format

The output follows the format of `allenai/IF_multi_constraints_upto5`:

```json
{
    "key": "dataset_name_12345",
    "messages": [
        {
            "role": "user",
            "content": "Write a poem about nature. Your response should contain at least 3 sentences. Include keywords ['forest', 'river'] in the response."
        }
    ],
    "ground_truth": "[{\"instruction_id\": [\"length_constraints:number_sentences\"], \"kwargs\": [{\"num_sentences\": 3, \"relation\": \"at least\"}]}, {\"instruction_id\": [\"keywords:existence\"], \"kwargs\": [{\"keywords\": [\"forest\", \"river\"]}]}]",
    "dataset": "dataset_name",
    "constraint_type": "multi",
    "constraint": "Your response should contain at least 3 sentences. Include keywords ['forest', 'river'] in the response."
}
```

## Available Constraints

The IFEvalG library provides 50+ constraint types across categories:

### Keywords
- `keywords:existence` - Include specific keywords
- `keywords:forbidden_words` - Exclude specific keywords
- `keywords:word_once` - Include keyword exactly once
- `keywords:frequency` - Keyword appears N times
- `keywords:palindrome` - Include a palindrome

### Length Constraints
- `length_constraints:number_sentences` - Sentence count requirement
- `length_constraints:number_paragraphs` - Paragraph count requirement
- `length_constraints:number_words` - Word count requirement

### Format
- `detectable_format:json_format` - Output in JSON
- `detectable_format:title` - Include a title in `<<>>`
- `detectable_format:number_bullet_lists` - Use bullet points
- `detectable_format:number_highlighted_sections` - Use markdown highlights

### Case/Language
- `change_case:english_capital` - All uppercase
- `change_case:english_lowercase` - All lowercase
- `language:response_language` - Respond in specific language

### Position
- `first_word:first_word_answer` - Start with specific word
- `last_word:last_word_answer` - End with specific word
- `startend:end_checker` - End with specific phrase

### Punctuation
- `punctuation:no_comma` - No commas allowed
- `punctuation:punctuation_dot` - No periods allowed

### Content
- `detectable_content:postscript` - Add a P.S.
- `startend:quotation` - Wrap in quotation marks

## Constraint Conflict Handling

The script automatically handles constraint conflicts. For example:
- `english_capital` and `english_lowercase` conflict
- `json_format` conflicts with most other format constraints
- Paragraph constraints conflict with each other

## Verification

Each constraint class includes a `check_following(response)` method that returns `True` if the response satisfies the constraint:

```python
from IFEvalG import instructions_registry

# Get a constraint checker
checker_class = instructions_registry.get_instruction_class("keywords:existence")
checker = checker_class("keywords:existence")
checker.build_description(keywords=["hello", "world"])

# Check if a response follows the constraint
response = "Hello world, this is a test."
is_following = checker.check_following(response)  # True
```

## Project Structure

```
.
├── IFEvalG/
│   ├── __init__.py
│   ├── instructions.py          # Constraint checker classes
│   ├── instructions_registry.py # Constraint registry and conflicts
│   └── instructions_util.py     # Utility functions
├── create_constraint_data.py    # Main data creation script
├── requirements.txt
└── README.md
```

## License

Apache License 2.0 (same as the original IFEvalG code from Google Research)

## Citation

If you use this code, please cite:

```bibtex
@misc{pyatkin2025generalizing,
    title={Generalizing Verifiable Instruction Following},
    author={Valentina Pyatkin and Saumya Malik and Victoria Graf and Hamish Ivison and Shengyi Huang and Pradeep Dasigi and Nathan Lambert and Hannaneh Hajishirzi},
    year={2025},
    eprint={TODO},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

## Acknowledgments

This project uses the IFEvalG constraint library from [AllenAI's open-instruct](https://github.com/allenai/open-instruct), which is based on Google Research's IFEval.
