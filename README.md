# Verifiable Constraint Data Creation

This repository provides tools for adding verifiable constraints to instruction-following data. It supports two approaches: heuristic constraints with programmatic verification (IFEvalG) and LLM-assisted constraint generation with prompt-based evaluation.

## Installation

```bash
pip install -r requirements.txt
```

NLTK data will be automatically downloaded on first use.

---

## Two Approaches to Creating Constraint-Augmented Data

### Approach 1: IFEvalG Heuristics (Python-Verifiable)

Uses the IFEvalG constraint library to sample non-conflicting constraints and append them to instructions. Constraints can be verified programmatically.

| Step | Script | Description |
|------|--------|-------------|
| Generate | `create_constraint_data.py` | Samples constraints and appends to instructions |
| Verify | `verify_constraints.py` | Checks responses against constraint ground truth |

```bash
# Generate constrained instructions
python create_constraint_data.py \
    --input_dataset allenai/tulu-3-sft-mixture \
    --save_to_disk ./output_dataset \
    --num_samples 1000 \
    --min_constraints 1 \
    --max_constraints 5

# Verify model responses
python verify_constraints.py \
    --input_file responses.jsonl \
    --output_file verified.jsonl
```

### Approach 2: vLLM + Prompt-Based (LLM-Assisted)

Uses a vLLM backend to generate constraints via prompts. Evaluation is done using LLM-based checklist matching.

| Step | Script | Prompt | Description |
|------|--------|--------|-------------|
| Augment | `augment_instructions_vllm.py` | `prompt/constraint_augmentation.txt` | Add constraints to instructions |
| Filter | `contradiction_check_vllm.py` | `prompt/contradiction_check.txt` | Label self-contradictory constraints |
| Extract | `checklist_extraction_vllm.py` | `prompt/checklist_extraction.txt` | Generate evaluation checklists |
| Evaluate | (use prompt directly) | `prompt/checklist_eval.txt` | Score responses against checklist |

```bash
# 1) Augment instructions with constraints
python augment_instructions_vllm.py \
    --input_dataset ./input_dataset \
    --save_to_disk ./augmented_dataset \
    --model "Qwen/Qwen3-30B-A3B-Instruct-2507" \
    --base_url "http://localhost:8000/v1" \
    --num_workers 64 --max_inflight 128

# 2) Check for self-contradictory constraints
python contradiction_check_vllm.py \
    --input_dataset ./augmented_dataset \
    --save_to_disk ./contradiction_labeled \
    --model "Qwen/Qwen3-30B-A3B-Instruct-2507" \
    --base_url "http://localhost:8000/v1"

# 3) Extract evaluation checklists
python checklist_extraction_vllm.py \
    --input_dataset ./augmented_dataset \
    --save_to_disk ./checklist_dataset \
    --model "Qwen/Qwen3-30B-A3B-Instruct-2507" \
    --base_url "http://localhost:8000/v1"
```

Shell scripts for cluster deployment are available in `scripts/`.

---

## Utility Scripts

### Upload Dataset to Hugging Face Hub

```bash
python upload_dataset_to_hf.py \
    --dataset_path ./my_dataset \
    --repo_id username/dataset_name \
    --private  # optional
```

### Deduplicate Instructions

Merges multiple datasets and removes duplicate instructions, tracking which dataset each row came from.

```bash
python dedup_instructions.py \
    --input_path ./dataset_a \
    --input_path ./dataset_b \
    --output_path ./deduped_dataset \
    --instruction_field instruction \
    --lower  # optional: case-insensitive dedup
```

---

## Script Reference

### `create_constraint_data.py`

| Argument | Description | Default |
|----------|-------------|---------|
| `--input_dataset` | HuggingFace dataset name or local path | Required |
| `--output_path` | Path to save output JSONL | — |
| `--save_to_disk` | Path to save as HF dataset | — |
| `--num_samples` | Number of samples to process | All |
| `--min_constraints` | Minimum constraints per example | 1 |
| `--max_constraints` | Maximum constraints per example | 5 |
| `--split` | Dataset split to use | train |
| `--seed` | Random seed | 42 |
| `--streaming` | Use streaming mode for large datasets | False |

### `verify_constraints.py`

| Argument | Description | Default |
|----------|-------------|---------|
| `--response` | Single response text to verify | — |
| `--constraint_id` | Constraint ID to check | — |
| `--kwargs` | JSON string of kwargs for constraint | `{}` |
| `--input_file` | Input JSONL file with responses | — |
| `--output_file` | Output JSONL file for results | — |
| `--response_field` | Field name containing response | response |
| `--list_constraints` | List all available constraint IDs | — |

### `augment_instructions_vllm.py` / `contradiction_check_vllm.py` / `checklist_extraction_vllm.py`

| Argument | Description | Default |
|----------|-------------|---------|
| `--input_dataset` | Dataset name, path, or JSON/JSONL file | Required |
| `--save_to_disk` | Path to save HF dataset | Required |
| `--push_to_hub` | Hub repo_id to push after saving | — |
| `--model` | Model name served by vLLM | openai/gpt-oss-120b |
| `--base_url` | Base URL for OpenAI-compatible API | http://localhost:8000/v1 |
| `--cache_path` | Disk cache directory | ~/.cache |
| `--system_prompt_path` | Path to system prompt file | (script-specific) |
| `--user_prompt_path` | Path to user prompt template | (script-specific) |
| `--instruction_field` | Field name for raw prompt | instruction |
| `--split` | Dataset split | train |
| `--streaming` | Streaming mode | False |
| `--max_samples` | Max samples to process | All |
| `--start_index` | Skip samples before this index | 0 |
| `--num_workers` | Number of threads | 4 |
| `--max_inflight` | Max in-flight requests | 32 |
| `--no_system` | Merge system prompt into user message | False |

### `upload_dataset_to_hf.py`

| Argument | Description | Default |
|----------|-------------|---------|
| `--dataset_path` | Local dataset directory | Required |
| `--repo_id` | Hub repo id (e.g. user/dataset) | Required |
| `--private` | Create repo as private | False |
| `--split` | Split name for DatasetDict | train |

### `dedup_instructions.py`

| Argument | Description | Default |
|----------|-------------|---------|
| `--input_path` | Dataset path (repeatable) | Required |
| `--output_path` | Path to save deduplicated dataset | Required |
| `--instruction_field` | Field to deduplicate by | instruction |
| `--lower` | Lowercase before dedup | False |
| `--no_strip` | Disable stripping whitespace | False |
| `--no_collapse_ws` | Disable collapsing whitespace | False |
| `--drop_empty` | Drop rows with empty instructions | False |

---

## Output Format

### IFEvalG Output (from `create_constraint_data.py`)

```json
{
    "key": "dataset_name_12345",
    "messages": [{"role": "user", "content": "Write a poem about nature. Your response should contain at least 3 sentences."}],
    "ground_truth": "[{\"instruction_id\": [\"length_constraints:number_sentences\"], \"kwargs\": [{\"num_sentences\": 3, \"relation\": \"at least\"}]}]",
    "dataset": "dataset_name",
    "constraint_type": "single",
    "constraint": "Your response should contain at least 3 sentences."
}
```

### vLLM Augmentation Output (from `augment_instructions_vllm.py`)

```json
{
    "key": "dataset_name_12345",
    "raw_prompt": "Write a poem about nature.",
    "augmented_prompt": "Write a poem about nature. Use exactly 4 stanzas. Each stanza must end with a question.",
    "messages": [{"role": "user", "content": "..."}],
    "dataset": "dataset_name",
    "model": "Qwen/Qwen3-30B-A3B-Instruct-2507"
}
```

---

## Available IFEvalG Constraints

The IFEvalG library provides 50+ constraint types:

| Category | Examples |
|----------|----------|
| Keywords | `keywords:existence`, `keywords:forbidden_words`, `keywords:frequency` |
| Length | `length_constraints:number_sentences`, `number_paragraphs`, `number_words` |
| Format | `detectable_format:json_format`, `title`, `number_bullet_lists` |
| Case | `change_case:english_capital`, `english_lowercase` |
| Language | `language:response_language` |
| Position | `first_word:first_word_answer`, `last_word:last_word_answer` |
| Punctuation | `punctuation:no_comma`, `punctuation_dot` |
| Content | `detectable_content:postscript`, `startend:quotation` |

Run `python verify_constraints.py --list_constraints` to see all available IDs.

---

## Programmatic Verification

```python
from IFEvalG import instructions_registry

checker_class = instructions_registry.get_instruction_class("keywords:existence")
checker = checker_class("keywords:existence")
checker.build_description(keywords=["hello", "world"])

response = "Hello world, this is a test."
is_following = checker.check_following(response)  # True
```

---

## Project Structure

```
.
├── IFEvalG/                         # Constraint library
│   ├── __init__.py
│   ├── instructions.py              # Constraint checker classes
│   ├── instructions_registry.py     # Constraint registry and conflicts
│   └── instructions_util.py         # Utility functions
├── prompt/                          # LLM prompts for vLLM pipeline
│   ├── constraint_augmentation.txt
│   ├── constraint_augmentation_user.txt
│   ├── contradiction_check.txt
│   ├── constradiction_check_user.txt
│   ├── checklist_extraction.txt
│   ├── checklist_extraction_user.txt
│   ├── checklist_eval.txt
│   └── checklist_eval_user.txt
├── scripts/                         # Cluster deployment scripts
│   ├── augment_instruction.sh
│   ├── contradiction_check.sh
│   └── checklist_extraction.sh
├── create_constraint_data.py        # IFEvalG constraint generation
├── verify_constraints.py            # IFEvalG constraint verification
├── augment_instructions_vllm.py     # vLLM constraint augmentation
├── contradiction_check_vllm.py      # vLLM contradiction detection
├── checklist_extraction_vllm.py     # vLLM checklist extraction
├── upload_dataset_to_hf.py          # Upload datasets to Hub
├── dedup_instructions.py            # Deduplicate instructions
├── chat.py                          # OpenAI-compatible chat client
├── requirements.txt
└── README.md
```

---

## License

Apache License 2.0 (same as the original IFEvalG code from Google Research)

## Citation

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
