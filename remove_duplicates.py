#!/usr/bin/env python3
"""Remove rows from magpie_creative_qwen_1M that exist in magpie_creative_qwen2.5."""

import hashlib
from datasets import load_from_disk
from tqdm import tqdm


def normalize_instruction(text: object) -> str:
    """Normalize instruction text for comparison."""
    if text is None:
        return ""
    if isinstance(text, str):
        value = text
    else:
        value = str(text)
    return " ".join(value.strip().split())


def main() -> None:
    # Load both datasets
    print("Loading magpie_creative_qwen2.5...")
    reference_ds = load_from_disk("magpie_creative_qwen2.5")
    print(f"  Loaded {len(reference_ds)} rows")

    print("Loading magpie_creative_qwen_1M...")
    source_ds = load_from_disk("magpie_creative_qwen_1M")
    print(f"  Loaded {len(source_ds)} rows")

    # Build set of instruction hashes from reference dataset
    print("Building reference instruction set...")
    reference_hashes = set()
    for row in tqdm(reference_ds, desc="Hashing reference"):
        normalized = normalize_instruction(row.get("instruction"))
        digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()
        reference_hashes.add(digest)
    print(f"  Found {len(reference_hashes)} unique instructions in reference")

    # Filter source dataset to remove duplicates
    print("Filtering source dataset...")
    keep_indices = []
    duplicates = 0
    for idx, row in tqdm(enumerate(source_ds), total=len(source_ds), desc="Filtering"):
        normalized = normalize_instruction(row.get("instruction"))
        digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()
        if digest not in reference_hashes:
            keep_indices.append(idx)
        else:
            duplicates += 1

    # Select and save
    deduped = source_ds.select(keep_indices)
    print(f"\nRemoved {duplicates} duplicate rows")
    print(f"Keeping {len(deduped)} rows (from {len(source_ds)})")

    print("Saving to magpie_creative_qwen_easy...")
    deduped.save_to_disk("magpie_creative_qwen_easy")
    print("Done!")


if __name__ == "__main__":
    main()
