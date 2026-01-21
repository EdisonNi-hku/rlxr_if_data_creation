#!/usr/bin/env python3
# Copyright 2025
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Verify whether responses follow the specified constraints.

This script can be used to:
1. Verify a single response against constraints
2. Batch verify responses from a JSONL file
3. Calculate constraint-following accuracy

Example usage:
    # Verify a single response
    python verify_constraints.py \
        --response "Hello world, this is a test." \
        --constraint_id "keywords:existence" \
        --kwargs '{"keywords": ["hello", "world"]}'

    # Batch verify from file
    python verify_constraints.py \
        --input_file responses.jsonl \
        --output_file verified.jsonl
"""

import argparse
import json

from tqdm import tqdm

from IFEvalG import instructions_registry


def _get_all_instruction_ids() -> list[str]:
    if hasattr(instructions_registry, "get_all_instruction_ids"):
        return list(instructions_registry.get_all_instruction_ids())
    if hasattr(instructions_registry, "INSTRUCTION_DICT"):
        return list(instructions_registry.INSTRUCTION_DICT.keys())
    if hasattr(instructions_registry, "FUNCTION_DICT"):
        return list(instructions_registry.FUNCTION_DICT.keys())
    return []


def _get_instruction_class(instruction_id: str):
    if hasattr(instructions_registry, "get_instruction_class"):
        return instructions_registry.get_instruction_class(instruction_id)
    if hasattr(instructions_registry, "INSTRUCTION_DICT"):
        return instructions_registry.INSTRUCTION_DICT[instruction_id]
    if hasattr(instructions_registry, "FUNCTION_DICT"):
        return instructions_registry.FUNCTION_DICT[instruction_id]
    raise KeyError(f"Unknown instruction id: {instruction_id}")


def verify_single_constraint(
    response: str,
    constraint_id: str,
    kwargs: dict | None = None,
) -> bool:
    """Verify if a response follows a single constraint.

    Args:
        response: The response text to verify.
        constraint_id: The constraint ID (e.g., "keywords:existence").
        kwargs: Optional keyword arguments for the constraint.

    Returns:
        True if the response follows the constraint, False otherwise.
    """
    checker_class = _get_instruction_class(constraint_id)
    checker = checker_class(constraint_id)

    if kwargs:
        checker.build_description(**kwargs)
    else:
        checker.build_description()

    try:
        return checker.check_following(response)
    except Exception as e:
        print(f"Error checking constraint {constraint_id}: {e}")
        return False


def verify_multiple_constraints(
    response: str,
    ground_truth: list[dict],
) -> tuple[bool, list[bool]]:
    """Verify if a response follows multiple constraints.

    Args:
        response: The response text to verify.
        ground_truth: List of constraint dicts with instruction_id and kwargs.

    Returns:
        Tuple of (all_passed, individual_results).
    """
    results = []

    for constraint_info in ground_truth:
        instruction_ids = constraint_info.get("instruction_id", [])
        kwargs_list = constraint_info.get("kwargs", [])

        for inst_id, kwargs in zip(instruction_ids, kwargs_list):
            passed = verify_single_constraint(response, inst_id, kwargs)
            results.append(passed)

    all_passed = all(results)
    return all_passed, results


def verify_from_file(
    input_file: str,
    output_file: str | None = None,
    response_field: str = "response",
) -> dict:
    """Verify responses from a JSONL file.

    Args:
        input_file: Path to input JSONL file with responses.
        output_file: Optional path to save verification results.
        response_field: Field name containing the response.

    Returns:
        Dictionary with verification statistics.
    """
    total = 0
    all_passed = 0
    constraint_results = []

    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    results = []
    for line in tqdm(lines, desc="Verifying"):
        example = json.loads(line.strip())
        total += 1

        response = example.get(response_field, "")
        ground_truth_str = example.get("ground_truth", "[]")

        # Parse ground_truth
        if isinstance(ground_truth_str, str):
            ground_truth = json.loads(ground_truth_str)
        else:
            ground_truth = ground_truth_str

        passed, individual = verify_multiple_constraints(response, ground_truth)

        if passed:
            all_passed += 1

        constraint_results.extend(individual)

        # Add verification result to example
        example["verification_passed"] = passed
        example["individual_results"] = individual
        results.append(example)

    # Calculate statistics
    stats = {
        "total_examples": total,
        "examples_passed": all_passed,
        "example_accuracy": all_passed / total if total > 0 else 0,
        "total_constraints": len(constraint_results),
        "constraints_passed": sum(constraint_results),
        "constraint_accuracy": (
            sum(constraint_results) / len(constraint_results)
            if constraint_results
            else 0
        ),
    }

    # Save results if output file specified
    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Verify whether responses follow specified constraints."
    )

    # Single verification mode
    parser.add_argument(
        "--response",
        type=str,
        help="Response text to verify",
    )
    parser.add_argument(
        "--constraint_id",
        type=str,
        help="Constraint ID to check (e.g., keywords:existence)",
    )
    parser.add_argument(
        "--kwargs",
        type=str,
        default="{}",
        help="JSON string of kwargs for the constraint",
    )

    # Batch verification mode
    parser.add_argument(
        "--input_file",
        type=str,
        help="Input JSONL file with responses to verify",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        help="Output JSONL file to save verification results",
    )
    parser.add_argument(
        "--response_field",
        type=str,
        default="response",
        help="Field name containing the response (default: response)",
    )

    # List available constraints
    parser.add_argument(
        "--list_constraints",
        action="store_true",
        help="List all available constraint IDs",
    )

    args = parser.parse_args()

    if args.list_constraints:
        print("Available constraint IDs:")
        for cid in sorted(_get_all_instruction_ids()):
            print(f"  - {cid}")
        return

    if args.response and args.constraint_id:
        # Single verification mode
        kwargs = json.loads(args.kwargs)
        result = verify_single_constraint(args.response, args.constraint_id, kwargs)
        print(f"Constraint: {args.constraint_id}")
        print(f"Response follows constraint: {result}")

    elif args.input_file:
        # Batch verification mode
        stats = verify_from_file(
            args.input_file,
            args.output_file,
            args.response_field,
        )
        print("\nVerification Results:")
        print(f"  Total examples: {stats['total_examples']}")
        print(f"  Examples passed: {stats['examples_passed']}")
        print(f"  Example accuracy: {stats['example_accuracy']:.2%}")
        print(f"  Total constraints: {stats['total_constraints']}")
        print(f"  Constraints passed: {stats['constraints_passed']}")
        print(f"  Constraint accuracy: {stats['constraint_accuracy']:.2%}")

        if args.output_file:
            print(f"\nResults saved to: {args.output_file}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
