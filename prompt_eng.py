#!/usr/bin/env python3
"""
Simple prompt engineering CLI for vLLM backends.

Usage:
    # Interactive mode
    python prompt_eng.py

    # Single prompt
    python prompt_eng.py -p "What is 2+2?"

    # With system prompt
    python prompt_eng.py -s "You are a helpful assistant" -p "Hello"

    # From file
    python prompt_eng.py -f prompt.txt

    # Custom model/endpoint
    python prompt_eng.py --model "Qwen/Qwen3-8B" --base-url "http://localhost:8000/v1"
"""

import argparse
import sys

from chat import LocalChat


GEN_CONFIG = {
        "temperature": 0.6,
        "top_p": 0.95,
        "extra_body": {"enable_thinking": True, "top_k": 20},
    }


def build_messages(system_prompt: str | None, user_prompt: str) -> list[dict]:
    """Build message list for the chat API."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    return messages


def print_response(reply: str, reasoning: str, show_reasoning: bool = True):
    """Pretty print the response."""
    if show_reasoning and reasoning:
        print("\n" + "=" * 60)
        print("REASONING:")
        print("=" * 60)
        print(reasoning)
    print("\n" + "=" * 60)
    print("RESPONSE:")
    print("=" * 60)
    print(reply)
    print("=" * 60 + "\n")


def interactive_mode(chat: LocalChat, system_prompt: str | None, show_reasoning: bool):
    """Run in interactive REPL mode."""
    print("Interactive prompt engineering mode. Type 'quit' or 'exit' to stop.")
    print("Commands: :system <prompt> - set system prompt")
    print("          :clear - clear system prompt")
    print("          :model - show current model")
    print("-" * 60)
    
    while True:
        try:
            user_input = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        
        if not user_input:
            continue
        
        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break
        
        # Handle commands
        if user_input.startswith(":system "):
            system_prompt = user_input[8:].strip()
            print(f"System prompt set to: {system_prompt[:50]}...")
            continue
        elif user_input == ":clear":
            system_prompt = None
            print("System prompt cleared.")
            continue
        elif user_input == ":model":
            print(f"Model: {chat.model}")
            print(f"Base URL: {chat.base_url}")
            continue
        
        # Send prompt
        messages = build_messages(system_prompt, user_input)
        reply, reasoning = chat.ask(messages)
        
        if reply:
            print_response(reply, reasoning, show_reasoning)
        else:
            print("[No response received]")


def main():
    parser = argparse.ArgumentParser(
        description="Prompt engineering CLI for vLLM backends",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "-p", "--prompt",
        type=str,
        default=None,
        help="User prompt to send",
    )
    parser.add_argument(
        "-s", "--system",
        type=str,
        default=None,
        help="System prompt",
    )
    parser.add_argument(
        "-f", "--file",
        type=str,
        default=None,
        help="Read user prompt from file",
    )
    parser.add_argument(
        "--system-file",
        type=str,
        default=None,
        help="Read system prompt from file",
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default="openai/gpt-oss-120b",
        help=f"Model name (default: openai/gpt-oss-120b). Available: {list(GENERATION_CONFIGS.keys())}",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:8000/v1",
        help="vLLM API base URL (default: http://localhost:8000/v1)",
    )
    parser.add_argument(
        "--cache",
        type=str,
        default=None,
        help="Cache directory path (default: ~/.cache)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable caching (use temp directory)",
    )
    parser.add_argument(
        "--no-reasoning",
        action="store_true",
        help="Don't show reasoning content in output",
    )
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Force interactive mode even with -p",
    )

    args = parser.parse_args()

    # Handle cache
    import os
    import tempfile
    if args.no_cache:
        cache_path = tempfile.mkdtemp()
    elif args.cache:
        cache_path = args.cache
    else:
        cache_path = os.path.expanduser("~/.cache")

    # Hardcoded generation config (overrides GENERATION_CONFIGS)
    gen_config = GEN_CONFIG

    # Initialize chat client
    chat = LocalChat(
        model=args.model,
        base_url=args.base_url,
        cache_path=cache_path,
        generation_config=gen_config,
    )

    # Read prompts from files if specified
    system_prompt = args.system
    if args.system_file:
        with open(args.system_file, "r", encoding="utf-8") as f:
            system_prompt = f.read().strip()

    user_prompt = args.prompt
    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            user_prompt = f.read().strip()

    show_reasoning = not args.no_reasoning

    # Decide mode
    if args.interactive or (user_prompt is None and not sys.stdin.isatty()):
        # Check if input is from pipe
        if not sys.stdin.isatty() and user_prompt is None:
            user_prompt = sys.stdin.read().strip()
            if user_prompt:
                messages = build_messages(system_prompt, user_prompt)
                reply, reasoning = chat.ask(messages)
                if reply:
                    print_response(reply, reasoning, show_reasoning)
                else:
                    print("[No response received]", file=sys.stderr)
                    sys.exit(1)
        else:
            interactive_mode(chat, system_prompt, show_reasoning)
    elif user_prompt:
        # Single prompt mode
        messages = build_messages(system_prompt, user_prompt)
        reply, reasoning = chat.ask(messages)
        if reply:
            print_response(reply, reasoning, show_reasoning)
        else:
            print("[No response received]", file=sys.stderr)
            sys.exit(1)
    else:
        # No prompt provided, enter interactive mode
        interactive_mode(chat, system_prompt, show_reasoning)


if __name__ == "__main__":
    main()
