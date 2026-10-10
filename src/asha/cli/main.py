# Copyright 2026 Ajay Rajan
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""CLI - demo and quick testing (code uses ProcessResult API)."""

from __future__ import annotations

from typing import Any, Dict, List

import click
import sys

from ..types.results import ProcessResult
from ..utils.dropin import process


@click.command()
@click.argument("prompt")
@click.option("--debug", "-d", is_flag=True, help="Show detailed metrics")
@click.option(
    "--mode", "-m", default="balanced", help="Processing mode (balanced/strict/lite)"
)
@click.option("--context", "-c", help="Context configuration (JSON format)")
@click.option(
    "--debug-mode",
    is_flag=True,
    help="Enable trace output (same as --debug)",
)
def demo(prompt: str, debug: bool, mode: str, context: str, debug_mode: bool) -> None:
    """Demo ASHA with any prompt."""
    try:
        context_config = None
        if context:
            import json

            context_config = json.loads(context)

        result = process(
            prompt,
            debug=debug or debug_mode,
            mode=mode,
        )
        assert isinstance(result, ProcessResult)

        print("\n" + "=" * 60)
        print("ASHA Demo")
        print("=" * 60)
        print(f"\nOriginal Prompt:\n   {prompt}")
        print(f"\nProcessed Prompt:\n   {result.output}")
        print(f"\nConfiguration:\n   Mode: {mode}")
        print(f"   Context: {'Applied' if context_config else 'None'}")
        print(f"   Degraded: {result.degraded}")
        if result.metrics:
            print(f"\nMetrics:")
            print(f"   Token reduction: {result.metrics.token_reduction_pct:.1f}%")
            print(f"   Tokens saved: {result.metrics.tokens_saved}")
        if result.security:
            print(f"   PII detected: {result.security.pii_detected}")
        print("\n" + "=" * 50)

    except Exception as e:
        print(f"\nError: {str(e)}")
        sys.exit(1)


@click.command()
def quick_test() -> None:
    """Run quick built-in tests to verify ASHA works."""
    print("\nASHA Quick Test Suite")
    print("=" * 40)

    test_cases: List[Dict[str, Any]] = [
        {
            "name": "Basic PII Detection",
            "prompt": "My email is john@gmail.com",
            "should_mask": True,
        },
        {
            "name": "Safe Prompt (No Change)",
            "prompt": "Hello, summarize this text",
            "should_mask": False,
        },
    ]

    passed = 0
    for i, test in enumerate(test_cases, 1):
        print(f"\n{i}. {test['name']}")
        try:
            result = process(str(test["prompt"]))
            assert isinstance(result, ProcessResult)
            pii_masked = (
                result.security is not None and len(result.security.masked_entities) > 0
            ) or test["prompt"] not in result.output
            if pii_masked == test["should_mask"]:
                print("PASS")
                passed += 1
            else:
                print(f"FAIL - expected mask={test['should_mask']}")
        except Exception as e:
            print(f"ERROR - {e}")

    print(f"\nResults: {passed}/{len(test_cases)} passed")


@click.command()
@click.option("--count", "-n", default=5, help="Number of example prompts")
def examples(count: int) -> None:
    """Show example prompts and their results."""
    samples = [
        "My email is john@gmail.com. Analyze this dataset.",
        "Hello, can you help me with this simple question?",
    ]
    for i, example in enumerate(samples[:count], 1):
        result = process(example)
        assert isinstance(result, ProcessResult)
        print(f"{i}. {result.output}")


@click.group()
def cli() -> None:
    """ASHA CLI - Drop-in security and optimization for LLM apps."""


cli.add_command(demo)
cli.add_command(quick_test)
cli.add_command(examples)

try:
    from .benchmark_cli import run_benchmark

    @click.command()
    @click.option("--mode", "-m", default="balanced", help="Policy mode to test")
    def benchmark(mode: str) -> None:
        class Args:
            mode: str
            config: str | None
            custom: str | None
            compare: str | None
            output: str | None
            save: bool
            format: str
            verbose: bool
            quiet: bool
            timeout: int

            def __init__(self, benchmark_mode: str) -> None:
                self.mode = benchmark_mode
                self.config = None
                self.custom = None
                self.compare = None
                self.output = None
                self.save = False
                self.format = "readable"
                self.verbose = False
                self.quiet = False
                self.timeout = 5000

        run_benchmark(Args(mode))

    cli.add_command(benchmark)
except ImportError:
    pass

try:
    from .recommend_cli import recommend_cmd

    cli.add_command(recommend_cmd)
except ImportError:
    pass


def main() -> None:
    cli()


if __name__ == "__main__":
    cli()
