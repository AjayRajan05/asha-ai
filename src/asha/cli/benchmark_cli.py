#!/usr/bin/env python3
# Copyright 2026 Ajay Rajan
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
ASHA Benchmark CLI Tool

Command-line interface for running benchmarks, comparing versions,
and tracking performance metrics.

Usage:
    asha benchmark                    # Run full benchmark suite
    asha benchmark --mode strict      # Test specific policy mode
    asha benchmark --compare old.json  # Compare with previous results
    asha benchmark --custom "prompt"  # Test custom prompt
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Protocol

from asha.core.benchmark import BenchmarkHarness
from asha.core.policy_config import PolicyConfig, get_preset
from asha.utils.dropin import process


class BenchmarkArgs(Protocol):
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


def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="ASHA Benchmark CLI Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Run full benchmark suite
  %(prog)s --mode strict                      # Test STRICT policy mode
  %(prog)s --config production                 # Use production preset
  %(prog)s --compare results_old.json          # Compare with previous results
  %(prog)s --custom "Test prompt with PII"     # Test custom prompt
  %(prog)s --output ./benchmarks              # Save results to directory
  %(prog)s --verbose --save                    # Verbose output and save results
        """,
    )

    # Benchmark options
    parser.add_argument(
        "--mode",
        "-m",
        choices=["strict", "balanced", "lite", "off"],
        default="balanced",
        help="Policy mode to test (default: balanced)",
    )

    parser.add_argument(
        "--config",
        "-c",
        choices=[
            "production",
            "development",
            "security_first",
            "performance",
            "compliance",
            "disabled",
        ],
        help="Use preset configuration",
    )

    parser.add_argument("--custom", help="Test custom prompt (single test)")

    parser.add_argument(
        "--compare", help="Compare with previous benchmark results (JSON file)"
    )

    # Output options
    parser.add_argument(
        "--output", "-o", help="Output directory for results (default: ./benchmarks)"
    )

    parser.add_argument(
        "--save", "-s", action="store_true", help="Save results to file"
    )

    parser.add_argument(
        "--format",
        "-f",
        choices=["readable", "json", "compact"],
        default="readable",
        help="Output format (default: readable)",
    )

    # Behavior options
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Quiet mode (minimal output)"
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=5000,
        help="Timeout per test in milliseconds (default: 5000)",
    )

    return parser


def run_benchmark(args: BenchmarkArgs) -> int:
    """Run benchmark based on arguments."""
    try:
        # Initialize benchmark harness
        output_dir = args.output or "./benchmarks"
        harness = BenchmarkHarness(output_dir=output_dir)

        # Set up configuration
        config = None
        if args.config:
            config = get_preset(args.config)
        elif args.mode:
            config = PolicyConfig.from_mode(args.mode)

        # Add custom test if provided
        if args.custom:
            harness.add_custom_test(
                name="custom_test",
                prompt=args.custom,
                expected_pii=1 if "@" in args.custom else 0,
                category="custom",
            )

        # Run benchmark
        if not args.quiet:
            print(f"🚀 ASHA Benchmark Tool")
            print(f"Mode: {args.mode or 'default'}")
            if args.config:
                print(f"Config: {args.config}")
            print(f"Output: {output_dir}")
            print()

        summary = harness.run_benchmark(config=config)

        # Output results
        if args.format == "json":
            print(json.dumps(summary.__dict__, indent=2))
        elif args.format == "compact":
            print(f"Results: {summary.passed_tests}/{summary.total_tests} passed")
            print(f"Avg Latency: {summary.avg_latency_ms:.1f}ms")
            print(f"P95 Latency: {summary.p95_latency_ms:.1f}ms")
            print(f"P99 Latency: {summary.p99_latency_ms:.1f}ms")
            print(f"Token Reduction: {summary.avg_token_reduction_percentage:.1f}%")
            print(f"False Positive Rate: {summary.false_positive_rate:.2f}")
        else:  # readable
            print("\n" + "=" * 60)
            print("BENCHMARK RESULTS")
            print("=" * 60)
            print(f"Total Tests: {summary.total_tests}")
            print(f"Passed: {summary.passed_tests}")
            print(f"Failed: {summary.failed_tests}")
            print(f"Success Rate: {summary.passed_tests/summary.total_tests*100:.1f}%")
            print()
            print("PERFORMANCE METRICS:")
            print(f"  Average Latency: {summary.avg_latency_ms:.1f}ms")
            print(f"  P95 Latency: {summary.p95_latency_ms:.1f}ms")
            print(f"  P99 Latency: {summary.p99_latency_ms:.1f}ms")
            print(f"  Token Reduction: {summary.avg_token_reduction_percentage:.1f}%")
            print(f"  Total PII Detected: {summary.total_pii_detected}")
            print(f"  Total Threats Blocked: {summary.total_threats_blocked}")
            print(f"  False Positive Rate: {summary.false_positive_rate:.2f}")
            print(f"  Benchmark Time: {summary.benchmark_time_ms:.1f}ms")

        # Save results if requested
        if args.save:
            results_file = harness.save_results()
            if not args.quiet:
                print(f"\n📁 Results saved to: {results_file}")

        # Compare with previous results if requested
        if args.compare:
            try:
                comparison = harness.compare_versions(args.compare)
                print(f"\n🔄 Version Comparison:")
                print(f"Current: {comparison['current_version']}")
                print(f"Previous: {comparison['previous_version']}")
                print(
                    f"Overall improvement: {'✅ YES' if comparison['overall_improvement'] else '❌ NO'}"
                )

                if args.verbose:
                    perf = comparison["performance_comparison"]
                    print(f"\nDetailed Changes:")
                    for metric, data in perf.items():
                        change = "↑" if data["improvement"] else "↓"
                        print(
                            f"  {metric}: {data['current']:.2f} vs {data['previous']:.2f} {change}"
                        )
            except Exception as e:
                print(f"❌ Comparison failed: {e}")
                return 1

        # Return exit code based on success
        return 0 if summary.failed_tests == 0 else 1

    except KeyboardInterrupt:
        print("\n❌ Benchmark interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


def run_single_test(prompt: str, mode: str = "balanced", verbose: bool = False) -> int:
    """Run single prompt test."""
    try:
        if not verbose:
            print(f"Testing prompt: {prompt[:50]}...")

        # Process prompt
        result = process(prompt, mode=mode, debug=verbose)

        metrics = result.metrics.to_dict() if result.metrics else {}
        print("Processed prompt successfully")
        print(f"  Tokens saved: {metrics.get('tokens_saved', 0)}")
        print(f"  Cost reduction: {metrics.get('cost_reduction', '0%')}")
        print(f"  PII detected: {metrics.get('pii_detected', [])}")
        print(f"  Risk level: {metrics.get('risk_level', 'unknown')}")
        print(f"  Processing time: {metrics.get('processing_time_ms', 0)}ms")
        print(f"  Optimized: {result.output[:50]}...")

        return 0

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return 1


def main() -> int:
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()

    # Handle conflicting options
    if args.quiet and args.verbose:
        print("❌ Cannot specify both --quiet and --verbose")
        return 1

    # Run appropriate command
    if args.custom and not args.config and not args.mode:
        # Single test mode
        return run_single_test(args.custom, verbose=args.verbose)
    else:
        # Benchmark mode
        return run_benchmark(args)


if __name__ == "__main__":
    sys.exit(main())
