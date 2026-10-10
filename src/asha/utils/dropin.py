# Copyright 2026 Ajay Rajan
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""
Drop-in utility functions (NEW - CRITICAL FOR ADOPTION)
from .dropin import process, optimize, sanitize

These functions provide the CRITICAL adoption path:
- process() - One-line prompt processing
- optimize() - Token optimization only
- sanitize() - Security only

File paths / bytes are coerced to text (PDF/DOCX/plaintext) before engines run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, BinaryIO, Optional, Union

import difflib

from ..core.policy_config import PolicyConfig
from ..runtime.processor import PromptProcessor
from ..runtime.resolve import resolve_process_call, resolve_sanitize_policy
from ..types.results import OptimizeResult, ProcessResult, SanitizeResult
from .dropin_helpers import coerce_process_output

# Re-export for backward compatibility
_coerce_process_output = coerce_process_output

PromptInput = Union[str, Path, bytes, BinaryIO]


def generate_text_diff(original: str, modified: str) -> str:
    """Generate a readable diff between original and modified text."""
    if original == modified:
        return "No changes made"

    diff_lines = list(
        difflib.unified_diff(
            original.splitlines(keepends=True),
            modified.splitlines(keepends=True),
            fromfile="original",
            tofile="modified",
            lineterm="",
        )
    )

    if not diff_lines:
        return "No changes made"

    return "\n".join(diff_lines)


def _coerce(prompt: PromptInput, *, filename: Optional[str] = None) -> str:
    from ..documents.coerce import coerce_prompt_input

    return coerce_prompt_input(prompt, filename=filename)


def process(
    prompt: PromptInput,
    mode: str = "balanced",
    *,
    profile: Optional[Any] = None,
    policy: Optional[PolicyConfig] = None,
    token_budget: int = 1200,
    trace: bool = False,
    debug: bool = False,
    max_retries: int = 0,
    timeout_seconds: Optional[float] = None,
    verbose: bool = False,
    log_level: str = "INFO",
    log_output: str = "console",
    log_file: Optional[str] = None,
    include_legacy_detail: bool = False,
    filename: Optional[str] = None,
) -> ProcessResult:
    """
    Process a prompt through ASHA's security and optimization layers.

    Args:
        prompt (PromptInput): The text string, byte stream, or file path to process.
        mode (str, optional): The policy mode ('strict', 'balanced', 'lite', 'off'). Defaults to "balanced".
        profile (Profile or ExecutionProfile, optional): Preset deployment profile or execution profile.
        policy (PolicyConfig, optional): Advanced runtime configuration.
        token_budget (int, optional): The maximum token limit. Defaults to 1200.
        trace (bool, optional): If true, populates the trace attribute with internal execution steps.
        debug (bool, optional): If true, populates the diff attribute with a text diff.
        max_retries (int, optional): Number of retries for detector engines.
        timeout_seconds (float, optional): Max timeout for execution.
        verbose (bool, optional): Enables verbose console logs.
        log_level (str, optional): Logging level ('INFO', 'DEBUG').
        log_output (str, optional): Output destination ('console', etc).
        log_file (str, optional): Path to a log file.
        include_legacy_detail (bool, optional): Includes legacy dict schema.
        filename (str, optional): Required if passing raw bytes for document extraction.

    Returns:
        ProcessResult: An object containing the processed output, security metrics, and execution trace.

    Raises:
        ASHAProcessingError: If mode is 'strict' and an unrecoverable threat is detected.
    """
    exec_profile = None
    if profile is not None:
        if hasattr(profile, "to_policy_config"):
            policy = policy or profile.to_policy_config()
            mode = getattr(profile, "mode", mode)
            token_budget = getattr(profile, "token_budget", token_budget)
            debug = debug or getattr(profile, "debug", False)
            verbose = verbose or getattr(profile, "verbose", False)
        elif hasattr(profile, "name") and hasattr(profile, "value"):
            exec_profile = profile

    text = _coerce(prompt, filename=filename)
    context, runtime_extras = resolve_process_call(
        mode=mode,
        policy=policy,
        profile=exec_profile,
        token_budget=token_budget,
        trace=trace,
        debug=debug,
        max_retries=max_retries,
        timeout_seconds=timeout_seconds,
        verbose=verbose,
        log_level=log_level,
        log_output=log_output,
        log_file=log_file,
        include_legacy_detail=include_legacy_detail,
    )

    return PromptProcessor().run_with_context(
        text,
        context,
        **runtime_extras,
    )


async def process_async(
    prompt: PromptInput,
    mode: str = "balanced",
    *,
    profile: Optional[Any] = None,
    policy: Optional[PolicyConfig] = None,
    token_budget: int = 1200,
    trace: bool = False,
    debug: bool = False,
    max_retries: int = 0,
    timeout_seconds: Optional[float] = None,
    verbose: bool = False,
    log_level: str = "INFO",
    log_output: str = "console",
    log_file: Optional[str] = None,
    include_legacy_detail: bool = False,
    filename: Optional[str] = None,
) -> ProcessResult:
    """
    Asynchronously process a prompt through ASHA's security and optimization layers.

    This is a coroutine equivalent of :func:`process` suitable for use in
    ``asyncio`` event loops, FastAPI endpoints, and other async contexts.
    Internally dispatches to a thread pool via ``asyncio.to_thread`` so ASHA's
    CPU-bound security and optimization pipeline never blocks the event loop.

    Args:
        prompt (PromptInput): The text string, byte stream, or file path to process.
        mode (str, optional): The policy mode ('strict', 'balanced', 'lite', 'off').
            Defaults to ``"balanced"``.
        profile (Profile or ExecutionProfile, optional): Preset deployment profile or execution profile.
        policy (PolicyConfig, optional): Advanced runtime configuration. When
            provided, overrides ``mode``.
        token_budget (int, optional): Maximum token limit for the output.
            Defaults to 1200.
        trace (bool, optional): If ``True``, populates ``result.trace`` with
            internal execution steps. Defaults to ``False``.
        debug (bool, optional): If ``True``, populates ``result.diff`` with a
            unified text diff of the changes. Defaults to ``False``.
        max_retries (int, optional): Number of retry attempts for detector engines
            on transient failures. Defaults to 0.
        timeout_seconds (float, optional): Maximum wall-clock time in seconds.
            If exceeded, raises ``asyncio.TimeoutError``. Defaults to ``None``.
        verbose (bool, optional): Emit detailed per-request logs to stdout.
            Defaults to ``False``.
        log_level (str, optional): Logging verbosity level. Defaults to ``"INFO"``.
        log_output (str, optional): Output destination for logs. Defaults to
            ``"console"``.
        log_file (str, optional): Absolute path to a log file. Defaults to ``None``.
        include_legacy_detail (bool, optional): Include the legacy ``detail`` dict
            schema in the result for backward compatibility. Defaults to ``False``.
        filename (str, optional): Required when ``prompt`` is raw bytes without a
            file extension, used for document-type detection.

    Returns:
        ProcessResult: An object containing the processed output, security metrics,
            and optional trace/diff data.

    Raises:
        ASHAProcessingError: If ``mode='strict'`` and an unrecoverable threat is
            detected.
        asyncio.TimeoutError: If ``timeout_seconds`` is exceeded.

    Examples:
        FastAPI endpoint::

            from fastapi import FastAPI
            from asha import process_async

            app = FastAPI()

            @app.post("/chat")
            async def chat(body: dict):
                result = await process_async(body["prompt"], mode="balanced")
                return {"safe_prompt": result.output}
    """
    import asyncio

    return await asyncio.to_thread(
        process,
        prompt,
        mode=mode,
        profile=profile,
        policy=policy,
        token_budget=token_budget,
        trace=trace,
        debug=debug,
        max_retries=max_retries,
        timeout_seconds=timeout_seconds,
        verbose=verbose,
        log_level=log_level,
        log_output=log_output,
        log_file=log_file,
        include_legacy_detail=include_legacy_detail,
        filename=filename,
    )


async def optimize_async(
    prompt: PromptInput,
    token_budget: int = 1200,
    trust_input: bool = False,
    *,
    filename: Optional[str] = None,
) -> OptimizeResult:
    """
    Asynchronously optimize a prompt's token count without security processing.

    Coroutine equivalent of :func:`optimize`. Dispatches to a thread pool so
    the event loop is never blocked by CPU-bound tokenisation and compression.

    Args:
        prompt (PromptInput): The text string, byte stream, or file path to optimize.
        token_budget (int, optional): Maximum token count for the output.
            Defaults to 1200.
        trust_input (bool, optional): If ``True``, skips optimization entirely
            and returns the input unchanged. Useful for trusted system prompts.
            Defaults to ``False``.
        filename (str, optional): Required when ``prompt`` is raw bytes without
            a recognizable file extension.

    Returns:
        OptimizeResult: An object containing the optimized output and compression
            metrics (token count before/after, processing time).

    Examples:
        ::

            from asha import optimize_async

            result = await optimize_async(long_rag_context, token_budget=2048)
            print(result.output)  # Compressed context under 2048 tokens
    """
    import asyncio

    return await asyncio.to_thread(
        optimize,
        prompt,
        token_budget=token_budget,
        trust_input=trust_input,
        filename=filename,
    )


async def sanitize_async(
    prompt: PromptInput,
    *,
    mode: str = "balanced",
    policy: Optional[PolicyConfig] = None,
    max_retries: int = 0,
    timeout_seconds: Optional[float] = None,
    filename: Optional[str] = None,
) -> SanitizeResult:
    """
    Asynchronously sanitize a prompt for security only (PII masking + threat detection).

    Coroutine equivalent of :func:`sanitize`. Runs security-only processing in a
    thread pool without performing token optimization. Use this when you want
    security without modifying the prompt structure.

    Args:
        prompt (PromptInput): The text string, byte stream, or file path to sanitize.
        mode (str, optional): The policy mode ('strict', 'balanced', 'lite', 'off').
            Defaults to ``"balanced"``.
        policy (PolicyConfig, optional): Advanced runtime configuration. When
            provided, overrides ``mode``.
        max_retries (int, optional): Number of retry attempts for detector engines
            on transient failures. Defaults to 0.
        timeout_seconds (float, optional): Maximum wall-clock time in seconds.
            Defaults to ``None``.
        filename (str, optional): Required when ``prompt`` is raw bytes without
            a recognizable file extension.

    Returns:
        SanitizeResult: An object containing the sanitized output and security
            metrics (PII masked count, threat level, security score).

    Raises:
        ASHAProcessingError: If ``mode='strict'`` and an unrecoverable threat is
            detected.

    Examples:
        ::

            from asha import sanitize_async

            result = await sanitize_async(
                user_input,
                mode="strict",
            )
            if not result.security_result.is_safe:
                raise ValueError("Unsafe input rejected")
    """
    import asyncio

    return await asyncio.to_thread(
        sanitize,
        prompt,
        mode=mode,
        policy=policy,
        max_retries=max_retries,
        timeout_seconds=timeout_seconds,
        filename=filename,
    )


def optimize(
    prompt: PromptInput,
    token_budget: int = 1200,
    trust_input: bool = False,
    *,
    filename: Optional[str] = None,
) -> OptimizeResult:
    """
    Token optimization only via MSDPC - no security or compile stages.

    Args:
        prompt (PromptInput): The text string, byte stream, or file path to optimize.
        token_budget (int, optional): The maximum token limit. Defaults to 1200.
        trust_input (bool, optional): If true, bypasses optimization completely.
        filename (str, optional): Required if passing raw bytes for document extraction.

    Returns:
        OptimizeResult: An object containing the optimized output and metrics.
    """
    from ..core.engines import optimize_tokens

    text = _coerce(prompt, filename=filename)
    if trust_input:
        from ..types.results import MetricsInfo

        return OptimizeResult(
            output=text,
            original=text,
            degraded=False,
            degraded_reason=None,
            metrics=MetricsInfo(0, 0.0, 0.0),
        )
    return optimize_tokens(text, token_budget=token_budget)


def sanitize(
    prompt: PromptInput,
    *,
    mode: str = "balanced",
    policy: Optional[PolicyConfig] = None,
    max_retries: int = 0,
    timeout_seconds: Optional[float] = None,
    filename: Optional[str] = None,
) -> SanitizeResult:
    """
    Sanitize a prompt for security only (PII masking, threat detection).

    Args:
        prompt (PromptInput): The text string, byte stream, or file path to sanitize.
        mode (str, optional): The policy mode ('strict', 'balanced', 'lite', 'off'). Defaults to "balanced".
        policy (PolicyConfig, optional): Advanced runtime configuration.
        max_retries (int, optional): Number of retries for detector engines.
        timeout_seconds (float, optional): Max timeout for execution.
        filename (str, optional): Required if passing raw bytes for document extraction.

    Returns:
        SanitizeResult: An object containing the sanitized output and security metrics.

    Raises:
        ASHAProcessingError: If mode is 'strict' and an unrecoverable threat is detected.
    """
    from ..core.engines import sanitize_text

    text = _coerce(prompt, filename=filename)
    safety, reversible, pii_mode = resolve_sanitize_policy(mode, policy)
    return sanitize_text(
        text,
        reversible=reversible,
        safety_mode=safety,
        mode=mode,
        max_retries=max_retries,
        timeout_seconds=timeout_seconds,
        pii_mode=pii_mode,
    )
