# Copyright 2026 Ajay Rajan
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Core compiler primitives - sanitize, compile, optimize without full pipeline."""

from __future__ import annotations

import concurrent.futures as cf
from typing import TYPE_CHECKING, Any, Optional
import atexit as _atexit

# Shared executor for timed engine calls to avoid per-call thread churn.
_TIMEOUT_EXECUTOR: cf.ThreadPoolExecutor = cf.ThreadPoolExecutor(
    max_workers=4, thread_name_prefix="asha-eng"
)
_atexit.register(_TIMEOUT_EXECUTOR.shutdown, wait=False)

if TYPE_CHECKING:
    from ..core.safety import SafetyMode

from ..core._ir.ir_builder import IRBuilder
from ..core.compiler.prompt_compiler import PromptCompiler
from ..core.compiler.optimizer_engine import PromptOptimizer
from ..core.security.service import run_security_only
from ..core.security.security_layer import SecurityLevel
from ..types.results import OptimizeResult, SanitizeResult


def sanitize_text(
    prompt: str,
    *,
    reversible: bool = False,
    safety_mode: Optional["SafetyMode"] = None,
    mode: str = "balanced",
    max_retries: int = 0,
    timeout_seconds: Optional[float] = None,
    pii_mode: str = "hybrid",
) -> SanitizeResult:
    """Security-only sanitization (PII masking, threat detection)."""
    from ..core.safety import (
        is_fail_closed,
        resolve_safety_mode,
        should_raise_on_failure,
    )
    from ..exceptions import ASHAProcessingError
    from ..utils.dropin_privacy import SECURITY_FAIL_CLOSED_PLACEHOLDER
    from ..utils.result_builders import build_sanitize_result
    from ..utils.dropin_privacy import security_field as _security_field
    from ..types.results import SecurityInfo

    if safety_mode is not None:
        safety = safety_mode
    else:
        safety = resolve_safety_mode(mode)
    fail_closed = is_fail_closed(safety)

    def _run() -> Any:
        return run_security_only(
            prompt,
            security_level=SecurityLevel.HIGH,
            reversible=reversible,
            pii_mode=pii_mode,
        )

    try:
        security_result = None
        for attempt in range(max_retries + 1):
            try:
                if timeout_seconds is not None:
                    fut = _TIMEOUT_EXECUTOR.submit(_run)
                    try:
                        security_result = fut.result(timeout=timeout_seconds)
                    except cf.TimeoutError:
                        fut.cancel()
                        raise TimeoutError(
                            f"sanitize_text() timed out after {timeout_seconds}s"
                        ) from None
                else:
                    security_result = _run()
                break
            except cf.TimeoutError:
                raise TimeoutError(
                    f"sanitize_text() timed out after {timeout_seconds}s"
                ) from None
            except Exception as exc:
                if attempt == max_retries:
                    raise exc

        from ..core.security.service import get_sanitized_content

        output = get_sanitized_content(security_result, prompt)
        return build_sanitize_result(
            output=output,
            original=prompt,
            safe=bool(_security_field(security_result, "is_safe", True)),
            degraded=False,
            degraded_reason=None,
            security_result=security_result,
        )
    except Exception as exc:
        if should_raise_on_failure(safety):
            raise ASHAProcessingError("ASHA security processing failed.") from exc
        if fail_closed:
            return SanitizeResult(
                output=SECURITY_FAIL_CLOSED_PLACEHOLDER,
                original=prompt,
                safe=False,
                degraded=True,
                degraded_reason=f"fail_closed:{type(exc).__name__}",
                security=SecurityInfo(
                    safe=False,
                    pii_detected=[],
                    threats=[],
                    masked_entities={},
                ),
            )
        return build_sanitize_result(
            output=prompt,
            original=prompt,
            safe=False,
            degraded=True,
            degraded_reason=f"detector_failed:{type(exc).__name__}",
            security_result=None,
        )


def compile_prompt(prompt: str) -> str:
    """Compile prompt via internal IR + PromptCompiler."""
    builder = IRBuilder()
    prompt_ir = builder.parse(prompt)
    compiler = PromptCompiler()
    return compiler.compile(prompt_ir)


def optimize_tokens(
    prompt: str,
    *,
    token_budget: int = 1200,
) -> OptimizeResult:
    """MSDPC token compression only - no security or IR compile stages."""
    from ..types.results import MetricsInfo

    builder = IRBuilder()
    ir = builder.parse(prompt)
    optimizer = PromptOptimizer(use_msdpc=True)
    try:
        output, metrics = optimizer.optimize(prompt, ir, token_budget=token_budget)
        reduction = float(metrics.get("token_reduction_percentage", 0))
        orig_tok = int(metrics.get("original_tokens", len(prompt.split())))
        opt_tok = int(metrics.get("optimized_tokens", len(output.split())))
        return OptimizeResult(
            output=output,
            original=prompt,
            degraded=False,
            degraded_reason=None,
            metrics=MetricsInfo(
                tokens_saved=max(0, orig_tok - opt_tok),
                token_reduction_pct=reduction,
                processing_time_ms=0.0,
            ),
        )
    except Exception as exc:
        return OptimizeResult(
            output=prompt,
            original=prompt,
            degraded=True,
            degraded_reason=f"optimize_failed:{type(exc).__name__}",
            metrics=MetricsInfo(0, 0.0, 0.0),
        )
