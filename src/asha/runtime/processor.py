# Copyright 2026 Ajay Rajan
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Prompt orchestrator - composes core engines directly (no stage registry)."""

from __future__ import annotations

import concurrent.futures as cf
import time
import uuid
from typing import Any, Dict, Optional

# Module-level executor for timed process() calls.
# Avoids spawning + tearing down a thread pool on every invocation under load.
# max_workers=4 covers typical async-gateway concurrency without over-allocating.
_TIMEOUT_EXECUTOR: cf.ThreadPoolExecutor = cf.ThreadPoolExecutor(
    max_workers=4, thread_name_prefix="asha-proc"
)
import atexit as _atexit
_atexit.register(_TIMEOUT_EXECUTOR.shutdown, wait=False)

from ..core.engines import compile_prompt, optimize_tokens
from ..core.ml_utils import check_pii_mode_feasibility, validate_pii_mode
from ..core.policy_gate import (
    create_passthrough_result,
    modification_disabled,
    optimization_disabled,
    security_disabled,
    should_passthrough,
)
from ..core.policy_resolution import build_pipeline_config
from ..core.safety import (
    SafetyMode,
    is_fail_closed,
    safety_mode_from_policy_mode,
    should_raise_on_failure,
)
from ..core.security.service import get_sanitized_content, run_security
from ..core.security.security_layer import SecurityLevel, SecurityResult
from ..exceptions import ASHAProcessingError
from ..types.results import ProcessResult
from ..utils.dropin_helpers import normalize_security_level
from ..utils.dropin_privacy import (
    SECURITY_FAIL_CLOSED_PLACEHOLDER,
    finalize_privacy_output as _finalize_privacy_output,
    privacy_fallback as _privacy_fallback,
    extract_pii_types as _extract_pii_types,
    security_field as _security_field,
)
from ..utils.metrics_hook import record_process_metrics
from ..utils.result_builders import build_process_result
from .profiles import ExecutionProfile, profile_to_stages
from .run_context import RunContext


def _security_result_to_dict(sr: Any) -> Any:
    return sr


class PromptProcessor:
    """Orchestrates security → compile → optimize via core engines."""

    def run(
        self,
        prompt: str,
        *,
        profile: ExecutionProfile = ExecutionProfile.STANDARD,
        mode: str = "balanced",
        token_budget: int = 1200,
        trace: bool = False,
        debug: bool = False,
        log_level: str = "INFO",
        log_output: str = "console",
        log_file: Optional[str] = None,
        max_retries: int = 0,
        timeout_seconds: Optional[float] = None,
        verbose: bool = False,
    ) -> ProcessResult:
        """Public API - use ``profile`` and ``mode`` only."""
        effective_profile = profile
        if mode.lower() == "off":
            effective_profile = ExecutionProfile.PASSTHROUGH
        sec, comp, opt = profile_to_stages(effective_profile)
        context = RunContext(
            mode=mode,
            profile=effective_profile,
            safety=safety_mode_from_policy_mode(mode),
            effective_privacy=mode.lower() != "off",
            security=sec,
            compile=comp,
            optimize=opt,
            token_budget=token_budget,
            trace=trace,
            debug=debug,
            log_level=log_level,
            log_output=log_output,
            log_file=log_file,
        )
        return self.run_with_context(
            prompt,
            context,
            max_retries=max_retries,
            timeout_seconds=timeout_seconds,
            verbose=verbose,
        )

    def run_with_context(
        self,
        prompt: str,
        context: RunContext,
        *,
        max_retries: int = 0,
        timeout_seconds: Optional[float] = None,
        verbose: bool = False,
    ) -> ProcessResult:
        """Internal entry used by ``process()`` after compat resolution."""
        if not prompt or not isinstance(prompt, str):
            return build_process_result(
                output="",
                original=str(prompt or ""),
                degraded=True,
                degraded_reason="invalid_prompt",
                privacy_applied=False,
                pipeline_result=None,
                privacy=False,
                include_metrics=True,
            )

        safety = context.safety
        fail_closed = is_fail_closed(safety)
        raise_on_failure = should_raise_on_failure(safety)
        security_flag = context.security
        compile_flag = context.compile
        optimize_flag = context.optimize
        privacy = context.effective_privacy

        try:
            pii_mode = validate_pii_mode(context.pii_mode)
            feasibility = check_pii_mode_feasibility(pii_mode)
            if not feasibility["feasible"]:
                if feasibility.get("requires_ml") and verbose:
                    print(
                        f"Warning: {feasibility.get('installation_instructions', '')}"
                    )
                    print("Falling back to lite (regex-only) PII detection")
                pii_mode = "lite"

            _, effective_privacy, policy_dict = build_pipeline_config(
                mode=context.mode,
                privacy=privacy,
                security=security_flag,
                compile=compile_flag,
                optimize=optimize_flag,
                safety_mode=safety,
                pii_mode=pii_mode,
                reversible=context.reversible,
                preserve_intent=context.preserve_intent,
                security_level=context.security_level,
                debug=context.debug,
            )
            policy_dict["security_level"] = normalize_security_level(
                context.security_level
            )

            def _execute() -> ProcessResult:
                return self._run_engines(
                    prompt,
                    effective_privacy=effective_privacy,
                    policy_dict=policy_dict,
                    context=context,
                    pii_mode=pii_mode,
                    fail_closed=fail_closed,
                )

            last_exc: Optional[Exception] = None
            for attempt in range(max_retries + 1):
                try:
                    if timeout_seconds is not None:
                        fut = _TIMEOUT_EXECUTOR.submit(_execute)
                        try:
                            return fut.result(timeout=timeout_seconds)
                        except cf.TimeoutError:
                            fut.cancel()
                            raise TimeoutError(
                                f"process() timed out after {timeout_seconds}s"
                            ) from None
                    return _execute()
                except cf.TimeoutError:
                    raise TimeoutError(
                        f"process() timed out after {timeout_seconds}s"
                    ) from None
                except ASHAProcessingError:
                    raise
                except Exception as exc:
                    last_exc = exc
                    if attempt == max_retries:
                        raise
                    if verbose:
                        print(
                            f"[ASHA] attempt {attempt + 1} failed "
                            f"({type(exc).__name__}), retrying..."
                        )

            raise last_exc or RuntimeError("Processing failed")

        except TimeoutError:
            legacy = _timeout_detail(prompt, timeout_seconds)
            return build_process_result(
                output=prompt,
                original=prompt,
                degraded=True,
                degraded_reason="timeout",
                privacy_applied=False,
                pipeline_result=legacy if context.include_legacy_detail else None,
                privacy=privacy and security_flag,
                include_metrics=True,
                legacy_detail=legacy if context.include_legacy_detail else None,
            )
        except ASHAProcessingError:
            raise
        except Exception as exc:
            if raise_on_failure:
                raise ASHAProcessingError(
                    "ASHA security processing failed; refusing to return "
                    "unprocessed prompt."
                ) from exc
            fallback = _privacy_fallback(
                prompt,
                privacy and security_flag,
                safety=safety,
            )
            degraded_reason = f"processor_failed:{type(exc).__name__}"
            record_process_metrics(
                prompt=prompt,
                optimized=fallback,
                latency_ms=0,
                success=False,
                pii_detected=_extract_pii_types(
                    None, prompt, privacy and security_flag
                ),
            )
            return build_process_result(
                output=fallback,
                original=prompt,
                degraded=True,
                degraded_reason=degraded_reason,
                privacy_applied=False,
                pipeline_result=None,
                privacy=privacy and security_flag,
                include_metrics=True,
            )

    def _run_engines(
        self,
        prompt: str,
        *,
        effective_privacy: bool,
        policy_dict: Dict[str, Any],
        context: RunContext,
        pii_mode: str,
        fail_closed: bool,
    ) -> ProcessResult:
        security = context.security
        compile_flag = context.compile
        optimize_flag = context.optimize
        safety = context.safety
        token_budget = context.token_budget
        preserve_intent = context.preserve_intent
        trace = context.trace
        debug = context.debug

        start = time.perf_counter()
        session_id = str(uuid.uuid4())
        trace_ctx = None
        if trace or debug:
            from ..core.trace_context import TraceContext

            trace_ctx = TraceContext(
                input_prompt=prompt,
                log_level=context.log_level,
                trace_enabled=trace,
                debug_enabled=debug,
                log_output=context.log_output,
                log_file=context.log_file,
            )

        if should_passthrough(policy_dict):
            passthrough = create_passthrough_result(prompt, policy_dict)
            return self._finalize(
                prompt,
                passthrough["prompts"]["optimized"],
                passthrough,
                effective_privacy,
                security,
                fail_closed,
                preserve_intent,
                include_legacy=context.include_legacy_detail,
                trace_ctx=trace_ctx,
                trace=trace,
                debug=debug,
            )

        security_result: Any = None
        working = prompt
        sanitized = prompt

        if security and effective_privacy and not security_disabled(policy_dict):
            if trace_ctx:
                trace_ctx.log_stage_start("security", prompt)
            security_result = self._run_security(
                prompt, policy_dict, safety, fail_closed
            )
            sanitized = get_sanitized_content(security_result, prompt)
            working = sanitized
            if trace_ctx:
                trace_ctx.log_stage_end("security", sanitized)

        compiled = working
        if compile_flag and not modification_disabled(policy_dict):
            if trace_ctx:
                trace_ctx.log_stage_start("compile", working)
            try:
                compiled = compile_prompt(working)
                working = compiled
            except Exception as exc:
                if should_raise_on_failure(safety):
                    raise ASHAProcessingError("ASHA compile stage failed.") from exc
                if fail_closed:
                    compiled = SECURITY_FAIL_CLOSED_PLACEHOLDER
                    working = compiled
                else:
                    compiled = working
            if trace_ctx:
                trace_ctx.log_stage_end("compile", compiled)

        token_reduction_pct = 0.0
        optimized = working
        if optimize_flag and not optimization_disabled(policy_dict):
            if trace_ctx:
                trace_ctx.log_stage_start("optimize", working)
            opt = optimize_tokens(working, token_budget=token_budget)
            optimized = opt.output
            if opt.metrics:
                token_reduction_pct = opt.metrics.token_reduction_pct
            if trace_ctx:
                trace_ctx.log_stage_end(
                    "optimize",
                    optimized,
                    metrics={"token_reduction_percentage": token_reduction_pct},
                )

        elapsed_ms = (time.perf_counter() - start) * 1000
        pipeline_result: Dict[str, Any] = {
            "success": True,
            "session_id": session_id,
            "prompts": {
                "original": prompt,
                "sanitized": sanitized,
                "compiled": compiled,
                "optimized": optimized,
            },
            "security_result": _security_result_to_dict(security_result),
            "optimization_metrics": {
                "token_reduction_percentage": token_reduction_pct,
            },
            "performance_metrics": {"total_pipeline_ms": elapsed_ms},
            "stage_metrics": {},
        }
        if trace_ctx:
            pipeline_result["trace"] = trace_ctx.get_trace_summary()
            if debug:
                pipeline_result["diff"] = trace_ctx.generate_diff()

        return self._finalize(
            prompt,
            optimized,
            pipeline_result,
            effective_privacy,
            security,
            fail_closed,
            preserve_intent,
            include_legacy=context.include_legacy_detail,
            trace_ctx=trace_ctx,
            trace=trace,
            debug=debug,
        )

    def _run_security(
        self,
        prompt: str,
        policy_dict: Dict[str, Any],
        safety: SafetyMode,
        fail_closed: bool,
    ) -> Any:
        try:
            return run_security(prompt, policy_dict)
        except Exception as exc:
            if should_raise_on_failure(safety):
                raise ASHAProcessingError("ASHA security processing failed.") from exc
            if fail_closed:
                return SecurityResult(
                    is_safe=False,
                    threat_level=SecurityLevel.HIGH,
                    detected_threats=[],
                    sanitized_content=SECURITY_FAIL_CLOSED_PLACEHOLDER,
                    masked_entities={},
                    security_score=0.0,
                    recommendations=["security_processing_failed"],
                    processing_time_ms=0.0,
                )
            return SecurityResult(
                is_safe=True,
                threat_level=SecurityLevel.LOW,
                detected_threats=[],
                sanitized_content=prompt,
                masked_entities={},
                security_score=0.0,
                recommendations=[],
                processing_time_ms=0.0,
            )

    def _finalize(
        self,
        prompt: str,
        optimized: str,
        pipeline_result: Dict[str, Any],
        effective_privacy: bool,
        security: bool,
        fail_closed: bool,
        preserve_intent: bool,
        *,
        include_legacy: bool,
        trace_ctx: Any,
        trace: bool,
        debug: bool,
    ) -> ProcessResult:
        del trace_ctx
        if not pipeline_result.get("success", True) and fail_closed:
            optimized = _privacy_fallback(
                prompt,
                effective_privacy,
                safety=SafetyMode.STRICT,
            )

        security_result = pipeline_result.get("security_result")
        pii_types = _extract_pii_types(
            security_result,
            prompt,
            effective_privacy and security,
            processed_text=optimized,
        )
        threats = _security_field(security_result, "detected_threats", []) or []
        preserve = preserve_intent and not pii_types and len(threats) == 0
        output = (
            prompt
            if preserve
            else _finalize_privacy_output(
                pipeline_result, effective_privacy and security
            )
        )

        privacy_applied = (
            security and effective_privacy and not pipeline_result.get("passthrough")
        )
        if security_result is not None:
            privacy_applied = privacy_applied and bool(
                _security_field(security_result, "is_safe", True)
                or _security_field(security_result, "sanitized_content")
            )

        legacy = pipeline_result if include_legacy else None
        proc_result = build_process_result(
            output=output,
            original=prompt,
            degraded=False,
            degraded_reason=None,
            privacy_applied=bool(privacy_applied),
            pipeline_result=pipeline_result,
            privacy=effective_privacy and security,
            include_metrics=True,
            trace=pipeline_result.get("trace") if trace else None,
            diff=pipeline_result.get("diff") if debug else None,
            legacy_detail=legacy,
        )

        record_process_metrics(
            prompt=prompt,
            optimized=output,
            latency_ms=(
                proc_result.metrics.processing_time_ms if proc_result.metrics else 0
            ),
            success=True,
            tokens_saved=(
                proc_result.metrics.tokens_saved if proc_result.metrics else 0
            ),
            pii_detected=pii_types,
        )
        return proc_result


def _timeout_detail(prompt: str, timeout_seconds: Optional[float]) -> Dict[str, Any]:
    return {
        "success": True,
        "timed_out": True,
        "timeout_stage": "processor",
        "prompts": {
            "original": prompt,
            "sanitized": prompt,
            "compiled": prompt,
            "optimized": prompt,
        },
        "security_result": None,
        "optimization_metrics": {"token_reduction_percentage": 0},
        "performance_metrics": {
            "total_pipeline_ms": 0,
            "timeout_seconds": timeout_seconds,
        },
        "stage_metrics": {},
    }
