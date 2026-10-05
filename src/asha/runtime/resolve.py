# Copyright 2026 Ajay Rajan
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Resolve process()/sanitize() arguments into RunContext (runtime hot path)."""

from __future__ import annotations

from typing import Any, Optional, cast

from ..core.policy_config import PolicyConfig, PolicyMode
from ..core.policy_resolution import policy_from_mode
from ..core.safety import SafetyMode, safety_mode_from_policy_mode
from .profiles import ExecutionProfile, profile_to_stages
from .run_context import RunContext

_PROCESS_KWARGS = frozenset(
    {
        "mode",
        "policy",
        "profile",
        "token_budget",
        "trace",
        "debug",
        "max_retries",
        "timeout_seconds",
        "verbose",
        "log_level",
        "log_output",
        "log_file",
        "include_legacy_detail",
    }
)


def _reject_unknown_kwargs(kwargs: dict[str, Any], *, allowed: frozenset[str]) -> None:
    unknown = set(kwargs) - allowed
    if unknown:
        raise TypeError(f"Unexpected keyword argument(s): {', '.join(sorted(unknown))}")


def _effective_policy(mode: str, policy: Optional[Any]) -> PolicyConfig:
    if policy is not None:
        if hasattr(policy, "to_policy_config"):
            return cast(PolicyConfig, policy.to_policy_config())
        return cast(PolicyConfig, policy)
    return policy_from_mode(mode)


def _stage_flags(
    *,
    effective_profile: ExecutionProfile,
    profile: Optional[ExecutionProfile],
    cfg: PolicyConfig,
) -> tuple[bool, bool, bool]:
    if effective_profile == ExecutionProfile.PASSTHROUGH:
        return False, False, False
    if profile is not None:
        return profile_to_stages(profile)
    return (
        cfg.enable_pii_detection or cfg.enable_injection_detection,
        cfg.allow_modification,
        cfg.enable_optimization,
    )


def resolve_process_call(
    *,
    mode: str = "balanced",
    policy: Optional[PolicyConfig] = None,
    profile: Optional[ExecutionProfile] = None,
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
) -> tuple[RunContext, dict[str, Any]]:
    """
    Normalize process() arguments into a :class:`RunContext` and runtime extras.

    Returns ``(context, extras)`` where extras holds ``max_retries``,
    ``timeout_seconds``, and ``verbose`` for the processor executor.
    """
    cfg = _effective_policy(mode, policy)
    effective_mode = cfg.mode.value if hasattr(cfg.mode, "value") else str(cfg.mode)

    if profile is not None:
        effective_profile = profile
    elif effective_mode == "off" or cfg.mode == PolicyMode.OFF:
        effective_profile = ExecutionProfile.PASSTHROUGH
    else:
        effective_profile = ExecutionProfile.STANDARD

    sec, comp, opt = _stage_flags(
        effective_profile=effective_profile,
        profile=profile,
        cfg=cfg,
    )
    safety = safety_mode_from_policy_mode(effective_mode)
    effective_privacy = effective_mode != "off"

    context = RunContext(
        mode=effective_mode,
        profile=effective_profile,
        safety=safety,
        effective_privacy=effective_privacy,
        security=sec,
        compile=comp,
        optimize=opt,
        token_budget=token_budget,
        pii_mode=cfg.pii_mode,
        reversible=cfg.reversible,
        preserve_intent=cfg.preserve_intent,
        security_level=cfg.security_level,
        trace=trace,
        debug=debug or cfg.debug_diff,
        log_level=log_level,
        log_output=log_output,
        log_file=log_file,
        include_legacy_detail=include_legacy_detail,
    )
    runtime_extras = {
        "max_retries": max_retries,
        "timeout_seconds": timeout_seconds,
        "verbose": verbose,
    }
    return context, runtime_extras


def resolve_sanitize_safety(mode: str = "balanced") -> SafetyMode:
    """Map a policy mode string to :class:`SafetyMode` for sanitize()."""
    return safety_mode_from_policy_mode(mode)


def resolve_sanitize_policy(
    mode: str = "balanced",
    policy: Optional[PolicyConfig] = None,
) -> tuple[SafetyMode, bool, str]:
    """Return ``(safety_mode, reversible, pii_mode)`` for sanitize()."""
    cfg = _effective_policy(mode, policy)
    effective_mode = cfg.mode.value if hasattr(cfg.mode, "value") else str(cfg.mode)
    return (
        safety_mode_from_policy_mode(effective_mode),
        cfg.reversible,
        getattr(cfg, "pii_mode", "hybrid"),
    )
