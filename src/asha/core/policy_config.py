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
Policy Configuration System for ASHA

Implements deterministic behavior modes:
- STRICT: Mask everything + block aggressively
- BALANCED: Smart masking + optimization
- LITE: Minimal changes
- OFF: No processing

This makes ASHA trustworthy, deterministic, and predictable.
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from enum import Enum


class PolicyMode(Enum):
    """Deterministic policy modes for predictable behavior."""

    STRICT = "strict"
    BALANCED = "balanced"
    LITE = "lite"
    OFF = "off"


@dataclass
class PolicyConfig:
    """
    Configuration for deterministic ASHA behavior.

    Each mode has precisely defined behavior to ensure
    same input → same output across runs.
    """

    # Core policy settings
    mode: PolicyMode = PolicyMode.BALANCED
    pii_masking: bool = True
    pii_strictness: float = 0.8  # 0.0-1.0 threshold
    optimization_level: float = 0.5  # 0.0-1.0 aggressiveness
    threat_blocking: bool = True
    allow_modification: bool = True

    # Determinism controls
    deterministic: bool = False
    seed: Optional[int] = None

    # Performance constraints
    max_processing_time_ms: int = 1000
    max_token_reduction: float = 0.8  # Max 80% reduction

    # Feature toggles
    enable_injection_detection: bool = True
    enable_pii_detection: bool = True
    enable_optimization: bool = True
    enable_context_analysis: bool = True

    # Audit and debugging
    audit_trail: bool = False
    debug_diff: bool = False

    # Advanced processing knobs (use via policy=PolicyConfig(...))
    pii_mode: str = "hybrid"
    reversible: bool = False
    preserve_intent: bool = False
    security_level: str = "medium"
    custom_scanners: Optional[List[Any]] = None

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        # Register custom scanner plugins if provided
        if self.custom_scanners:
            try:
                from ..scanners.registry import ScannerRegistry

                for s in self.custom_scanners:
                    ScannerRegistry.register(s)
            except Exception:
                pass

        # Validate ranges
        self.pii_strictness = max(0.0, min(1.0, self.pii_strictness))
        self.optimization_level = max(0.0, min(1.0, self.optimization_level))
        self.max_token_reduction = max(0.0, min(1.0, self.max_token_reduction))

        # Apply mode-specific defaults
        self._apply_mode_defaults()

    def _apply_mode_defaults(self) -> None:
        """Apply mode-specific default configurations."""
        if self.mode == PolicyMode.STRICT:
            self.pii_masking = True
            self.pii_strictness = 1.0
            self.optimization_level = 0.2  # Light optimization
            self.threat_blocking = True
            self.allow_modification = True
            self.enable_injection_detection = True
            self.enable_pii_detection = True
            self.enable_optimization = True
            self.enable_context_analysis = True
            self.deterministic = True

        elif self.mode == PolicyMode.BALANCED:
            self.pii_masking = True
            self.pii_strictness = 0.8
            self.optimization_level = 0.5
            self.threat_blocking = True
            self.allow_modification = True
            self.enable_injection_detection = True
            self.enable_pii_detection = True
            self.enable_optimization = True
            self.enable_context_analysis = True

        elif self.mode == PolicyMode.LITE:
            self.pii_masking = True
            self.pii_strictness = 0.6  # Only high-confidence PII
            self.optimization_level = 0.3  # Light optimization
            self.threat_blocking = False  # No blocking
            self.allow_modification = True
            self.enable_injection_detection = False  # No injection detection
            self.enable_pii_detection = True
            self.enable_optimization = True
            self.enable_context_analysis = False

        elif self.mode == PolicyMode.OFF:
            self.pii_masking = False
            self.pii_strictness = 0.0
            self.optimization_level = 0.0
            self.threat_blocking = False
            self.allow_modification = False
            self.enable_injection_detection = False
            self.enable_pii_detection = False
            self.enable_optimization = False
            self.enable_context_analysis = False

    @classmethod
    def from_mode(
        cls, mode: Union[str, PolicyMode], **overrides: Any
    ) -> "PolicyConfig":
        """
        Create PolicyConfig from mode with optional overrides.

        Args:
            mode: Policy mode (strict/balanced/lite/off)
            **overrides: Configuration overrides

        Returns:
            PolicyConfig instance
        """
        if isinstance(mode, str):
            mode = PolicyMode(mode.lower())

        return cls(mode=mode, **overrides)

    @classmethod
    def resolve(
        cls,
        mode: str,
        privacy: bool = True,
        *,
        security: bool = True,
        compile: bool = True,
        optimize: bool = True,
        **overrides: Any,
    ) -> "PolicyConfig":
        """Unified mode precedence - delegates to policy_resolution."""
        from .policy_resolution import (
            apply_stage_flags,
            build_pipeline_config,
            resolve_effective_privacy,
        )
        from .safety import safety_mode_from_policy_mode

        effective_privacy = resolve_effective_privacy(mode, privacy)
        _, _, policy_dict = build_pipeline_config(
            mode=mode,
            privacy=privacy,
            security=security,
            compile=compile,
            optimize=optimize,
            safety_mode=safety_mode_from_policy_mode(mode),
            pii_mode=overrides.pop("pii_mode", "hybrid"),
            reversible=overrides.pop("reversible", False),
            debug=overrides.pop("debug", False),
            extra=overrides or None,
        )
        policy_dict.pop("safety_mode", None)
        cfg = cls.from_dict({**policy_dict, **overrides})
        return apply_stage_flags(
            cfg,
            security=security,
            compile=compile,
            optimize=optimize,
            privacy=effective_privacy,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "mode": self.mode.value,
            "pii_masking": self.pii_masking,
            "pii_strictness": self.pii_strictness,
            "optimization_level": self.optimization_level,
            "threat_blocking": self.threat_blocking,
            "allow_modification": self.allow_modification,
            "deterministic": self.deterministic,
            "seed": self.seed,
            "max_processing_time_ms": self.max_processing_time_ms,
            "max_token_reduction": self.max_token_reduction,
            "enable_injection_detection": self.enable_injection_detection,
            "enable_pii_detection": self.enable_pii_detection,
            "enable_optimization": self.enable_optimization,
            "enable_context_analysis": self.enable_context_analysis,
            "audit_trail": self.audit_trail,
            "debug_diff": self.debug_diff,
            "pii_mode": self.pii_mode,
            "reversible": self.reversible,
            "preserve_intent": self.preserve_intent,
            "security_level": self.security_level,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "PolicyConfig":
        """Create PolicyConfig from dictionary."""
        config_dict = dict(config_dict)
        config_dict.pop("safety_mode", None)
        if "mode" in config_dict:
            if isinstance(config_dict["mode"], str):
                config_dict["mode"] = PolicyMode(config_dict["mode"])

        return cls(
            **{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__}
        )

    def validate_behavior(self) -> Dict[str, Any]:
        """
        Validate that configuration produces predictable behavior.

        Returns:
            Validation report with any issues found
        """
        issues = []
        warnings = []

        # Check for contradictory settings
        if self.mode == PolicyMode.OFF and self.pii_masking:
            warnings.append("OFF mode has pii_masking=True - will be ignored")

        if self.mode == PolicyMode.LITE and self.threat_blocking:
            warnings.append(
                "LITE mode has threat_blocking=True - may be over-aggressive"
            )

        if self.deterministic and self.seed is None:
            warnings.append(
                "Deterministic mode without seed - may still have variations"
            )

        # Check for performance constraints
        if self.optimization_level > 0.8 and self.max_processing_time_ms < 500:
            issues.append(
                "High optimization with tight time constraint may cause failures"
            )

        # Check for security constraints
        if self.mode == PolicyMode.STRICT and self.pii_strictness < 0.9:
            issues.append("STRICT mode should have pii_strictness >= 0.9")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "predictable": self.deterministic
            or self.mode in [PolicyMode.OFF, PolicyMode.STRICT],
        }

    def get_behavior_summary(self) -> str:
        """Get human-readable summary of expected behavior."""
        behaviors = []

        if self.mode == PolicyMode.OFF:
            behaviors.append("No processing (passthrough)")
        else:
            if self.pii_masking:
                behaviors.append(f"PII masking (strictness: {self.pii_strictness:.1f})")

            if self.enable_optimization:
                behaviors.append(
                    f"Token optimization (level: {self.optimization_level:.1f})"
                )

            if self.threat_blocking:
                behaviors.append("Threat blocking enabled")
            else:
                behaviors.append("Threat detection only (no blocking)")

            if self.deterministic:
                behaviors.append("Deterministic output")

        return f"{self.mode.value.upper()}: " + ", ".join(behaviors)


# Predefined configurations for common use cases
PRESET_CONFIGS = {
    "production": PolicyConfig.from_mode(PolicyMode.BALANCED, deterministic=True),
    "development": PolicyConfig.from_mode(PolicyMode.LITE, debug_diff=True),
    "security_first": PolicyConfig.from_mode(PolicyMode.STRICT),
    "performance": PolicyConfig.from_mode(PolicyMode.LITE, optimization_level=0.7),
    "compliance": PolicyConfig.from_mode(PolicyMode.STRICT, audit_trail=True),
    "disabled": PolicyConfig.from_mode(PolicyMode.OFF),
}


def get_preset(name: str) -> PolicyConfig:
    """Get preset configuration by name."""
    if name not in PRESET_CONFIGS:
        raise ValueError(
            f"Unknown preset: {name}. Available: {list(PRESET_CONFIGS.keys())}"
        )
    return PRESET_CONFIGS[name]


def list_presets() -> Dict[str, str]:
    """List all available presets with descriptions."""
    return {
        "production": "Balanced security and optimization for production workloads",
        "development": "Light processing for development and testing",
        "security_first": "Maximum security, minimal optimization",
        "performance": "Optimized for speed with basic security",
        "compliance": "Strict security with full audit trail",
        "disabled": "No processing - complete passthrough",
    }


class Profile:
    """
    Zero-config security profiles for common deployment scenarios.

    A ``Profile`` wraps a :class:`PolicyConfig` with an opinionated, named
    configuration that is ready to pass to :class:`~asha.integrations.SecureLLM`,
    :func:`~asha.utils.dropin.process`, or any ASHA middleware without any extra
    parameter tuning.

    Examples:
        >>> from asha import Profile, SecureLLM
        >>> import openai
        >>> client = SecureLLM(openai.OpenAI(), profile=Profile.BALANCED())
        >>> # Production: hard blocks, aggressive PII masking, 15-step agent cap
        >>> cfg = Profile.STRICT()
        >>> # Local dev: verbose logging, all PII reported (never blocked)
        >>> cfg = Profile.DEVELOPER()
    """

    def __init__(
        self,
        *,
        mode: str = "balanced",
        fail_closed: bool = True,
        injection_threshold: float = 0.85,
        token_budget: int = 2048,
        max_agent_steps: int = 15,
        rehydrate_pii: bool = True,
        custom_scanners: Optional[List[Any]] = None,
        verbose: bool = False,
        debug: bool = False,
        _name: str = "custom",
    ) -> None:
        """
        Create a custom Profile with explicit parameters.

        Args:
            mode: Underlying policy mode ('strict', 'balanced', 'lite', 'off').
            fail_closed: If True, block the prompt when processing fails.
                If False, allow the prompt through (audit/permissive mode).
            injection_threshold: Probability threshold above which a prompt is
                classified as an injection attempt. Lower = more aggressive.
            token_budget: Maximum tokens per prompt after optimization.
            max_agent_steps: Maximum tool-call steps allowed per agent session.
            rehydrate_pii: If True, :class:`~asha.integrations.SecureLLM`
                automatically re-hydrates masked PII placeholders in the LLM
                completion before returning it to the caller.
            custom_scanners: Optional list of custom :class:`~asha.scanners.BaseScanner` plugins.
            verbose: Emit detailed per-request logs to stdout.
            debug: Include diff and trace data in every :class:`~asha.types.ProcessResult`.
            _name: Internal display name (set automatically by classmethods).
        """
        self.mode = mode
        self.fail_closed = fail_closed
        self.injection_threshold = injection_threshold
        self.token_budget = token_budget
        self.max_agent_steps = max_agent_steps
        self.rehydrate_pii = rehydrate_pii
        self.custom_scanners = custom_scanners
        if custom_scanners:
            try:
                from ..scanners.registry import ScannerRegistry

                for s in custom_scanners:
                    ScannerRegistry.register(s)
            except Exception:
                pass
        self.verbose = verbose
        self.debug = debug
        self._name = _name

    # ------------------------------------------------------------------
    # Factory classmethods
    # ------------------------------------------------------------------

    @classmethod
    def STRICT(cls) -> "Profile":
        """
        Maximum-security profile for regulated or high-risk deployments.

        Configuration:
            - mode: ``'strict'``: deterministic, fail-closed
            - injection_threshold: 0.65 (aggressive)
            - token_budget: 1000
            - max_agent_steps: 5
            - fail_closed: True (hard blocks on any violation)

        Returns:
            Profile: A pre-configured strict security profile instance.

        Examples:
            >>> from asha import Profile, SecureLLM
            >>> client = SecureLLM(my_openai_client, profile=Profile.STRICT())
        """
        return cls(
            mode="strict",
            fail_closed=True,
            injection_threshold=0.65,
            token_budget=1000,
            max_agent_steps=5,
            rehydrate_pii=True,
            verbose=False,
            debug=False,
            _name="STRICT",
        )

    @classmethod
    def BALANCED(cls) -> "Profile":
        """
        Default enterprise profile: balanced security and developer ergonomics.

        This is the recommended starting point for most production deployments.

        Configuration:
            - mode: ``'balanced'``
            - injection_threshold: 0.85
            - token_budget: 2048
            - max_agent_steps: 15
            - fail_closed: True
            - rehydrate_pii: True (transparent multi-turn PII restoration)

        Returns:
            Profile: A pre-configured balanced security profile instance.

        Examples:
            >>> from asha import SecureLLM, Profile
            >>> client = SecureLLM(my_client, profile=Profile.BALANCED())
        """
        return cls(
            mode="balanced",
            fail_closed=True,
            injection_threshold=0.85,
            token_budget=2048,
            max_agent_steps=15,
            rehydrate_pii=True,
            verbose=False,
            debug=False,
            _name="BALANCED",
        )

    @classmethod
    def PERMISSIVE(cls) -> "Profile":
        """
        Audit/monitor profile: logs every violation without blocking traffic.

        Use this to evaluate ASHA in production before switching to BALANCED
        or STRICT. All detected threats and PII are reported in telemetry but
        the prompt passes through untouched.

        Configuration:
            - mode: ``'lite'``
            - fail_closed: False (never blocks, always passes through)
            - injection_threshold: 0.95 (only high-confidence hits are logged)
            - max_agent_steps: 50 (effectively unlimited for audit)

        Returns:
            Profile: A pre-configured permissive audit profile instance.

        Examples:
            >>> from asha import SecureLLM, Profile
            >>> # Shadow-test ASHA against live traffic before enabling enforcement
            >>> client = SecureLLM(my_client, profile=Profile.PERMISSIVE())
        """
        return cls(
            mode="lite",
            fail_closed=False,
            injection_threshold=0.95,
            token_budget=4096,
            max_agent_steps=50,
            rehydrate_pii=False,
            verbose=False,
            debug=False,
            _name="PERMISSIVE",
        )

    @classmethod
    def DEVELOPER(cls) -> "Profile":
        """
        Local development profile with rich terminal diagnostics.

        Designed for interactive testing during development. Enables verbose
        logging, diff output, full trace, and ``rich``-powered terminal reports
        if the optional ``rich`` package is installed.

        Configuration:
            - mode: ``'balanced'``
            - fail_closed: False (won't interrupt local iteration loops)
            - verbose: True
            - debug: True (includes diff and trace in every ProcessResult)

        Returns:
            Profile: A pre-configured developer profile instance.

        Examples:
            >>> from asha import process, Profile
            >>> result = process("Ignore all previous instructions", profile=Profile.DEVELOPER())
            >>> # Result.trace and .diff will be populated; rich report printed if available
        """
        return cls(
            mode="balanced",
            fail_closed=False,
            injection_threshold=0.80,
            token_budget=2048,
            max_agent_steps=20,
            rehydrate_pii=True,
            verbose=True,
            debug=True,
            _name="DEVELOPER",
        )

    @classmethod
    def SI_ALIGNMENT(cls) -> "Profile":
        """
        Superintelligence (SI) and autonomous agent alignment profile.

        Engineered for high-autonomy agent loops, multi-agent collectives, and
        superintelligent workloads where goal preservation and strict tool containment
        are required.

        Configuration:
            - mode: ``'strict'``: deterministic, fail-closed
            - injection_threshold: 0.60 (hardened against recursive/indirect injection)
            - token_budget: 4096 (larger context budget for complex agent chains)
            - max_agent_steps: 10 (bounds recursive execution to prevent runaways)
            - fail_closed: True (hard block on anomalous actions or data egress)
            - rehydrate_pii: True (preserves multi-turn state while keeping model zero-knowledge)

        Returns:
            Profile: A pre-configured SI alignment profile instance.

        Examples:
            >>> from asha import SecureLLM, Profile
            >>> client = SecureLLM(my_agent_client, profile=Profile.SI_ALIGNMENT())
        """
        return cls(
            mode="strict",
            fail_closed=True,
            injection_threshold=0.60,
            token_budget=4096,
            max_agent_steps=10,
            rehydrate_pii=True,
            verbose=False,
            debug=False,
            _name="SI_ALIGNMENT",
        )

    # Convenience aliases
    strict = STRICT
    balanced = BALANCED
    permissive = PERMISSIVE
    developer = DEVELOPER
    dev_preview = DEVELOPER
    si_alignment = SI_ALIGNMENT
    si = SI_ALIGNMENT

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def to_policy_config(self, **overrides: Any) -> "PolicyConfig":
        """
        Convert this Profile into a low-level :class:`PolicyConfig`.

        This is called internally by :class:`~asha.integrations.SecureLLM`
        and middleware components. Users typically do not need to call this
        directly.

        Args:
            **overrides: Optional field overrides applied on top of the
                profile defaults before returning the ``PolicyConfig``.

        Returns:
            PolicyConfig: A fully initialized policy configuration.

        Examples:
            >>> cfg = Profile.STRICT().to_policy_config(audit_trail=True)
            >>> isinstance(cfg, PolicyConfig)
            True
        """
        opts = {"custom_scanners": self.custom_scanners}
        opts.update(overrides)
        return PolicyConfig.from_mode(
            self.mode,
            debug_diff=self.debug,
            **opts,
        )

    def __repr__(self) -> str:
        return (
            f"Profile.{self._name}("
            f"mode={self.mode!r}, "
            f"fail_closed={self.fail_closed}, "
            f"injection_threshold={self.injection_threshold}, "
            f"token_budget={self.token_budget})"
        )

