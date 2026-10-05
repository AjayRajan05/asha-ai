"""
Canonical security execution path.

Used by sanitize(), Processor security step, and privacy fallbacks so all security-only
processing shares one code path.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, List, Optional, Union

from .security_layer import SecurityLayer, SecurityLevel, SecurityResult, ThreatType

logger = logging.getLogger(__name__)


def read_security_field(security_result: Any, field: str, default: Any = None) -> Any:
    """Read a field from SecurityResult dataclass or serialized dict."""
    if security_result is None:
        return default
    if isinstance(security_result, dict):
        return security_result.get(field, default)
    return getattr(security_result, field, default)


def get_sanitized_content(security_result: Any, fallback: str = "") -> str:
    """Return sanitized text from any security result representation."""
    content = read_security_field(security_result, "sanitized_content", fallback)
    return content if content is not None else fallback


def normalize_security_level(
    level: Union[str, SecurityLevel],
    *,
    default: SecurityLevel = SecurityLevel.MEDIUM,
) -> SecurityLevel:
    """Convert config string or enum to SecurityLevel."""
    if isinstance(level, SecurityLevel):
        return level
    try:
        return SecurityLevel[level.upper()]
    except KeyError:
        return default


def _apply_safety_classifier(
    content: str,
    *,
    threat_blocking: bool,
) -> Optional[Any]:
    """Always run rule-based SafetyClassifier.

    Heavy HuggingFace toxic-bert is opt-in via ``ASHA_ENABLE_HF_SAFETY=1``.
    ``ASHA_DISABLE_ML`` no longer skips this path (it only gates spaCy NER).
    """
    import os

    try:
        from ..safety_classifier import SafetyClassifier

        enable_hf = os.environ.get("ASHA_ENABLE_HF_SAFETY", "").lower() in (
            "1",
            "true",
            "yes",
        )
        classifier = SafetyClassifier(enable_ml=enable_hf)
        return classifier.classify_safety(content)
    except ImportError:
        return None
    except Exception:
        return None


def _build_hybrid_masked_entities(pii_entities: List[Any]) -> Dict[str, Any]:
    """Normalize hybrid PII entities to masked_entities dict format."""
    masked_entities: Dict[str, Any] = {}
    for i, entity in enumerate(pii_entities):
        raw_type = (
            getattr(entity, "category", None)
            or getattr(entity, "entity_type", None)
            or getattr(entity, "pii_type", None)
            or "unknown"
        )
        if hasattr(raw_type, "value"):
            entity_type = raw_type.value
        else:
            entity_type = str(raw_type)
        masked_entities[f"entity_{i}"] = {
            "type": entity_type,
            "original_length": len(getattr(entity, "text", "")),
            "severity": getattr(entity, "confidence", 0.8),
        }
    return masked_entities


def _merge_with_safety(
    base: SecurityResult,
    safety_result: Optional[Any],
    *,
    hybrid_pii_used: bool = False,
) -> SecurityResult:
    """Overlay safety classifier metadata onto a SecurityResult."""
    if not safety_result:
        return SecurityResult(
            is_safe=base.is_safe,
            threat_level=base.threat_level,
            detected_threats=base.detected_threats,
            sanitized_content=base.sanitized_content,
            masked_entities=base.masked_entities,
            security_score=base.security_score,
            recommendations=base.recommendations,
            processing_time_ms=base.processing_time_ms,
            hybrid_pii_used=hybrid_pii_used,
            safety_classifier_used=False,
            masking_map=getattr(base, "masking_map", {}) or {},
        )

    extra_threats: List[ThreatType] = list(base.detected_threats)
    safety_threats = getattr(safety_result, "threats", None) or []
    for threat in safety_threats:
        if threat not in extra_threats:
            extra_threats.append(threat)

    threat_level = base.threat_level
    safety_level = getattr(safety_result, "safety_level", None)
    if safety_level is not None and not getattr(safety_result, "is_safe", True):
        try:
            threat_level = SecurityLevel[safety_level.name.upper()]
        except (AttributeError, KeyError):
            pass

    # Never downgrade an unsafe verdict from base
    final_is_safe = bool(base.is_safe and getattr(safety_result, "is_safe", True))
    if not final_is_safe and threat_level == SecurityLevel.LOW:
        if any(
            t in (ThreatType.INJECTION, ThreatType.MALICIOUS_CONTENT, ThreatType.SYSTEM_MANIPULATION)
            for t in extra_threats
        ):
            threat_level = SecurityLevel.HIGH
        else:
            threat_level = SecurityLevel.MEDIUM

    return SecurityResult(
        is_safe=final_is_safe,
        threat_level=threat_level,
        detected_threats=extra_threats,
        sanitized_content=base.sanitized_content,
        masked_entities=base.masked_entities,
        security_score=(
            base.security_score
            if not base.is_safe
            else getattr(safety_result, "confidence_score", base.security_score)
        ),
        recommendations=getattr(safety_result, "recommendations", base.recommendations)
        or base.recommendations,
        processing_time_ms=base.processing_time_ms,
        hybrid_pii_used=hybrid_pii_used,
        safety_classifier_used=True,
        masking_map=getattr(base, "masking_map", {}) or {},
    )


def run_security(
    content: str,
    config: Optional[Dict[str, Any]] = None,
) -> SecurityResult:
    """
    Run the canonical security-only path (PII masking + threat analysis).

    Args:
        content: Text to secure.
        config: Optional settings:
            - security_level: str or SecurityLevel (default MEDIUM)
            - pii_mode: "hybrid" (default) | "lite" (regex-only; alias "rule") | "ml_only"
            - threat_blocking: block unsafe content (default True)
            - debug_enabled: enable hybrid detector debug (default False)

    Returns:
        SecurityResult dataclass.
    """
    config = config or {}
    enable_pii = config.get("enable_pii_detection", True)
    enable_injection = config.get("enable_injection_detection", True)
    pii_masking = config.get("pii_masking", config.get("privacy", True))
    if not pii_masking:
        enable_pii = False

    if not enable_pii and not enable_injection:
        return SecurityResult(
            is_safe=True,
            threat_level=SecurityLevel.LOW,
            detected_threats=[],
            sanitized_content=content,
            masked_entities={},
            security_score=0.0,
            recommendations=[],
            processing_time_ms=0.0,
        )

    security_level = normalize_security_level(
        config.get("security_level", SecurityLevel.MEDIUM)
    )
    from ..ml_utils import validate_pii_mode

    raw_pii_mode = config.get("pii_mode", "hybrid")
    try:
        pii_mode = validate_pii_mode(str(raw_pii_mode))
    except ValueError:
        pii_mode = "hybrid"
    threat_blocking = config.get("threat_blocking", True)
    debug_enabled = config.get("debug_enabled", False)
    # Auto-upgrade: ensemble when hardened deps+artifacts exist; else lite.
    # Explicit injection_mode / ASHA_INJECTION_MODE always wins.
    from .injection_detector import resolve_injection_mode

    raw_injection = config.get("injection_mode", None)
    injection_mode = resolve_injection_mode(
        None if raw_injection is None else str(raw_injection)
    )

    if not enable_pii:
        pii_mode = "lite"

    layer = SecurityLayer(
        security_level,
        reversible_masking=bool(config.get("reversible", False)),
        injection_mode=str(injection_mode),
    )
    hybrid_detector = None

    # lite (and deprecated "rule" alias) → SecurityLayer/PIIDetector path.
    # hybrid / ml_only → layered HybridPIIDetector pipeline.
    if enable_pii and pii_mode not in ("lite",):
        try:
            from ..hybrid_pii import HybridPIIDetector

            hybrid_detector = HybridPIIDetector(
                pii_mode=pii_mode,
                debug_enabled=debug_enabled,
            )
        except ImportError as exc:
            hybrid_detector = None
            msg = (
                f"pii_mode={pii_mode!r} requested but HybridPIIDetector import failed "
                f"({exc}); falling back to regex/lite SecurityLayer path. "
                "Install asha[ml] or asha[hardened] for hybrid PII."
            )
            warnings.warn(msg, UserWarning, stacklevel=2)
            logger.warning(msg)
    if hybrid_detector:
        # Evaluate threat detection on raw unmasked content so PII masking never blinds injection checks
        scored = layer.process(content, mask_pii=False)
        pii_target = scored.sanitized_content if scored.sanitized_content else content
        pii_result = hybrid_detector.detect_and_mask(pii_target)
        masked_entities = _build_hybrid_masked_entities(pii_result.entities)
        reversible = bool(config.get("reversible", False))
        masking_map = dict(pii_result.masking_map or {}) if reversible else {}
        base = SecurityResult(
            is_safe=scored.is_safe,
            threat_level=scored.threat_level,
            detected_threats=list(scored.detected_threats),
            sanitized_content=pii_result.masked_text,
            masked_entities=masked_entities,
            security_score=float(scored.security_score),
            recommendations=list(scored.recommendations),
            processing_time_ms=float(scored.processing_time_ms),
            hybrid_pii_used=True,
            safety_classifier_used=False,
            masking_map=masking_map if reversible else {},
        )
    else:
        base = layer.process(content, mask_pii=enable_pii)

    if enable_injection:
        safety_result = _apply_safety_classifier(
            base.sanitized_content, threat_blocking=threat_blocking
        )
    else:
        safety_result = None

    return _merge_with_safety(
        base,
        safety_result,
        hybrid_pii_used=hybrid_detector is not None,
    )


def run_security_only(
    content: str,
    *,
    security_level: Union[str, SecurityLevel] = SecurityLevel.MEDIUM,
    pii_mode: str = "hybrid",
    threat_blocking: bool = True,
    debug_enabled: bool = False,
    reversible: bool = False,
    injection_mode: Optional[str] = None,
) -> SecurityResult:
    """Convenience wrapper for security-only public APIs (e.g. sanitize()).

    Defaults match :func:`run_security` / ``SecurityLayer`` (MEDIUM + hybrid PII).
    ``sanitize_text`` may still pass ``SecurityLevel.HIGH`` explicitly.
    """
    cfg: Dict[str, Any] = {
        "security_level": security_level,
        "pii_mode": pii_mode,
        "threat_blocking": threat_blocking,
        "debug_enabled": debug_enabled,
        "reversible": reversible,
    }
    if injection_mode is not None:
        cfg["injection_mode"] = injection_mode
    return run_security(content, cfg)
