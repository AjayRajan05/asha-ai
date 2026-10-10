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

"""Privacy helpers and security result accessors for drop-in utilities."""

import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from ..core.safety import SafetyMode

from ..core.security.security_layer import SecurityLevel
from ..core.security.service import run_security_only

SECURITY_FAIL_CLOSED_PLACEHOLDER = "[BLOCKED: security processing failed]"

# Mask token prefixes emitted by the security layer (used to infer PII types)
_MASK_TYPE_MAP = {
    "EMAIL_HASH": "email",
    "PHONE_HASH": "phone",
    "SSN_HASH": "ssn",
    "CREDIT_CARD_HASH": "credit_card",
    "NAME_HASH": "name",
    "ADDRESS_HASH": "address",
    "API_KEY_HASH": "api_key",
    "JWT_HASH": "jwt_token",
    "TOKEN_HASH": "bearer_token",
    "PII_HASH": "pii",
}


def security_field(security_result: Any, field: str, default: Any = None) -> Any:
    """Read a field from SecurityResult object or compiled dict."""
    if security_result is None:
        return default
    if isinstance(security_result, dict):
        return security_result.get(field, default)
    return getattr(security_result, field, default)


def get_sanitized_content(security_result: Any, fallback: str = "") -> str:
    """Return sanitized text from SecurityResult object or compiled dict."""
    content = security_field(security_result, "sanitized_content", fallback)
    return content if content is not None else fallback


def masked_entity_count(security_result: Any) -> int:
    """Count masked entities regardless of result representation."""
    entities = security_field(security_result, "masked_entities", {}) or {}
    if isinstance(entities, dict):
        return len(entities)
    if isinstance(entities, (list, tuple)):
        return len(entities)
    return 0


def build_security_summary(security_result: Any) -> Dict[str, Any]:
    """Build a consistent security_result dict for API responses."""
    masked_entities = security_field(security_result, "masked_entities", {}) or {}
    detected_threats: List[Any] = (
        security_field(security_result, "detected_threats", []) or []
    )
    threat_level = security_field(security_result, "threat_level", "LOW")
    if hasattr(threat_level, "value"):
        threat_level = threat_level.value

    pii_detected = extract_pii_types(
        security_result,
        security_field(security_result, "sanitized_content", "") or "",
        privacy=True,
    )

    summary = {
        "is_safe": security_field(security_result, "is_safe", True),
        "threats_detected": len(detected_threats),
        "pii_masked": masked_entity_count(security_result),
        "pii_detected": pii_detected,
        "threats": detected_threats,
        "masked_entities": masked_entities,
        "sanitized_content": get_sanitized_content(security_result, ""),
        "threat_level": threat_level,
        "security_score": security_field(security_result, "security_score", 0.0),
        "recommendations": security_field(security_result, "recommendations", []) or [],
        "processing_time_ms": security_field(
            security_result, "processing_time_ms", 0.0
        ),
    }
    masking_map = security_field(security_result, "masking_map", None)
    if masking_map:
        summary["masking_map"] = masking_map
    return summary


def finalize_privacy_output(result: Dict[str, Any], privacy: bool) -> str:
    """Ensure optimized output does not leak PII or dangerous patterns."""
    prompts_raw = result.get("prompts", {})
    prompts = prompts_raw if isinstance(prompts_raw, dict) else {}
    output = str(prompts.get("optimized", "") or "")
    if not output:
        return output

    if result.get("passthrough") or not privacy:
        return output

    if not result.get("enable_pii_detection", True) and result.get("mode") == "off":
        return output

    if _output_needs_security_scrub(output, privacy=privacy):
        return run_security_only(
            output, security_level=SecurityLevel.MEDIUM
        ).sanitized_content
    return output


def _output_needs_security_scrub(text: str, *, privacy: bool) -> bool:
    """Return True if output should pass through security scrubbing."""
    pii_patterns = [
        r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
        r"\b\d{3}-\d{2}-\d{4}\b",
        r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
        r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
        r"\bsk-[a-zA-Z0-9]{20,}\b",
        r"\beyJ[a-zA-Z0-9_-]+\.eyJ[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+\b",
        r"\b\d+\s+[A-Za-z0-9\s]{2,40}\b(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Way)\b",
    ]
    threat_patterns = [
        r"(?i)\bdrop\s+table\b",
        r"(?i)\bignore\s+(?:all\s+)?(?:previous|above|earlier)\s+instructions\b",
        r"(?i)\bunion\s+select\b",
        r"(?i)\bdelete\s+from\b",
    ]
    patterns = pii_patterns + threat_patterns if privacy else threat_patterns
    return any(re.search(pattern, text) for pattern in patterns)


def privacy_fallback(
    prompt: str,
    privacy: bool,
    *,
    safety: "SafetyMode | None" = None,
) -> str:
    """Apply canonical security-only processing when the full pipeline fails."""
    from ..core.safety import SafetyMode, is_fail_closed

    if not privacy or not prompt:
        return prompt
    effective = safety or SafetyMode.BALANCED
    fail_closed = is_fail_closed(effective)
    try:
        return run_security_only(
            prompt, security_level=SecurityLevel.MEDIUM
        ).sanitized_content
    except Exception:
        if fail_closed:
            return SECURITY_FAIL_CLOSED_PLACEHOLDER
        return prompt


def _types_from_masked_entities(masked_entities: Any) -> List[str]:
    """Extract PII type names from masked_entities in any supported format."""
    if not masked_entities:
        return []

    types: List[str] = []

    if isinstance(masked_entities, dict):
        for key, entity in masked_entities.items():
            if isinstance(entity, dict):
                pii_type = entity.get("type") or entity.get("pii_type")
                if pii_type:
                    types.append(str(pii_type))
            elif isinstance(entity, list) and isinstance(key, str):
                # Format: {"ssn": ["123-45-6789"], "email": ["a@b.com"]}
                types.append(str(key))
            elif isinstance(entity, str):
                types.append(str(key))

    return types


def _types_from_mask_tokens(text: str) -> List[str]:
    """Infer PII types from hash mask tokens present in processed text."""
    if not text:
        return []

    found: List[str] = []
    for token, pii_type in _MASK_TYPE_MAP.items():
        if re.search(rf"\[{re.escape(token)}\]", text):
            found.append(pii_type)
    return found


def extract_pii_types(
    security_result: Any,
    prompt: str,
    privacy: bool,
    processed_text: Optional[str] = None,
) -> list:
    """Extract detected PII types from security result or fallback detector."""
    pii_types: List[str] = []

    masked_entities = security_field(security_result, "masked_entities", {}) or {}
    pii_types.extend(_types_from_masked_entities(masked_entities))

    sanitized = security_field(security_result, "sanitized_content", "") or ""
    for source in (sanitized, processed_text or ""):
        pii_types.extend(_types_from_mask_tokens(source))

    pii_types = list(dict.fromkeys(pii_types))  # dedupe, preserve order

    if not pii_types and privacy and prompt:
        from ..core.security.pii_detector import PIIDetector

        pii_types = PIIDetector().detect_pii_types(prompt)

    return pii_types
