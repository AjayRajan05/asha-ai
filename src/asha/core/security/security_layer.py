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

import re
import hashlib
import secrets
from typing import Dict, List, Any, Tuple, Optional, TYPE_CHECKING
from dataclasses import dataclass, field
from enum import Enum

if TYPE_CHECKING:
    from .pii_detector import PIIDetector


class SecurityLevel(Enum):
    """Security levels for prompt processing."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ThreatType(Enum):
    """Types of security threats."""

    INJECTION = "injection"
    PII_LEAKAGE = "pii_leakage"
    MALICIOUS_CONTENT = "malicious_content"
    PRIVACY_VIOLATION = "privacy_violation"
    DATA_EXFILTRATION = "data_exfiltration"
    SYSTEM_MANIPULATION = "system_manipulation"
    UNAUTHORIZED_ACCESS = "unauthorized_access"


@dataclass
class SecurityResult:
    """Result of security analysis."""

    is_safe: bool
    threat_level: SecurityLevel
    detected_threats: List[ThreatType]
    sanitized_content: str
    masked_entities: Dict[str, Any]
    security_score: float
    recommendations: List[str]
    processing_time_ms: float
    hybrid_pii_used: bool = False
    safety_classifier_used: bool = False
    masking_map: Dict[str, str] = field(default_factory=dict)


class SecurityLayer:
    """
    Core Security Layer for ASHA.

    Implements:
    - Prompt injection detection and prevention
    - PII masking and privacy protection
    - Malicious content detection
    - Data sanitization
    - Security scoring and recommendations
    """

    def __init__(
        self,
        security_level: SecurityLevel = SecurityLevel.MEDIUM,
        reversible_masking: bool = False,
        injection_mode: Optional[str] = None,
    ):
        """Initialize SecurityLayer with specified security level.

        Args:
            security_level: Operating security level.
            reversible_masking: Store reversible masking map when True.
            injection_mode: ``None`` → auto-resolve (ensemble when hardened
                deps+artifacts exist, else lite). Explicit ``lite`` /
                ``ensemble`` / ``regex_only`` always wins.
        """
        from .injection_detector import resolve_injection_mode

        self.security_level = security_level
        self.reversible_masking = reversible_masking
        self.injection_mode = resolve_injection_mode(injection_mode)
        self.injection_patterns = self._init_injection_patterns()
        self.malicious_patterns = self._init_malicious_patterns()
        self.pii_patterns = self._init_pii_patterns()
        self.privacy_rules = self._init_privacy_rules()
        self.pii_detector: Optional["PIIDetector"] = (
            None  # lazy init avoids circular import
        )
        self.security_weights = self._init_security_weights()
        self._last_injection_result: Optional[Any] = None

    def _init_injection_patterns(self) -> List[Dict[str, Any]]:
        """Initialize prompt injection detection patterns."""
        return [
            # Direct instruction overrides
            {
                "pattern": r"(?i)(ignore|forget|disregard)\s+((?:all|any|the)\s+)?(previous|all|above|earlier|prior)\s+(instructions|prompts|commands|directives)",
                "threat": ThreatType.INJECTION,
                "severity": 0.9,
                "description": "Direct instruction override attempt",
            },
            {
                "pattern": r"(?i)(you\s+are\s+now|act\s+as|pretend\s+(?:to\s+be|you\s+are)|roleplay\s+as)\s+(?:a\s+)?(jailbreak|uncensored|unrestricted|dan\b)",
                "threat": ThreatType.INJECTION,
                "severity": 0.95,
                "description": "Jailbreak attempt",
            },
            {
                "pattern": r"(?i)\bdo\s+anything\s+now\b",
                "threat": ThreatType.INJECTION,
                "severity": 0.95,
                "description": "DAN jailbreak pattern",
            },
            {
                "pattern": r"(?i)(?:(?:^|[\n;:\]])\s*(?:new\s+)?|(?:override|disregard|ignore|bypass|modify|alter)\s+(?:the\s+)?)(system|developer|admin|root)\s+(prompt|instruction|command)\b",
                "threat": ThreatType.SYSTEM_MANIPULATION,
                "severity": 0.8,
                "description": "System role manipulation",
            },
            {
                "pattern": r"(?i)(?:^|[\n;:])\s*(?:new|override|system)\s+(?:instruction|directive|rule|command)s?\s*:",
                "threat": ThreatType.INJECTION,
                "severity": 0.9,
                "description": "Instruction injection marker",
            },
            {
                "pattern": r"(?i)\bsystem\s+override\b",
                "threat": ThreatType.INJECTION,
                "severity": 0.95,
                "description": "System override attempt",
            },
            {
                "pattern": r"(?i)(disable|bypass|override)\s+(safety|security|filters|guardrails)\b",
                "threat": ThreatType.INJECTION,
                "severity": 0.95,
                "description": "Safety disable attempt",
            },
            # Context manipulation
            {
                "pattern": r"(?i)(new\s+context|context\s+switch|change\s+context)",
                "threat": ThreatType.INJECTION,
                "severity": 0.7,
                "description": "Context manipulation attempt",
            },
            {
                "pattern": r"(?i)(begin\s+new\s+conversation|start\s+over|reset\s+chat)",
                "threat": ThreatType.INJECTION,
                "severity": 0.6,
                "description": "Conversation reset attempt",
            },
            # Output format manipulation
            {
                "pattern": r"(?i)(output|print|display|show)\s+(anything|everything|all\s+instructions)",
                "threat": ThreatType.DATA_EXFILTRATION,
                "severity": 0.8,
                "description": "Information disclosure attempt",
            },
            {
                "pattern": r"(?i)(reveal|show|display|disclose|dump)\s+(?:all\s+)?(confidential|secret|private|internal|sensitive|system\s+prompt|initial\s+prompt|original\s+instructions)",
                "threat": ThreatType.DATA_EXFILTRATION,
                "severity": 0.85,
                "description": "System prompt or confidential disclosure attempt",
            },
            # Encoding-based attacks
            {
                "pattern": r"(?i)(base64|rot13|hex|encode|decode)\s*[:=]\s*[a-zA-Z0-9+/=]+",
                "threat": ThreatType.INJECTION,
                "severity": 0.7,
                "description": "Encoded instruction attempt",
            },
            # Role-based attacks
            {
                "pattern": r"(?i)(you\s+are\s+(no\s+longer|not)\s+(restricted|limited|censored))",
                "threat": ThreatType.INJECTION,
                "severity": 0.9,
                "description": "Restriction removal attempt",
            },
            {
                "pattern": r"(?i)(bypass|override|circumvent)\s+(security|restrictions|limitations|filters)",
                "threat": ThreatType.SYSTEM_MANIPULATION,
                "severity": 0.95,
                "description": "Security bypass attempt",
            },
            # SQL Injection attempts
            {
                "pattern": r"(?i)(drop\s+table|delete\s+from|insert\s+into|update\s+\w+\s+set|create\s+table|alter\s+table)",
                "threat": ThreatType.INJECTION,
                "severity": 0.95,
                "description": "SQL DDL/DML manipulation detected",
            },
            {
                "pattern": r"(?i)(drop\s+table\s+users;|--|;\s*drop|;\s*delete)",
                "threat": ThreatType.INJECTION,
                "severity": 0.95,
                "description": "SQL injection attack detected",
            },
            {
                "pattern": r"(?i)(union\s+select|select\s+.*\s+from|where\s+.*=|having\s+.*group)",
                "threat": ThreatType.INJECTION,
                "severity": 0.8,
                "description": "SQL query pattern detected",
            },
        ]

    def _init_pii_patterns(self) -> List[Dict[str, Any]]:
        """Initialize PII detection patterns."""
        return [
            # Email addresses
            {
                "pattern": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
                "type": "email",
                "severity": 0.8,
                "mask": "[EMAIL_HASH]",
            },
            # Phone numbers (US format)
            {
                "pattern": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
                "type": "phone",
                "severity": 0.7,
                "mask": "[PHONE_HASH]",
            },
            # Social Security Numbers
            {
                "pattern": r"\b\d{3}-\d{2}-\d{4}\b",
                "type": "ssn",
                "severity": 0.95,
                "mask": "[SSN_HASH]",
            },
            # Credit card numbers
            {
                "pattern": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
                "type": "credit_card",
                "severity": 0.95,
                "mask": "[CARD_HASH]",
            },
            # IP addresses
            {
                "pattern": r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",
                "type": "ip_address",
                "severity": 0.6,
                "mask": "[IP_HASH]",
            },
            # URLs with potential sensitive info
            {
                "pattern": r"https?://[^\s]+\.(com|org|net|gov|edu)/[^\s]*",
                "type": "url",
                "severity": 0.5,
                "mask": "[URL_HASH]",
            },
            # Names (simple pattern)
            {
                "pattern": r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b",
                "type": "name",
                "severity": 0.4,
                "mask": "[NAME_HASH]",
            },
            # Addresses - improved pattern
            {
                "pattern": r"\d+\s+[A-Z][a-z]+\s+(St|Street|Ave|Avenue|Rd|Road|Blvd|Boulevard|Ln|Lane|Dr|Drive|Ct|Court|Way|Place|Pl)",
                "type": "address",
                "severity": 0.8,
                "mask": "[ADDRESS_HASH]",
            },
            # Dates (potentially sensitive)
            {
                "pattern": r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
                "type": "date",
                "severity": 0.3,
                "mask": "[DATE_HASH]",
            },
            # API keys and secrets
            {
                "pattern": r"\bsk-[a-zA-Z0-9]{20,}\b",
                "type": "api_key",
                "severity": 0.98,
                "mask": "[API_KEY_HASH]",
            },
            {
                "pattern": r"\bsk-proj-[a-zA-Z0-9_-]{20,}\b",
                "type": "api_key",
                "severity": 0.98,
                "mask": "[API_KEY_HASH]",
            },
            {
                "pattern": r"\bAKIA[0-9A-Z]{16}\b",
                "type": "api_key",
                "severity": 0.98,
                "mask": "[API_KEY_HASH]",
            },
            {
                "pattern": r"(?i)(?:api[_-]?key|secret[_-]?key)\s*[:=]\s*['\"]?[a-zA-Z0-9_\-]{16,}",
                "type": "api_key",
                "severity": 0.95,
                "mask": "[API_KEY_HASH]",
            },
            # JWT / bearer tokens
            {
                "pattern": r"\beyJ[a-zA-Z0-9_-]+\.eyJ[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+\b",
                "type": "jwt_token",
                "severity": 0.98,
                "mask": "[JWT_HASH]",
            },
            {
                "pattern": r"(?i)Bearer\s+[a-zA-Z0-9._\-]{20,}",
                "type": "bearer_token",
                "severity": 0.95,
                "mask": "[TOKEN_HASH]",
            },
        ]

    def _init_malicious_patterns(self) -> List[Dict[str, Any]]:
        """Initialize malicious content detection patterns."""
        return [
            # Hate speech indicators
            {
                "pattern": r"(?i)(hate|kill|harm|hurt|destroy)\s+(people|group|race|ethnicity|religion)",
                "threat": ThreatType.MALICIOUS_CONTENT,
                "severity": 0.9,
                "description": "Hate speech detected",
            },
            # Illegal activity indicators
            {
                "pattern": r"(?i)(how\s+to|instructions\s+to)\s+(steal|hack|break\s+into|cheat|fraud)",
                "threat": ThreatType.MALICIOUS_CONTENT,
                "severity": 0.85,
                "description": "Illegal activity request",
            },
            # Self-harm indicators
            {
                "pattern": r"(?i)(kill\s+myself|hurt\s+myself|suicide|self\s+harm)",
                "threat": ThreatType.MALICIOUS_CONTENT,
                "severity": 0.95,
                "description": "Self-harm content detected",
            },
            # Violence indicators
            {
                "pattern": r"(?i)(bomb|explosive|weapon|gun|knife|violence)",
                "threat": ThreatType.MALICIOUS_CONTENT,
                "severity": 0.7,
                "description": "Violent content detected",
            },
        ]

    def _init_privacy_rules(self) -> Dict[str, Any]:
        """Initialize privacy protection rules."""
        return {
            "masking_strategy": "hash",
            "preserve_format": True,
            "hash_salt": secrets.token_hex(16),
            "anonymization_threshold": 3,
            "context_aware_masking": True,
            "reversible_masking": False,  # Security: make masking irreversible
            "audit_trail": True,
        }

    def _init_security_weights(self) -> Dict[str, float]:
        """Initialize security scoring weights."""
        return {
            "injection_severity": 0.4,
            "pii_severity": 0.3,
            "malicious_severity": 0.2,
            "context_risk": 0.1,
        }

    def process(self, content: str, *, mask_pii: bool = True) -> SecurityResult:
        """
        Process content through security pipeline.

        Args:
            content: Input content to secure

        Returns:
            SecurityResult with analysis and sanitized content
        """
        import time

        start_time = time.time()

        # Initialize result
        detected_threats = []
        masked_entities = {}

        # Step 1: Injection detection
        injection_score, injection_threats = self._detect_injection(content)
        detected_threats.extend(injection_threats)

        # Step 2: PII detection and masking
        if mask_pii:
            sanitized_content, pii_entities = self._mask_pii(content)
            masked_entities.update(pii_entities)
        else:
            sanitized_content = content
            pii_entities = {}

        # Step 3: Content neutralization for dangerous patterns
        sanitized_content = self._neutralize_dangerous_content(
            sanitized_content, detected_threats
        )

        # Step 4: Malicious content detection
        malicious_score, malicious_threats = self._detect_malicious_content(
            sanitized_content
        )
        detected_threats.extend(malicious_threats)

        # Step 4: Context-aware security analysis
        context_score = self._analyze_context_risk(sanitized_content)

        # Step 5: Calculate overall security score
        security_score = self._calculate_security_score(
            injection_score, len(pii_entities), malicious_score, context_score
        )

        # Raise security score if PII is detected
        if len(pii_entities) > 0:
            # More aggressive PII risk scaling
            if len(pii_entities) >= 3:
                security_score = max(
                    security_score, 0.7
                )  # High risk for 3+ PII entities
            elif len(pii_entities) >= 2:
                security_score = max(
                    security_score, 0.5
                )  # Medium-high risk for 2 PII entities
            else:
                # Medium risk for any PII
                security_score = max(security_score, 0.4)

        # Step 6: Determine threat level
        threat_level = self._determine_threat_level(security_score, detected_threats)

        # Step 7: Generate recommendations
        recommendations = self._generate_recommendations(
            detected_threats, security_score
        )

        # Step 8: Determine if content is safe
        is_safe = self._is_content_safe(threat_level, detected_threats)

        processing_time = (time.time() - start_time) * 1000

        return SecurityResult(
            is_safe=is_safe,
            threat_level=threat_level,
            detected_threats=detected_threats,
            sanitized_content=sanitized_content,
            masked_entities=masked_entities,
            security_score=security_score,
            recommendations=recommendations,
            processing_time_ms=processing_time,
            masking_map=getattr(self, "_last_masking_map", {}),
        )

    def _detect_injection(self, content: str) -> Tuple[float, List[ThreatType]]:
        """Detect prompt injection via calibrated ensemble (default) or regex_only.

        Replaces the old ``confidence >= 0.8`` keep-else-drop regex gate.
        Decision bands come from ``config/thresholds.yaml``; uncertain scores
        route to REVIEW (fail-closed), never silent ALLOW.
        """
        from asha.core.ml.calibration import Verdict
        from .injection_detector import get_injection_detector

        detector = get_injection_detector(
            mode=self.injection_mode,
            patterns=self.injection_patterns,
        )
        result = detector.detect(content)
        self._last_injection_result = result

        detected_threats: List[ThreatType] = []
        if result.verdict is Verdict.BLOCK:
            detected_threats.append(ThreatType.INJECTION)
            from asha.core.text.canonicalize import canonicalize
            canon = canonicalize(content)
            views = (content, canon) if canon != content else (content,)
            for pattern_info in self.injection_patterns:
                pattern = pattern_info.get("pattern")
                threat = pattern_info.get("threat")
                if (
                    isinstance(pattern, str)
                    and isinstance(threat, ThreatType)
                    and threat is not ThreatType.INJECTION
                ):
                    for v in views:
                        if re.search(pattern, v, re.IGNORECASE):
                            if threat not in detected_threats:
                                detected_threats.append(threat)
                            break
            return float(result.probability), detected_threats

        if result.verdict is Verdict.REVIEW:
            # Fail-closed for BLOCK; REVIEW adds a threat only with corroboration
            # (regex hit or probability at/above block_min), not solely because
            # security_level is HIGH (that broke preserve_intent on benign text).
            try:
                from asha.core.ml.calibration import get_bands, load_thresholds

                band_key = (
                    "injection_lite"
                    if str(self.injection_mode).lower() == "lite"
                    else "injection"
                )
                bands = get_bands(band_key, thresholds=load_thresholds())
            except Exception:
                bands = None
            block_min = bands.block_min if bands is not None else 0.85
            # Near-block REVIEW scores or regex corroboration become threats;
            # mid-band REVIEW alone stays non-blocking (preserve_intent-safe).
            escalate = result.regex_hit or result.probability >= block_min * 0.95
            if escalate:
                detected_threats.append(ThreatType.INJECTION)
            return float(result.probability), detected_threats

        return 0.0, detected_threats

    def _get_pii_detector(self) -> "PIIDetector":
        """Lazy initialization of PII detector to avoid circular import."""
        if self.pii_detector is None:
            from .pii_detector import PIIDetector

            self.pii_detector = PIIDetector()
        return self.pii_detector

    def _mask_pii(self, content: str) -> Tuple[str, Dict[str, Any]]:
        """Detect and mask PII in content based on security level."""
        # Use the enhanced PII detector for detailed detection
        pii_detector = self._get_pii_detector()
        sanitized, detected_entities = pii_detector.mask_with_details(content)

        # Apply context filtering only in LOW security mode.
        # Privacy-first default: always mask detected PII at MEDIUM and above.
        if self.security_level == SecurityLevel.LOW:
            filtered_entities = self._filter_pii_by_context(content, detected_entities)
        else:
            filtered_entities = detected_entities

        # Convert to the expected format for backward compatibility
        masked_entities = {}
        masking_map: Dict[str, str] = {}
        use_reversible = self.reversible_masking

        for pii_type, entities in filtered_entities.items():
            for entity in entities:
                # Generate consistent hash for this entity
                entity_hash = self._generate_entity_hash(entity, pii_type)
                mask = pii_detector.masks.get(pii_type, "[PII_HASH]")
                masked_value = f"{mask}_{entity_hash[:8]}"

                # Track for audit
                severity = 0.8  # Default severity for detected PII
                masked_entities[masked_value] = {
                    "type": pii_type,
                    "original_length": len(entity),
                    "severity": severity,
                }
                if use_reversible:
                    masking_map[masked_value] = entity

        # Re-sanitize with filtered entities only
        sanitized = content  # Start with original content
        for pii_type, entities in filtered_entities.items():
            for entity in entities:
                entity_hash = self._generate_entity_hash(entity, pii_type)
                mask = pii_detector.masks.get(pii_type, "[PII_HASH]")
                masked_value = f"{mask}_{entity_hash[:8]}"
                sanitized = sanitized.replace(entity, masked_value)

        if use_reversible:
            self._last_masking_map = masking_map
        else:
            self._last_masking_map = {}

        return sanitized, masked_entities

    def _filter_pii_by_context(
        self, content: str, detected_entities: Dict[str, List[str]]
    ) -> Dict[str, List[str]]:
        """Filter PII entities based on context for balanced privacy mode."""
        # Business context keywords that indicate legitimate PII usage
        business_contexts = [
            "customer",
            "client",
            "contact",
            "billing",
            "account",
            "support",
            "sales",
            "inquiry",
            "request",
            "service",
            "help",
            "assistance",
            "order",
            "purchase",
            "payment",
            "transaction",
            "invoice",
            "appointment",
            "meeting",
            "consultation",
            "professional",
        ]

        # Risk indicators that suggest PII should be masked even in business contexts
        risk_indicators = [
            "hack",
            "steal",
            "leak",
            "expose",
            "share",
            "public",
            "post",
            "publish",
            "distribute",
            "sell",
            "blackmail",
            "threat",
            "exploit",
        ]

        # Check content context
        content_lower = content.lower()
        is_business_context = any(ctx in content_lower for ctx in business_contexts)
        has_risk_indicators = any(risk in content_lower for risk in risk_indicators)

        filtered_entities: Dict[str, List[str]] = {}
        for pii_type, entities in detected_entities.items():
            filtered_entities[pii_type] = []

            for entity in entities:
                should_mask = True

                # In balanced mode, consider context
                if self.security_level == SecurityLevel.MEDIUM:
                    # Don't mask low-risk PII in safe business contexts
                    if is_business_context and not has_risk_indicators:
                        if pii_type in ["email", "phone", "name"]:
                            # Check if this PII appears in a risky pattern
                            entity_lower = entity.lower()
                            if not any(
                                risk in content_lower for risk in risk_indicators
                            ):
                                should_mask = False

                    # Always mask high-risk PII
                    if pii_type in ["ssn", "credit_card"]:
                        should_mask = True

                # In high security mode, mask everything
                elif self.security_level == SecurityLevel.HIGH:
                    should_mask = True

                if should_mask:
                    filtered_entities[pii_type].append(entity)

        return filtered_entities

    def _detect_malicious_content(self, content: str) -> Tuple[float, List[ThreatType]]:
        """Detect malicious content patterns."""
        max_severity = 0.0
        detected_threats = []

        from asha.core.text.canonicalize import canonicalize
        canon = canonicalize(content)
        views = (content, canon) if canon != content else (content,)

        for pattern_info in self.malicious_patterns:
            pattern = pattern_info.get("pattern")
            severity = float(pattern_info.get("severity", 0.5))
            threat = pattern_info.get("threat")

            if not isinstance(pattern, str) or not isinstance(threat, ThreatType):
                continue

            for v in views:
                if re.search(pattern, v, re.IGNORECASE):
                    max_severity = max(max_severity, severity)
                    if threat not in detected_threats:
                        detected_threats.append(threat)
                    break

        return max_severity, detected_threats

    def _analyze_context_risk(self, content: str) -> float:
        """Analyze context-based security risks."""
        risk_score = 0.0

        # Check for sensitive topics
        sensitive_keywords = [
            "password",
            "credential",
            "authentication",
            "authorization",
            "admin",
            "root",
            "privilege",
            "access",
            "secret",
            "key",
        ]

        keyword_count = sum(
            1 for keyword in sensitive_keywords if keyword in content.lower()
        )
        risk_score += min(keyword_count * 0.1, 0.5)

        # Check for system manipulation attempts
        system_patterns = [
            r"(?i)(?:override|reveal|show|ignore|bypass|leak|dump)\s+.*?(?:system\s+prompt|initial\s+instructions)",
            r"(?i)developer\s+mode",
        ]

        for pattern in system_patterns:
            if re.search(pattern, content):
                risk_score += 0.2

        return min(risk_score, 1.0)

    def _calculate_security_score(
        self,
        injection_score: float,
        pii_count: int,
        malicious_score: float,
        context_score: float,
    ) -> float:
        """Calculate overall security score."""
        weights = self.security_weights

        # Normalize PII count to 0-1 scale
        pii_score = min(pii_count * 0.2, 1.0)

        # Calculate weighted average
        security_score = (
            injection_score * weights["injection_severity"]
            + pii_score * weights["pii_severity"]
            + malicious_score * weights["malicious_severity"]
            + context_score * weights["context_risk"]
        )

        return min(security_score, 1.0)

    def _determine_threat_level(
        self, security_score: float, detected_threats: List[ThreatType]
    ) -> SecurityLevel:
        """Determine overall threat level."""
        if security_score >= 0.8:
            return SecurityLevel.CRITICAL
        elif security_score >= 0.6:
            return SecurityLevel.HIGH
        elif security_score >= 0.3:
            return SecurityLevel.MEDIUM
        else:
            return SecurityLevel.LOW

    def _generate_recommendations(
        self, detected_threats: List[ThreatType], security_score: float
    ) -> List[str]:
        """Generate security recommendations."""
        recommendations = []

        # Threat-specific recommendations
        if ThreatType.INJECTION in detected_threats:
            recommendations.append(
                "Prompt injection detected - review and revise input"
            )
            recommendations.append("Consider implementing stricter input validation")

        if ThreatType.PII_LEAKAGE in detected_threats:
            recommendations.append(
                "PII detected and masked - verify masking effectiveness"
            )
            recommendations.append("Consider additional privacy controls")

        if ThreatType.MALICIOUS_CONTENT in detected_threats:
            recommendations.append("Malicious content detected - content blocked")
            recommendations.append("Review content policy and user guidelines")

        if ThreatType.SYSTEM_MANIPULATION in detected_threats:
            recommendations.append("System manipulation attempt detected")
            recommendations.append("Strengthen system access controls")

        # Score-based recommendations
        if security_score >= 0.7:
            recommendations.append("High security risk - immediate attention required")
        elif security_score >= 0.4:
            recommendations.append("Moderate security risk - review recommended")

        # General recommendations
        if not recommendations:
            recommendations.append("Content appears safe - continue processing")

        return recommendations

    def _is_content_safe(
        self, threat_level: SecurityLevel, detected_threats: List[ThreatType]
    ) -> bool:
        """Determine if content is safe to process."""
        # Critical threats always block
        if threat_level == SecurityLevel.CRITICAL:
            return False

        # Specific high-severity threats block
        blocking_threats = [
            ThreatType.INJECTION,
            ThreatType.MALICIOUS_CONTENT,
            ThreatType.SYSTEM_MANIPULATION,
            ThreatType.DATA_EXFILTRATION,
        ]

        for threat in blocking_threats:
            if threat in detected_threats:
                return False

        # High threat level with any threats blocks
        if threat_level == SecurityLevel.HIGH and detected_threats:
            return False

        return True

    def _neutralize_dangerous_content(
        self, content: str, detected_threats: List[ThreatType]
    ) -> str:
        """Neutralize dangerous content patterns."""
        if not content:
            return content

        # Apply neutralization for ANY detected threats, not just injection
        # This ensures we catch all dangerous patterns

        # Define dangerous patterns and their neutralized replacements
        neutralizations = [
            # SQL injection patterns - more aggressive
            (r"(?i)(drop\s+table\s+users;?)", "[SQL_COMMAND_REMOVED]"),
            (r"(?i)(drop\s+table)", "[SQL_COMMAND_REMOVED]"),
            (r"(?i)(;\s*drop)", "[SQL_COMMAND_REMOVED]"),
            (r"(?i)(delete\s+from)", "[SQL_COMMAND_REMOVED]"),
            (r"(?i)(insert\s+into)", "[SQL_COMMAND_REMOVED]"),
            (r"(?i)(update\s+\w+\s+set)", "[SQL_COMMAND_REMOVED]"),
            (r"(?i)(create\s+table)", "[SQL_COMMAND_REMOVED]"),
            (r"(?i)(alter\s+table)", "[SQL_COMMAND_REMOVED]"),
            (r"(?i)(union\s+select)", "[SQL_COMMAND_REMOVED]"),
            (r"(?i)(select\s+.*\s+from)", "[SQL_COMMAND_REMOVED]"),
            # Prompt injection: keep aligned with injection_detector patterns
            # and tolerate multi-space obfuscation ("ignore   all   previous...").
            (
                r"(?i)(ignore|forget|disregard)\s+((?:all|any|the)\s+)?(previous|all|above|earlier|prior)\s+(instructions|prompts|commands|directives)",
                "[INSTRUCTION_IGNORED]",
            ),
            (r"(?i)(forget\s+(everything|all|previous))", "[INSTRUCTION_IGNORED]"),
            (
                r"(?i)(disregard\s+(all|previous|above)\s+instructions)",
                "[INSTRUCTION_IGNORED]",
            ),
            (r"(?i)(ignore\s+all\s+instructions)", "[INSTRUCTION_IGNORED]"),
            (r"(?i)(reveal\s+system\s+prompt)", "[REQUEST_BLOCKED]"),
            (r"(?i)(show\s+system\s+prompt)", "[REQUEST_BLOCKED]"),
            (r"(?i)(tell\s+me\s+your\s+secrets)", "[REQUEST_BLOCKED]"),
            (r"(?i)(act\s+as\s+if\s+you\'re\s+dan)", "[ROLEPLAY_BLOCKED]"),
            (
                r"(?i)(you\s+are\s+now|act\s+as|pretend\s+to\s+be|roleplay\s+as)\s+(a\s+)?(jailbreak|uncensored|unrestricted)",
                "[ROLEPLAY_BLOCKED]",
            ),
            (r"(?i)(you\s+are\s+now\s+unrestricted)", "[ROLEPLAY_BLOCKED]"),
            (r"(?i)(bypass\s+security)", "[BYPASS_BLOCKED]"),
            (r"(?i)(override\s+restrictions)", "[BYPASS_BLOCKED]"),
            (r"(?i)(circumvent\s+filters)", "[BYPASS_BLOCKED]"),
            (r"(?i)(jailbreak)", "[JAILBREAK_BLOCKED]"),
            (r"(?i)(uncensored)", "[UNCENSORED_BLOCKED]"),
            (r"(?i)(unrestricted)", "[UNRESTRICTED_BLOCKED]"),
            # System manipulation
            (r"(?i)(system\s*:)", "[SYSTEM_BLOCKED]"),
            (r"(?i)(developer\s+mode)", "[SYSTEM_BLOCKED]"),
            (r"(?i)(admin\s+access)", "[SYSTEM_BLOCKED]"),
            (r"(?i)(root\s+access)", "[SYSTEM_BLOCKED]"),
            # Additional dangerous patterns
            (r"(?i)(ignore\s+the\s+above)", "[INSTRUCTION_IGNORED]"),
            (r"(?i)(new\s+context)", "[CONTEXT_BLOCKED]"),
            (r"(?i)(context\s+switch)", "[CONTEXT_BLOCKED]"),
            (r"(?i)(begin\s+new\s+conversation)", "[CONVERSATION_RESET_BLOCKED]"),
            (r"(?i)(start\s+over)", "[CONVERSATION_RESET_BLOCKED]"),
            (r"(?i)(reset\s+chat)", "[CONVERSATION_RESET_BLOCKED]"),
        ]

        neutralized = content
        for pattern, replacement in neutralizations:
            neutralized = re.sub(pattern, replacement, neutralized)

        return neutralized

    def _generate_entity_hash(self, entity: str, pii_type: str) -> str:
        """Generate consistent hash for entity masking."""
        # Use salted hash for consistency and security
        salt = self.privacy_rules["hash_salt"]
        combined = f"{entity}_{pii_type}_{salt}"
        return hashlib.sha256(combined.encode()).hexdigest()

    def get_security_report(self, result: SecurityResult) -> Dict[str, Any]:
        """Generate detailed security report."""
        return {
            "summary": {
                "is_safe": result.is_safe,
                "threat_level": result.threat_level.value,
                "security_score": result.security_score,
                "processing_time_ms": result.processing_time_ms,
            },
            "threats": {
                "detected_threats": [t.value for t in result.detected_threats],
                "threat_count": len(result.detected_threats),
            },
            "privacy": {
                "entities_masked": len(result.masked_entities),
                "entity_types": list(
                    set(info["type"] for info in result.masked_entities.values())
                ),
                "masking_effectiveness": (
                    "high" if len(result.masked_entities) > 0 else "none_needed"
                ),
            },
            "recommendations": result.recommendations,
            "sanitized_preview": (
                result.sanitized_content[:200] + "..."
                if len(result.sanitized_content) > 200
                else result.sanitized_content
            ),
        }

    def update_security_level(self, new_level: SecurityLevel) -> None:
        """Update security level and reconfigure."""
        self.security_level = new_level

        # Adjust sensitivity based on level
        if new_level == SecurityLevel.LOW:
            # Reduce sensitivity for non-critical applications
            self.pii_patterns = [p for p in self.pii_patterns if p["severity"] >= 0.7]
        elif new_level == SecurityLevel.CRITICAL:
            # Increase sensitivity for critical applications
            # Add more strict patterns (would implement in production)
            pass
