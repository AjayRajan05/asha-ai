"""
Regex-based PII Detector - High precision pattern matching
"""

import re
from typing import Callable, Dict, List, Optional, Tuple
from ...stages.base_stage import PIIEntity
from ....security.patterns import compile_pii_patterns, is_example_email


class RegexDetector:
    """
    Regex-based PII detector with high precision pattern matching.

    This detector uses carefully crafted regular expressions to identify
    PII patterns with very high precision and low false positive rate.
    """

    def __init__(self) -> None:
        """Initialize regex detector with comprehensive pattern library."""
        self.patterns = self._compile_patterns()
        self.confidence_weights = self._get_confidence_weights()

    def _compile_patterns(self) -> Dict[str, List[re.Pattern]]:
        """Compile all regex patterns for efficient matching."""
        return compile_pii_patterns()

    def _get_confidence_weights(self) -> Dict[str, float]:
        """Get confidence weights for different PII types."""
        return {
            "email": 0.95,
            "phone": 0.90,
            "ssn": 0.95,
            "credit_card": 0.90,
            "aadhaar": 0.92,
            "pan": 0.88,
            "gstin": 0.90,
            "ip_address": 0.85,
            "url": 0.80,
            "zip_code": 0.75,
            "date_of_birth": 0.70,
            "address": 0.65,
            "name": 0.60,
            "api_key": 0.95,
            "jwt_token": 0.95,
            "bearer_token": 0.90,
        }

    def detect(self, text: str) -> List[PIIEntity]:
        """
        Detect PII entities using regex patterns.

        Args:
            text: Input text to analyze

        Returns:
            List of detected PII entities
        """
        from ....security.checksums import validate_pii_checksum
        from ....text.canonicalize import canonicalize, expand_for_matching

        entities: List[PIIEntity] = []
        scan_texts = [text]
        canon = canonicalize(text)
        if canon and canon not in scan_texts:
            scan_texts.append(canon)
        for view in expand_for_matching(text):
            if view not in scan_texts:
                scan_texts.append(view)

        for scan_text in scan_texts:
            entities.extend(
                self._detect_on_text(scan_text, text, validate_pii_checksum)
            )

        return self._dedupe_entities(entities)

    def _dedupe_entities(self, entities: List[PIIEntity]) -> List[PIIEntity]:
        seen: set[Tuple[str, str]] = set()
        out: List[PIIEntity] = []
        for ent in sorted(entities, key=lambda e: (-e.confidence, e.start)):
            key = (ent.pii_type, ent.text)
            if key in seen:
                continue
            seen.add(key)
            out.append(ent)
        return out

    def _find_flexible_span(self, text: str, literal: str) -> Optional[Tuple[int, int]]:
        if not literal:
            return None
        pat = r"[\s\u200b-\u200d\ufeff\ufe00-\ufe0f]*".join(
            re.escape(c) for c in literal
        )
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.start(), m.end()
        return None

    def _detect_on_text(
        self,
        scan_text: str,
        original_text: str,
        validate_pii_checksum: Callable[[str, str], Tuple[bool, str]],
    ) -> List[PIIEntity]:
        entities = []
        use_original_offsets = scan_text is original_text

        for pii_type, patterns in self.patterns.items():
            for pattern in patterns:
                matches = pattern.finditer(scan_text)

                for match in matches:
                    if pii_type == "email" and not self._validate_email(match.group()):
                        continue
                    if pii_type == "ip_address" and not self._validate_ip_address(
                        match.group()
                    ):
                        continue
                    if pii_type == "phone" and not self._validate_phone(match.group()):
                        continue

                    base_conf = self.confidence_weights.get(pii_type, 0.8)
                    base_conf = self._apply_context_boost(
                        original_text, pii_type, match.start(), match.end(), base_conf
                    )
                    checksum_ok, algo = validate_pii_checksum(pii_type, match.group())
                    if not checksum_ok:
                        base_conf *= 0.45

                    if use_original_offsets:
                        start, end = match.start(), match.end()
                    else:
                        located = self._find_flexible_span(original_text, match.group())
                        if located:
                            start, end = located
                        else:
                            continue

                    entity = PIIEntity(
                        text=match.group(),
                        start=start,
                        end=end,
                        pii_type=pii_type,
                        confidence=base_conf,
                        context=self._get_context(original_text, start, end),
                        metadata={
                            "detector": "regex",
                            "pattern": pattern.pattern,
                            "checksum_algo": algo,
                            "checksum_valid": checksum_ok,
                            "validation_passed": checksum_ok
                            or algo == "none"
                            or pii_type
                            not in ("credit_card", "aadhaar", "pan", "gstin"),
                            "scan_view": (
                                "canonical" if not use_original_offsets else "raw"
                            ),
                        },
                    )
                    entities.append(entity)

        return entities

    _CONTEXT_CUES: Tuple[Tuple[str, str, float], ...] = (
        ("phone", "phone", 0.10),
        ("mobile", "phone", 0.08),
        ("ssn", "ssn", 0.12),
        ("social security", "ssn", 0.12),
        ("aadhaar", "aadhaar", 0.12),
        ("aadhar", "aadhaar", 0.10),
        ("email", "email", 0.10),
        ("credit card", "credit_card", 0.12),
        ("card", "credit_card", 0.08),
        ("pan", "pan", 0.10),
    )

    def _apply_context_boost(
        self,
        text: str,
        pii_type: str,
        match_start: int,
        match_end: int,
        confidence: float,
    ) -> float:
        window = 60
        start = max(0, match_start - window)
        end = min(len(text), match_end + window)
        context = text[start:end].lower()
        boost = 0.0
        for phrase, cue_type, weight in self._CONTEXT_CUES:
            if cue_type == pii_type and phrase in context:
                boost = max(boost, weight)
        return min(1.0, confidence + boost)

    def _validate_match(self, pii_type: str, match_text: str) -> bool:
        """Additional validation for certain PII types (hard gates)."""
        if pii_type == "email":
            return self._validate_email(match_text)
        elif pii_type == "ip_address":
            return self._validate_ip_address(match_text)
        elif pii_type == "phone":
            return self._validate_phone(match_text)
        return True

    def _validate_email(self, email: str) -> bool:
        """Validate email format."""
        if is_example_email(email):
            return False
        # Basic email validation
        if "@" not in email or "." not in email:
            return False

        local, domain = email.split("@", 1)

        # Check local part
        if len(local) < 1 or len(local) > 64:
            return False

        # Check domain part
        if len(domain) < 4 or len(domain) > 253:
            return False

        # Check for valid domain structure
        if "." not in domain:
            return False

        domain_parts = domain.split(".")
        if len(domain_parts) < 2:
            return False

        # Check TLD
        tld = domain_parts[-1]
        if len(tld) < 2:
            return False

        return True

    def _validate_credit_card(self, card_number: str) -> bool:
        """Validate credit card using Luhn algorithm."""
        # Remove non-digits
        digits = re.sub(r"\D", "", card_number)

        # Check length
        if len(digits) not in [13, 14, 15, 16]:
            return False

        # Luhn algorithm
        total = 0
        reverse_digits = digits[::-1]

        for i, digit in enumerate(reverse_digits):
            n = int(digit)
            if i % 2 == 1:  # Every second digit
                n *= 2
                if n > 9:
                    n -= 9
            total += n

        return total % 10 == 0

    def _validate_ip_address(self, ip: str) -> bool:
        """Validate IP address format."""
        parts = ip.split(".")

        if len(parts) != 4:
            return False

        for part in parts:
            try:
                num = int(part)
                if num < 0 or num > 255:
                    return False
            except ValueError:
                return False

        return True

    def _validate_phone(self, phone: str) -> bool:
        """Validate phone number format (US + Indian)."""
        digits = re.sub(r"\D", "", phone)

        # Indian mobile: 10 digits starting 6-9, optional 91 country code.
        if len(digits) == 10 and digits[0] in "6789":
            return True
        if len(digits) == 12 and digits.startswith("91") and digits[2] in "6789":
            return True

        # US-style: 10 or 11 digits with country code
        if len(digits) not in [10, 11]:
            return False

        if len(digits) >= 10:
            area_code = digits[-10:-7] if len(digits) == 11 else digits[:3]
            if area_code.startswith(("0", "1")):
                return False

        return True

    def _get_context(
        self, text: str, start: int, end: int, window_size: int = 50
    ) -> str:
        """Get context around a match."""
        context_start = max(0, start - window_size)
        context_end = min(len(text), end + window_size)
        return text[context_start:context_end]

    def get_pattern_count(self) -> Dict[str, int]:
        """Get count of patterns by type."""
        return {pii_type: len(patterns) for pii_type, patterns in self.patterns.items()}

    def add_custom_pattern(
        self, pii_type: str, pattern: str, confidence: float = 0.8
    ) -> None:
        """Add a custom regex pattern."""
        if pii_type not in self.patterns:
            self.patterns[pii_type] = []

        compiled_pattern = re.compile(pattern, re.IGNORECASE)
        self.patterns[pii_type].append(compiled_pattern)
        self.confidence_weights[pii_type] = confidence
