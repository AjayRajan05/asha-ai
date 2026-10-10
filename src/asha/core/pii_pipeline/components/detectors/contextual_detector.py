"""
Contextual PII Detector - Context-aware PII detection
"""

import re
from typing import List, Dict, Any
from ...stages.base_stage import PIIEntity


class ContextualDetector:
    """
    Contextual PII detector using context-aware rules and patterns.

    This detector uses contextual analysis to identify PII that might
    not match exact patterns but is clearly PII based on surrounding context.
    """

    def __init__(self) -> None:
        """Initialize contextual detector with context rules."""
        self.contextual_rules = self._load_contextual_rules()
        self.confidence_weights = self._get_confidence_weights()

    def _load_contextual_rules(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load contextual detection rules."""
        return {
            "email": [
                {
                    "indicators": [
                        "my email",
                        "email me",
                        "contact me at",
                        "reach me at",
                        "my address",
                    ],
                    "boosters": ["personal", "private", "confidential", "direct"],
                    "reducers": [
                        "example",
                        "demo",
                        "test",
                        "sample",
                        "fake",
                        "placeholder",
                    ],
                    "patterns": [
                        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
                    ],
                    "confidence_boost": 0.15,
                    "confidence_reduction": 0.25,
                }
            ],
            "phone": [
                {
                    "indicators": [
                        "my phone",
                        "call me",
                        "my number",
                        "contact me",
                        "reach me",
                    ],
                    "boosters": ["personal", "private", "mobile", "cell", "direct"],
                    "reducers": ["example", "demo", "test", "sample", "fake"],
                    "patterns": [r"\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b"],
                    "confidence_boost": 0.15,
                    "confidence_reduction": 0.20,
                }
            ],
            "name": [
                {
                    "indicators": [
                        "my name is",
                        "i am",
                        "call me",
                        "known as",
                        "people call me",
                    ],
                    "boosters": ["real name", "actual name", "full name"],
                    "reducers": [
                        "example",
                        "demo",
                        "test",
                        "sample",
                        "placeholder",
                        "fake",
                    ],
                    "patterns": [r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b"],
                    "confidence_boost": 0.10,
                    "confidence_reduction": 0.20,
                }
            ],
            "address": [
                {
                    "indicators": [
                        "my address",
                        "live at",
                        "home address",
                        "my home",
                        "residence",
                    ],
                    "boosters": ["personal", "private", "home", "residential"],
                    "reducers": ["example", "demo", "test", "sample", "fake"],
                    "patterns": [
                        r"\d+\s+[A-Z][a-z]+\s+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd)\b"
                    ],
                    "confidence_boost": 0.10,
                    "confidence_reduction": 0.20,
                }
            ],
            "ssn": [
                {
                    "indicators": [
                        "social security",
                        "ssn",
                        "my ssn",
                        "tax id",
                        "identification",
                    ],
                    "boosters": ["personal", "private", "confidential"],
                    "reducers": ["example", "demo", "test", "sample", "fake"],
                    "patterns": [r"\b\d{3}-\d{2}-\d{4}\b"],
                    "confidence_boost": 0.20,
                    "confidence_reduction": 0.30,
                }
            ],
            "credit_card": [
                {
                    "indicators": [
                        "credit card",
                        "card number",
                        "payment method",
                        "my card",
                    ],
                    "boosters": ["personal", "private", "confidential"],
                    "reducers": [
                        "example",
                        "demo",
                        "test",
                        "sample",
                        "fake",
                        "test card",
                    ],
                    "patterns": [r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"],
                    "confidence_boost": 0.15,
                    "confidence_reduction": 0.25,
                }
            ],
        }

    def _get_confidence_weights(self) -> Dict[str, float]:
        """Get base confidence weights for different PII types."""
        return {
            "email": 0.80,
            "phone": 0.75,
            "name": 0.70,
            "address": 0.65,
            "ssn": 0.85,
            "credit_card": 0.80,
        }

    def detect(self, text: str) -> List[PIIEntity]:
        """
        Detect PII entities using contextual analysis.

        Args:
            text: Input text to analyze

        Returns:
            List of detected PII entities
        """
        entities = []
        text_lower = text.lower()

        for pii_type, rules in self.contextual_rules.items():
            for rule in rules:
                # Find contextual indicators
                for indicator in rule["indicators"]:
                    if indicator in text_lower:
                        # Look for PII patterns near the indicator
                        indicator_pos = text_lower.find(indicator)
                        if indicator_pos != -1:
                            # Search in a window after the indicator
                            start = indicator_pos + len(indicator)
                            end = min(len(text), start + 100)
                            window = text[start:end]

                            # Try to find specific patterns in this window
                            for pattern in rule["patterns"]:
                                matches = re.finditer(pattern, window, re.IGNORECASE)

                                for match in matches:
                                    # Calculate contextual confidence
                                    base_confidence = self.confidence_weights.get(
                                        pii_type, 0.7
                                    )
                                    contextual_confidence = (
                                        self._calculate_contextual_confidence(
                                            window, match, rule, base_confidence
                                        )
                                    )

                                    if (
                                        contextual_confidence > 0.5
                                    ):  # Only include if confidence is reasonable
                                        entity_start = start + match.start()
                                        entity_end = start + match.end()

                                        entity = PIIEntity(
                                            text=match.group(),
                                            start=entity_start,
                                            end=entity_end,
                                            pii_type=pii_type,
                                            confidence=contextual_confidence,
                                            context=self._get_context(
                                                text, entity_start, entity_end
                                            ),
                                            metadata={
                                                "detector": "contextual",
                                                "indicator": indicator,
                                                "context_window": window,
                                                "base_confidence": base_confidence,
                                                "contextual_adjustment": contextual_confidence
                                                - base_confidence,
                                                "validation_method": "contextual_analysis",
                                            },
                                        )
                                        entities.append(entity)

        return entities

    def _calculate_contextual_confidence(
        self, window: str, match: re.Match, rule: Dict[str, Any], base_confidence: float
    ) -> float:
        """Calculate confidence based on contextual analysis."""
        confidence = base_confidence
        window_lower = window.lower()

        # Apply boosters
        for booster in rule.get("boosters", []):
            if booster in window_lower:
                confidence += rule.get("confidence_boost", 0.1)

        # Apply reducers
        for reducer in rule.get("reducers", []):
            if reducer in window_lower:
                confidence -= rule.get("confidence_reduction", 0.2)

        # Additional contextual analysis
        confidence += self._analyze_proximity(window, match)
        confidence += self._analyze_sentence_structure(window, match)

        # Ensure confidence stays within bounds
        return max(0.0, min(1.0, confidence))

    def _analyze_proximity(self, window: str, match: re.Match) -> float:
        """Analyze proximity of match to contextual indicators."""
        match_start = match.start()
        match_end = match.end()
        window_length = len(window)

        # Closer to start of window (near indicator) gets higher confidence
        proximity_score = 1.0 - (match_start / window_length)

        return proximity_score * 0.05  # Small boost based on proximity

    def _analyze_sentence_structure(self, window: str, match: re.Match) -> float:
        """Analyze sentence structure around the match."""
        # Get the sentence containing the match
        sentences = re.split(r"[.!?]+", window)
        match_text = match.group()

        for sentence in sentences:
            if match_text in sentence:
                # Check if match is in a possessive or ownership context
                possessive_indicators = ["my", "his", "her", "our", "their"]
                if any(
                    indicator in sentence.lower() for indicator in possessive_indicators
                ):
                    return 0.1  # Boost for possessive context

                # Check if match is in a contact context
                contact_indicators = ["contact", "reach", "call", "email", "send"]
                if any(
                    indicator in sentence.lower() for indicator in contact_indicators
                ):
                    return 0.08  # Boost for contact context

                break

        return 0.0

    def _get_context(
        self, text: str, start: int, end: int, window_size: int = 50
    ) -> str:
        """Get context around a match."""
        context_start = max(0, start - window_size)
        context_end = min(len(text), end + window_size)
        return text[context_start:context_end]

    def add_contextual_rule(self, pii_type: str, rule: Dict[str, Any]) -> None:
        """Add a new contextual rule."""
        if pii_type not in self.contextual_rules:
            self.contextual_rules[pii_type] = []

        # Ensure rule has required fields
        required_fields = ["indicators", "patterns"]
        for field in required_fields:
            if field not in rule:
                rule[field] = []

        # Set defaults for optional fields
        rule.setdefault("boosters", [])
        rule.setdefault("reducers", [])
        rule.setdefault("confidence_boost", 0.1)
        rule.setdefault("confidence_reduction", 0.2)

        self.contextual_rules[pii_type].append(rule)

    def add_indicator(
        self, pii_type: str, indicator: str, confidence_adjustment: float = 0.1
    ) -> None:
        """Add a contextual indicator for a PII type."""
        if pii_type in self.contextual_rules:
            for rule in self.contextual_rules[pii_type]:
                if indicator not in rule["indicators"]:
                    rule["indicators"].append(indicator)
                    rule["confidence_boost"] = confidence_adjustment

    def add_booster(
        self, pii_type: str, booster: str, confidence_boost: float = 0.1
    ) -> None:
        """Add a confidence booster for a PII type."""
        if pii_type in self.contextual_rules:
            for rule in self.contextual_rules[pii_type]:
                if booster not in rule["boosters"]:
                    rule["boosters"].append(booster)
                    rule["confidence_boost"] = confidence_boost

    def add_reducer(
        self, pii_type: str, reducer: str, confidence_reduction: float = 0.2
    ) -> None:
        """Add a confidence reducer for a PII type."""
        if pii_type in self.contextual_rules:
            for rule in self.contextual_rules[pii_type]:
                if reducer not in rule["reducers"]:
                    rule["reducers"].append(reducer)
                    rule["confidence_reduction"] = confidence_reduction
