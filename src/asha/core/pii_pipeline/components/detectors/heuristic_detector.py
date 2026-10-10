"""
Heuristic PII Detector - Pattern-based detection with heuristics
"""

import re
from typing import List, Dict
from ...stages.base_stage import PIIEntity


class HeuristicDetector:
    """
    Heuristic PII detector using pattern-based heuristics.

    This detector uses intelligent heuristics and pattern recognition
    to identify PII that might not match exact regex patterns.
    """

    def __init__(self) -> None:
        """Initialize heuristic detector with pattern heuristics."""
        self.patterns = self._compile_heuristic_patterns()
        self.confidence_weights = self._get_confidence_weights()

    def _compile_heuristic_patterns(self) -> Dict[str, List[re.Pattern]]:
        """Compile heuristic patterns for PII detection."""
        patterns = {
            "name": [
                # Name patterns
                re.compile(r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b"),  # First Last
                # First M. Last
                re.compile(r"\b[A-Z][a-z]+\s+[A-Z]\.\s+[A-Z][a-z]+\b"),
                re.compile(r"\bMr\.?\s+[A-Z][a-z]+\b", re.IGNORECASE),  # Mr. First
                re.compile(r"\bMs\.?\s+[A-Z][a-z]+\b", re.IGNORECASE),  # Ms. First
                re.compile(r"\bDr\.?\s+[A-Z][a-z]+\b", re.IGNORECASE),  # Dr. First
                re.compile(r"\bProf\.?\s+[A-Z][a-z]+\b", re.IGNORECASE),  # Prof. First
            ],
            "address": [
                # Address patterns
                re.compile(
                    r"\d+\s+[A-Z][a-z]+\s+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd)\b",
                    re.IGNORECASE,
                ),
                re.compile(
                    r"\d+\s+[A-Z][a-z]+\s+[A-Z][a-z]+\s+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd)\b",
                    re.IGNORECASE,
                ),
                re.compile(
                    r"\d+\s+[A-Z][a-z]+\s+[A-Z][a-z]+\s+[A-Z][a-z]+\s+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd)\b",
                    re.IGNORECASE,
                ),
                re.compile(
                    r"\d+\s+\d+\s+[A-Z][a-z]+\s+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd)\b",
                    re.IGNORECASE,
                ),  # 123 Main St
            ],
            "zip_code": [
                # ZIP code patterns
                re.compile(r"\b\d{5}\b"),
                re.compile(r"\b\d{5}-\d{4}\b"),
                re.compile(r"\b[A-Z]{2}\s+\d{5}\b"),  # State + ZIP
            ],
            "date_of_birth": [
                # Date patterns
                re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b"),
                re.compile(r"\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b"),
                re.compile(
                    r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b",
                    re.IGNORECASE,
                ),
                re.compile(
                    r"\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}\b",
                    re.IGNORECASE,
                ),
            ],
            "organization": [
                # Organization patterns
                re.compile(
                    r"\b[A-Z][a-z]+\s+(?:Inc|Corp|LLC|Ltd|Co|Company|Corporation)\b",
                    re.IGNORECASE,
                ),
                re.compile(
                    r"\b(?:The\s+)?[A-Z][a-z]+\s+(?:University|College|Institute|Hospital)\b",
                    re.IGNORECASE,
                ),
                re.compile(
                    r"\b[A-Z][a-z]+\s+(?:Bank|Insurance|Financial|Group)\b",
                    re.IGNORECASE,
                ),
            ],
            "job_title": [
                # Job title patterns
                re.compile(
                    r"\b(?:Senior|Junior|Lead|Chief|Head|Principal)\s+[A-Z][a-z]+\b",
                    re.IGNORECASE,
                ),
                re.compile(
                    r"\b(?:Software|Data|Marketing|Sales|Financial)\s+(?:Engineer|Manager|Analyst|Director)\b",
                    re.IGNORECASE,
                ),
                re.compile(
                    r"\b(?:President|CEO|CTO|CFO|VP|Vice President)\b", re.IGNORECASE
                ),
            ],
            "location": [
                # Location patterns
                re.compile(r"\b[A-Z][a-z]+,\s+[A-Z]{2}\b"),  # City, State
                re.compile(
                    r"\b[A-Z][a-z]+,\s+[A-Z][a-z]+,\s+[A-Z]{2}\b"
                ),  # City, County, State
                re.compile(
                    r"\b\d+\s+[A-Z][a-z]+\s+(?:Street|St|Ave|Rd|Dr|Ln),\s+[A-Z][a-z]+,\s+[A-Z]{2}\s+\d{5}\b",
                    re.IGNORECASE,
                ),
            ],
        }

        return patterns

    def _get_confidence_weights(self) -> Dict[str, float]:
        """Get confidence weights for different PII types."""
        return {
            "name": 0.65,
            "address": 0.60,
            "zip_code": 0.75,
            "date_of_birth": 0.55,
            "organization": 0.50,
            "job_title": 0.45,
            "location": 0.55,
        }

    def detect(self, text: str) -> List[PIIEntity]:
        """
        Detect PII entities using heuristic patterns.

        Args:
            text: Input text to analyze

        Returns:
            List of detected PII entities
        """
        entities = []

        for pii_type, patterns in self.patterns.items():
            for pattern in patterns:
                matches = pattern.finditer(text)

                for match in matches:
                    # Additional heuristic validation
                    if not self._validate_heuristic_match(
                        pii_type, match.group(), text, match.start(), match.end()
                    ):
                        continue

                    entity = PIIEntity(
                        text=match.group(),
                        start=match.start(),
                        end=match.end(),
                        pii_type=pii_type,
                        confidence=self.confidence_weights.get(pii_type, 0.6),
                        context=self._get_context(text, match.start(), match.end()),
                        metadata={
                            "detector": "heuristic",
                            "pattern": pattern.pattern,
                            "validation_method": "heuristic",
                        },
                    )
                    entities.append(entity)

        return entities

    def _validate_heuristic_match(
        self, pii_type: str, match_text: str, full_text: str, start: int, end: int
    ) -> bool:
        """Additional heuristic validation for matches."""
        if pii_type == "name":
            return self._validate_name_heuristic(match_text, full_text, start, end)
        elif pii_type == "address":
            return self._validate_address_heuristic(match_text, full_text, start, end)
        elif pii_type == "date_of_birth":
            return self._validate_date_heuristic(match_text, full_text, start, end)
        elif pii_type == "organization":
            return self._validate_organization_heuristic(
                match_text, full_text, start, end
            )

        return True

    def _validate_name_heuristic(
        self, name: str, full_text: str, start: int, end: int
    ) -> bool:
        """Validate name using heuristics."""
        # Check if it's likely a name
        words = name.split()

        if len(words) == 2:
            first, last = words

            # Both words should start with uppercase
            if not (first[0].isupper() and last[0].isupper()):
                return False

            # Words should be reasonable length
            if len(first) < 2 or len(last) < 2:
                return False

            # Words should be alphabetic
            if not (first.isalpha() and last.isalpha()):
                return False

            # Check surrounding context for name indicators
            context = self._get_context(full_text, start, end, 30).lower()
            name_indicators = ["name", "called", "known as", "my name is", "i am"]

            # Higher confidence if name indicators are nearby
            if any(indicator in context for indicator in name_indicators):
                return True

        return True

    def _validate_address_heuristic(
        self, address: str, full_text: str, start: int, end: int
    ) -> bool:
        """Validate address using heuristics."""
        # Should start with a number
        if not re.match(r"^\d+", address.strip()):
            return False

        # Should contain street type
        street_types = [
            "street",
            "st",
            "avenue",
            "ave",
            "road",
            "rd",
            "drive",
            "dr",
            "lane",
            "ln",
            "boulevard",
            "blvd",
        ]
        if not any(street_type in address.lower() for street_type in street_types):
            return False

        return True

    def _validate_date_heuristic(
        self, date: str, full_text: str, start: int, end: int
    ) -> bool:
        """Validate date using heuristics."""
        # Check if it's likely a birth date (not just any date)
        context = self._get_context(full_text, start, end, 30).lower()

        birth_indicators = ["born", "birth", "birthday", "age", "dob", "date of birth"]
        if any(indicator in context for indicator in birth_indicators):
            return True

        # If no birth indicators, be more conservative
        return False

    def _validate_organization_heuristic(
        self, org: str, full_text: str, start: int, end: int
    ) -> bool:
        """Validate organization using heuristics."""
        # Should have company suffix
        company_suffixes = [
            "inc",
            "corp",
            "llc",
            "ltd",
            "co",
            "company",
            "corporation",
            "university",
            "college",
            "hospital",
        ]
        if not any(suffix in org.lower() for suffix in company_suffixes):
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

    def add_custom_heuristic(
        self, pii_type: str, pattern: str, confidence: float = 0.6
    ) -> None:
        """Add a custom heuristic pattern."""
        if pii_type not in self.patterns:
            self.patterns[pii_type] = []

        compiled_pattern = re.compile(pattern, re.IGNORECASE)
        self.patterns[pii_type].append(compiled_pattern)
        self.confidence_weights[pii_type] = confidence
