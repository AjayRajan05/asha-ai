"""
Stage 1: Normalization Layer - Text preprocessing and standardization
"""

import re
import unicodedata
from .base_stage import BaseStage, StageResult, PIIContext


class NormalizationStage(BaseStage):
    """
    Normalization Stage - Preprocesses and standardizes input text for better PII detection.

    This stage handles:
    - Unicode normalization
    - Selective lowercasing
    - Removing obfuscation patterns
    - Standardizing formats
    - Cleaning whitespace
    """

    def __init__(self) -> None:
        super().__init__("normalization")

        # Obfuscation patterns: only bracketed/parenthesized forms.
        # Never rewrite bare "at"/"dot" (breaks "email me at alice@..." and
        # "cancellation").
        self.obfuscation_patterns = {
            r"\s*\[\s*at\s*\]\s*": "@",
            r"\s*\[\s*dot\s*\]\s*": ".",
            r"\s*\(\s*at\s*\)\s*": "@",
            r"\s*\(\s*dot\s*\)\s*": ".",
            r"[-\s]\*[-\s]": "-",
        }

        # Format standardization: credit cards first so phone/SSN patterns
        # cannot carve a 16-digit PAN into ###-###-#### fragments.
        self.format_patterns = [
            (
                r"(\d{4})[.\s-]*(\d{4})[.\s-]*(\d{4})[.\s-]*(\d{4})",
                r"\1\2\3\4",
            ),
            (
                r"(?<!\d)(\d{3})[.\s-]+(\d{3})[.\s-]+(\d{4})(?!\d)",
                r"\1-\2-\3",
            ),
            (
                r"(?<!\d)(\d{3})[.\s-]+(\d{2})[.\s-]+(\d{4})(?!\d)",
                r"\1-\2-\3",
            ),
        ]

    def execute(self, context: PIIContext) -> StageResult:
        """
        Execute normalization on input text.

        Args:
            context: PII pipeline context

        Returns:
            StageResult with normalized text
        """
        original_text = context.current_text
        normalized_text = original_text

        # Step 1: Unicode normalization
        normalized_text = self._normalize_unicode(normalized_text)

        # Step 2: Remove obfuscation patterns
        normalized_text = self._remove_obfuscation(normalized_text)

        # Step 3: Standardize formats
        normalized_text = self._standardize_formats(normalized_text)

        # Step 4: Clean whitespace
        normalized_text = self._clean_whitespace(normalized_text)

        # Step 5: Preserve case for context (selective lowercase)
        normalized_text = self._selective_lowercase(normalized_text)

        # Update context
        context.update_text(normalized_text)

        self.add_debug_info(
            context,
            "Text normalization completed",
            {
                "original_length": len(original_text),
                "normalized_length": len(normalized_text),
                "changes_made": original_text != normalized_text,
            },
        )

        return StageResult(
            success=True,
            stage_name=self.stage_name,
            execution_time_ms=0.0,  # Will be set by base class
            processed_text=normalized_text,
            metadata={
                "original_length": len(original_text),
                "normalized_length": len(normalized_text),
                "changes_made": original_text != normalized_text,
                "normalization_steps": [
                    "unicode_normalization",
                    "obfuscation_removal",
                    "format_standardization",
                    "whitespace_cleaning",
                    "selective_lowercase",
                ],
            },
        )

    def _normalize_unicode(self, text: str) -> str:
        """Normalize Unicode characters"""
        # Normalize to NFKC form (compatibility decomposition)
        normalized = unicodedata.normalize("NFKC", text)

        # Convert common Unicode equivalents to ASCII
        unicode_to_ascii = {
            '"': '"',
            '"': '"',
            """: "'",
            """: "'",
            "-": "-",  # en dash
            " - ": "--",  # em dash
            "…": "...",  # ellipsis
        }

        for unicode_char, ascii_char in unicode_to_ascii.items():
            normalized = normalized.replace(unicode_char, ascii_char)

        return normalized

    def _remove_obfuscation(self, text: str) -> str:
        """Remove common obfuscation patterns"""
        normalized = text

        # Apply obfuscation patterns
        for pattern, replacement in self.obfuscation_patterns.items():
            normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)

        return normalized

    def _standardize_formats(self, text: str) -> str:
        """Standardize common PII formats"""
        normalized = text

        # Apply format standardization patterns (ordered list of tuples).
        for pattern, replacement in self.format_patterns:
            normalized = re.sub(pattern, replacement, normalized)

        return normalized

    def _clean_whitespace(self, text: str) -> str:
        """Clean and normalize whitespace"""
        # Replace multiple whitespace with single space
        normalized = re.sub(r"\s+", " ", text)

        # Remove leading/trailing whitespace
        normalized = normalized.strip()

        # Fix spacing around punctuation
        normalized = re.sub(r"\s+([.,!?;:])", r"\1", normalized)
        normalized = re.sub(r"([.,!?;:])\s+", r"\1 ", normalized)

        return normalized

    def _selective_lowercase(self, text: str) -> str:
        """Apply selective lowercase to improve detection"""
        # For PII detection, we generally want to lowercase
        # but preserve some context for better understanding
        words = text.split()
        normalized_words = []

        for word in words:
            # Keep words that might be proper names or important context
            if self._should_preserve_case(word):
                normalized_words.append(word)
            else:
                normalized_words.append(word.lower())

        return " ".join(normalized_words)

    def _should_preserve_case(self, word: str) -> bool:
        """Determine if a word should preserve its case"""
        # Preserve case for:
        # - All uppercase words (acronyms)
        # - Words that start sentences
        # - Words that might be proper names (first letter capital)
        # - Short words that might be important context

        if word.isupper() and len(word) > 1:
            return True

        if word[0].isupper() and word[1:].islower():
            return True  # Likely a proper name

        if len(word) <= 2:
            return True  # Short words like "I", "US", "UK"

        # Check if word contains numbers (might be an ID or code)
        if any(char.isdigit() for char in word):
            return True

        return False

    def validate_input(self, context: PIIContext) -> bool:
        """Validate input for normalization stage"""
        current_text: object = context.current_text
        if not current_text:
            return False

        if not isinstance(current_text, str):
            return False

        return True

    def fallback(self, context: PIIContext) -> StageResult:
        """Fallback normalization - basic cleanup only"""
        text = context.current_text

        # Basic cleanup
        fallback_text = re.sub(r"\s+", " ", text.strip())

        context.update_text(fallback_text)

        self.add_debug_info(
            context,
            "Normalization fallback used",
            {"reason": "main_normalization_failed", "fallback_type": "basic_cleanup"},
        )

        return StageResult(
            success=True,
            stage_name=self.stage_name,
            execution_time_ms=0.0,
            processed_text=fallback_text,
            metadata={"fallback_used": True, "fallback_type": "basic_cleanup"},
        )
