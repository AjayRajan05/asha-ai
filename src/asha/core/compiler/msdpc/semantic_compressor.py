"""
Semantic Compressor - Extractive compression for prompt optimization
"""

import re
from typing import List
from dataclasses import dataclass


@dataclass
class CompressionResult:
    """Result of semantic compression."""

    compressed_text: str
    tokens_removed: int
    tokens_remaining: int
    compression_ratio: float
    preserved_keywords: List[str]


class SemanticCompressor:
    """Extractive semantic compression for prompts."""

    def __init__(self) -> None:
        """Initialize compression patterns and rules."""
        # Filler words and phrases to remove
        self.filler_words = {
            "actually",
            "basically",
            "basically",
            "basically",
            "basically",
            "basically",
            "actually",
            "literally",
            "really",
            "very",
            "quite",
            "rather",
            "somewhat",
            "somehow",
            "anyway",
            "anyhow",
            "whatever",
            "however",
            "therefore",
            "moreover",
            "furthermore",
            "nevertheless",
            "nonetheless",
            "consequently",
            "accordingly",
            "hence",
            "thus",
            "otherwise",
            "meanwhile",
            "further",
            "besides",
            "indeed",
            "certainly",
            "definitely",
            "absolutely",
            "completely",
            "totally",
            "entirely",
            "perfectly",
            "exactly",
            "precisely",
            "specifically",
            "particularly",
            "especially",
            "mainly",
            "mostly",
            "primarily",
            "chiefly",
            "principally",
            "essentially",
            "fundamentally",
            "ultimately",
            "eventually",
            "finally",
            "initially",
            "originally",
        }

        # Conversational fillers
        self.conversational_fillers = [
            r"\b(hey|hi|hello|good morning|good afternoon|good evening)\b",
            r"\b(bro|dude|man|guys|folks|everyone)\b",
            r"\b(please|kindly|could you|would you|can you)\b",
            r"\b(thank you|thanks|appreciate|grateful)\b",
            r"\b(excuse me|pardon me|sorry)\b",
            r"\b(you know|you see|you understand)\b",
            r"\b(i mean|i think|i feel|i believe)\b",
            r"\b(to be honest|honestly|frankly|truthfully)\b",
            r"\b(as far as i know|as far as i can tell)\b",
            r"\b(in my opinion|from my perspective)\b",
        ]

        # Redundant phrases
        self.redundant_phrases = [
            r"\b(the fact that|due to the fact that|in spite of the fact that)\b",
            r"\b(in order to|so as to|for the purpose of)\b",
            r"\b(at this point in time|at the present time|at this moment)\b",
            r"\b(in the event that|in the case that|should it happen that)\b",
            r"\b(on the other hand|on the contrary|in contrast)\b",
            r"\b(as a matter of fact|in reality|in actual fact)\b",
            r"\b(first and foremost|above all|most importantly)\b",
            r"\b(last but not least|finally|in conclusion)\b",
        ]

        # Important keywords to preserve
        self.preservation_keywords = {
            "analyze",
            "analysis",
            "examine",
            "investigate",
            "study",
            "review",
            "summarize",
            "summary",
            "condense",
            "compress",
            "recap",
            "generate",
            "create",
            "make",
            "produce",
            "write",
            "develop",
            "extract",
            "pull",
            "get",
            "obtain",
            "retrieve",
            "isolate",
            "classify",
            "categorize",
            "group",
            "sort",
            "organize",
            "label",
            "compare",
            "comparison",
            "versus",
            "against",
            "contrast",
            "validate",
            "verify",
            "check",
            "confirm",
            "ensure",
            "test",
            "debug",
            "fix",
            "repair",
            "resolve",
            "troubleshoot",
            "solve",
            "translate",
            "convert",
            "transform",
            "change",
            "modify",
            "dataset",
            "data",
            "database",
            "table",
            "text",
            "document",
            "code",
            "program",
            "function",
            "script",
            "algorithm",
            "email",
            "phone",
            "address",
            "contact",
            "information",
            "pattern",
            "trend",
            "insight",
            "anomaly",
            "error",
            "issue",
            "quickly",
            "fast",
            "accurately",
            "precisely",
            "briefly",
            "detailed",
        }

    def compress(self, text: str, preserve_structure: bool = True) -> CompressionResult:
        """
        Compress text by removing filler words and redundant phrases.

        Args:
            text: Input text to compress
            preserve_structure: Whether to preserve sentence structure

        Returns:
            CompressionResult with compressed text and metrics
        """
        original_tokens = self._count_tokens(text)
        preserved_keywords = self._find_preserved_keywords(text)

        # Step 1: Remove conversational fillers
        compressed = text
        for pattern in self.conversational_fillers:
            compressed = re.sub(pattern, "", compressed, flags=re.IGNORECASE)

        # Step 2: Remove redundant phrases
        for pattern in self.redundant_phrases:
            compressed = re.sub(pattern, "", compressed, flags=re.IGNORECASE)

        # Step 3: Remove filler words
        words = compressed.split()
        filtered_words = []

        for word in words:
            clean_word = word.lower().strip(".,!?;:")
            if clean_word not in self.filler_words:
                filtered_words.append(word)

        compressed = " ".join(filtered_words)

        # Step 4: Clean up extra whitespace and punctuation
        compressed = re.sub(r"\s+", " ", compressed)  # Multiple spaces
        compressed = re.sub(
            r"\s+([.,!?;:])", r"\1", compressed
        )  # Space before punctuation
        compressed = re.sub(
            r"([.,!?;:])\s+", r"\1 ", compressed
        )  # Multiple spaces after punctuation
        compressed = compressed.strip()

        # Step 5: Preserve structure if requested
        if preserve_structure:
            compressed = self._preserve_sentence_structure(compressed)

        # Calculate metrics
        final_tokens = self._count_tokens(compressed)
        tokens_removed = original_tokens - final_tokens
        compression_ratio = (
            tokens_removed / original_tokens if original_tokens > 0 else 0
        )

        return CompressionResult(
            compressed_text=compressed,
            tokens_removed=tokens_removed,
            tokens_remaining=final_tokens,
            compression_ratio=compression_ratio,
            preserved_keywords=preserved_keywords,
        )

    def _count_tokens(self, text: str) -> int:
        """Count approximate tokens in text."""
        # Simple token count - split by whitespace and punctuation
        tokens = re.findall(r"\b\w+\b", text)
        return len(tokens)

    def _find_preserved_keywords(self, text: str) -> List[str]:
        """Find important keywords that should be preserved."""
        words = re.findall(r"\b\w+\b", text.lower())
        preserved = [word for word in words if word in self.preservation_keywords]
        return list(set(preserved))

    def _preserve_sentence_structure(self, text: str) -> str:
        """Preserve basic sentence structure while compressing."""
        # Ensure text starts with capital letter
        if text and text[0].islower():
            text = text[0].upper() + text[1:]

        # Ensure proper ending punctuation
        if text and text[-1] not in ".!?":
            text += "."

        return text

    def get_compression_summary(self, result: CompressionResult) -> str:
        """Generate human-readable compression summary."""
        return f"Removed {result.tokens_removed} tokens ({result.compression_ratio:.1%}) | Preserved: {', '.join(result.preserved_keywords[:5])}"
