"""
Token Pruner - Algorithmic removal of low-value tokens
"""

import re
from typing import Dict, Tuple
from dataclasses import dataclass


@dataclass
class PruningResult:
    """Result of token pruning."""

    pruned_text: str
    tokens_removed: int
    tokens_remaining: int
    pruning_ratio: float
    removed_categories: Dict[str, int]


class TokenPruner:
    """Algorithmic token pruning for prompt optimization."""

    def __init__(self) -> None:
        """Initialize pruning rules and categories."""
        # Stopwords that can be safely removed
        self.stopwords = {
            "a",
            "an",
            "the",
            "but",
            "while",
            "when",
            "where",
            "how",
            "why",
            "what",
            "which",
            "who",
            "whom",
            "whose",
            "this",
            "that",
            "these",
            "those",
            "i",
            "you",
            "he",
            "she",
            "it",
            "we",
            "they",
            "me",
            "him",
            "her",
            "us",
            "them",
            "my",
            "your",
            "his",
            "her",
            "its",
            "our",
            "their",
            "am",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "shall",
            "should",
            "can",
            "could",
            "may",
            "might",
            "must",
            "ought",
            "for",
            "with",
            "by",
            "at",
            "on",
            "in",
            "to",
            "from",
            "of",
            "off",
            "out",
            "up",
            "down",
            "over",
            "under",
            "through",
            "between",
            "among",
            "around",
        }

        # Redundant adjectives and adverbs
        self.redundant_modifiers = {
            "very",
            "quite",
            "rather",
            "somewhat",
            "somehow",
            "really",
            "actually",
            "literally",
            "basically",
            "essentially",
            "fundamentally",
            "practically",
            "virtually",
            "nearly",
            "almost",
            "approximately",
            "roughly",
            "about",
            "generally",
            "typically",
            "usually",
            "normally",
            "commonly",
            "often",
            "frequently",
            "regularly",
            "sometimes",
            "occasionally",
            "rarely",
            "seldom",
            "hardly",
            "scarcely",
            "barely",
            "merely",
            "simply",
            "just",
        }

        # Redundant phrases
        self.redundant_phrases = [
            r"\b(in order to|so as to)\b",
            r"\b(due to the fact that|owing to the fact that)\b",
            r"\b(at this point in time|at the present time)\b",
            r"\b(in the event that|in the case that)\b",
            r"\b(as far as i know|as far as i can tell)\b",
            r"\b(for all intents and purposes|for all practical purposes)\b",
            r"\b(in the final analysis|when all is said and done)\b",
            r"\b(be that as it may|nevertheless)\b",
        ]

        # Important content words to preserve
        self.preservation_words = {
            "and",
            "or",
            "if",
            "then",
            "else",
            "not",
            "no",
            "never",
            "nor",
            "because",
            "since",
            "therefore",
            "unless",
            "except",
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
            "abstract",
            "generate",
            "create",
            "make",
            "produce",
            "write",
            "develop",
            "build",
            "extract",
            "pull",
            "get",
            "obtain",
            "retrieve",
            "isolate",
            "separate",
            "classify",
            "categorize",
            "group",
            "sort",
            "organize",
            "label",
            "tag",
            "compare",
            "comparison",
            "versus",
            "against",
            "contrast",
            "differentiate",
            "validate",
            "verify",
            "check",
            "confirm",
            "ensure",
            "test",
            "prove",
            "debug",
            "fix",
            "repair",
            "resolve",
            "troubleshoot",
            "solve",
            "correct",
            "translate",
            "convert",
            "transform",
            "change",
            "modify",
            "adapt",
            "dataset",
            "data",
            "database",
            "table",
            "text",
            "document",
            "file",
            "code",
            "program",
            "function",
            "script",
            "algorithm",
            "method",
            "email",
            "phone",
            "address",
            "contact",
            "information",
            "details",
            "pattern",
            "trend",
            "insight",
            "anomaly",
            "error",
            "issue",
            "problem",
            "quickly",
            "fast",
            "accurately",
            "precisely",
            "exactly",
            "correctly",
            "briefly",
            "short",
            "concise",
            "succinct",
            "detailed",
            "thorough",
        }

    def prune(self, text: str, aggressive: bool = False) -> PruningResult:
        """
        Prune low-value tokens from text.

        Args:
            text: Input text to prune
            aggressive: Whether to use aggressive pruning

        Returns:
            PruningResult with pruned text and metrics
        """
        original_tokens = self._count_tokens(text)
        removed_categories = {}

        # Step 1: Remove redundant phrases
        pruned_text, phrase_removals = self._remove_redundant_phrases(text)
        removed_categories["redundant_phrases"] = phrase_removals

        # Step 2: Remove stopwords (conservative)
        if aggressive:
            pruned_text, stopword_removals = self._remove_stopwords(pruned_text)
            removed_categories["stopwords"] = stopword_removals

        # Step 3: Remove redundant modifiers
        pruned_text, modifier_removals = self._remove_redundant_modifiers(pruned_text)
        removed_categories["redundant_modifiers"] = modifier_removals

        # Step 4: Remove duplicate phrases
        pruned_text, duplicate_removals = self._remove_duplicate_phrases(pruned_text)
        removed_categories["duplicate_phrases"] = duplicate_removals

        # Step 5: Clean up formatting
        pruned_text = self._clean_formatting(pruned_text)

        # Calculate metrics
        final_tokens = self._count_tokens(pruned_text)
        tokens_removed = original_tokens - final_tokens
        pruning_ratio = tokens_removed / original_tokens if original_tokens > 0 else 0

        return PruningResult(
            pruned_text=pruned_text,
            tokens_removed=tokens_removed,
            tokens_remaining=final_tokens,
            pruning_ratio=pruning_ratio,
            removed_categories=removed_categories,
        )

    def _count_tokens(self, text: str) -> int:
        """Count approximate tokens in text."""
        tokens = re.findall(r"\b\w+\b", text)
        return len(tokens)

    def _remove_redundant_phrases(self, text: str) -> Tuple[str, int]:
        """Remove redundant phrases from text."""
        removed = 0
        for pattern in self.redundant_phrases:
            matches = len(re.findall(pattern, text, re.IGNORECASE))
            if matches > 0:
                text = re.sub(pattern, "", text, flags=re.IGNORECASE)
                removed += matches

        return text, removed

    def _remove_stopwords(self, text: str) -> Tuple[str, int]:
        """Remove stopwords while preserving important words."""
        words = text.split()
        filtered_words = []
        removed = 0

        for word in words:
            clean_word = word.lower().strip(".,!?;:")
            # Only remove if it's a stopword AND not a preservation word
            if (
                clean_word in self.stopwords
                and clean_word not in self.preservation_words
            ):
                removed += 1
            else:
                filtered_words.append(word)

        return " ".join(filtered_words), removed

    def _remove_redundant_modifiers(self, text: str) -> Tuple[str, int]:
        """Remove redundant adjectives and adverbs."""
        words = text.split()
        filtered_words = []
        removed = 0

        for word in words:
            clean_word = word.lower().strip(".,!?;:")
            if clean_word in self.redundant_modifiers:
                removed += 1
            else:
                filtered_words.append(word)

        return " ".join(filtered_words), removed

    def _remove_duplicate_phrases(self, text: str) -> Tuple[str, int]:
        """Remove duplicate phrases and clauses."""
        # Simple duplicate removal - split into sentences and remove duplicates
        sentences = re.split(r"[.!?]+", text)
        unique_sentences = []
        removed = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and sentence not in unique_sentences:
                unique_sentences.append(sentence)
            elif sentence:
                removed += 1

        return ". ".join(unique_sentences), removed

    def _clean_formatting(self, text: str) -> str:
        """Clean up formatting after pruning."""
        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text)

        # Fix spacing around punctuation
        text = re.sub(r"\s+([.,!?;:])", r"\1", text)
        text = re.sub(r"([.,!?;:])\s+", r"\1 ", text)

        # Remove leading/trailing whitespace
        text = text.strip()

        # Ensure proper capitalization
        if text and text[0].islower():
            text = text[0].upper() + text[1:]

        return text

    def get_pruning_summary(self, result: PruningResult) -> str:
        """Generate human-readable pruning summary."""
        return f"Removed {result.tokens_removed} tokens ({result.pruning_ratio:.1%}) | Categories: {len(result.removed_categories)}"
