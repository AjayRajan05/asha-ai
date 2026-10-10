"""
Intent Extractor - Rule-based NLP for prompt intent analysis
"""

import re
from typing import List, Tuple
from dataclasses import dataclass
from enum import Enum


class IntentType(Enum):
    """Intent types for prompt classification."""

    ANALYZE = "analyze"
    SUMMARIZE = "summarize"
    GENERATE = "generate"
    EXTRACT = "extract"
    CLASSIFY = "classify"
    COMPARE = "compare"
    VALIDATE = "validate"
    DEBUG = "debug"
    TRANSLATE = "translate"
    TRANSFORM = "transform"


@dataclass
class IntentExtractionResult:
    """Result of intent extraction."""

    intent: IntentType
    object: str
    constraints: List[str]
    confidence: float
    extracted_entities: List[str]


class IntentExtractor:
    """Rule-based intent extraction for prompt analysis."""

    def __init__(self) -> None:
        """Initialize intent extraction patterns."""
        self.intent_patterns = {
            IntentType.ANALYZE: [
                r"\b(analyze|analysis|examine|investigate|study|review|audit|inspect)\b",
                r"\b(find|identify|discover|detect|uncover|reveal)\b.*\b(pattern|trend|insight|anomaly)\b",
            ],
            IntentType.SUMMARIZE: [
                r"\b(summarize|summary|condense|compress|recap|abstract|outline)\b",
                r"\b(brief|short|concise|succinct|compact)\b.*\b(version|summary|overview)\b",
            ],
            IntentType.GENERATE: [
                r"\b(generate|create|make|produce|write|develop|build|compose)\b",
                r"\b(new|fresh|original|unique|custom)\b.*\b(content|text|code|data)\b",
            ],
            IntentType.EXTRACT: [
                r"\b(extract|pull|get|obtain|retrieve|isolate|separate)\b",
                r"\b(find|locate|identify|search|look for)\b.*\b(email|phone|address|name|data)\b",
            ],
            IntentType.CLASSIFY: [
                r"\b(classify|categorize|group|sort|organize|label|tag)\b",
                r"\b(type|kind|category|class|group)\b.*\b(into|as)\b",
            ],
            IntentType.COMPARE: [
                r"\b(compare|comparison|versus|vs|against|contrast|differentiate)\b",
                r"\b(similar|different|better|worse|pros|cons)\b.*\b(than|between)\b",
            ],
            IntentType.VALIDATE: [
                r"\b(validate|validation|verify|check|confirm|ensure|test)\b",
                r"\b(correct|proper|valid|accurate|right)\b.*\b(format|structure|syntax)\b",
            ],
            IntentType.DEBUG: [
                r"\b(debug|debugging|fix|repair|resolve|troubleshoot|solve)\b",
                r"\b(error|issue|problem|bug|fault|mistake)\b.*\b(in|with|for)\b",
            ],
            IntentType.TRANSLATE: [
                r"\b(translate|translation|convert|transform|change)\b",
                r"\b(from|to)\b.*\b(language|lang|format)\b",
            ],
            IntentType.TRANSFORM: [
                r"\b(transform|convert|change|modify|adapt|alter)\b",
                r"\b(format|structure|layout|style)\b.*\b(to|into)\b",
            ],
        }

        self.object_patterns = [
            r"\b(dataset|data|database|table)\b",
            r"\b(text|document|file|content|article)\b",
            r"\b(code|program|function|script|algorithm)\b",
            r"\b(email|phone|address|contact|information)\b",
            r"\b(image|picture|photo|graphic)\b",
            r"\b(list|array|collection|set|group)\b",
        ]

        self.constraint_patterns = [
            r"\b(quickly|fast|rapid|swift|immediate)\b",
            r"\b(accurately|precisely|exactly|correctly)\b",
            r"\b(briefly|short|concise|succinct)\b",
            r"\b(detailed|thorough|comprehensive|complete)\b",
            r"\b(safely|securely|protected)\b",
            r"\b(efficiently|optimized|performance)\b",
        ]

    def extract_intent(self, prompt: str) -> IntentExtractionResult:
        """
        Extract intent, object, and constraints from prompt.

        Args:
            prompt: Input prompt to analyze

        Returns:
            IntentExtractionResult with extracted information
        """
        # Normalize prompt
        normalized_prompt = prompt.lower().strip()

        # Extract intent
        intent, confidence = self._extract_intent_type(normalized_prompt)

        # Extract object
        object_text = self._extract_object(normalized_prompt)

        # Extract constraints
        constraints = self._extract_constraints(normalized_prompt)

        # Extract entities
        entities = self._extract_entities(normalized_prompt)

        return IntentExtractionResult(
            intent=intent,
            object=object_text,
            constraints=constraints,
            confidence=confidence,
            extracted_entities=entities,
        )

    def _extract_intent_type(self, prompt: str) -> Tuple[IntentType, float]:
        """Extract primary intent type and confidence."""
        intent_scores = {}

        for intent_type, patterns in self.intent_patterns.items():
            score = 0
            matches = 0

            for pattern in patterns:
                pattern_matches = len(re.findall(pattern, prompt, re.IGNORECASE))
                if pattern_matches > 0:
                    matches += 1
                    score += pattern_matches * 10  # Weight by number of matches

            if matches > 0:
                intent_scores[intent_type] = score / len(patterns)

        if not intent_scores:
            return IntentType.ANALYZE, 0.5  # Default fallback

        # Get best intent
        best_intent = max(intent_scores, key=lambda k: intent_scores[k])
        # Normalize to 0-1
        confidence = min(intent_scores[best_intent] / 20, 1.0)

        return best_intent, confidence

    def _extract_object(self, prompt: str) -> str:
        """Extract the object being acted upon."""
        for pattern in self.object_patterns:
            matches = re.findall(pattern, prompt, re.IGNORECASE)
            if matches:
                return str(matches[0])

        # Fallback: look for nouns after intent verbs
        intent_words = ["analyze", "summarize", "generate", "extract", "classify"]
        for word in intent_words:
            pattern = rf"{word}\s+(\w+)"
            match = re.search(pattern, prompt, re.IGNORECASE)
            if match:
                return match.group(1)

        return "content"  # Default fallback

    def _extract_constraints(self, prompt: str) -> List[str]:
        """Extract constraints and requirements."""
        constraints = []

        for pattern in self.constraint_patterns:
            matches = re.findall(pattern, prompt, re.IGNORECASE)
            constraints.extend(matches)

        return list(set(constraints))  # Remove duplicates

    def _extract_entities(self, prompt: str) -> List[str]:
        """Extract named entities from prompt."""
        entities = []

        # Email patterns
        email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        entities.extend(re.findall(email_pattern, prompt))

        # Phone patterns
        phone_pattern = r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"
        entities.extend(re.findall(phone_pattern, prompt))

        # URL patterns
        url_pattern = r'\bhttps?://[^\s<>"]+\b'
        entities.extend(re.findall(url_pattern, prompt))

        return entities

    def get_intent_summary(self, result: IntentExtractionResult) -> str:
        """Generate human-readable intent summary."""
        return f"Intent: {result.intent.value} | Object: {result.object} | Constraints: {', '.join(result.constraints)} | Confidence: {result.confidence:.2f}"
