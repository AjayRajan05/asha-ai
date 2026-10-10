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
from typing import Dict, List, Any
from dataclasses import dataclass
from enum import Enum


class IntentType(Enum):
    """Intent types for prompt analysis."""

    ANALYZE = "analyze"
    GENERATE = "generate"
    SUMMARIZE = "summarize"
    TRANSLATE = "translate"
    CLASSIFY = "classify"
    EXTRACT = "extract"
    TRANSFORM = "transform"
    COMPARE = "compare"
    EXPLAIN = "explain"
    CREATE = "create"
    UNKNOWN = "unknown"


class ObjectType(Enum):
    """Object types for prompt analysis."""

    TEXT = "text"
    DATA = "data"
    DATASET = "dataset"
    CODE = "code"
    IMAGE = "image"
    DOCUMENT = "document"
    TABLE = "table"
    CHART = "chart"
    REPORT = "report"
    EMAIL = "email"
    MESSAGE = "message"
    UNKNOWN = "unknown"


@dataclass
class PromptAST:
    """Abstract Syntax Tree for prompt analysis."""

    original_prompt: str
    intent: IntentType
    objects: List[ObjectType]
    entities: List[str]
    keywords: List[str]
    complexity_score: float
    pii_indicators: List[str]
    structure_type: str
    metadata: Dict[str, Any]


class PromptParser:
    """Enhanced prompt parser with comprehensive analysis."""

    def __init__(self) -> None:
        """Initialize parser with intent and object patterns."""
        self.intent_patterns = {
            IntentType.ANALYZE: [
                r"\banalyze\b",
                r"\bexamine\b",
                r"\binspect\b",
                r"\breview\b",
                r"\binvestigate\b",
                r"\bstudy\b",
                r"\bevaluate\b",
                r"\bassess\b",
            ],
            IntentType.GENERATE: [
                r"\bgenerate\b",
                r"\bcreate\b",
                r"\bproduce\b",
                r"\bmake\b",
                r"\bbuild\b",
                r"\bdevelop\b",
                r"\bwrite\b",
                r"\bcompose\b",
            ],
            IntentType.SUMMARIZE: [
                r"\bsummarize\b",
                r"\bsummarise\b",
                r"\bcondense\b",
                r"\babridge\b",
                r"\bbrief\b",
                r"\boutline\b",
                r"\brecap\b",
                r"\bsynopsis\b",
            ],
            IntentType.TRANSLATE: [
                r"\btranslate\b",
                r"\bconvert\b",
                r"\btransform\b",
                r"\badapt\b",
                r"\blocalize\b",
                r"\btranscribe\b",
            ],
            IntentType.CLASSIFY: [
                r"\bclassify\b",
                r"\bcategorize\b",
                r"\bgroup\b",
                r"\bsort\b",
                r"\blabel\b",
                r"\btag\b",
                r"\borganize\b",
            ],
            IntentType.EXTRACT: [
                r"\bextract\b",
                r"\bpull\b",
                r"\bget\b",
                r"\bobtain\b",
                r"\bretrieve\b",
                r"\bfind\b",
                r"\bidentify\b",
            ],
            IntentType.COMPARE: [
                r"\bcompare\b",
                r"\bcontrast\b",
                r"\bdiff\b",
                r"\bversus\b",
                r"\bagainst\b",
                r"\bvs\b",
                r"\bthan\b",
            ],
            IntentType.EXPLAIN: [
                r"\bexplain\b",
                r"\bdescribe\b",
                r"\bdetail\b",
                r"\belaborate\b",
                r"\bclarify\b",
                r"\bdefine\b",
                r"\bdemonstrate\b",
            ],
        }

        self.object_patterns = {
            ObjectType.DATASET: [
                r"\bdataset\b",
                r"\bdata\b",
                r"\btable\b",
                r"\brecords\b",
                r"\bentries\b",
                r"\brows\b",
                r"\bcsv\b",
                r"\bexcel\b",
            ],
            ObjectType.CODE: [
                r"\bcode\b",
                r"\bprogram\b",
                r"\bscript\b",
                r"\bfunction\b",
                r"\bclass\b",
                r"\bmethod\b",
                r"\balgorithm\b",
                r"\bpython\b",
            ],
            ObjectType.DOCUMENT: [
                r"\bdocument\b",
                r"\bfile\b",
                r"\btext\b",
                r"\barticle\b",
                r"\bpaper\b",
                r"\breport\b",
                r"\bessay\b",
            ],
            ObjectType.EMAIL: [
                r"\bemail\b",
                r"\bmail\b",
                r"\bmessage\b",
                r"\bletter\b",
                r"\bnote\b",
                r"\bcommunication\b",
            ],
            ObjectType.IMAGE: [
                r"\bimage\b",
                r"\bpicture\b",
                r"\bphoto\b",
                r"\bgraphic\b",
                r"\bchart\b",
                r"\bplot\b",
                r"\bgraph\b",
                r"\bdiagram\b",
            ],
        }

        self.pii_indicators = [
            "email",
            "phone",
            "address",
            "ssn",
            "social security",
            "credit card",
            "name",
            "personal",
            "private",
            "confidential",
            "contact",
            "information",
            "details",
            "identification",
        ]

    def parse(self, prompt: str) -> PromptAST:
        """
        Parse prompt and return comprehensive AST.

        Args:
            prompt: Input prompt to analyze

        Returns:
            PromptAST with detailed analysis
        """
        # Normalize prompt
        normalized_prompt = prompt.lower().strip()

        # Detect intent
        intent = self._detect_intent(normalized_prompt)

        # Detect objects
        objects = self._detect_objects(normalized_prompt)

        # Extract entities
        entities = self._extract_entities(prompt)

        # Extract keywords
        keywords = self._extract_keywords(normalized_prompt)

        # Calculate complexity
        complexity_score = self._calculate_complexity(prompt)

        # Detect PII indicators
        pii_indicators = self._detect_pii_indicators(normalized_prompt)

        # Determine structure type
        structure_type = self._determine_structure_type(prompt)

        return PromptAST(
            original_prompt=prompt,
            intent=intent,
            objects=objects,
            entities=entities,
            keywords=keywords,
            complexity_score=complexity_score,
            pii_indicators=pii_indicators,
            structure_type=structure_type,
            metadata={
                "word_count": len(prompt.split()),
                "char_count": len(prompt),
                "has_questions": "?" in prompt,
                "has_commands": any(
                    cmd in normalized_prompt for cmd in ["please", "can you", "help"]
                ),
                "formality_level": self._assess_formality(prompt),
            },
        )

    def _detect_intent(self, prompt: str) -> IntentType:
        """Detect primary intent from prompt."""
        intent_scores = {}

        for intent_type, patterns in self.intent_patterns.items():
            score = 0
            for pattern in patterns:
                matches = re.findall(pattern, prompt, re.IGNORECASE)
                score += len(matches)
            intent_scores[intent_type] = score

        # Return intent with highest score
        if intent_scores:
            best_intent = max(intent_scores, key=lambda k: intent_scores[k])
            return best_intent if intent_scores[best_intent] > 0 else IntentType.UNKNOWN

        return IntentType.UNKNOWN

    def _detect_objects(self, prompt: str) -> List[ObjectType]:
        """Detect object types in prompt."""
        detected_objects = []

        for obj_type, patterns in self.object_patterns.items():
            for pattern in patterns:
                if re.search(pattern, prompt, re.IGNORECASE):
                    detected_objects.append(obj_type)
                    break

        return detected_objects or [ObjectType.UNKNOWN]

    def _extract_entities(self, prompt: str) -> List[str]:
        """Extract named entities from prompt."""
        entities = []

        # Extract quoted strings
        quoted = re.findall(r'"([^"]+)"', prompt)
        entities.extend(quoted)

        # Extract single quotes
        single_quoted = re.findall(r"'([^']+)'", prompt)
        entities.extend(single_quoted)

        # Extract proper nouns (simplified)
        proper_nouns = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", prompt)
        entities.extend(proper_nouns)

        return list(set(entities))  # Remove duplicates

    def _extract_keywords(self, prompt: str) -> List[str]:
        """Extract important keywords from prompt."""
        # Remove common stop words
        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
        }

        words = re.findall(r"\b\w+\b", prompt.lower())
        keywords = [word for word in words if word not in stop_words and len(word) > 2]

        # Return most frequent words
        from collections import Counter

        word_freq = Counter(keywords)
        return [word for word, _ in word_freq.most_common(10)]

    def _calculate_complexity(self, prompt: str) -> float:
        """Calculate complexity score (0.0 to 1.0)."""
        factors = {
            "length": min(len(prompt.split()) / 100, 1.0),  # Word count factor
            "punctuation": len(re.findall(r"[^\w\s]", prompt))
            / 20,  # Punctuation density
            "questions": prompt.count("?") * 0.1,  # Question complexity
            "nested": len(
                re.findall(
                    r"\b(?:if|when|while|because|although|since)\b", prompt.lower()
                )
            )
            * 0.1,
            "technical": len(
                re.findall(
                    r"\b(?:algorithm|function|method|class|variable|parameter)\b",
                    prompt.lower(),
                )
            )
            * 0.15,
        }

        return min(sum(factors.values()), 1.0)

    def _detect_pii_indicators(self, prompt: str) -> List[str]:
        """Detect PII-related indicators in prompt."""
        found_indicators = []

        for indicator in self.pii_indicators:
            if indicator in prompt:
                found_indicators.append(indicator)

        return found_indicators

    def _determine_structure_type(self, prompt: str) -> str:
        """Determine the structural type of the prompt."""
        if "?" in prompt:
            return "question"
        elif any(
            cmd in prompt.lower() for cmd in ["please", "can you", "help", "show me"]
        ):
            return "request"
        elif any(cond in prompt.lower() for cond in ["if", "when", "while", "because"]):
            return "conditional"
        elif ":" in prompt or "-" in prompt:
            return "structured"
        else:
            return "simple"

    def _assess_formality(self, prompt: str) -> str:
        """Assess formality level of the prompt."""
        informal_indicators = [
            "hey",
            "hi",
            "yo",
            "what's up",
            "gonna",
            "wanna",
            "gotta",
        ]
        formal_indicators = [
            "please",
            "would you",
            "could you",
            "thank you",
            "regarding",
            "concerning",
        ]

        prompt_lower = prompt.lower()

        if any(indicator in prompt_lower for indicator in informal_indicators):
            return "informal"
        elif any(indicator in prompt_lower for indicator in formal_indicators):
            return "formal"
        else:
            return "neutral"
