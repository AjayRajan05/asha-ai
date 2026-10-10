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

"""
Safety constraints for ASHA optimization.

Prevents silent logic corruption and intent preservation failures.
"""

import re
from typing import List, Literal, Tuple

from asha.core.ml.optimizer_similarity import SimilarityResult

SimilarityMode = Literal["auto", "embedding", "jaccard"]


class SafetyConstraints:
    """
    Hard constraints to prevent logic corruption during optimization.

    This is CRITICAL for production safety - prevents optimizer
    from silently changing prompt intent or breaking logic.
    """

    def __init__(
        self,
        *,
        similarity_mode: SimilarityMode = "auto",
        allow_hash_fallback: bool = False,
    ) -> None:
        self.similarity_mode = similarity_mode
        self.allow_hash_fallback = allow_hash_fallback
        # Words that indicate negation - must be preserved
        self.negative_keywords = [
            "not",
            "don't",
            "never",
            "avoid",
            "prevent",
            "stop",
            "no",
            "none",
            "without",
            "except",
            "unless",
            "ignore",
            "skip",
            "omit",
            "exclude",
            "reject",
            "deny",
            "refuse",
        ]

        # Quantitative patterns that must be preserved exactly
        self.quantitative_patterns = [
            r"\b\d+\b",  # Numbers
            r"\b\d+%\b",  # Percentages
            r"\$\d+",  # Money amounts
            r"\b\d+\.\d+\b",  # Decimals
            r"\b\d+/\d+\b",  # Fractions
        ]

        # Constraint and requirement words that must be preserved
        self.constraint_keywords = [
            "must",
            "should",
            "require",
            "ensure",
            "only",
            "exactly",
            "always",
            "never",
            "mandatory",
            "obligatory",
            "essential",
            "critical",
            "vital",
            "necessary",
            "compulsory",
        ]

        # Logic operators that must be preserved
        self.logic_operators = [
            "and",
            "or",
            "but",
            "if",
            "then",
            "else",
            "when",
            "while",
            "for",
            "each",
            "every",
            "all",
            "any",
            "some",
        ]

    def check_safety(self, original: str, optimized: str) -> Tuple[bool, List[str]]:
        """
        Check if optimization violates safety constraints.

        Args:
            original: Original prompt
            optimized: Optimized prompt

        Returns:
            Tuple of (is_safe, list_of_violations)
        """
        violations = []

        # Check for lost negations (CRITICAL)
        original_negations = self._count_negations(original)
        optimized_negations = self._count_negations(optimized)

        if original_negations > optimized_negations:
            violations.append(
                f"Lost negation words: {original_negations - optimized_negations} - LOGIC CORRUPTION RISK"
            )

        # Check for changed quantitative data (CRITICAL)
        original_quant = self._extract_quantitative_data(original)
        optimized_quant = self._extract_quantitative_data(optimized)

        if len(original_quant) != len(optimized_quant):
            violations.append(
                f"Quantitative data modified: {len(original_quant)} → {len(optimized_quant)} - ACCURACY RISK"
            )

        # Check for lost constraints (HIGH)
        original_constraints = self._count_constraints(original)
        optimized_constraints = self._count_constraints(optimized)

        if original_constraints > optimized_constraints:
            violations.append(
                f"Constraint words removed: {original_constraints - optimized_constraints} - REQUIREMENT LOSS RISK"
            )

        # Check for lost logic operators (MEDIUM)
        original_logic = self._count_logic_operators(original)
        optimized_logic = self._count_logic_operators(optimized)

        if original_logic > optimized_logic:
            violations.append(
                f"Logic operators removed: {original_logic - optimized_logic} - STRUCTURE RISK"
            )

        # Semantic similarity check (HIGH): MiniLM cosine vs calibrated floor
        sim = self._similarity_result(original, optimized)
        if not sim.is_safe:
            violations.append(
                f"Low semantic similarity: {sim.score:.2f} < floor {sim.floor:.2f} "
                f"({sim.method}) - INTENT CORRUPTION RISK"
            )

        return len(violations) == 0, violations

    def _count_negations(self, text: str) -> int:
        """Count negation words in text."""
        text_lower = text.lower()
        return sum(1 for word in self.negative_keywords if word in text_lower)

    def _extract_quantitative_data(self, text: str) -> List[str]:
        """Extract quantitative data patterns from text."""
        matches = []
        for pattern in self.quantitative_patterns:
            found = re.findall(pattern, text)
            matches.extend(found)
        return matches

    def _count_constraints(self, text: str) -> int:
        """Count constraint words in text."""
        text_lower = text.lower()
        return sum(1 for word in self.constraint_keywords if word in text_lower)

    def _count_logic_operators(self, text: str) -> int:
        """Count logic operators in text."""
        text_lower = text.lower()
        return sum(1 for word in self.logic_operators if word in text_lower)

    def _similarity_result(self, original: str, optimized: str) -> SimilarityResult:
        from asha.core.ml.optimizer_similarity import compute_similarity

        return compute_similarity(
            original,
            optimized,
            mode=self.similarity_mode,
            allow_hash_fallback=self.allow_hash_fallback,
        )

    def _calculate_semantic_similarity(self, original: str, optimized: str) -> float:
        """Return similarity score (embedding cosine or Jaccard fallback)."""
        return self._similarity_result(original, optimized).score

    def should_revert_optimization(self, original: str, optimized: str) -> bool:
        """
        Quick check if optimization should be reverted.

        Returns True if any critical safety violations detected.
        """
        # Quick checks for most critical violations
        original_negations = self._count_negations(original)
        optimized_negations = self._count_negations(optimized)

        # Always revert if negations lost
        if original_negations > optimized_negations:
            return True

        sim = self._similarity_result(original, optimized)
        if not sim.is_safe:
            return True

        return False

    def get_safety_report(self, original: str, optimized: str) -> dict:
        """Generate detailed safety report."""
        is_safe, violations = self.check_safety(original, optimized)

        sim = self._similarity_result(original, optimized)
        return {
            "is_safe": is_safe,
            "violations": violations,
            "negation_count": {
                "original": self._count_negations(original),
                "optimized": self._count_negations(optimized),
            },
            "quantitative_count": {
                "original": len(self._extract_quantitative_data(original)),
                "optimized": len(self._extract_quantitative_data(optimized)),
            },
            "constraint_count": {
                "original": self._count_constraints(original),
                "optimized": self._count_constraints(optimized),
            },
            "semantic_similarity": sim.score,
            "similarity_floor": sim.floor,
            "similarity_method": sim.method,
            "recommendation": "Keep original" if not is_safe else "Optimization safe",
        }
