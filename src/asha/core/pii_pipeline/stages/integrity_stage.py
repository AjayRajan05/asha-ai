"""
Stage 6: Semantic Integrity Check - Ensure prompt meaning is preserved after masking
"""

import re
from typing import Dict, Any, List, Tuple
from .base_stage import BaseStage, StageResult, PIIContext, PIIEntity


class IntegrityStage(BaseStage):
    """
    Semantic Integrity Stage - Ensures prompt meaning is preserved after masking.

    This stage handles:
    - Semantic validation of masked text
    - Grammar and syntax checking
    - Coherence assessment
    - Masking impact analysis
    - Integrity scoring
    """

    def __init__(self) -> None:
        super().__init__("integrity")

        # Integrity configuration
        self.integrity_config = {
            "grammar_patterns": {
                "sentence_structure": r"^[A-Z][^.!?]*[.!?]$",
                "question_structure": r"^[A-Z][^.!?]*\?$",
                "proper_spacing": r"^[^.!?]*[.!?]\s*$",
                "no_trailing_spaces": r"[^ ]+$",
            },
            "semantic_patterns": {
                "broken_sentences": [
                    r"\[\w+\]\s*\[\w+\]",  # Multiple consecutive masks
                    r"\[\w+\]\s*[,.!?]\s*$",  # Mask at sentence end
                    r"^\s*\[\w+\]\s*[,.!?]",  # Mask at sentence start
                ],
                "incomplete_phrases": [
                    r"\[\w+\]\s+is",
                    r"\[\w+\]\s+are",
                    r"\[\w+\]\s+was",
                    r"\[\w+\]\s+were",
                    r"my\s+\[\w+\]\s+is",
                    r"contact\s+\[\w+\]\s+at",
                ],
                "contextual_breaks": [
                    r"\[\w+\]\s+and\s+\[\w+\]",
                    r"\[\w+\]\s+or\s+\[\w+\]",
                    r"\[\w+\]\s+but\s+\[\w+\]",
                ],
            },
            "integrity_thresholds": {
                "minimum_coherence": 0.7,
                "maximum_masking_ratio": 0.4,
                "minimum_sentence_completeness": 0.8,
            },
            "repair_strategies": {
                "add_connectors": True,
                "fix_grammar": True,
                "improve_flow": True,
            },
        }

    def execute(self, context: PIIContext) -> StageResult:
        """
        Check semantic integrity of masked text.

        Args:
            context: PII pipeline context

        Returns:
            StageResult with integrity assessment
        """
        masked_text = context.current_text
        original_text = context.original_text
        entities = context.entities.copy()
        config = context.config.get("integrity", self.integrity_config)

        if not entities:
            return StageResult(
                success=True,
                stage_name=self.stage_name,
                execution_time_ms=0.0,
                processed_text=masked_text,
                entities=[],
                metadata={"no_entities": True},
            )

        # Step 1: Analyze grammar and syntax
        grammar_score = self._analyze_grammar(masked_text, config)

        # Step 2: Check semantic coherence
        coherence_score = self._check_coherence(
            masked_text, original_text, entities, config
        )

        # Step 3: Assess masking impact
        impact_score = self._assess_masking_impact(
            masked_text, original_text, entities, config
        )

        # Step 4: Calculate overall integrity score
        overall_integrity = self._calculate_integrity_score(
            grammar_score, coherence_score, impact_score, config
        )

        # Step 5: Apply repairs if needed
        repaired_text = masked_text
        repairs_applied: List[str] = []

        if overall_integrity < config.get("integrity_thresholds", {}).get(
            "minimum_coherence", 0.7
        ):
            repaired_text, repairs_applied = self._apply_repairs(
                masked_text, entities, config
            )
            # Recalculate integrity after repairs
            if repairs_applied:
                grammar_score = self._analyze_grammar(repaired_text, config)
                coherence_score = self._check_coherence(
                    repaired_text, original_text, entities, config
                )
                overall_integrity = self._calculate_integrity_score(
                    grammar_score, coherence_score, impact_score, config
                )

        self.add_debug_info(
            context,
            "Semantic integrity check completed",
            {
                "grammar_score": grammar_score,
                "coherence_score": coherence_score,
                "impact_score": impact_score,
                "overall_integrity": overall_integrity,
                "repairs_applied": len(repairs_applied),
                "integrity_passed": overall_integrity
                >= config.get("integrity_thresholds", {}).get("minimum_coherence", 0.7),
            },
        )

        return StageResult(
            success=overall_integrity
            >= config.get("integrity_thresholds", {}).get("minimum_coherence", 0.7),
            stage_name=self.stage_name,
            execution_time_ms=0.0,
            processed_text=repaired_text,
            entities=entities,
            metadata={
                "integrity_scores": {
                    "grammar": grammar_score,
                    "coherence": coherence_score,
                    "impact": impact_score,
                    "overall": overall_integrity,
                },
                "repairs_applied": repairs_applied,
                "integrity_passed": overall_integrity
                >= config.get("integrity_thresholds", {}).get("minimum_coherence", 0.7),
                "integrity_stats": self._get_integrity_stats(
                    masked_text, original_text, entities
                ),
            },
        )

    def _analyze_grammar(self, text: str, config: Dict[str, Any]) -> float:
        """Analyze grammar and syntax of masked text"""
        grammar_patterns = config.get("grammar_patterns", {})
        score = 0.0
        total_checks = 0

        # Check sentence structure
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        for sentence in sentences:
            total_checks += 1

            # Basic sentence structure check
            if sentence and sentence[0].isupper():
                score += 0.3

            # Check for proper ending (if not the last sentence)
            if sentence and any(char in sentence for char in [".", "!", "?"]):
                score += 0.2

            # Check for no trailing spaces
            if sentence and not sentence.endswith(" "):
                score += 0.2

            # Check for reasonable length
            if 5 <= len(sentence) <= 200:
                score += 0.3

        # Normalize score
        if total_checks > 0:
            score = score / total_checks

        return min(1.0, score)

    def _check_coherence(
        self,
        masked_text: str,
        original_text: str,
        entities: List[PIIEntity],
        config: Dict[str, Any],
    ) -> float:
        """Check semantic coherence of masked text"""
        semantic_patterns = config.get("semantic_patterns", {})
        score = 1.0  # Start with perfect score

        # Check for broken patterns
        broken_patterns = semantic_patterns.get("broken_sentences", [])
        for pattern in broken_patterns:
            matches = re.findall(pattern, masked_text)
            score -= len(matches) * 0.2

        # Check for incomplete phrases
        incomplete_patterns = semantic_patterns.get("incomplete_phrases", [])
        for pattern in incomplete_patterns:
            matches = re.findall(pattern, masked_text)
            score -= len(matches) * 0.15

        # Check for contextual breaks
        contextual_breaks = semantic_patterns.get("contextual_breaks", [])
        for pattern in contextual_breaks:
            matches = re.findall(pattern, masked_text)
            score -= len(matches) * 0.1

        # Check masking density
        mask_count = len(re.findall(r"\[\w+.*?\]", masked_text))
        word_count = len(masked_text.split())

        if word_count > 0:
            masking_ratio = mask_count / word_count
            max_ratio = config.get("integrity_thresholds", {}).get(
                "maximum_masking_ratio", 0.4
            )

            if masking_ratio > max_ratio:
                score -= (masking_ratio - max_ratio) * 0.5

        # Ensure score doesn't go below 0
        return max(0.0, score)

    def _assess_masking_impact(
        self,
        masked_text: str,
        original_text: str,
        entities: List[PIIEntity],
        config: Dict[str, Any],
    ) -> float:
        """Assess the impact of masking on text meaning"""
        # Calculate text similarity metrics
        original_words = set(original_text.lower().split())
        masked_words = set(masked_text.lower().split())

        # Remove mask tokens from comparison
        non_mask_words = {word for word in masked_words if not word.startswith("[")}

        # Calculate overlap
        if original_words:
            overlap = len(original_words.intersection(non_mask_words))
            similarity = overlap / len(original_words)
        else:
            similarity = 1.0

        # Consider entity importance
        entity_importance = self._calculate_entity_importance(entities)
        impact_penalty = (
            sum(entity_importance.values()) / len(entities) if entities else 0
        )

        # Calculate final impact score
        impact_score = similarity * (1 - impact_penalty * 0.3)

        return max(0.0, min(1.0, impact_score))

    def _calculate_entity_importance(
        self, entities: List[PIIEntity]
    ) -> Dict[str, float]:
        """Calculate importance scores for entities"""
        importance = {}

        for entity in entities:
            # Higher confidence = higher importance
            base_importance = entity.confidence

            # Entity type importance
            type_importance = {
                "name": 0.8,
                "email": 0.9,
                "phone": 0.7,
                "address": 0.6,
                "ssn": 1.0,
                "credit_card": 1.0,
            }

            type_score = type_importance.get(entity.pii_type, 0.5)

            # Length importance (longer entities might be more important)
            length_score = min(1.0, len(entity.text) / 20)

            # Combined importance
            importance[entity.text] = (base_importance + type_score + length_score) / 3

        return importance

    def _calculate_integrity_score(
        self,
        grammar_score: float,
        coherence_score: float,
        impact_score: float,
        config: Dict[str, Any],
    ) -> float:
        """Calculate overall integrity score"""
        # Weight the different components
        weights = {"grammar": 0.3, "coherence": 0.4, "impact": 0.3}

        overall_score = (
            grammar_score * weights["grammar"]
            + coherence_score * weights["coherence"]
            + impact_score * weights["impact"]
        )

        return overall_score

    def _apply_repairs(
        self, text: str, entities: List[PIIEntity], config: Dict[str, Any]
    ) -> Tuple[str, List[str]]:
        """Apply repairs to improve text integrity"""
        repaired_text = text
        repairs_applied = []

        repair_strategies = config.get("repair_strategies", {})

        # Fix consecutive masks
        if repair_strategies.get("add_connectors", True):
            pattern = r"(\[\w+\])\s*(\[\w+\])"
            matches = re.findall(pattern, repaired_text)
            for match in matches:
                repaired_text = repaired_text.replace(
                    f"{match[0]} {match[1]}", f"{match[0]} and {match[1]}"
                )
                repairs_applied.append(
                    f"Added connector between {match[0]} and {match[1]}"
                )

        # Fix incomplete phrases
        if repair_strategies.get("fix_grammar", True):
            incomplete_patterns = [
                (r"\[\w+\]\s+is", "[MASKED] is"),
                (r"\[\w+\]\s+are", "[MASKED] are"),
                (r"my\s+\[\w+\]\s+is", "my [MASKED] is"),
                (r"contact\s+\[\w+\]\s+at", "contact [MASKED] at"),
            ]

            for pattern, replacement in incomplete_patterns:
                if re.search(pattern, repaired_text):
                    repaired_text = re.sub(pattern, replacement, repaired_text)
                    repairs_applied.append(f"Fixed incomplete phrase: {pattern}")

        # Improve flow
        if repair_strategies.get("improve_flow", True):
            # Fix spacing around masks
            repaired_text = re.sub(r"\s*\[\w+\]\s*", " [MASKED] ", repaired_text)
            repaired_text = re.sub(r"\s+", " ", repaired_text).strip()

            if repaired_text != text:
                repairs_applied.append("Improved text flow and spacing")

        return repaired_text, repairs_applied

    def _get_integrity_stats(
        self, masked_text: str, original_text: str, entities: List[PIIEntity]
    ) -> Dict[str, Any]:
        """Get integrity statistics"""
        return {
            "original_length": len(original_text),
            "masked_length": len(masked_text),
            "length_change": len(masked_text) - len(original_text),
            "mask_count": len(re.findall(r"\[\w+.*?\]", masked_text)),
            "word_count_original": len(original_text.split()),
            "word_count_masked": len(masked_text.split()),
            "entity_count": len(entities),
            "masking_ratio": (
                len(re.findall(r"\[\w+.*?\]", masked_text)) / len(masked_text.split())
                if masked_text.split()
                else 0
            ),
        }

    def validate_input(self, context: PIIContext) -> bool:
        """Validate input for integrity stage"""
        entities: object = context.entities
        if not entities:
            return True  # No entities is valid

        if not isinstance(entities, list):
            return False

        if not context.current_text:
            return False

        if not context.original_text:
            return False

        for entity in entities:
            if not isinstance(entity, PIIEntity):
                return False

        return True

    def fallback(self, context: PIIContext) -> StageResult:
        """Fallback integrity check - basic validation only"""
        masked_text = context.current_text
        original_text = context.original_text
        entities = context.entities

        # Basic integrity check
        basic_score = 0.8  # Assume decent integrity

        # Check for obvious issues
        if (
            "[MASKED]" in masked_text
            and masked_text.count("[MASKED]") > len(masked_text.split()) * 0.5
        ):
            basic_score = 0.5

        # Check for broken patterns
        if "[MASKED] [MASKED]" in masked_text:
            basic_score -= 0.2

        self.add_debug_info(
            context,
            "Integrity check fallback used",
            {
                "reason": "main_integrity_failed",
                "fallback_type": "basic_validation",
                "basic_score": basic_score,
            },
        )

        return StageResult(
            success=basic_score >= 0.5,
            stage_name=self.stage_name,
            execution_time_ms=0.0,
            processed_text=masked_text,
            entities=entities,
            metadata={
                "fallback_used": True,
                "fallback_type": "basic_validation",
                "basic_integrity_score": basic_score,
            },
        )
