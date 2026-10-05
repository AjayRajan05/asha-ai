"""
Stage 4: Context Resolution Engine - Understand context and adjust detection decisions
"""

import re
from typing import Dict, Any, List
from .base_stage import BaseStage, StageResult, PIIContext, PIIEntity


class ContextStage(BaseStage):
    """
    Context Resolution Stage - Understands context and adjusts PII detection decisions.

    This stage handles:
    - Intent detection (teaching vs real data)
    - Role analysis (example vs personal)
    - Contextual confidence adjustment
    - Entity relationship analysis
    - Semantic context understanding
    """

    def __init__(self) -> None:
        super().__init__("context")

        # Context resolution configuration
        self.context_config = {
            "intent_patterns": {
                "teaching": {
                    "keywords": [
                        "example",
                        "demo",
                        "test",
                        "sample",
                        "illustration",
                        "tutorial",
                    ],
                    "confidence_reduction": 0.3,
                    "patterns": [
                        r"for example",
                        r"here is an example",
                        r"this is a test",
                        r"sample data",
                        r"demo account",
                    ],
                },
                "documentation": {
                    "keywords": [
                        "documentation",
                        "manual",
                        "guide",
                        "reference",
                        "readme",
                    ],
                    "confidence_reduction": 0.2,
                    "patterns": [
                        r"in this documentation",
                        r"for reference",
                        r"see the manual",
                    ],
                },
                "personal": {
                    "keywords": ["my", "personal", "private", "real", "actual"],
                    "confidence_boost": 0.15,
                    "patterns": [
                        r"my email is",
                        r"my phone number",
                        r"contact me at",
                        r"reach me at",
                        r"my address is",
                    ],
                },
            },
            "entity_relationships": {
                "name_email": {
                    "boost": 0.1,
                    "patterns": [
                        r"(\w+\s+\w+)\s*<([^>]+)>",  # Name <email>
                        r"(\w+\s+\w+)\s*\(\s*([^)]+)\s*\)",  # Name (email)
                    ],
                },
                "address_phone": {
                    "boost": 0.05,
                    "patterns": [
                        r"(\d+[^,.!?]*\d+)\s*[,.]?\s*(\d{3}[-.\s]\d{3}[-.\s]\d{4})"
                    ],
                },
            },
            "semantic_context": {
                "business_context": {
                    "keywords": ["company", "corporate", "business", "office", "work"],
                    "confidence_adjustment": 0.05,
                },
                "personal_context": {
                    "keywords": ["personal", "private", "home", "family", "individual"],
                    "confidence_adjustment": 0.1,
                },
                "technical_context": {
                    "keywords": [
                        "code",
                        "programming",
                        "development",
                        "technical",
                        "system",
                    ],
                    "confidence_adjustment": -0.1,
                },
            },
        }

    def execute(self, context: PIIContext) -> StageResult:
        """
        Analyze context and adjust entity confidence scores.

        Args:
            context: PII pipeline context

        Returns:
            StageResult with context-adjusted entities
        """
        entities = context.entities.copy()
        config = context.config.get("context", self.context_config)

        # Intent is a property of the text, not of surviving entities; always
        # detect it (teaching placeholders may leave the entity list empty).
        intent = self._detect_intent(context.current_text, config)

        if not entities:
            return StageResult(
                success=True,
                stage_name=self.stage_name,
                execution_time_ms=0.0,
                entities=[],
                metadata={"no_entities": True, "detected_intent": intent},
            )

        # Step 2: Apply intent-based adjustments
        entities = self._apply_intent_adjustments(entities, intent, config)

        # Step 3: Analyze entity relationships
        entities = self._analyze_entity_relationships(entities, context, config)

        # Step 4: Apply semantic context adjustments
        entities = self._apply_semantic_context(entities, context, config)

        # Step 5: Final context validation
        entities = self._validate_context(entities, context, config)

        self.add_debug_info(
            context,
            "Context resolution completed",
            {
                "detected_intent": intent,
                "total_entities": len(context.entities),
                "adjusted_entities": len(entities),
                "context_adjustments": self._count_adjustments(entities),
            },
        )

        return StageResult(
            success=True,
            stage_name=self.stage_name,
            execution_time_ms=0.0,
            entities=entities,
            metadata={
                "detected_intent": intent,
                "total_entities": len(context.entities),
                "adjusted_entities": len(entities),
                "context_adjustments": self._count_adjustments(entities),
                "context_stats": self._get_context_stats(entities, intent),
            },
        )

    def _detect_intent(self, text: str, config: Dict[str, Any]) -> str:
        """Detect the overall intent of the text"""
        # Strip emails/URLs so domain tokens like example.com do not trigger
        # teaching intent for real contact data.
        text_lower = re.sub(
            r"[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}",
            " ",
            text.lower(),
        )
        text_lower = re.sub(r"https?://\S+", " ", text_lower)
        intent_scores: Dict[str, int] = {}

        for intent_name, intent_config in config.get("intent_patterns", {}).items():
            score = 0

            # Check keywords as whole words
            for keyword in intent_config.get("keywords", []):
                if re.search(rf"\b{re.escape(keyword)}\b", text_lower):
                    score += 1

            # Check patterns
            for pattern in intent_config.get("patterns", []):
                if re.search(pattern, text_lower):
                    score += 2

            intent_scores[intent_name] = score

        # Return intent with highest score
        if intent_scores:
            detected_intent = max(intent_scores, key=lambda k: intent_scores[k])
            if intent_scores[detected_intent] > 0:
                return detected_intent

        return "neutral"

    _STRUCTURED_PII = frozenset(
        {
            "email",
            "phone",
            "ssn",
            "credit_card",
            "aadhaar",
            "pan",
            "api_key",
            "jwt_token",
            "bearer_token",
        }
    )

    def _apply_intent_adjustments(
        self, entities: List[PIIEntity], intent: str, config: Dict[str, Any]
    ) -> List[PIIEntity]:
        """Apply intent-based confidence adjustments"""
        intent_patterns = config.get("intent_patterns", {})

        if intent not in intent_patterns:
            return entities

        intent_config = intent_patterns[intent]

        for entity in entities:
            original_confidence = entity.confidence

            # Structured identifiers keep detection strength under teaching/docs
            # intent: "example" often appears inside domains (example.com) without
            # meaning the value is a placeholder.
            if (
                intent in ("teaching", "documentation")
                and entity.pii_type in self._STRUCTURED_PII
            ):
                from ...security.patterns import is_example_email

                if entity.pii_type == "email" and is_example_email(entity.text):
                    entity.confidence -= intent_config.get("confidence_reduction", 0.3)
                # else: leave structured confidence untouched
            elif intent == "teaching":
                entity.confidence -= intent_config.get("confidence_reduction", 0.3)
            elif intent == "documentation":
                entity.confidence -= intent_config.get("confidence_reduction", 0.2)
            elif intent == "personal":
                entity.confidence += intent_config.get("confidence_boost", 0.15)

            # Ensure confidence stays within bounds
            entity.confidence = max(0.0, min(1.0, entity.confidence))

            # Store adjustment information
            entity.metadata["intent_adjustment"] = {
                "intent": intent,
                "adjustment": entity.confidence - original_confidence,
                "original_confidence": original_confidence,
            }

        return entities

    def _analyze_entity_relationships(
        self, entities: List[PIIEntity], context: PIIContext, config: Dict[str, Any]
    ) -> List[PIIEntity]:
        """Analyze relationships between entities"""
        relationships = config.get("entity_relationships", {})
        text = context.current_text

        for relationship_name, relationship_config in relationships.items():
            for pattern in relationship_config.get("patterns", []):
                matches = re.finditer(pattern, text, re.IGNORECASE)

                for match in matches:
                    # Find entities that match the pattern groups
                    self._boost_related_entities(
                        entities, match, relationship_config.get("boost", 0.1)
                    )

        return entities

    def _boost_related_entities(
        self, entities: List[PIIEntity], match: re.Match, boost: float
    ) -> None:
        """Boost confidence of entities that are related"""
        # Find entities that match the pattern groups
        for i, group in enumerate(match.groups()):
            for entity in entities:
                if entity.text.lower() == group.lower():
                    original_confidence = entity.confidence
                    entity.confidence = min(1.0, entity.confidence + boost)

                    # Store relationship boost information
                    if "relationship_boosts" not in entity.metadata:
                        entity.metadata["relationship_boosts"] = []

                    entity.metadata["relationship_boosts"].append(
                        {
                            "boost": boost,
                            "original_confidence": original_confidence,
                            "new_confidence": entity.confidence,
                            "related_text": group,
                        }
                    )

    def _apply_semantic_context(
        self, entities: List[PIIEntity], context: PIIContext, config: Dict[str, Any]
    ) -> List[PIIEntity]:
        """Apply semantic context adjustments"""
        semantic_config = config.get("semantic_context", {})
        text_lower = context.current_text.lower()

        # Determine semantic context
        detected_contexts = []
        for context_name, context_config in semantic_config.items():
            keywords = context_config.get("keywords", [])
            if any(keyword in text_lower for keyword in keywords):
                detected_contexts.append(context_name)

        # Apply context adjustments
        for entity in entities:
            original_confidence = entity.confidence

            for context_name in detected_contexts:
                context_config = semantic_config[context_name]
                adjustment = context_config.get("confidence_adjustment", 0)
                entity.confidence += adjustment

            # Ensure confidence stays within bounds
            entity.confidence = max(0.0, min(1.0, entity.confidence))

            # Store semantic context information
            entity.metadata["semantic_context"] = {
                "detected_contexts": detected_contexts,
                "adjustment": entity.confidence - original_confidence,
                "original_confidence": original_confidence,
            }

        return entities

    def _validate_context(
        self, entities: List[PIIEntity], context: PIIContext, config: Dict[str, Any]
    ) -> List[PIIEntity]:
        """Final context validation and filtering"""
        validated_entities = []

        for entity in entities:
            # Additional validation rules
            if self._is_valid_context(entity, context):
                validated_entities.append(entity)
            else:
                # Mark as invalid in metadata
                entity.metadata["context_validation"] = "invalid"
                entity.metadata["validation_reason"] = self._get_validation_reason(
                    entity, context
                )

        return validated_entities

    def _is_valid_context(self, entity: PIIEntity, context: PIIContext) -> bool:
        """Check if entity is valid in current context"""
        text_lower = context.current_text.lower()
        entity_text_lower = entity.text.lower()

        from ...security.patterns import is_example_email

        # True teaching placeholders (user@example.com), not every *@example.com
        if entity.pii_type == "email" and is_example_email(entity.text):
            return False

        # Soft types only: shared substring like "example" in a domain must not
        # invalidate structured PII that should still be masked.
        if entity.pii_type not in self._STRUCTURED_PII:
            example_indicators = ["example", "sample", "test", "demo", "fake"]
            for indicator in example_indicators:
                if indicator in text_lower and indicator in entity_text_lower:
                    return False

        # Rule: If confidence is too low after context adjustment, mark as invalid
        if entity.confidence < 0.3:
            return False

        return True

    def _get_validation_reason(self, entity: PIIEntity, context: PIIContext) -> str:
        """Get reason for context validation failure"""
        text_lower = context.current_text.lower()
        entity_text_lower = entity.text.lower()

        from ...security.patterns import is_example_email

        if entity.pii_type == "email" and is_example_email(entity.text):
            return "Known placeholder/teaching email"

        if entity.pii_type not in self._STRUCTURED_PII:
            example_indicators = ["example", "sample", "test", "demo", "fake"]
            for indicator in example_indicators:
                if indicator in text_lower and indicator in entity_text_lower:
                    return f"Contains example indicator: {indicator}"

        # Check confidence
        if entity.confidence < 0.3:
            return f"Low confidence: {entity.confidence:.2f}"

        return "Unknown validation failure"

    def _count_adjustments(self, entities: List[PIIEntity]) -> Dict[str, int]:
        """Count different types of adjustments made"""
        counts = {
            "intent_adjustments": 0,
            "relationship_boosts": 0,
            "semantic_adjustments": 0,
            "context_validations": 0,
        }

        for entity in entities:
            if "intent_adjustment" in entity.metadata:
                counts["intent_adjustments"] += 1

            if "relationship_boosts" in entity.metadata:
                counts["relationship_boosts"] += len(
                    entity.metadata["relationship_boosts"]
                )

            if "semantic_context" in entity.metadata:
                counts["semantic_adjustments"] += 1

            if "context_validation" in entity.metadata:
                counts["context_validations"] += 1

        return counts

    def _get_context_stats(
        self, entities: List[PIIEntity], intent: str
    ) -> Dict[str, Any]:
        """Get context resolution statistics"""
        if not entities:
            return {}

        return {
            "detected_intent": intent,
            "total_entities": len(entities),
            "average_confidence_after_context": sum(e.confidence for e in entities)
            / len(entities),
            "entities_by_type": self._count_entities_by_type(entities),
            "confidence_by_intent": self._get_confidence_by_intent(entities, intent),
        }

    def _count_entities_by_type(self, entities: List[PIIEntity]) -> Dict[str, int]:
        """Count entities by type"""
        counts: Dict[str, int] = {}
        for entity in entities:
            counts[entity.pii_type] = counts.get(entity.pii_type, 0) + 1
        return counts

    def _get_confidence_by_intent(
        self, entities: List[PIIEntity], intent: str
    ) -> Dict[str, Any]:
        """Get confidence statistics by intent"""
        confidences = [e.confidence for e in entities]

        return {
            "intent": intent,
            "average_confidence": sum(confidences) / len(confidences),
            "max_confidence": max(confidences),
            "min_confidence": min(confidences),
        }

    def validate_input(self, context: PIIContext) -> bool:
        """Validate input for context stage"""
        entities: object = context.entities
        if not entities:
            return True  # No entities is valid

        if not isinstance(entities, list):
            return False

        for entity in entities:
            if not isinstance(entity, PIIEntity):
                return False

        return True

    def fallback(self, context: PIIContext) -> StageResult:
        """Fallback context resolution - basic intent detection"""
        entities = context.entities.copy()

        # Basic intent detection
        text_lower = context.current_text.lower()
        intent = "neutral"

        if any(word in text_lower for word in ["example", "test", "sample", "demo"]):
            intent = "teaching"
        elif any(word in text_lower for word in ["my", "personal", "private"]):
            intent = "personal"

        # Apply basic adjustments (structured PII kept intact under teaching)
        for entity in entities:
            if intent == "teaching" and entity.pii_type not in self._STRUCTURED_PII:
                entity.confidence *= 0.7
            elif intent == "personal":
                entity.confidence *= 1.1

            entity.confidence = max(0.0, min(1.0, entity.confidence))
            entity.metadata["fallback_context"] = {"intent": intent}

        self.add_debug_info(
            context,
            "Context resolution fallback used",
            {
                "reason": "main_context_failed",
                "fallback_type": "basic_intent",
                "detected_intent": intent,
                "entities_processed": len(entities),
            },
        )

        return StageResult(
            success=True,
            stage_name=self.stage_name,
            execution_time_ms=0.0,
            entities=entities,
            metadata={
                "fallback_used": True,
                "fallback_type": "basic_intent",
                "detected_intent": intent,
                "entities_processed": len(entities),
            },
        )
