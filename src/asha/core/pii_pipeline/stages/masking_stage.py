"""
Stage 5: Masking Strategy Engine - Apply advanced masking strategies
"""

import hashlib
import re
from typing import Dict, Any
from .base_stage import BaseStage, StageResult, PIIContext, PIIEntity


class MaskingStage(BaseStage):
    """
    Masking Strategy Stage - Applies advanced masking strategies with consistent hashing.

    This stage handles:
    - Consistent hash-based masking
    - Multiple masking strategies
    - Reference preservation
    - Reversible mapping (optional)
    - Masking validation
    """

    def __init__(self) -> None:
        super().__init__("masking")

        # Masking configuration
        self.masking_config = {
            "strategies": {
                "hash": {
                    "format": "[{type}]_{hash}",
                    "hash_length": 8,
                    "preserve_length": False,
                },
                "preserve_length": {"format": "[{type}_{length}]", "mask_char": "*"},
                "partial": {
                    "format": "{first}{mask}{last}",
                    "mask_char": "*",
                    "preserve_chars": 2,
                },
                "generic": {"format": "[{type}]", "simple": True},
            },
            "default_strategy": "hash",
            # Structured identifiers always use stable hash tokens that match
            # the lite PIIDetector contract ([PHONE_HASH], [CREDIT_CARD_HASH], …).
            "entity_type_strategies": {
                "email": "stable_hash",
                "phone": "stable_hash",
                "ssn": "stable_hash",
                "credit_card": "stable_hash",
                "aadhaar": "stable_hash",
                "pan": "stable_hash",
                "gstin": "stable_hash",
                "api_key": "stable_hash",
                "jwt_token": "stable_hash",
                "bearer_token": "stable_hash",
                "name": "hash",
                "address": "hash",
                "zip_code": "hash",
                "ip_address": "hash",
                "url": "hash",
            },
            "stable_tokens": {
                "email": "[EMAIL_HASH]",
                "phone": "[PHONE_HASH]",
                "ssn": "[SSN_HASH]",
                "credit_card": "[CREDIT_CARD_HASH]",
                "aadhaar": "[AADHAAR_HASH]",
                "pan": "[PAN_HASH]",
                "gstin": "[GSTIN_HASH]",
                "api_key": "[API_KEY_HASH]",
                "jwt_token": "[JWT_HASH]",
                "bearer_token": "[TOKEN_HASH]",
            },
            "hash_salt": "asha_pii_masking_v1",
            "case_sensitive": True,
            "min_mask_confidence": 0.55,
        }

    def execute(self, context: PIIContext) -> StageResult:
        """
        Apply masking strategies to detected entities.

        Args:
            context: PII pipeline context

        Returns:
            StageResult with masked text and mapping
        """
        entities = context.entities.copy()
        text = context.current_text
        config = context.config.get("masking", self.masking_config)

        if not entities:
            return StageResult(
                success=True,
                stage_name=self.stage_name,
                execution_time_ms=0.0,
                processed_text=text,
                entities=[],
                metadata={"no_entities": True},
            )

        # Step 1: Sort entities by position (reverse order for masking)
        entities.sort(key=lambda e: e.start, reverse=True)

        # Step 2: Create entity mapping
        entity_mapping = {}

        # Step 3: Apply masking
        masked_text = text
        masking_stats: Dict[str, Any] = {
            "total_entities": len(entities),
            "masked_entities": 0,
            "by_type": {},
            "by_strategy": {},
        }

        for entity in entities:
            # Skip low-confidence entities: hybrid default must not mask
            # dictionary/heuristic name FPs that scoring left in the pipeline.
            if float(entity.confidence) < float(
                config.get("min_mask_confidence", 0.55)
            ):
                continue
            if entity.pii_type == "name":
                from .scoring_stage import _is_stopword_heavy_name

                if _is_stopword_heavy_name(entity.text):
                    continue

            # Determine masking strategy
            strategy = self._determine_strategy(entity, config)

            # Generate mask
            mask = self._generate_mask(entity, strategy, config)

            # Apply mask to text
            start, end = entity.start, entity.end
            masked_text = masked_text[:start] + mask + masked_text[end:]

            # Store mapping
            entity_mapping[entity.text] = {
                "mask": mask,
                "strategy": strategy,
                "entity_type": entity.pii_type,
                "confidence": entity.confidence,
                "position": (start, end),
            }

            # Update stats
            masking_stats["masked_entities"] += 1
            masking_stats["by_type"][entity.pii_type] = (
                masking_stats["by_type"].get(entity.pii_type, 0) + 1
            )
            masking_stats["by_strategy"][strategy] = (
                masking_stats["by_strategy"].get(strategy, 0) + 1
            )

        # Step 4: Validate masking
        validation_result = self._validate_masking(masked_text, entity_mapping, config)

        self.add_debug_info(
            context,
            "Masking completed",
            {
                "total_entities": len(entities),
                "masked_entities": masking_stats["masked_entities"],
                "validation_passed": validation_result["valid"],
                "strategies_used": list(masking_stats["by_strategy"].keys()),
            },
        )

        return StageResult(
            success=validation_result["valid"],
            stage_name=self.stage_name,
            execution_time_ms=0.0,
            processed_text=masked_text,
            entities=entities,
            metadata={
                "entity_mapping": entity_mapping,
                "masking_stats": masking_stats,
                "validation_result": validation_result,
                "strategies_used": list(masking_stats["by_strategy"].keys()),
            },
        )

    def _determine_strategy(self, entity: PIIEntity, config: Dict[str, Any]) -> str:
        """Determine masking strategy for an entity"""
        # Check entity type specific strategy
        entity_type_strategies = config.get("entity_type_strategies", {})
        if entity.pii_type in entity_type_strategies:
            return str(entity_type_strategies[entity.pii_type])

        # Check confidence-based strategy
        if entity.confidence >= 0.85:
            return "hash"
        elif entity.confidence >= 0.6:
            return "partial"
        else:
            return "generic"

    def _generate_mask(
        self, entity: PIIEntity, strategy: str, config: Dict[str, Any]
    ) -> str:
        """Generate mask for an entity based on strategy"""
        strategies = config.get("strategies", {})
        strategy_config = strategies.get(strategy, strategies.get("hash", {}))

        if strategy == "hash":
            return self._generate_hash_mask(entity, strategy_config, config)
        elif strategy == "stable_hash":
            return self._generate_stable_hash_mask(entity, config)
        elif strategy == "preserve_length":
            return self._generate_preserve_length_mask(entity, strategy_config, config)
        elif strategy == "partial":
            return self._generate_partial_mask(entity, strategy_config, config)
        elif strategy == "generic":
            return self._generate_generic_mask(entity, strategy_config, config)
        else:
            return self._generate_hash_mask(entity, strategies.get("hash", {}), config)

    def _generate_stable_hash_mask(
        self, entity: PIIEntity, config: Dict[str, Any]
    ) -> str:
        """Lite-compatible tokens for structured identifiers.

        Phones/cards/SSN use fixed ``[TYPE_HASH]`` labels (match lite detector).
        Emails use content-addressed ``[EMAIL]_xxxxxxxx`` so distinct addresses
        get distinct tokens (collision-safe) while remaining reversible.
        """
        if entity.pii_type == "email":
            return self._generate_hash_mask(
                entity,
                {"format": "[{type}]_{hash}", "hash_length": 8},
                config,
            )
        tokens = config.get(
            "stable_tokens", self.masking_config.get("stable_tokens", {})
        )
        token = tokens.get(entity.pii_type)
        if token:
            return str(token)
        return f"[{entity.pii_type.upper()}_HASH]"

    def _generate_hash_mask(
        self, entity: PIIEntity, strategy_config: Dict[str, Any], config: Dict[str, Any]
    ) -> str:
        """Generate hash-based mask"""
        format_template = strategy_config.get("format", "[{type}]_{hash}")
        hash_length = strategy_config.get("hash_length", 8)
        salt = config.get("hash_salt", self.masking_config["hash_salt"])

        # Generate consistent hash
        hash_input = f"{entity.text}_{entity.pii_type}_{salt}"
        if config.get("case_sensitive", True):
            hash_input = hash_input

        hash_object = hashlib.sha256(hash_input.encode())
        hash_hex = hash_object.hexdigest()[:hash_length]

        return str(format_template.format(type=entity.pii_type.upper(), hash=hash_hex))

    def _generate_preserve_length_mask(
        self, entity: PIIEntity, strategy_config: Dict[str, Any], config: Dict[str, Any]
    ) -> str:
        """Generate length-preserving mask"""
        format_template = strategy_config.get("format", "[{type}_{length}]")
        mask_char = strategy_config.get("mask_char", "*")

        return str(
            format_template.format(
                type=entity.pii_type.upper(), length=len(entity.text)
            )
        )

    def _generate_partial_mask(
        self, entity: PIIEntity, strategy_config: Dict[str, Any], config: Dict[str, Any]
    ) -> str:
        """Generate partial mask"""
        format_template = strategy_config.get("format", "{first}{mask}{last}")
        mask_char = strategy_config.get("mask_char", "*")
        preserve_chars = strategy_config.get("preserve_chars", 2)

        text = entity.text

        if len(text) <= preserve_chars * 2:
            # If text is too short, use generic mask
            return f"[{entity.pii_type.upper()}]"

        first_chars = text[:preserve_chars]
        last_chars = text[-preserve_chars:]
        mask_length = len(text) - (preserve_chars * 2)
        mask = mask_char * mask_length

        return str(
            format_template.format(first=first_chars, mask=mask, last=last_chars)
        )

    def _generate_generic_mask(
        self, entity: PIIEntity, strategy_config: Dict[str, Any], config: Dict[str, Any]
    ) -> str:
        """Generate generic mask"""
        if entity.pii_type in ("api_key", "jwt_token", "bearer_token"):
            return "[REDACTED]"
        format_template = strategy_config.get("format", "[{type}]")

        return str(format_template.format(type=entity.pii_type.upper()))

    def _validate_masking(
        self, masked_text: str, entity_mapping: Dict[str, Any], config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate masked text"""
        errors = []
        warnings = []

        # Check for unmasked PII
        original_texts = list(entity_mapping.keys())
        for original_text in original_texts:
            if original_text in masked_text:
                errors.append(f"Unmasked PII found: {original_text}")

        # Check for broken masking patterns
        mask_patterns = [
            r"\[\w+\]_\w+",  # Hash masks
            r"\[\w+_\d+\]",  # Length masks
            r"\[\w+\]",  # Generic masks
        ]

        for pattern in mask_patterns:
            matches = re.findall(pattern, masked_text)
            for match in matches:
                if not match.endswith("]"):
                    warnings.append(f"Potentially broken mask: {match}")

        # Check for excessive masking
        total_masked = len(re.findall(r"\[\w+.*?\]", masked_text))
        total_length = len(masked_text)
        masking_ratio = total_masked / total_length if total_length > 0 else 0

        if masking_ratio > 0.5:
            warnings.append(f"High masking ratio: {masking_ratio:.2f}")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "masking_ratio": masking_ratio,
        }

    def validate_input(self, context: PIIContext) -> bool:
        """Validate input for masking stage"""
        if not context.entities:
            return True

        if not context.current_text:
            return False

        return all(isinstance(entity, PIIEntity) for entity in context.entities)

    def fallback(self, context: PIIContext) -> StageResult:
        """Fallback masking - simple generic masks"""
        entities = context.entities.copy()
        text = context.current_text

        # Sort entities by position (reverse order for masking)
        entities.sort(key=lambda e: e.start, reverse=True)

        masked_text = text
        entity_mapping = {}

        for entity in entities:
            # Simple generic mask
            mask = f"[{entity.pii_type.upper()}]"

            # Apply mask
            start, end = entity.start, entity.end
            masked_text = masked_text[:start] + mask + masked_text[end:]

            # Store mapping
            entity_mapping[entity.text] = {
                "mask": mask,
                "strategy": "generic",
                "entity_type": entity.pii_type,
                "confidence": entity.confidence,
            }

        self.add_debug_info(
            context,
            "Masking fallback used",
            {
                "reason": "main_masking_failed",
                "fallback_type": "generic_masking",
                "entities_masked": len(entities),
            },
        )

        return StageResult(
            success=True,
            stage_name=self.stage_name,
            execution_time_ms=0.0,
            processed_text=masked_text,
            entities=entities,
            metadata={
                "fallback_used": True,
                "fallback_type": "generic_masking",
                "entity_mapping": entity_mapping,
                "entities_masked": len(entities),
            },
        )
