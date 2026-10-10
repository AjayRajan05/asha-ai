"""
Multi-Stage PII Detection Pipeline - Main Orchestrator

This is the main entry point for the advanced multi-stage PII detection pipeline.
It orchestrates all 7 stages to provide comprehensive PII detection and masking.
"""

from typing import Dict, Any, List, Optional
from .stages import (
    NormalizationStage,
    DetectionStage,
    ScoringStage,
    ContextStage,
    MaskingStage,
    IntegrityStage,
    VerificationStage,
)
from .stages.base_stage import PIIContext, create_pii_context, PIIEntity


class PIIPipeline:
    """
    Multi-Stage PII Detection Pipeline

    Advanced PII detection using a 7-stage compiler-style pipeline:
    1. Normalization Layer - Text preprocessing and standardization
    2. Multi-Detector Engine - Parallel PII detection
    3. Confidence Scoring Engine - Calculate and adjust confidence scores
    4. Context Resolution Engine - Understand context and adjust decisions
    5. Masking Strategy Engine - Apply advanced masking strategies
    6. Semantic Integrity Check - Ensure meaning is preserved
    7. Verification Layer - Final validation and quality assurance
    """

    def __init__(
        self, config: Optional[Dict[str, Any]] = None, debug_enabled: bool = False
    ) -> None:
        """
        Initialize the PII pipeline.

        Args:
            config: Configuration dictionary for all stages
            debug_enabled: Enable debug logging
        """
        self.config = config or {}
        self.debug_enabled = debug_enabled

        # Initialize all stages
        self.stages = {
            "normalization": NormalizationStage(),
            "detection": DetectionStage(),
            "scoring": ScoringStage(),
            "context": ContextStage(),
            "masking": MaskingStage(),
            "integrity": IntegrityStage(),
            "verification": VerificationStage(),
        }

        # Default configuration
        self.default_config: Dict[str, Dict[str, Any]] = {
            "normalization": {
                "enable_unicode_normalization": True,
                "enable_obfuscation_removal": True,
                "enable_format_standardization": True,
                "enable_selective_lowercase": True,
            },
            "detection": {
                "enable_regex": True,
                "enable_heuristics": True,
                "enable_dictionary": True,
                "enable_contextual": True,
                "parallel_execution": True,
                "min_confidence": 0.3,
            },
            "scoring": {
                "base_confidence_weights": {
                    "regex": 0.9,
                    "heuristic": 0.6,
                    "dictionary": 0.7,
                    "contextual": 0.8,
                },
                "thresholds": {
                    "high_confidence": 0.85,
                    "medium_confidence": 0.6,
                    "low_confidence": 0.3,
                    "minimum_confidence": 0.55,
                },
            },
            "context": {
                "intent_patterns": {
                    "teaching": {
                        "keywords": ["example", "demo", "test", "sample"],
                        "confidence_reduction": 0.15,
                    },
                    "personal": {
                        "keywords": ["my", "personal", "private"],
                        "confidence_boost": 0.15,
                    },
                }
            },
            "masking": {
                "default_strategy": "hash",
                "entity_type_strategies": {
                    "email": "stable_hash",
                    "phone": "stable_hash",
                    "ssn": "stable_hash",
                    "credit_card": "stable_hash",
                    "aadhaar": "stable_hash",
                    "pan": "stable_hash",
                    "gstin": "stable_hash",
                    "name": "hash",
                    "address": "hash",
                },
                "stable_tokens": {
                    "email": "[EMAIL_HASH]",
                    "phone": "[PHONE_HASH]",
                    "ssn": "[SSN_HASH]",
                    "credit_card": "[CREDIT_CARD_HASH]",
                    "aadhaar": "[AADHAAR_HASH]",
                    "pan": "[PAN_HASH]",
                    "gstin": "[GSTIN_HASH]",
                },
                "hash_salt": "asha_pii_masking_v1",
                "min_mask_confidence": 0.55,
            },
            "integrity": {
                "integrity_thresholds": {
                    "minimum_coherence": 0.5,
                    "maximum_masking_ratio": 0.6,
                }
            },
            "verification": {
                "quality_thresholds": {
                    "minimum_masking_coverage": 0.8,
                    "maximum_false_positive_rate": 0.1,
                }
            },
        }

        # Merge with provided config
        self._merge_config()

    def _merge_config(self) -> None:
        """Merge default config with provided config"""
        for stage_name, stage_config in self.default_config.items():
            if stage_name not in self.config:
                self.config[stage_name] = {}

            # Deep merge
            for key, value in stage_config.items():
                if key not in self.config[stage_name]:
                    self.config[stage_name][key] = value

    def process(
        self, text: str, config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process text through the complete PII pipeline.

        Args:
            text: Input text to process
            config: Optional override configuration

        Returns:
            Complete processing result with all stages and metrics
        """
        # Input validation
        if not text or not isinstance(text, str):
            return self._create_error_result("Invalid input text", text)

        # Merge runtime config
        runtime_config = self.config.copy()
        if config:
            for stage_name, stage_config in config.items():
                if stage_name not in runtime_config:
                    runtime_config[stage_name] = {}
                runtime_config[stage_name].update(stage_config)

        # Create pipeline context
        context = create_pii_context(text, runtime_config, self.debug_enabled)

        # Process through all stages
        try:
            final_result = self._process_stages(context)
            return final_result

        except Exception as e:
            if self.debug_enabled:
                print(f"[PII Pipeline] Critical error: {e}")

            return self._create_error_result(str(e), text, context)

    def _process_stages(self, context: PIIContext) -> Dict[str, Any]:
        """Process text through all pipeline stages."""
        stage_order = [
            "normalization",
            "detection",
            "scoring",
            "context",
            "masking",
            "integrity",
            "verification",
        ]

        for stage_name in stage_order:
            if self.debug_enabled:
                print(f"[PII Pipeline] Executing stage: {stage_name}")

            stage = self.stages[stage_name]
            result = stage.process(context)

            if not result.success:
                if self.debug_enabled:
                    print(f"[PII Pipeline] Stage {stage_name} failed: {result.error}")

                # Attempt stage fallback before failing
                if hasattr(stage, "fallback"):
                    fallback_result = stage.fallback(context)
                    if fallback_result.success:
                        context.add_stage_result(fallback_result)
                        continue

                if result.error and "fallback" in str(result.error).lower():
                    continue

                raise RuntimeError(
                    f"Stage {stage_name} failed: {result.error or 'unknown error'}"
                )

        # Return final result
        return self._create_final_result(context)

    def _create_final_result(self, context: PIIContext) -> Dict[str, Any]:
        """Create final result from pipeline context."""
        verification_result = context.get_stage_result("verification")

        return {
            "success": True,
            "session_id": context.session_id,
            "original_text": context.original_text,
            "masked_text": context.current_text,
            "entities": [self._entity_to_dict(entity) for entity in context.entities],
            "stage_results": self._compile_stage_results(context),
            "verification_report": (
                verification_result.metadata.get("verification_report")
                if verification_result
                else None
            ),
            "masking_map": self._extract_masking_map(context),
            "processing_summary": {
                "total_stages": len(context.stage_results),
                "successful_stages": len(context.get_successful_stages()),
                "failed_stages": len(context.get_failed_stages()),
                "total_execution_time_ms": context.get_total_execution_time(),
                "entities_detected": len(context.entities),
                "masks_applied": len(
                    [entity for entity in context.entities if entity.confidence >= 0.5]
                ),
            },
            "pipeline_info": {
                "version": "2.0",
                "architecture": "multi_stage_deterministic",
                "stages": list(self.stages.keys()),
                "config_used": context.config,
            },
        }

    def _entity_to_dict(self, entity: PIIEntity) -> Dict[str, Any]:
        """Convert PIIEntity to dictionary."""
        return {
            "text": entity.text,
            "start": entity.start,
            "end": entity.end,
            "pii_type": entity.pii_type,
            "confidence": entity.confidence,
            "context": entity.context,
            "metadata": entity.metadata,
        }

    def _compile_stage_results(self, context: PIIContext) -> Dict[str, Any]:
        """Compile results from all stages."""
        stage_results = {}

        for stage_name, result in context.stage_results.items():
            stage_results[stage_name] = {
                "success": result.success,
                "execution_time_ms": result.execution_time_ms,
                "entities_count": len(result.entities) if result.entities else 0,
                "metadata": result.metadata,
                "error": result.error,
            }

        return stage_results

    def _extract_masking_map(self, context: PIIContext) -> Dict[str, str]:
        """Build token -> original map from masking stage metadata."""
        masking_result = context.get_stage_result("masking")
        if not masking_result or not masking_result.metadata:
            return {}
        entity_mapping = masking_result.metadata.get("entity_mapping", {})
        token_map: Dict[str, str] = {}
        for original, info in entity_mapping.items():
            if isinstance(info, dict) and "mask" in info:
                token_map[info["mask"]] = original
        return token_map

    def _create_error_result(
        self, error: str, original_text: str, context: Optional[PIIContext] = None
    ) -> Dict[str, Any]:
        """Create error result."""
        return {
            "success": False,
            "error": error,
            "original_text": original_text,
            "masked_text": original_text,  # Return original on error
            "entities": [],
            "stage_results": (self._compile_stage_results(context) if context else {}),
            "verification_report": None,
            "processing_summary": {
                "total_stages": len(context.stage_results) if context else 0,
                "successful_stages": (
                    len(context.get_successful_stages()) if context else 0
                ),
                "failed_stages": len(context.get_failed_stages()) if context else 0,
                "total_execution_time_ms": (
                    context.get_total_execution_time() if context else 0
                ),
                "entities_detected": 0,
                "masks_applied": 0,
            },
            "pipeline_info": {
                "version": "2.0",
                "architecture": "multi_stage_deterministic",
                "stages": list(self.stages.keys()),
                "error_occurred": True,
            },
        }

    def get_stage_info(self) -> Dict[str, Any]:
        """Get information about all pipeline stages."""
        return {
            "stages": list(self.stages.keys()),
            "total_stages": len(self.stages),
            "configuration": self.config,
            "debug_enabled": self.debug_enabled,
            "architecture": "multi_stage_deterministic",
            "version": "2.0",
        }

    def configure_stage(self, stage_name: str, config: Dict[str, Any]) -> None:
        """
        Configure a specific pipeline stage.

        Args:
            stage_name: Name of the stage to configure
            config: Configuration dictionary for the stage
        """
        if stage_name not in self.stages:
            raise ValueError(f"Unknown stage: {stage_name}")

        if stage_name not in self.config:
            self.config[stage_name] = {}

        self.config[stage_name].update(config)

    def get_entities_by_type(
        self, text: str, pii_types: Optional[List[str]] = None
    ) -> List[PIIEntity]:
        """
        Get entities of specific types from text.

        Args:
            text: Input text to process
            pii_types: List of PII types to filter for

        Returns:
            List of PII entities
        """
        result = self.process(text)

        entities = [PIIEntity(**entity) for entity in result.get("entities", [])]

        if pii_types:
            entities = [entity for entity in entities if entity.pii_type in pii_types]

        return entities

    def get_masked_text(self, text: str) -> str:
        """
        Get masked text (convenience method).

        Args:
            text: Input text to process

        Returns:
            Masked text
        """
        result = self.process(text)
        masked = result.get("masked_text", text)
        return str(masked)

    def get_verification_report(self, text: str) -> Dict[str, Any]:
        """
        Get detailed verification report.

        Args:
            text: Input text to process

        Returns:
            Verification report
        """
        result = self.process(text)
        report = result.get("verification_report", {})
        return dict(report) if isinstance(report, dict) else {}

    def reset(self) -> None:
        """Reset pipeline to initial state."""
        self.stages = {
            "normalization": NormalizationStage(),
            "detection": DetectionStage(),
            "scoring": ScoringStage(),
            "context": ContextStage(),
            "masking": MaskingStage(),
            "integrity": IntegrityStage(),
            "verification": VerificationStage(),
        }

    def __len__(self) -> int:
        """Return number of stages in pipeline."""
        return len(self.stages)

    def __getitem__(self, stage_name: str) -> Any:
        """Get a specific stage by name."""
        return self.stages.get(stage_name)

    def __contains__(self, stage_name: str) -> bool:
        """Check if stage exists in pipeline."""
        return stage_name in self.stages


# Convenience functions for backward compatibility
def detect_pii(text: str, config: Optional[Dict[str, Any]] = None) -> List[PIIEntity]:
    """
    Convenience function to detect PII in text.

    Args:
        text: Input text to process
        config: Optional configuration

    Returns:
        List of detected PII entities
    """
    pipeline = PIIPipeline(config)
    result = pipeline.process(text)
    return [PIIEntity(**entity) for entity in result.get("entities", [])]


def mask_pii(text: str, config: Optional[Dict[str, Any]] = None) -> str:
    """
    Convenience function to mask PII in text.

    Args:
        text: Input text to process
        config: Optional configuration

    Returns:
        Masked text
    """
    pipeline = PIIPipeline(config)
    return pipeline.get_masked_text(text)


def verify_pii_masking(
    text: str, config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Convenience function to verify PII masking.

    Args:
        text: Input text to process
        config: Optional configuration

    Returns:
        Verification report
    """
    pipeline = PIIPipeline(config)
    return pipeline.get_verification_report(text)
