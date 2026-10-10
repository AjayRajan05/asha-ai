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
Hybrid PII Engine - Multi-Stage Pipeline Implementation

This implementation uses the advanced 7-stage PIIPipeline internally
while providing the same interface as the original HybridPIIDetector.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from .pii_pipeline.pii_pipeline import PIIPipeline


@dataclass
class PIIEntity:
    """Detected PII entity with confidence and context (v1 compatible)."""

    text: str
    category: Any  # Supports string or PIICategory enum
    confidence: float
    start_pos: int
    end_pos: int
    context: str
    source: str
    metadata: Dict[str, Any]


@dataclass
class HybridResult:
    """Result of hybrid PII detection (v1 compatible)."""

    entities: List[PIIEntity]
    masked_text: str
    confidence_score: float
    processing_time_ms: float
    method_breakdown: Dict[str, Any]
    masking_map: Optional[Dict[str, str]] = None


class HybridPIIDetector:
    """
    Main Hybrid PII Detector - Multi-Stage Pipeline Wrapper

    This class provides backward compatibility while using the new advanced
    multi-stage PII detection pipeline internally.
    """

    def __init__(
        self,
        pii_mode: str = "hybrid",
        model_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        debug_enabled: bool = False,
    ) -> None:
        """
        Initialize the hybrid PII detector.

        Args:
            pii_mode: PII detection mode (hybrid, lite, ml_only)
            model_name: Custom model name for ML detection
            config: Configuration dictionary
            debug_enabled: Enable debug logging
        """
        from .ml_utils import validate_pii_mode

        self.pii_mode = validate_pii_mode(pii_mode)
        self.config = config or {}
        self.debug_enabled = debug_enabled

        # Add pii_mode to config for the pipeline
        if "detection" not in self.config:
            self.config["detection"] = {}
        self.config["detection"]["mode"] = pii_mode
        if model_name:
            self.config["detection"]["model_name"] = model_name

        # Initialize the new multi-stage pipeline
        self.pipeline = PIIPipeline(
            config=self.config, debug_enabled=self.debug_enabled
        )

    def detect_and_mask(self, text: str, mask_pii: bool = True) -> HybridResult:
        """
        Detect and optionally mask PII in text.

        Args:
            text: Input text to process
            mask_pii: Whether to mask detected PII

        Returns:
            HybridResult with detected entities and masked text
        """
        # Process through the pipeline
        # If mask_pii is False, we might need to adjust config temporarily
        runtime_config = {}
        if not mask_pii:
            runtime_config["masking"] = {"enabled": False}

        result = self.pipeline.process(text, runtime_config)

        # Convert to v1 compatible format
        entities = []
        for entity_data in result.get("entities", []):
            entity = PIIEntity(
                text=entity_data["text"],
                category=entity_data["pii_type"],
                confidence=entity_data["confidence"],
                start_pos=entity_data["start"],
                end_pos=entity_data["end"],
                context=entity_data.get("context", ""),
                source="modular_pipeline",
                metadata=entity_data.get("metadata", {}),
            )
            entities.append(entity)

        # Calculate overall confidence
        confidence_score = 0.0
        if entities:
            confidence_score = sum(entity.confidence for entity in entities) / len(
                entities
            )

        # Get processing time
        processing_summary = result.get("processing_summary", {})
        processing_time_ms = processing_summary.get("total_execution_time_ms", 0.0)

        # Create method breakdown
        method_breakdown = {
            "pipeline_version": "2.0",
            "stages_used": result.get("pipeline_info", {}).get("stages", []),
            "entities_count": len(entities),
            "actual_mode": self.pii_mode,
        }

        return HybridResult(
            entities=entities,
            masked_text=result.get("masked_text", text) if mask_pii else text,
            confidence_score=confidence_score,
            processing_time_ms=processing_time_ms,
            method_breakdown=method_breakdown,
            masking_map=result.get("masking_map") or {},
        )

    def detect(self, text: str) -> List[PIIEntity]:
        """Detect PII entities (v1 compatible)."""
        result = self.detect_and_mask(text, mask_pii=False)
        return result.entities

    def get_detection_stats(self) -> Dict[str, Any]:
        """Get detection statistics."""
        return self.pipeline.get_stage_info()


# Convenience functions for backward compatibility
def detect_pii(
    text: str, enable_ml: bool = True, mask_pii: bool = True
) -> HybridResult:
    """
    Detect PII in text using hybrid approach (v1 compatible).

    Args:
        text: Input text to analyze
        enable_ml: Enable ML-based detection
        mask_pii: Whether to mask detected PII

    Returns:
        HybridResult with detection results
    """
    mode = "hybrid" if enable_ml else "lite"
    detector = HybridPIIDetector(pii_mode=mode)
    return detector.detect_and_mask(text, mask_pii)
