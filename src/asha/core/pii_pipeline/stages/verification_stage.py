"""
Stage 7: Verification Layer - Final validation and quality assurance
"""

import re
from typing import Dict, Any, List
from .base_stage import BaseStage, StageResult, PIIContext, PIIEntity


class VerificationStage(BaseStage):
    """
    Verification Stage - Final validation and quality assurance of PII detection and masking.

    This stage handles:
    - Final PII leak detection
    - Masking consistency validation
    - Quality assurance checks
    - Performance metrics collection
    - Final result validation
    """

    def __init__(self) -> None:
        super().__init__("verification")

        # Verification configuration
        self.verification_config = {
            "pii_leak_patterns": {
                "email": [
                    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
                    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\.[A-Z|a-z]{2,}\b",
                ],
                "phone": [
                    r"\b\d{3}-\d{3}-\d{4}\b",
                    r"\b\(\d{3}\)\s*\d{3}-\d{4}\b",
                    r"\b\d{3}\.\d{3}\.\d{4}\b",
                    r"\b\d{10}\b",
                ],
                "ssn": [r"\b\d{3}-\d{2}-\d{4}\b", r"\b\d{3}\s*\d{2}\s*\d{4}\b"],
                "credit_card": [
                    r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
                    r"\b\d{4}\s\d{4}\s\d{4}\s\d{4}\b",
                ],
                "api_key": [
                    r"\bsk-[a-zA-Z0-9]{20,}\b",
                    r"\bsk-proj-[a-zA-Z0-9_-]{20,}\b",
                    r"\bAKIA[0-9A-Z]{16}\b",
                ],
                "jwt_token": [
                    r"\beyJ[a-zA-Z0-9_-]+\.eyJ[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+\b",
                ],
                "bearer_token": [r"(?i)Bearer\s+[a-zA-Z0-9._\-]{20,}"],
            },
            "masking_consistency": {
                "same_entity_same_mask": True,
                "mask_format_validation": True,
                "no_partial_masks": True,
            },
            "quality_thresholds": {
                "minimum_masking_coverage": 0.8,
                "maximum_false_positive_rate": 0.1,
                "minimum_processing_quality": 0.7,
            },
            "performance_metrics": {
                "track_execution_time": True,
                "track_memory_usage": True,
                "track_accuracy": True,
            },
        }

    def execute(self, context: PIIContext) -> StageResult:
        """
        Perform final verification of PII detection and masking.

        Args:
            context: PII pipeline context

        Returns:
            StageResult with verification results
        """
        masked_text = context.current_text
        original_text = context.original_text
        entities = context.entities.copy()
        config = context.config.get("verification", self.verification_config)

        # Step 1: Check for PII leaks
        leak_detection = self._check_pii_leaks(masked_text, config)

        # Step 2: Validate masking consistency
        consistency_check = self._validate_masking_consistency(
            masked_text, entities, config
        )

        # Step 3: Quality assurance
        quality_check = self._perform_quality_assurance(
            original_text, masked_text, entities, config
        )

        # Step 4: Collect performance metrics
        performance_metrics = self._collect_performance_metrics(context, config)

        # Step 5: Final validation
        final_validation = self._final_validation(
            leak_detection, consistency_check, quality_check, config, context
        )

        # Step 6: Generate verification report
        verification_report = self._generate_verification_report(
            leak_detection,
            consistency_check,
            quality_check,
            performance_metrics,
            final_validation,
        )

        self.add_debug_info(
            context,
            "Verification completed",
            {
                "pii_leaks_found": len(leak_detection["leaks"]),
                "consistency_passed": consistency_check["passed"],
                "quality_score": quality_check["overall_score"],
                "final_validation_passed": final_validation["passed"],
                "verification_passed": final_validation["passed"],
            },
        )

        return StageResult(
            success=final_validation["passed"],
            stage_name=self.stage_name,
            execution_time_ms=0.0,
            processed_text=masked_text,
            entities=entities,
            metadata={
                "verification_report": verification_report,
                "leak_detection": leak_detection,
                "consistency_check": consistency_check,
                "quality_check": quality_check,
                "performance_metrics": performance_metrics,
                "final_validation": final_validation,
            },
        )

    def _check_pii_leaks(self, text: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Check for any remaining PII leaks in masked text"""
        leak_patterns = config.get("pii_leak_patterns", {})
        leaks = []

        for pii_type, patterns in leak_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)

                for match in matches:
                    leak = {
                        "type": pii_type,
                        "text": match.group(),
                        "start": match.start(),
                        "end": match.end(),
                        "pattern": pattern,
                    }
                    leaks.append(leak)

        return {
            "leaks": leaks,
            "leak_count": len(leaks),
            "leak_types": list(set(leak["type"] for leak in leaks)),
            "passed": len(leaks) == 0,
        }

    def _validate_masking_consistency(
        self, text: str, entities: List[PIIEntity], config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate masking consistency"""
        consistency_config = config.get("masking_consistency", {})
        issues = []

        # Check for same entity same mask
        if consistency_config.get("same_entity_same_mask", True):
            entity_masks: Dict[str, List[str]] = {}

            # Extract masks from text
            mask_pattern = r"\[(\w+)(?:_[^]]+)?\]"
            matches = re.finditer(mask_pattern, text)

            for match in matches:
                mask_type = match.group(1)
                mask_text = match.group()

                if mask_type not in entity_masks:
                    entity_masks[mask_type] = []
                entity_masks[mask_type].append(mask_text)

            # Check for consistency within types
            for mask_type, masks in entity_masks.items():
                unique_masks = set(masks)
                if len(unique_masks) > 1:
                    issues.append(
                        {
                            "type": "inconsistent_masking",
                            "entity_type": mask_type,
                            "unique_masks": list(unique_masks),
                            "count": len(masks),
                        }
                    )

        # Check mask format validation
        if consistency_config.get("mask_format_validation", True):
            invalid_masks = re.findall(r"\[[^\]]*\[", text)  # Unclosed brackets
            for invalid_mask in invalid_masks:
                issues.append(
                    {
                        "type": "invalid_mask_format",
                        "mask": invalid_mask,
                        "issue": "unclosed_bracket",
                    }
                )

        # Check for partial masks
        if consistency_config.get("no_partial_masks", True):
            partial_masks = re.findall(r"\[\w+_[a-zA-Z]", text)  # Incomplete hash masks
            for partial_mask in partial_masks:
                issues.append(
                    {
                        "type": "partial_mask",
                        "mask": partial_mask,
                        "issue": "incomplete_hash",
                    }
                )

        return {
            "issues": issues,
            "issue_count": len(issues),
            "issue_types": list(set(issue["type"] for issue in issues)),
            "passed": len(issues) == 0,
        }

    def _perform_quality_assurance(
        self,
        original_text: str,
        masked_text: str,
        entities: List[PIIEntity],
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Perform quality assurance checks"""
        quality_thresholds = config.get("quality_thresholds", {})

        # Calculate masking coverage
        total_entities_count = len(entities)
        masked_tags_count = len(re.findall(r"\[\w+.*?\]", masked_text))
        coverage = (
            masked_tags_count / total_entities_count
            if total_entities_count > 0
            else 1.0
        )

        # Estimate false positive rate (simplified)
        # This would ideally use ground truth data
        estimated_fp_rate = self._estimate_false_positive_rate(entities, masked_text)

        # Calculate processing quality
        processing_quality = self._calculate_processing_quality(
            original_text, masked_text, entities
        )

        overall_score = (coverage + (1 - estimated_fp_rate) + processing_quality) / 3

        return {
            "masking_coverage": coverage,
            "estimated_false_positive_rate": estimated_fp_rate,
            "processing_quality": processing_quality,
            "overall_score": overall_score,
            "passed": (
                coverage >= quality_thresholds.get("minimum_masking_coverage", 0.8)
                and estimated_fp_rate
                <= quality_thresholds.get("maximum_false_positive_rate", 0.1)
                and processing_quality
                >= quality_thresholds.get("minimum_processing_quality", 0.7)
            ),
        }

    def _estimate_false_positive_rate(
        self, entities: List[PIIEntity], masked_text: str
    ) -> float:
        """Estimate false positive rate (simplified heuristic)"""
        if not entities:
            return 0.0

        # Estimate false positive rate (simplified heuristic)
        low_confidence_entities = [e for e in entities if e.confidence < 0.3]
        estimated_fp_rate = len(low_confidence_entities) / len(entities)

        # Teaching/example context - reduce FP penalty, do not inflate above 1.0
        text_lower = masked_text.lower()
        if any(word in text_lower for word in ["example", "test", "sample", "demo"]):
            estimated_fp_rate = min(estimated_fp_rate, 0.15)

        return min(1.0, estimated_fp_rate)

    def _calculate_processing_quality(
        self, original_text: str, masked_text: str, entities: List[PIIEntity]
    ) -> float:
        """Calculate overall processing quality"""
        quality_score = 0.0

        # Text preservation (should be similar length and structure)
        length_ratio = len(masked_text) / len(original_text) if original_text else 0
        if 0.7 <= length_ratio <= 1.3:  # Reasonable length range
            quality_score += 0.3

        # Entity coverage
        if entities:
            high_confidence_entities = [e for e in entities if e.confidence >= 0.7]
            coverage_score = len(high_confidence_entities) / len(entities)
            quality_score += coverage_score * 0.4

        # Mask quality (consistent formatting)
        masks = re.findall(r"\[\w+.*?\]", masked_text)
        if masks:
            consistent_masks = all(
                mask.startswith("[") and mask.endswith("]") for mask in masks
            )
            if consistent_masks:
                quality_score += 0.3

        return min(1.0, quality_score)

    def _collect_performance_metrics(
        self, context: PIIContext, config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Collect performance metrics"""
        performance_config = config.get("performance_metrics", {})
        metrics: Dict[str, Any] = {}

        # Execution time metrics
        if performance_config.get("track_execution_time", True):
            total_time = sum(
                result.execution_time_ms for result in context.stage_results.values()
            )
            metrics["total_execution_time_ms"] = total_time
            metrics["average_stage_time_ms"] = (
                total_time / len(context.stage_results) if context.stage_results else 0
            )

            # Stage-specific times
            for stage_name, result in context.stage_results.items():
                metrics[f"{stage_name}_time_ms"] = result.execution_time_ms

        # Entity processing metrics
        metrics["total_entities_detected"] = len(context.entities)
        metrics["entities_by_type"] = {}
        for entity in context.entities:
            entity_type = entity.pii_type
            metrics["entities_by_type"][entity_type] = (
                metrics["entities_by_type"].get(entity_type, 0) + 1
            )

        # Text processing metrics
        metrics["original_text_length"] = len(context.original_text)
        metrics["final_text_length"] = len(context.current_text)
        metrics["text_length_change"] = len(context.current_text) - len(
            context.original_text
        )

        # Masking metrics
        mask_count = len(re.findall(r"\[\w+.*?\]", context.current_text))
        metrics["total_masks_applied"] = mask_count
        metrics["masking_ratio"] = (
            mask_count / len(context.current_text.split())
            if context.current_text.split()
            else 0
        )

        return metrics

    def _final_validation(
        self,
        leak_detection: Dict[str, Any],
        consistency_check: Dict[str, Any],
        quality_check: Dict[str, Any],
        config: Dict[str, Any],
        context: PIIContext,
    ) -> Dict[str, Any]:
        """Perform final validation"""
        # Leaks and consistency are hard failures; quality is advisory
        passed = leak_detection["passed"] and consistency_check["passed"]

        if context.debug_enabled:
            print(f"[verification] Leak check: {leak_detection['passed']}")
            print(f"[verification] Consistency check: {consistency_check['passed']}")
            print(f"[verification] Quality check: {quality_check['passed']}")
            if not quality_check["passed"]:
                print(f"[verification] Quality details: {quality_check}")

        critical_issues = []
        warnings = []

        if not leak_detection["passed"]:
            critical_issues.append(
                f"PII leaks detected: {leak_detection['leak_count']}"
            )

        if not consistency_check["passed"]:
            critical_issues.append(
                f"Consistency issues: {consistency_check['issue_count']}"
            )

        if not quality_check["passed"]:
            warnings.append(
                f"Quality score below threshold: {quality_check['overall_score']:.2f}"
            )

        return {
            "passed": passed,
            "critical_issues": critical_issues,
            "warnings": warnings,
            "overall_score": quality_check["overall_score"],
        }

    def _generate_verification_report(
        self,
        leak_detection: Dict[str, Any],
        consistency_check: Dict[str, Any],
        quality_check: Dict[str, Any],
        performance_metrics: Dict[str, Any],
        final_validation: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate comprehensive verification report"""
        return {
            "summary": {
                "verification_passed": final_validation["passed"],
                "overall_score": quality_check["overall_score"],
                "critical_issues_count": len(final_validation["critical_issues"]),
                "warnings_count": len(final_validation["warnings"]),
            },
            "pii_leak_analysis": leak_detection,
            "masking_consistency": consistency_check,
            "quality_assessment": quality_check,
            "performance_metrics": performance_metrics,
            "final_validation": final_validation,
            "recommendations": self._generate_recommendations(
                final_validation, quality_check, leak_detection
            ),
        }

    def _generate_recommendations(
        self,
        final_validation: Dict[str, Any],
        quality_check: Dict[str, Any],
        leak_detection: Dict[str, Any],
    ) -> List[str]:
        """Generate recommendations based on verification results"""
        recommendations = []

        if not final_validation["passed"]:
            recommendations.append("Address critical issues before using this result")

        if leak_detection["leak_count"] > 0:
            recommendations.append(
                f"Review {leak_detection['leak_count']} PII leaks detected"
            )

        if quality_check["masking_coverage"] < 0.8:
            recommendations.append(
                "Consider increasing detection sensitivity for better coverage"
            )

        if quality_check["estimated_false_positive_rate"] > 0.1:
            recommendations.append(
                "Review confidence thresholds to reduce false positives"
            )

        if quality_check["processing_quality"] < 0.7:
            recommendations.append("Improve text processing and masking strategies")

        if not recommendations:
            recommendations.append("Verification passed successfully")

        return recommendations

    def validate_input(self, context: PIIContext) -> bool:
        """Validate input for verification stage"""
        entities: object = context.entities
        if not entities:
            return True  # No entities is valid

        if not isinstance(entities, list):
            return False

        if not context.current_text:
            return False

        if not context.original_text:
            return False

        if not context.stage_results:
            return False

        for entity in entities:
            if not isinstance(entity, PIIEntity):
                return False

        return True

    def fallback(self, context: PIIContext) -> StageResult:
        """Fallback verification - basic checks only"""
        masked_text = context.current_text
        entities = context.entities

        # Basic leak detection
        basic_leaks = []
        email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        matches = re.finditer(email_pattern, masked_text, re.IGNORECASE)

        for match in matches:
            basic_leaks.append(
                {
                    "type": "email",
                    "text": match.group(),
                    "start": match.start(),
                    "end": match.end(),
                }
            )

        passed = len(basic_leaks) == 0

        verification_report = {
            "summary": {
                "verification_passed": passed,
                "fallback_used": True,
                "fallback_type": "basic_leak_check",
            },
            "pii_leak_analysis": {
                "leaks": basic_leaks,
                "leak_count": len(basic_leaks),
                "passed": passed,
            },
        }

        self.add_debug_info(
            context,
            "Verification fallback used",
            {
                "reason": "main_verification_failed",
                "fallback_type": "basic_leak_check",
                "basic_leaks": len(basic_leaks),
                "passed": passed,
            },
        )

        return StageResult(
            success=passed,
            stage_name=self.stage_name,
            execution_time_ms=0.0,
            processed_text=masked_text,
            entities=entities,
            metadata={
                "fallback_used": True,
                "fallback_type": "basic_leak_check",
                "verification_report": verification_report,
                "basic_leak_detection": {
                    "leaks": basic_leaks,
                    "leak_count": len(basic_leaks),
                    "passed": passed,
                },
            },
        )
