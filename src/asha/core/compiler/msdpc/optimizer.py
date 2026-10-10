"""
MSDPC Optimizer V2 - Main orchestrator for Multi-Stage Deterministic Prompt Compiler

This is the main entry point that coordinates all MSDPC components.
"""

import time
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass

from .intent_extractor import IntentExtractor, IntentType, IntentExtractionResult
from .semantic_compressor import SemanticCompressor
from .structure_builder import StructureBuilder
from .token_pruner import TokenPruner
from .output_shaper import OutputShaper
from .template_engine import TemplateEngine
from .metrics_engine import MetricsEngine


@dataclass
class OptimizationMetrics:
    """Comprehensive metrics for MSDPC optimization."""

    original_tokens: int
    optimized_tokens: int
    token_reduction_percentage: float
    processing_time_ms: float
    stages_applied: List[str]
    clarity_score: float
    structure_score: float
    efficiency_score: float
    intent_preserved: bool


class MSDPCOptimizer:
    """
    Multi-Stage Deterministic Prompt Compiler

    Orchestrates all MSDPC components to achieve 30-60% token reduction
    while maintaining intent and improving clarity.
    """

    def __init__(self) -> None:
        """Initialize all MSDPC components."""
        self.intent_extractor = IntentExtractor()
        self.semantic_compressor = SemanticCompressor()
        self.structure_builder = StructureBuilder()
        self.token_pruner = TokenPruner()
        self.output_shaper = OutputShaper()
        self.template_engine = TemplateEngine()
        self.metrics_engine = MetricsEngine()

        # Configuration
        self.config = {
            "enable_structure_building": True,
            "enable_output_shaping": True,
            "enable_templates": False,
            "target_reduction": 0.4,
            "aggressive_pruning": False,
        }

    def configure(
        self,
        enable_structure_building: bool = True,
        enable_output_shaping: bool = True,
        enable_templates: bool = False,
        target_reduction: float = 0.4,
        aggressive_pruning: bool = False,
    ) -> None:
        """
        Configure MSDPC optimization parameters.

        Args:
            enable_structure_building: Enable structural rewriting
            enable_output_shaping: Enable output shaping
            enable_templates: Enable template engine
            target_reduction: Target token reduction percentage
            aggressive_pruning: Use aggressive token pruning
        """
        self.config.update(
            {
                "enable_structure_building": enable_structure_building,
                "enable_output_shaping": enable_output_shaping,
                "enable_templates": enable_templates,
                "target_reduction": target_reduction,
                "aggressive_pruning": aggressive_pruning,
            }
        )

    def optimize(
        self, prompt: str, token_budget: Optional[int] = None
    ) -> Tuple[str, OptimizationMetrics]:
        """
        Optimize prompt through MSDPC pipeline.

        Args:
            prompt: Original prompt to optimize
            token_budget: Optional token limit to enforce

        Returns:
            Tuple of (optimized_prompt, optimization_metrics)
        """
        start_time = time.time()
        stages_applied = []

        # Count original tokens
        original_tokens = self._count_tokens(prompt)

        # Stage 1: Intent Extraction
        intent_result = self.intent_extractor.extract_intent(prompt)
        stages_applied.append("intent_extraction")

        # Stage 2: Semantic Compression
        compression_result = self.semantic_compressor.compress(prompt)
        current_prompt = compression_result.compressed_text
        stages_applied.append("semantic_compression")

        # Stage 3: Structure Building (if enabled)
        if self.config["enable_structure_building"]:
            structure_result = self.structure_builder.build_structure(
                current_prompt, intent_result
            )
            current_prompt = structure_result.structured_prompt
            stages_applied.append("structure_building")

        # Stage 4: Token Pruning
        pruning_result = self.token_pruner.prune(
            current_prompt, bool(self.config["aggressive_pruning"])
        )
        current_prompt = pruning_result.pruned_text
        stages_applied.append("token_pruning")

        # Stage 5: Output Shaping (if enabled)
        if self.config["enable_output_shaping"]:
            shaping_result = self.output_shaper.shape_output(
                current_prompt, intent_result.intent, intent_result.constraints
            )
            current_prompt = shaping_result.shaped_prompt
            stages_applied.append("output_shaping")

        # Stage 6: Template Engine (if enabled)
        if self.config["enable_templates"]:
            components = self._build_template_components(intent_result, current_prompt)
            template_result = self.template_engine.apply_template(
                intent_result.intent, components
            )
            current_prompt = template_result.templated_prompt
            stages_applied.append("template_engine")

        # Stage 7: Token Budget Enforcement (if token_budget specified)
        if token_budget is not None and token_budget > 0:
            current_tokens = self._count_tokens(current_prompt)
            if current_tokens > token_budget:
                budget_prune = self.token_pruner.prune(current_prompt, aggressive=True)
                current_prompt = budget_prune.pruned_text
                stages_applied.append("budget_pruning")
                words = current_prompt.split()
                if len(words) > token_budget:
                    current_prompt = " ".join(words[:token_budget])
                    stages_applied.append("budget_truncation")

        # Calculate final metrics
        processing_time = (time.time() - start_time) * 1000
        optimized_tokens = self._count_tokens(current_prompt)
        tokens_saved = original_tokens - optimized_tokens
        token_reduction_percentage = (
            ((tokens_saved) / original_tokens * 100)
            if original_tokens > 0 and tokens_saved > 0
            else 0.0
        )

        # Calculate quality scores
        quality_metrics = self.metrics_engine.calculate_metrics(prompt, current_prompt)

        # Check if intent was preserved
        intent_preserved = self._check_intent_preservation(
            prompt, current_prompt, intent_result
        )

        metrics = OptimizationMetrics(
            original_tokens=original_tokens,
            optimized_tokens=optimized_tokens,
            token_reduction_percentage=token_reduction_percentage,
            processing_time_ms=processing_time,
            stages_applied=stages_applied,
            clarity_score=quality_metrics.clarity_score,
            structure_score=quality_metrics.structure_score,
            efficiency_score=quality_metrics.efficiency_score,
            intent_preserved=intent_preserved,
        )

        return current_prompt, metrics

    def _count_tokens(self, text: str) -> int:
        """Count approximate tokens in text."""
        import re

        tokens = re.findall(r"\b\w+\b", text)
        return len(tokens)

    def _build_template_components(
        self, intent_result: IntentExtractionResult, prompt: str
    ) -> Dict[str, str]:
        """Build components for template engine."""
        components = {
            "object": intent_result.object,
            "focus": (
                ", ".join(intent_result.constraints)
                if intent_result.constraints
                else "general"
            ),
            "purpose": "analysis",
            "style": "professional",
        }

        # Add intent-specific components
        if intent_result.intent == IntentType.ANALYZE:
            components["focus"] = components.get("focus", "patterns and insights")
        elif intent_result.intent == IntentType.SUMMARIZE:
            components["style"] = "concise"
        elif intent_result.intent == IntentType.GENERATE:
            components["type"] = "content"

        return components

    def _check_intent_preservation(
        self, original: str, optimized: str, intent_result: IntentExtractionResult
    ) -> bool:
        """Check if the original intent was preserved in optimization."""
        # Extract intent from optimized prompt
        optimized_intent = self.intent_extractor.extract_intent(optimized)

        # Check if intent type matches
        intent_match = optimized_intent.intent == intent_result.intent

        # Check if key object is preserved
        object_match = intent_result.object.lower() in optimized.lower()

        # Check if constraints are preserved (at least one)
        constraint_match = (
            any(
                constraint.lower() in optimized.lower()
                for constraint in intent_result.constraints
            )
            if intent_result.constraints
            else True
        )

        return intent_match and (object_match or constraint_match)

    def get_optimization_summary(self, metrics: OptimizationMetrics) -> str:
        """Generate human-readable optimization summary."""
        summary_parts = [
            f"Token Reduction: {metrics.token_reduction_percentage:.1f}%",
            f"Processing Time: {metrics.processing_time_ms:.1f}ms",
            f"Stages Applied: {', '.join(metrics.stages_applied)}",
            f"Quality Scores: Clarity {metrics.clarity_score:.2f}, Structure {metrics.structure_score:.2f}, Efficiency {metrics.efficiency_score:.2f}",
            f"Intent Preserved: {'✅' if metrics.intent_preserved else '❌'}",
        ]

        return " | ".join(summary_parts)

    def benchmark(self, prompts: List[str]) -> Dict[str, Any]:
        """
        Benchmark MSDPC optimizer on multiple prompts.

        Args:
            prompts: List of prompts to benchmark

        Returns:
            Benchmark results with aggregated metrics
        """
        results: List[Dict[str, Any]] = []

        for prompt in prompts:
            optimized, metrics = self.optimize(prompt)
            results.append(
                {"original": prompt, "optimized": optimized, "metrics": metrics}
            )

        # Calculate aggregates
        total_prompts = len(results)
        avg_reduction = (
            sum(
                r["metrics"].token_reduction_percentage
                for r in results
                if isinstance(r["metrics"], OptimizationMetrics)
            )
            / total_prompts
        )
        avg_time = (
            sum(
                r["metrics"].processing_time_ms
                for r in results
                if isinstance(r["metrics"], OptimizationMetrics)
            )
            / total_prompts
        )
        avg_clarity = (
            sum(
                r["metrics"].clarity_score
                for r in results
                if isinstance(r["metrics"], OptimizationMetrics)
            )
            / total_prompts
        )
        avg_structure = (
            sum(
                r["metrics"].structure_score
                for r in results
                if isinstance(r["metrics"], OptimizationMetrics)
            )
            / total_prompts
        )
        avg_efficiency = (
            sum(
                r["metrics"].efficiency_score
                for r in results
                if isinstance(r["metrics"], OptimizationMetrics)
            )
            / total_prompts
        )
        intent_preserved_rate = (
            sum(
                1
                for r in results
                if isinstance(r["metrics"], OptimizationMetrics)
                and r["metrics"].intent_preserved
            )
            / total_prompts
        )

        return {
            "total_prompts": total_prompts,
            "average_reduction": avg_reduction,
            "average_processing_time": avg_time,
            "average_clarity_score": avg_clarity,
            "average_structure_score": avg_structure,
            "average_efficiency_score": avg_efficiency,
            "intent_preserved_rate": intent_preserved_rate,
            "detailed_results": results,
        }
