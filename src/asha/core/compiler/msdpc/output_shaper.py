"""
Output Shaper - Add output constraints to reduce output tokens
"""

import re
from typing import List, Optional
from dataclasses import dataclass

from .intent_extractor import IntentType


@dataclass
class ShapingResult:
    """Result of output shaping."""

    shaped_prompt: str
    output_constraints: List[str]
    estimated_reduction: float
    shaping_type: str


class OutputShaper:
    """Add output constraints to reduce output token usage."""

    def __init__(self) -> None:
        """Initialize output shaping templates."""
        self.output_templates = {
            IntentType.ANALYZE: [
                "Respond with key insights only",
                "Provide concise analysis",
                "Focus on main findings",
                "Limit to 3-5 key points",
            ],
            IntentType.SUMMARIZE: [
                "Provide brief summary",
                "Keep under 100 words",
                "Focus on main points only",
                "Exclude minor details",
            ],
            IntentType.GENERATE: [
                "Generate concise output",
                "Keep response brief",
                "Focus on essential elements",
                "Avoid unnecessary elaboration",
            ],
            IntentType.EXTRACT: [
                "Extract only requested information",
                "Provide direct answers only",
                "No explanations needed",
                "List results concisely",
            ],
            IntentType.CLASSIFY: [
                "Provide classification only",
                "No explanations needed",
                "Direct category assignment",
                "Brief justification only if required",
            ],
            IntentType.COMPARE: [
                "Focus on key differences",
                "Provide concise comparison",
                "Highlight main contrasts",
                "Limit to essential points",
            ],
            IntentType.VALIDATE: [
                "Provide validation result only",
                "Brief error description if any",
                "No detailed explanations",
                "Direct pass/fail response",
            ],
            IntentType.DEBUG: [
                "Provide solution directly",
                "Brief explanation only",
                "Focus on fix only",
                "Avoid lengthy analysis",
            ],
            IntentType.TRANSLATE: [
                "Provide translation only",
                "No explanations needed",
                "Direct translation result",
                "Maintain original meaning",
            ],
            IntentType.TRANSFORM: [
                "Provide transformed result only",
                "No explanations needed",
                "Direct transformation",
                "Follow specified format",
            ],
        }

        self.constraint_patterns = {
            "length": [
                r"\b(under|less than|below|no more than)\s+\d+\s+(words|tokens|characters)\b",
                r"\b(limit to|keep to|restrict to)\s+\d+\s+(words|tokens|characters)\b",
                r"\b(brief|short|concise|succinct|compact)\b",
            ],
            "format": [
                r"\b(bullet points|numbered list|outline|summary)\b",
                r"\b(table|chart|graph|diagram)\b",
                r"\b(json|xml|csv|yaml)\b",
            ],
            "content": [
                r"\b(key|main|essential|important|critical)\s+(points|insights|findings)\b",
                r"\b(exclude|omit|ignore|skip)\s+(minor|unnecessary|extra)\b",
                r"\b(only|just|directly|simply)\s+(the|result|answer|output)\b",
            ],
        }

    def shape_output(
        self,
        prompt: str,
        intent_type: IntentType,
        existing_constraints: Optional[List[str]] = None,
    ) -> ShapingResult:
        """
        Add output shaping constraints to prompt.

        Args:
            prompt: Original prompt
            intent_type: Type of intent
            existing_constraints: Existing constraints from prompt

        Returns:
            ShapingResult with shaped prompt and constraints
        """
        if existing_constraints is None:
            existing_constraints = []

        # Get appropriate shaping templates
        templates = self.output_templates.get(
            intent_type, self.output_templates[IntentType.ANALYZE]
        )

        # Select best template based on existing constraints
        selected_template = self._select_template(templates, existing_constraints)

        # Add shaping constraint to prompt
        shaped_prompt = self._add_shaping_constraint(prompt, selected_template)

        # Determine shaping type
        shaping_type = self._determine_shaping_type(selected_template)

        # Estimate output reduction
        estimated_reduction = self._estimate_reduction(selected_template, intent_type)

        return ShapingResult(
            shaped_prompt=shaped_prompt,
            output_constraints=[selected_template],
            estimated_reduction=estimated_reduction,
            shaping_type=shaping_type,
        )

    def _select_template(
        self, templates: List[str], existing_constraints: List[str]
    ) -> str:
        """Select the best shaping template based on existing constraints."""
        # Check for existing length constraints
        has_length_constraint = any(
            re.search(pattern, " ".join(existing_constraints), re.IGNORECASE)
            for patterns in self.constraint_patterns.values()
            for pattern in patterns
        )

        if has_length_constraint:
            # If length constraint exists, use a complementary constraint
            for template in templates:
                if "key" in template.lower() or "main" in template.lower():
                    return template

        # Otherwise, use the most effective template
        for template in templates:
            if "brief" in template.lower() or "concise" in template.lower():
                return template

        return templates[0]  # Fallback

    def _add_shaping_constraint(self, prompt: str, constraint: str) -> str:
        """Add shaping constraint to prompt."""
        # Check if prompt already ends with punctuation
        if prompt.endswith((".", "!", "?")):
            shaped_prompt = f"{prompt} {constraint}."
        else:
            shaped_prompt = f"{prompt}. {constraint}."

        return shaped_prompt

    def _determine_shaping_type(self, template: str) -> str:
        """Determine the type of shaping being applied."""
        if "word" in template.lower() or "token" in template.lower():
            return "length_constraint"
        elif "format" in template.lower() or "list" in template.lower():
            return "format_constraint"
        elif "key" in template.lower() or "main" in template.lower():
            return "content_constraint"
        else:
            return "general_constraint"

    def _estimate_reduction(self, template: str, intent_type: IntentType) -> float:
        """Estimate the percentage reduction in output tokens."""
        base_reduction = 0.3  # 30% average reduction

        # Adjust based on template type
        if "under" in template.lower() or "limit" in template.lower():
            base_reduction += 0.2  # Additional 20% for explicit limits
        elif "brief" in template.lower() or "concise" in template.lower():
            base_reduction += 0.1  # Additional 10% for brevity
        elif "key" in template.lower() or "main" in template.lower():
            base_reduction += 0.15  # Additional 15% for key points only

        # Adjust based on intent type
        if intent_type in [IntentType.SUMMARIZE, IntentType.ANALYZE]:
            base_reduction += 0.1  # These tend to generate longer outputs
        elif intent_type in [IntentType.VALIDATE, IntentType.CLASSIFY]:
            base_reduction += 0.05  # These are already shorter

        return min(base_reduction, 0.6)  # Cap at 60% reduction

    def get_shaping_summary(self, result: ShapingResult) -> str:
        """Generate human-readable shaping summary."""
        return f"Type: {result.shaping_type} | Estimated reduction: {result.estimated_reduction:.1%} | Constraint: {result.output_constraints[0]}"
