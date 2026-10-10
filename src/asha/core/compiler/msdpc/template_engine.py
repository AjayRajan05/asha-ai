"""
Template Engine - Intent-to-template mapping
"""

import re
from typing import Dict, List
from dataclasses import dataclass

from .intent_extractor import IntentType


@dataclass
class TemplateResult:
    """Result of template application."""

    templated_prompt: str
    template_used: str
    template_type: str
    coverage_score: float


class TemplateEngine:
    """Map intents to predefined prompt templates."""

    def __init__(self) -> None:
        """Initialize template library."""
        self.templates = {
            IntentType.ANALYZE: {
                "basic": "Analyze {object} for {focus}",
                "detailed": "Perform comprehensive analysis of {object} focusing on {focus}. Provide insights and recommendations.",
                "quick": "Quick analysis of {object} - identify {focus}",
                "structured": "ANALYSIS REQUEST:\nOBJECT: {object}\nFOCUS: {focus}\nOUTPUT: insights and findings",
            },
            IntentType.SUMMARIZE: {
                "basic": "Summarize {object}",
                "detailed": "Provide comprehensive summary of {object} including key points and main findings.",
                "brief": "Brief summary of {object} - main points only",
                "structured": "SUMMARIZATION REQUEST:\nSOURCE: {object}\nSTYLE: {style}\nOUTPUT: concise summary",
            },
            IntentType.GENERATE: {
                "basic": "Generate {type} for {purpose}",
                "detailed": "Create detailed {type} for {purpose} with all necessary components and best practices.",
                "quick": "Quick {type} generation for {purpose}",
                "structured": "GENERATION REQUEST:\nTYPE: {type}\nPURPOSE: {purpose}\nOUTPUT: new content",
            },
            IntentType.EXTRACT: {
                "basic": "Extract {target} from {source}",
                "detailed": "Carefully extract {target} information from {source}, ensuring accuracy and completeness.",
                "quick": "Quick extraction of {target} from {source}",
                "structured": "EXTRACTION REQUEST:\nSOURCE: {source}\nTARGET: {target}\nOUTPUT: extracted data",
            },
            IntentType.CLASSIFY: {
                "basic": "Classify {items} into {categories}",
                "detailed": "Systematically classify {items} into appropriate {categories} with reasoning for each classification.",
                "quick": "Quick classification of {items} into {categories}",
                "structured": "CLASSIFICATION REQUEST:\nITEMS: {items}\nCATEGORIES: {categories}\nOUTPUT: classifications",
            },
            IntentType.COMPARE: {
                "basic": "Compare {items} based on {criteria}",
                "detailed": "Thoroughly compare {items} using {criteria}, highlighting similarities, differences, and recommendations.",
                "quick": "Quick comparison of {items} using {criteria}",
                "structured": "COMPARISON REQUEST:\nITEMS: {items}\nCRITERIA: {criteria}\nOUTPUT: comparison analysis",
            },
            IntentType.VALIDATE: {
                "basic": "Validate {target} against {rules}",
                "detailed": "Comprehensively validate {target} against {rules}, identifying any issues and providing recommendations.",
                "quick": "Quick validation of {target} using {rules}",
                "structured": "VALIDATION REQUEST:\nTARGET: {target}\nRULES: {rules}\nOUTPUT: validation result",
            },
            IntentType.DEBUG: {
                "basic": "Debug {problem} in {context}",
                "detailed": "Systematically debug {problem} in {context}, identifying root causes and providing solutions.",
                "quick": "Quick debug of {problem} in {context}",
                "structured": "DEBUG REQUEST:\nPROBLEM: {problem}\nCONTEXT: {context}\nOUTPUT: solution",
            },
            IntentType.TRANSLATE: {
                "basic": "Translate {content} from {source_lang} to {target_lang}",
                "detailed": "Accurately translate {content} from {source_lang} to {target_lang}, preserving meaning and context.",
                "quick": "Quick translation of {content} to {target_lang}",
                "structured": "TRANSLATION REQUEST:\nCONTENT: {content}\nFROM: {source_lang}\nTO: {target_lang}\nOUTPUT: translated text",
            },
            IntentType.TRANSFORM: {
                "basic": "Transform {input} to {format}",
                "detailed": "Completely transform {input} into {format} with proper structure and validation.",
                "quick": "Quick transformation of {input} to {format}",
                "structured": "TRANSFORMATION REQUEST:\nINPUT: {input}\nFORMAT: {format}\nOUTPUT: transformed content",
            },
        }

        self.template_types = {
            "basic": {"complexity": "low", "length": "short"},
            "detailed": {"complexity": "high", "length": "long"},
            "quick": {"complexity": "low", "length": "very_short"},
            "structured": {"complexity": "medium", "length": "medium"},
        }

    def apply_template(
        self,
        intent_type: IntentType,
        components: Dict[str, str],
        template_preference: str = "basic",
    ) -> TemplateResult:
        """
        Apply appropriate template based on intent and components.

        Args:
            intent_type: Type of intent
            components: Dictionary of components to fill template
            template_preference: Preferred template type

        Returns:
            TemplateResult with templated prompt and metrics
        """
        # Get available templates for intent
        intent_templates = self.templates.get(
            intent_type, self.templates[IntentType.ANALYZE]
        )

        # Select best template
        selected_template = self._select_best_template(
            intent_templates, template_preference, components
        )

        # Fill template with components
        templated_prompt = self._fill_template(selected_template, components)

        # Calculate coverage score
        coverage_score = self._calculate_coverage(selected_template, components)

        return TemplateResult(
            templated_prompt=templated_prompt,
            template_used=selected_template,
            template_type=template_preference,
            coverage_score=coverage_score,
        )

    def _select_best_template(
        self, templates: Dict[str, str], preference: str, components: Dict[str, str]
    ) -> str:
        """Select the best template based on preference and components."""
        # Try preference first
        if preference in templates:
            template = templates[preference]
            if self._can_fill_template(template, components):
                return template

        # Try other templates in order of preference
        preference_order = ["structured", "basic", "quick", "detailed"]

        for pref in preference_order:
            if pref in templates:
                template = templates[pref]
                if self._can_fill_template(template, components):
                    return template

        # Fallback to any available template
        return list(templates.values())[0]

    def _can_fill_template(self, template: str, components: Dict[str, str]) -> bool:
        """Check if template can be filled with available components."""
        # Extract placeholders from template
        placeholders = re.findall(r"\{(\w+)\}", template)

        # Check if all placeholders have corresponding components
        for placeholder in placeholders:
            if placeholder not in components or not components[placeholder]:
                return False

        return True

    def _fill_template(self, template: str, components: Dict[str, str]) -> str:
        """Fill template with component values."""
        filled_template = template

        for placeholder, value in components.items():
            if value:
                filled_template = filled_template.replace(f"{{{placeholder}}}", value)

        return filled_template

    def _calculate_coverage(self, template: str, components: Dict[str, str]) -> float:
        """Calculate how well the template covers available components."""
        placeholders = re.findall(r"\{(\w+)\}", template)
        if not placeholders:
            return 1.0

        covered = sum(
            1
            for placeholder in placeholders
            if placeholder in components and components[placeholder]
        )

        return covered / len(placeholders)

    def get_available_templates(self, intent_type: IntentType) -> List[str]:
        """Get list of available templates for intent type."""
        templates = self.templates.get(intent_type, {})
        return list(templates.keys())

    def get_template_info(self, template_type: str) -> Dict[str, str]:
        """Get information about template type."""
        return self.template_types.get(template_type, {})

    def get_template_summary(self, result: TemplateResult) -> str:
        """Generate human-readable template summary."""
        return f"Type: {result.template_type} | Coverage: {result.coverage_score:.2f} | Template: {result.template_used[:50]}..."
