"""
Structure Builder - Convert prompts into structured instructions
"""

import re
from typing import Dict, List, Optional
from dataclasses import dataclass

from .intent_extractor import IntentType, IntentExtractionResult


@dataclass
class StructureResult:
    """Result of structure building."""

    structured_prompt: str
    original_format: str
    new_format: str
    structure_score: float
    components: Dict[str, str]


class StructureBuilder:
    """Convert prompts into structured, clear instructions."""

    def __init__(self) -> None:
        """Initialize structure templates and rules."""
        self.structure_templates = {
            IntentType.ANALYZE: "TASK: {intent}\nINPUT: {object}\nFOCUS: {constraints}\nOUTPUT: insights",
            IntentType.SUMMARIZE: "TASK: {intent}\nINPUT: {object}\nSTYLE: {constraints}\nOUTPUT: summary",
            IntentType.GENERATE: "TASK: {intent}\nTYPE: {object}\nREQUIREMENTS: {constraints}\nOUTPUT: new_content",
            IntentType.EXTRACT: "TASK: {intent}\nSOURCE: {object}\nTARGET: {constraints}\nOUTPUT: extracted_data",
            IntentType.CLASSIFY: "TASK: {intent}\nITEMS: {object}\nCATEGORIES: {constraints}\nOUTPUT: classifications",
            IntentType.COMPARE: "TASK: {intent}\nITEMS: {object}\nCRITERIA: {constraints}\nOUTPUT: comparison",
            IntentType.VALIDATE: "TASK: {intent}\nTARGET: {object}\nRULES: {constraints}\nOUTPUT: validation_result",
            IntentType.DEBUG: "TASK: {intent}\nPROBLEM: {object}\nCONTEXT: {constraints}\nOUTPUT: solution",
            IntentType.TRANSLATE: "TASK: {intent}\nFROM: {object}\nTO: {constraints}\nOUTPUT: translated_content",
            IntentType.TRANSFORM: "TASK: {intent}\nINPUT: {object}\nFORMAT: {constraints}\nOUTPUT: transformed_content",
        }

        self.structure_keywords = {
            "task",
            "input",
            "output",
            "focus",
            "style",
            "type",
            "requirements",
            "source",
            "target",
            "items",
            "categories",
            "criteria",
            "rules",
            "problem",
            "context",
            "solution",
            "from",
            "to",
            "format",
        }

    def build_structure(
        self, prompt: str, intent_result: IntentExtractionResult
    ) -> StructureResult:
        """
        Convert prompt into structured format.

        Args:
            prompt: Original prompt
            intent_result: Extracted intent information

        Returns:
            StructureResult with structured prompt and metrics
        """
        # Detect original format
        original_format = self._detect_format(prompt)

        # Get template for intent
        template = self.structure_templates.get(
            intent_result.intent, self.structure_templates[IntentType.ANALYZE]
        )

        # Build structured components
        components = self._build_components(intent_result, prompt)

        # Fill template
        structured_prompt = template.format(
            intent=intent_result.intent.value,
            object=components.get("object", intent_result.object),
            constraints=components.get(
                "constraints", ", ".join(intent_result.constraints)
            ),
        )

        # Calculate structure score
        structure_score = self._calculate_structure_score(structured_prompt)

        # Determine new format
        new_format = "structured_key_value"

        return StructureResult(
            structured_prompt=structured_prompt,
            original_format=original_format,
            new_format=new_format,
            structure_score=structure_score,
            components=components,
        )

    def _detect_format(self, prompt: str) -> str:
        """Detect the format of the original prompt."""
        # Check for structured formats
        if re.search(r"^\w+:\s*.+", prompt, re.MULTILINE):
            return "key_value"
        elif re.search(r"^\d+\.\s*.+", prompt, re.MULTILINE):
            return "numbered_list"
        elif re.search(r"^-\s*.+", prompt, re.MULTILINE):
            return "bullet_list"
        elif re.search(r"^#+\s*.+", prompt, re.MULTILINE):
            return "markdown"
        elif "\n" in prompt:
            return "multiline"
        else:
            return "plain_text"

    def _build_components(
        self, intent_result: IntentExtractionResult, original_prompt: str
    ) -> Dict[str, str]:
        """Build structured components from intent and original prompt."""
        components = {
            "object": original_prompt,
            "constraints": (
                ", ".join(intent_result.constraints)
                if intent_result.constraints
                else "standard"
            ),
        }

        # Extract additional context from original prompt
        context = self._extract_context(original_prompt)
        if context:
            components["context"] = context

        # Extract specific requirements
        requirements = self._extract_requirements(original_prompt)
        if requirements:
            components["requirements"] = ", ".join(requirements)

        return components

    def _extract_context(self, prompt: str) -> Optional[str]:
        """Extract contextual information from prompt."""
        # Look for context indicators
        context_patterns = [
            r"given\s+(.+?),?\s+(?:please|can|could)",
            r"based\s+on\s+(.+?),?\s+(?:please|can|could)",
            r"using\s+(.+?),?\s+(?:please|can|could)",
            r"with\s+(.+?),?\s+(?:please|can|could)",
        ]

        for pattern in context_patterns:
            match = re.search(pattern, prompt, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        return None

    def _extract_requirements(self, prompt: str) -> List[str]:
        """Extract specific requirements from prompt."""
        requirements = []

        # Look for requirement indicators
        requirement_patterns = [
            r"make\s+sure\s+(.+)",
            r"ensure\s+(.+)",
            r"include\s+(.+)",
            r"add\s+(.+)",
            r"provide\s+(.+)",
        ]

        for pattern in requirement_patterns:
            matches = re.findall(pattern, prompt, re.IGNORECASE)
            requirements.extend(matches)

        return [req.strip() for req in requirements if req.strip()]

    def _calculate_structure_score(self, structured_prompt: str) -> float:
        """Calculate how well-structured the prompt is."""
        score = 0.0

        # Check for key-value structure
        if re.search(r"^\w+:\s*.+", structured_prompt, re.MULTILINE):
            score += 0.4

        # Check for multiple components
        components = len(re.findall(r"^\w+:\s*.+", structured_prompt, re.MULTILINE))
        if components >= 3:
            score += 0.3
        elif components >= 2:
            score += 0.2

        # Check for clear task definition
        if re.search(r"^TASK:\s*\w+", structured_prompt, re.MULTILINE):
            score += 0.2

        # Check for output specification
        if re.search(r"^OUTPUT:\s*\w+", structured_prompt, re.MULTILINE):
            score += 0.1

        return min(score, 1.0)

    def get_structure_summary(self, result: StructureResult) -> str:
        """Generate human-readable structure summary."""
        return f"Format: {result.original_format} → {result.new_format} | Score: {result.structure_score:.2f} | Components: {len(result.components)}"
