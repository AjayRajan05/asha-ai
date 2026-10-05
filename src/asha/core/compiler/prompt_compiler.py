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

import re
from typing import Dict, List, Any, Optional, Union
from .._ir.prompt_ir import (
    PromptIR,
    IntentType,
    EntityType,
    ConstraintType,
    PrivacyLevel,
)


class PromptCompiler:
    """
    Prompt Compiler - Converts Prompt IR into optimized, executable prompts.

    This is the core compiler that transforms structured IR into
    model-ready prompts with appropriate formatting, context, and instructions.
    """

    def __init__(self) -> None:
        """Initialize prompt compiler with templates and rules."""
        self.templates = self._init_templates()
        self.compilation_rules = self._init_compilation_rules()
        self.formatting_rules = self._init_formatting_rules()

    def _init_templates(self) -> Dict[IntentType, str]:
        """Initialize prompt templates for different intents."""
        return {
            IntentType.ANALYZE: """You are an expert data analyst. Your task is to {task} the {entity}.

{context}

Constraints:
{constraints}

Please provide a thorough analysis with clear insights and recommendations.""",
            IntentType.GENERATE: """You are a creative content generator. Your task is to {task} {entity}.

{context}

Requirements:
{constraints}

Generate high-quality, original content that meets all specified requirements.""",
            IntentType.SUMMARIZE: """You are an expert summarizer. Your task is to {task} the {entity}.

{context}

Summary requirements:
{constraints}

Provide a concise yet comprehensive summary that captures all key points.""",
            IntentType.TRANSLATE: """You are a professional translator. Your task is to {task} the {entity}.

{context}

Translation guidelines:
{constraints}

Ensure accurate translation while preserving meaning and context.""",
            IntentType.CLASSIFY: """You are an expert classifier. Your task is to {task} the {entity}.

{context}

Classification criteria:
{constraints}

Provide clear classification with reasoning and confidence scores.""",
            IntentType.EXTRACT: """You are a data extraction specialist. Your task is to {task} from the {entity}.

{context}

Extraction requirements:
{constraints}

Extract all relevant information accurately and completely.""",
            IntentType.COMPARE: """You are an expert analyst. Your task is to {task} between {entity}.

{context}

Comparison criteria:
{constraints}

Provide a detailed comparison highlighting similarities, differences, and insights.""",
            IntentType.EXPLAIN: """You are an expert educator. Your task is to {task} the {entity}.

{context}

Explanation requirements:
{constraints}

Provide clear, comprehensive explanations with examples and context.""",
            IntentType.CREATE: """You are a creative professional. Your task is to {task} {entity}.

{context}

Creation guidelines:
{constraints}

Create original, high-quality output that meets all specifications.""",
            IntentType.MODIFY: """You are an expert editor. Your task is to {task} the {entity}.

{context}

Modification requirements:
{constraints}

Make appropriate changes while maintaining quality and coherence.""",
            IntentType.VALIDATE: """You are a quality assurance expert. Your task is to {task} the {entity}.

{context}

Validation criteria:
{constraints}

Provide thorough validation with clear results and recommendations.""",
            IntentType.SEARCH: """You are an expert researcher. Your task is to {task} for {entity}.

{context}

Search criteria:
{constraints}

Conduct comprehensive search and provide relevant, accurate results.""",
            IntentType.DEBUG: """You are an expert debugger. Your task is to {task} the {entity}.

{context}

Debugging approach:
{constraints}

Identify issues systematically and provide clear solutions.""",
            IntentType.OPTIMIZE: """You are an optimization expert. Your task is to {task} the {entity}.

{context}

Optimization goals:
{constraints}

Provide effective optimization strategies and implementations.""",
            IntentType.ABSTAIN: """You are a careful assistant. Process the following {entity} without assuming a specific task verb.

{context}

Requirements:
{constraints}

Follow the user's wording closely. Prefer clarifying questions over inventing intent.""",
        }

    def _init_compilation_rules(self) -> Dict[str, Any]:
        """Initialize compilation rules and transformations."""
        return {
            "intent_verbs": {
                IntentType.ANALYZE: "analyze",
                IntentType.GENERATE: "generate",
                IntentType.SUMMARIZE: "summarize",
                IntentType.TRANSLATE: "translate",
                IntentType.CLASSIFY: "classify",
                IntentType.EXTRACT: "extract",
                IntentType.COMPARE: "compare",
                IntentType.EXPLAIN: "explain",
                IntentType.CREATE: "create",
                IntentType.MODIFY: "modify",
                IntentType.VALIDATE: "validate",
                IntentType.SEARCH: "search",
                IntentType.DEBUG: "debug",
                IntentType.OPTIMIZE: "optimize",
                IntentType.ABSTAIN: "process",
            },
            "entity_descriptions": {
                EntityType.DATASET: "dataset",
                EntityType.TEXT: "text content",
                EntityType.CODE: "code",
                EntityType.DOCUMENT: "document",
                EntityType.IMAGE: "image",
                EntityType.DATA: "data",
                EntityType.MODEL: "model",
                EntityType.API: "API",
                EntityType.SYSTEM: "system",
                EntityType.USER: "user information",
                EntityType.BUSINESS: "business data",
                EntityType.FINANCIAL: "financial information",
                EntityType.MEDICAL: "medical data",
                EntityType.LEGAL: "legal document",
            },
            "constraint_instructions": {
                ConstraintType.PRIVACY: "Ensure all sensitive information is protected and privacy is maintained.",
                ConstraintType.ACCURACY: "Provide highly accurate and precise results with minimal errors.",
                ConstraintType.SPEED: "Focus on efficiency and provide results quickly.",
                ConstraintType.COST: "Optimize for cost-effectiveness and resource efficiency.",
                ConstraintType.FORMAT: "Follow the specified format requirements precisely.",
                ConstraintType.LENGTH: "Adhere to the specified length requirements.",
                ConstraintType.STYLE: "Maintain the specified writing style and tone.",
                ConstraintType.LANGUAGE: "Use the specified language consistently.",
                ConstraintType.SECURITY: "Ensure all security requirements are met.",
                ConstraintType.COMPLIANCE: "Follow all relevant compliance requirements.",
            },
        }

    def _init_formatting_rules(self) -> Dict[str, str]:
        """Initialize formatting rules for different contexts."""
        return {
            "system_prefix": "SYSTEM:\n",
            "task_prefix": "TASK:\n",
            "context_prefix": "CONTEXT:\n",
            "constraints_prefix": "REQUIREMENTS:\n",
            "section_separator": "\n---\n",
            "bullet_point": "• ",
            "numbered_list": "{index}. ",
            "code_block": "```\n{content}\n```",
            "emphasis": "**{text}**",
        }

    def compile(
        self,
        ir: Union[PromptIR, str],
        optimization_level: str = "standard",
        preserve_intent: bool = True,
    ) -> str:
        """
        Compile Prompt IR into executable prompt while preserving user content.

        Args:
            ir: Prompt Intermediate Representation or prompt string
            optimization_level: "minimal", "standard", or "comprehensive"
            preserve_intent: If True (default), strictly preserve original prompt content

        Returns:
            Compiled prompt ready for LLM
        """
        # Select compilation strategy based on optimization level
        if optimization_level == "minimal":
            return self._compile_minimal(ir, preserve_intent=preserve_intent)
        elif optimization_level == "comprehensive":
            return self._compile_comprehensive(ir, preserve_intent=preserve_intent)
        else:
            return self._compile_standard(ir, preserve_intent=preserve_intent)

    def _compile_standard(
        self, ir: Union[PromptIR, str], preserve_intent: bool = True
    ) -> str:
        """Standard compilation with balanced detail, preserving user prompt."""
        if isinstance(ir, str):
            prompt_text = ir
            ir = PromptIR(
                intent=IntentType.ANALYZE,
                entity=EntityType.TEXT,
                constraints=[],
                privacy=PrivacyLevel.PUBLIC,
                original_prompt=ir,
                extracted_entities=[],
                parameters={},
            )
        else:
            prompt_text = getattr(ir, "original_prompt", "") or ""

        if not prompt_text.strip():
            return ""

        # Build constraints text if present
        constraints = getattr(ir, "constraints", [])
        constraints_text = (
            self._build_constraints_text(constraints) if constraints else ""
        )

        # If intent is abstain or no specific constraints, keep prompt text clean
        intent = getattr(ir, "intent", IntentType.ABSTAIN)
        if (
            intent == IntentType.ABSTAIN
            or not constraints
            or constraints_text == "No specific constraints."
        ):
            return self._post_process_prompt(prompt_text, ir)

        parts = [prompt_text.strip()]
        if (
            constraints_text
            and constraints_text != "No specific constraints."
            and constraints_text.lower() not in prompt_text.lower()
        ):
            parts.append(f"Requirements:\n{constraints_text}")

        compiled = "\n\n".join(parts)
        return self._post_process_prompt(compiled, ir)

    def _compile_minimal(
        self, ir: Union[PromptIR, str], preserve_intent: bool = True
    ) -> str:
        """Minimal compilation for maximum token efficiency."""
        if isinstance(ir, str):
            prompt_text = ir
            ir = PromptIR(
                intent=IntentType.ANALYZE,
                entity=EntityType.TEXT,
                constraints=[],
                privacy=PrivacyLevel.PUBLIC,
                original_prompt=ir,
                extracted_entities=[],
                parameters={},
            )
        else:
            prompt_text = getattr(ir, "original_prompt", "") or ""

        if not prompt_text.strip():
            return ""

        return self._post_process_prompt(prompt_text, ir)

    def _compile_comprehensive(
        self, ir: Union[PromptIR, str], preserve_intent: bool = True
    ) -> str:
        """Comprehensive compilation with structured guidance."""
        if isinstance(ir, str):
            prompt_text = ir
            ir = PromptIR(
                intent=IntentType.ANALYZE,
                entity=EntityType.TEXT,
                constraints=[],
                privacy=PrivacyLevel.PUBLIC,
                original_prompt=ir,
                extracted_entities=[],
                parameters={},
            )
        else:
            prompt_text = getattr(ir, "original_prompt", "") or ""

        if not prompt_text.strip():
            return ""

        sections = []

        # System role
        system_role = self._generate_system_role(ir)
        sections.append(f"{self.formatting_rules['system_prefix']}{system_role}")

        # Task definition - user's actual prompt text!
        sections.append(f"{self.formatting_rules['task_prefix']}{prompt_text.strip()}")

        # Context
        context_val = getattr(ir, "context", None)
        if (
            context_val
            and context_val.lower() not in prompt_text.lower()
        ):
            context_section = self._build_detailed_context(ir)
            sections.append(
                f"{self.formatting_rules['context_prefix']}{context_section}"
            )

        # Constraints and requirements
        if getattr(ir, "constraints", None):
            constraints_section = self._build_detailed_constraints(ir)
            if (
                constraints_section
                and constraints_section != "Standard quality and ethical guidelines."
            ):
                sections.append(
                    f"{self.formatting_rules['constraints_prefix']}{constraints_section}"
                )

        # Output format
        output_format = self._generate_output_format(ir)
        if output_format:
            sections.append(f"OUTPUT FORMAT:\n{output_format}")

        compiled = self.formatting_rules["section_separator"].join(sections)
        return self._post_process_prompt(compiled, ir)

    def _build_constraints_text(self, constraints: List[ConstraintType]) -> str:
        """Build constraints section text."""
        if not constraints:
            return "No specific constraints."

        constraint_items = []
        for constraint in constraints:
            instruction = self.compilation_rules["constraint_instructions"].get(
                constraint
            )
            if instruction:
                constraint_items.append(
                    f"{self.formatting_rules['bullet_point']}{instruction}"
                )

        return (
            "\n".join(constraint_items)
            if constraint_items
            else "Standard quality requirements."
        )

    def _build_context_text(self, ir: PromptIR) -> str:
        """Build context section text."""
        context_parts = []

        if ir.context:
            context_parts.append(f"Additional context: {ir.context}")

        if ir.extracted_entities:
            entities_text = f"Relevant entities: {', '.join(ir.extracted_entities[:5])}"
            context_parts.append(entities_text)

        if ir.parameters:
            params_text = f"Parameters: {self._format_parameters(ir.parameters)}"
            context_parts.append(params_text)

        return (
            "\n".join(context_parts)
            if context_parts
            else "No additional context provided."
        )

    def _build_detailed_context(self, ir: PromptIR) -> str:
        """Build detailed context section."""
        context_items = []

        if ir.context:
            context_items.append(f"• Background: {ir.context}")

        if ir.extracted_entities:
            context_items.append(f"• Key entities: {', '.join(ir.extracted_entities)}")

        if ir.parameters:
            context_items.append(
                f"• Parameters: {self._format_parameters(ir.parameters)}"
            )

        if ir.urgency:
            context_items.append(f"• Urgency level: {ir.urgency}")

        if ir.complexity_score:
            complexity_level = ir.get_complexity_level()
            context_items.append(f"• Complexity: {complexity_level}")

        return "\n".join(context_items) if context_items else "Standard context."

    def _build_detailed_constraints(self, ir: PromptIR) -> str:
        """Build detailed constraints section."""
        constraint_items = []

        for constraint in ir.constraints:
            instruction = self.compilation_rules["constraint_instructions"].get(
                constraint
            )
            if instruction:
                constraint_items.append(f"• {instruction}")

        # Add privacy-specific constraints
        if ir.requires_privacy_masking():
            constraint_items.append(
                "• Ensure all sensitive data is properly masked and anonymized"
            )

        # Add optimization-specific constraints
        if ir.optimization_targets:
            targets_text = ", ".join(ir.optimization_targets)
            constraint_items.append(f"• Optimize for: {targets_text}")

        return (
            "\n".join(constraint_items)
            if constraint_items
            else "Standard quality and ethical guidelines."
        )

    def _format_parameters(self, parameters: Dict[str, Any]) -> str:
        """Format parameters for inclusion in prompt."""
        formatted_parts = []

        for key, value in parameters.items():
            if isinstance(value, list):
                formatted_parts.append(f"{key}: {', '.join(map(str, value))}")
            else:
                formatted_parts.append(f"{key}: {value}")

        return ", ".join(formatted_parts)

    def _generate_system_role(self, ir: PromptIR) -> str:
        """Generate appropriate system role based on intent and entity."""
        role_mapping = {
            (
                IntentType.ANALYZE,
                EntityType.DATASET,
            ): "You are an expert data scientist and analyst.",
            (
                IntentType.ANALYZE,
                EntityType.CODE,
            ): "You are an expert software analyst and code reviewer.",
            (
                IntentType.GENERATE,
                EntityType.CODE,
            ): "You are an expert software developer and programmer.",
            (
                IntentType.GENERATE,
                EntityType.TEXT,
            ): "You are an expert content writer and communicator.",
            (
                IntentType.DEBUG,
                EntityType.CODE,
            ): "You are an expert software debugger and problem solver.",
            (
                IntentType.OPTIMIZE,
                EntityType.SYSTEM,
            ): "You are an expert system optimizer and performance engineer.",
        }

        role = role_mapping.get((ir.intent, ir.entity))
        if not role:
            # Default role based on intent
            intent_roles = {
                IntentType.ANALYZE: "You are an expert analyst with deep domain knowledge.",
                IntentType.GENERATE: "You are an expert creator with strong creative skills.",
                IntentType.SUMMARIZE: "You are an expert summarizer with excellent comprehension.",
                IntentType.EXPLAIN: "You are an expert educator with strong communication skills.",
                IntentType.VALIDATE: "You are an expert quality assurance specialist.",
            }
            role = intent_roles.get(
                ir.intent, "You are an expert assistant with comprehensive knowledge."
            )

        # Add privacy context if needed
        if ir.requires_privacy_masking():
            role += " You are trained to handle sensitive data responsibly and maintain strict privacy."

        return role

    def _generate_output_format(self, ir: PromptIR) -> Optional[str]:
        """Generate output format instructions."""
        format_requirements = []

        if ConstraintType.FORMAT in ir.constraints:
            format_requirements.append("Follow the specified format precisely")

        if ir.intent in [IntentType.ANALYZE, IntentType.COMPARE]:
            format_requirements.append("Provide clear, structured output with sections")

        if ir.intent in [IntentType.CLASSIFY, IntentType.VALIDATE]:
            format_requirements.append("Include confidence scores and reasoning")

        if ir.intent == IntentType.EXTRACT:
            format_requirements.append("Present extracted data in organized format")

        return (
            "\n".join(f"• {req}" for req in format_requirements)
            if format_requirements
            else None
        )

    def _post_process_prompt(self, prompt: str, ir: PromptIR) -> str:
        """Apply post-processing optimizations while preserving structure."""
        # Normalize trailing whitespace per line and limit blank lines
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in prompt.splitlines()]
        cleaned = "\n".join(lines)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        cleaned = cleaned.replace(" ,", ",").replace(" .", ".")

        # Add final optimization hints if needed
        if ir.is_cost_sensitive():
            cleaned += "\n\nFocus on efficiency and cost-effectiveness."

        if ir.is_time_sensitive():
            cleaned += "\n\nPrioritize speed and quick results."

        return cleaned.strip()

    def get_compilation_metrics(
        self, original_prompt: str, compiled_prompt: str, ir: PromptIR
    ) -> Dict[str, Any]:
        """Get compilation metrics for analysis."""
        original_tokens = self._estimate_tokens(original_prompt)
        compiled_tokens = self._estimate_tokens(compiled_prompt)
        tokens_saved = original_tokens - compiled_tokens

        return {
            "original_tokens": original_tokens,
            "compiled_tokens": compiled_tokens,
            "token_reduction": max(0, tokens_saved),
            "token_reduction_percentage": (
                (tokens_saved / original_tokens * 100)
                if original_tokens > 0 and tokens_saved > 0
                else 0.0
            ),
            "intent": ir.intent.value,
            "entity": ir.entity.value,
            "complexity_level": ir.get_complexity_level(),
            "constraint_count": len(ir.constraints),
            "privacy_level": ir.privacy.value,
        }

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation."""
        return int(len(text.split()) * 1.3)
