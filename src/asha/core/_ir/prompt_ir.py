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

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum


class IntentType(Enum):
    """Intent types for prompt classification."""

    ANALYZE = "analyze"
    GENERATE = "generate"
    SUMMARIZE = "summarize"
    TRANSLATE = "translate"
    CLASSIFY = "classify"
    EXTRACT = "extract"
    COMPARE = "compare"
    EXPLAIN = "explain"
    CREATE = "create"
    MODIFY = "modify"
    VALIDATE = "validate"
    SEARCH = "search"
    DEBUG = "debug"
    CODE = "code"
    OPTIMIZE = "optimize"
    ABSTAIN = "abstain"


class EntityType(Enum):
    """Entity types for object classification."""

    DATASET = "dataset"
    TEXT = "text"
    CODE = "code"
    DOCUMENT = "document"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    DATA = "data"
    MODEL = "model"
    API = "api"
    SYSTEM = "system"
    USER = "user"
    BUSINESS = "business"
    FINANCIAL = "financial"
    MEDICAL = "medical"
    LEGAL = "legal"


class ConstraintType(Enum):
    """Constraint types for prompt requirements."""

    PRIVACY = "privacy"
    ACCURACY = "accuracy"
    SPEED = "speed"
    COST = "cost"
    FORMAT = "format"
    LENGTH = "length"
    TOKEN_LIMIT = "token_limit"
    STYLE = "style"
    TONE = "tone"
    LANGUAGE = "language"
    COMPLEXITY = "complexity"
    SECURITY = "security"
    COMPLIANCE = "compliance"


class PrivacyLevel(Enum):
    """Privacy levels for data handling."""

    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    MASKED = "masked"


@dataclass
class PromptIR:
    """
    Prompt Intermediate Representation (IR) - structured representation of a prompt.

    This is the core data structure that enables compiler-style optimizations
    and transformations in ASHA.
    """

    # Core components
    intent: IntentType
    entity: EntityType
    constraints: List[ConstraintType]
    privacy: PrivacyLevel

    # Additional metadata
    original_prompt: str
    extracted_entities: List[str]
    parameters: Dict[str, Any]
    context: Optional[str] = None
    urgency: Optional[str] = None
    complexity_score: Optional[float] = None
    token_estimate: Optional[int] = None

    # Optimization hints
    optimization_targets: List[str] = field(
        default_factory=lambda: ["tokens", "accuracy"]
    )
    fallback_intents: List[IntentType] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Post-initialization processing."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert IR to dictionary representation."""
        return {
            "intent": self.intent.value,
            "entity": self.entity.value,
            "constraints": [c.value for c in self.constraints],
            "privacy": self.privacy.value,
            "original_prompt": self.original_prompt,
            "extracted_entities": self.extracted_entities,
            "parameters": self.parameters,
            "context": self.context,
            "urgency": self.urgency,
            "complexity_score": self.complexity_score,
            "token_estimate": self.token_estimate,
            "optimization_targets": self.optimization_targets,
            "fallback_intents": [i.value for i in self.fallback_intents],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PromptIR":
        """Create IR from dictionary representation."""
        return cls(
            intent=IntentType(data["intent"]),
            entity=EntityType(data["entity"]),
            constraints=[ConstraintType(c) for c in data["constraints"]],
            privacy=PrivacyLevel(data["privacy"]),
            original_prompt=data["original_prompt"],
            extracted_entities=data.get("extracted_entities", []),
            parameters=data.get("parameters", {}),
            context=data.get("context"),
            urgency=data.get("urgency"),
            complexity_score=data.get("complexity_score"),
            token_estimate=data.get("token_estimate"),
            optimization_targets=data.get(
                "optimization_targets", ["tokens", "accuracy"]
            ),
            fallback_intents=[IntentType(i) for i in data.get("fallback_intents", [])],
        )

    def add_constraint(self, constraint: ConstraintType) -> None:
        """Add a constraint to the IR."""
        if constraint not in self.constraints:
            self.constraints.append(constraint)

    def remove_constraint(self, constraint: ConstraintType) -> None:
        """Remove a constraint from the IR."""
        if constraint in self.constraints:
            self.constraints.remove(constraint)

    def add_parameter(self, key: str, value: Any) -> None:
        """Add a parameter to the IR."""
        self.parameters[key] = value

    def get_complexity_level(self) -> str:
        """Get complexity level based on score."""
        if self.complexity_score is None:
            return "unknown"
        elif self.complexity_score < 0.3:
            return "low"
        elif self.complexity_score < 0.7:
            return "medium"
        else:
            return "high"

    def requires_privacy_masking(self) -> bool:
        """Check if privacy masking is required."""
        return self.privacy in [
            PrivacyLevel.CONFIDENTIAL,
            PrivacyLevel.RESTRICTED,
            PrivacyLevel.MASKED,
        ]

    def is_cost_sensitive(self) -> bool:
        """Check if cost optimization is important."""
        return ConstraintType.COST in self.constraints

    def is_time_sensitive(self) -> bool:
        """Check if time optimization is important."""
        return ConstraintType.SPEED in self.constraints or self.urgency == "high"

    def get_optimization_priority(self) -> List[str]:
        """Get optimization priorities based on constraints."""
        priorities = []

        if ConstraintType.COST in self.constraints:
            priorities.append("cost")
        if ConstraintType.SPEED in self.constraints:
            priorities.append("speed")
        if ConstraintType.TOKEN_LIMIT in self.constraints:
            priorities.append("tokens")
        if ConstraintType.ACCURACY in self.constraints:
            priorities.append("accuracy")

        return priorities or ["tokens", "accuracy"]  # Default priorities

    def __str__(self) -> str:
        """String representation of IR."""
        return f"PromptIR(intent={self.intent.value}, entity={self.entity.value}, privacy={self.privacy.value})"

    def __repr__(self) -> str:
        """Detailed string representation."""
        return self.to_dict().__repr__()


class IRTransform:
    """
    Transformation operations for Prompt IR.

    Enables compiler-style optimizations and transformations.
    """

    @staticmethod
    def optimize_for_tokens(ir: PromptIR) -> PromptIR:
        """Optimize IR for token efficiency."""
        # Create a copy to avoid mutating original
        optimized = PromptIR(
            intent=ir.intent,
            entity=ir.entity,
            constraints=ir.constraints.copy(),
            privacy=ir.privacy,
            original_prompt=ir.original_prompt,
            extracted_entities=ir.extracted_entities.copy(),
            parameters=ir.parameters.copy(),
            context=ir.context,
            urgency=ir.urgency,
            complexity_score=ir.complexity_score,
            token_estimate=ir.token_estimate,
            optimization_targets=["tokens"]
            + [t for t in ir.optimization_targets if t != "tokens"],
            fallback_intents=ir.fallback_intents.copy(),
        )

        # Add token optimization constraint
        optimized.add_constraint(ConstraintType.TOKEN_LIMIT)

        return optimized

    @staticmethod
    def enhance_privacy(ir: PromptIR) -> PromptIR:
        """Enhance privacy level of IR."""
        enhanced = PromptIR(
            intent=ir.intent,
            entity=ir.entity,
            constraints=ir.constraints.copy(),
            privacy=PrivacyLevel.MASKED,  # Upgrade to masked
            original_prompt=ir.original_prompt,
            extracted_entities=ir.extracted_entities.copy(),
            parameters=ir.parameters.copy(),
            context=ir.context,
            urgency=ir.urgency,
            complexity_score=ir.complexity_score,
            token_estimate=ir.token_estimate,
            optimization_targets=ir.optimization_targets.copy(),
            fallback_intents=ir.fallback_intents.copy(),
        )

        # Add privacy constraint
        enhanced.add_constraint(ConstraintType.PRIVACY)

        return enhanced

    @staticmethod
    def simplify_intent(ir: PromptIR) -> PromptIR:
        """Simplify intent for faster processing."""
        # Map complex intents to simpler ones
        intent_mapping = {
            IntentType.OPTIMIZE: IntentType.ANALYZE,
            IntentType.DEBUG: IntentType.ANALYZE,
            IntentType.VALIDATE: IntentType.ANALYZE,
            IntentType.MODIFY: IntentType.CREATE,
        }

        simplified_intent = intent_mapping.get(ir.intent, ir.intent)

        return PromptIR(
            intent=simplified_intent,
            entity=ir.entity,
            constraints=ir.constraints.copy(),
            privacy=ir.privacy,
            original_prompt=ir.original_prompt,
            extracted_entities=ir.extracted_entities.copy(),
            parameters=ir.parameters.copy(),
            context=ir.context,
            urgency=ir.urgency,
            complexity_score=ir.complexity_score,
            token_estimate=ir.token_estimate,
            optimization_targets=["speed"]
            + [t for t in ir.optimization_targets if t != "speed"],
            fallback_intents=ir.fallback_intents.copy(),
        )
