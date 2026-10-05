"""
Metrics Engine - Deterministic scoring for prompt quality
"""

import re
from typing import List
from dataclasses import dataclass


@dataclass
class QualityMetrics:
    """Quality metrics for prompt evaluation."""

    clarity_score: float
    structure_score: float
    efficiency_score: float
    overall_score: float
    issues: List[str]
    recommendations: List[str]


class MetricsEngine:
    """Deterministic scoring engine for prompt quality assessment."""

    def __init__(self) -> None:
        """Initialize scoring criteria and weights."""
        self.weights = {"clarity": 0.35, "structure": 0.35, "efficiency": 0.30}

        self.clarity_indicators = {
            "positive": [
                r"\b(clear|specific|precise|exact|detailed|explicit)\b",
                r"\b(analyze|summarize|generate|extract|classify)\b",
                r"\b(dataset|data|text|code|document)\b",
                r"\b(quickly|accurately|briefly|thoroughly)\b",
            ],
            "negative": [
                r"\b(maybe|perhaps|possibly|might|could|would)\b",
                r"\b(like|kind of|sort of|approximately|roughly)\b",
                r"\b(stuff|things|something|anything)\b",
                r"\b(hey|hi|hello|bro|dude)\b",
            ],
        }

        self.structure_indicators = {
            "positive": [
                r"^TASK:\s*\w+",  # Task definition
                r"^INPUT:\s*\w+",  # Input specification
                r"^OUTPUT:\s*\w+",  # Output specification
                r"\d+\.\s+",  # Numbered lists
                r"^-\s+",  # Bullet points
                r"\w+:\s+\w+",  # Key-value pairs
            ],
            "negative": [
                r"\.{2,}",  # Multiple periods
                r"\s{2,}",  # Multiple spaces
                r"[A-Z]{3,}",  # All caps words
                r"\b(\w+)\s+\1\b",  # Repeated words
            ],
        }

        self.efficiency_indicators = {
            "positive": [
                r"\b(brief|short|concise|succinct|compact)\b",
                r"\b(key|main|essential|important|critical)\b",
                r"\b(only|just|directly|simply)\b",
                r"\b(under \d+|less than \d+|limit to \d+)\b",
            ],
            "negative": [
                r"\b(very|quite|rather|somewhat|really|actually)\b",
                r"\b(the fact that|due to the fact that|in order to)\b",
                r"\b(as far as i know|in my opinion|i think)\b",
                r"\b(for all intents and purposes|when all is said and done)\b",
            ],
        }

    def calculate_metrics(
        self, original_prompt: str, optimized_prompt: str
    ) -> QualityMetrics:
        """
        Calculate quality metrics for prompt comparison.

        Args:
            original_prompt: Original prompt before optimization
            optimized_prompt: Optimized prompt after processing

        Returns:
            QualityMetrics with scores and recommendations
        """
        # Calculate individual scores
        clarity_score = self._calculate_clarity_score(optimized_prompt)
        structure_score = self._calculate_structure_score(optimized_prompt)
        efficiency_score = self._calculate_efficiency_score(optimized_prompt)

        # Calculate overall score
        overall_score = (
            clarity_score * self.weights["clarity"]
            + structure_score * self.weights["structure"]
            + efficiency_score * self.weights["efficiency"]
        )

        # Identify issues
        issues = self._identify_issues(optimized_prompt)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            clarity_score, structure_score, efficiency_score, issues
        )

        return QualityMetrics(
            clarity_score=clarity_score,
            structure_score=structure_score,
            efficiency_score=efficiency_score,
            overall_score=overall_score,
            issues=issues,
            recommendations=recommendations,
        )

    def _calculate_clarity_score(self, prompt: str) -> float:
        """Calculate clarity score based on language clarity."""
        positive_score = 0
        negative_score = 0

        # Count positive indicators
        for pattern in self.clarity_indicators["positive"]:
            matches = len(re.findall(pattern, prompt, re.IGNORECASE))
            positive_score += matches

        # Count negative indicators
        for pattern in self.clarity_indicators["negative"]:
            matches = len(re.findall(pattern, prompt, re.IGNORECASE))
            negative_score += matches

        # Calculate normalized score
        total_indicators = positive_score + negative_score
        if total_indicators == 0:
            return 0.7  # Neutral score for no indicators

        clarity_ratio = positive_score / total_indicators
        return min(clarity_ratio, 1.0)

    def _calculate_structure_score(self, prompt: str) -> float:
        """Calculate structure score based on prompt organization."""
        positive_score = 0.0
        negative_score = 0

        # Count positive structure indicators
        for pattern in self.structure_indicators["positive"]:
            if re.search(pattern, prompt, re.MULTILINE | re.IGNORECASE):
                positive_score += 1

        # Count negative structure indicators
        for pattern in self.structure_indicators["negative"]:
            matches = len(re.findall(pattern, prompt, re.IGNORECASE))
            negative_score += matches

        # Bonus for multiple structure types
        structure_types = 0
        if re.search(r"^TASK:\s*\w+", prompt, re.MULTILINE | re.IGNORECASE):
            structure_types += 1
        if re.search(r"^\w+:\s+\w+", prompt, re.MULTILINE):
            structure_types += 1
        if re.search(r"\d+\.\s+", prompt, re.MULTILINE):
            structure_types += 1

        positive_score += structure_types * 0.5

        # Calculate normalized score
        total_score = positive_score - negative_score
        normalized_score = max(0, min(1, (total_score + 2) / 4))  # Normalize to 0-1

        return normalized_score

    def _calculate_efficiency_score(self, prompt: str) -> float:
        """Calculate efficiency score based on conciseness."""
        positive_score = 0
        negative_score = 0

        # Count positive efficiency indicators
        for pattern in self.efficiency_indicators["positive"]:
            matches = len(re.findall(pattern, prompt, re.IGNORECASE))
            positive_score += matches

        # Count negative efficiency indicators
        for pattern in self.efficiency_indicators["negative"]:
            matches = len(re.findall(pattern, prompt, re.IGNORECASE))
            negative_score += matches

        # Calculate length efficiency (prefer shorter prompts)
        word_count = len(re.findall(r"\b\w+\b", prompt))
        if word_count <= 10:
            length_score = 1.0
        elif word_count <= 20:
            length_score = 0.8
        elif word_count <= 30:
            length_score = 0.6
        else:
            length_score = 0.4

        # Calculate normalized score
        total_indicators = positive_score + negative_score
        if total_indicators == 0:
            indicator_score = 0.7  # Neutral score
        else:
            indicator_score = positive_score / total_indicators

        # Combine indicator and length scores
        efficiency_score = (indicator_score * 0.7) + (length_score * 0.3)

        return min(efficiency_score, 1.0)

    def _identify_issues(self, prompt: str) -> List[str]:
        """Identify specific issues in the prompt."""
        issues = []

        # Check for ambiguity
        if re.search(
            r"\b(maybe|perhaps|possibly|might|could|would)\b", prompt, re.IGNORECASE
        ):
            issues.append("Contains ambiguous language")

        # Check for filler words
        if re.search(
            r"\b(very|quite|rather|somewhat|really|actually)\b", prompt, re.IGNORECASE
        ):
            issues.append("Contains filler words")

        # Check for conversational tone
        if re.search(
            r"\b(hey|hi|hello|bro|dude|thanks|please)\b", prompt, re.IGNORECASE
        ):
            issues.append("Too conversational")

        # Check for redundancy
        if re.search(r"\b(\w+)\s+\1\b", prompt, re.IGNORECASE):
            issues.append("Contains repeated words")

        # Check for lack of structure
        if not re.search(r"^\w+:\s+\w+", prompt, re.MULTILINE):
            if len(prompt.split()) > 15:
                issues.append("Lacks clear structure")

        return issues

    def _generate_recommendations(
        self, clarity: float, structure: float, efficiency: float, issues: List[str]
    ) -> List[str]:
        """Generate improvement recommendations."""
        recommendations = []

        if clarity < 0.6:
            recommendations.append("Use more specific and direct language")
            recommendations.append("Remove ambiguous words like 'maybe' or 'possibly'")

        if structure < 0.6:
            recommendations.append("Add clear structure with TASK/INPUT/OUTPUT format")
            recommendations.append("Use bullet points or numbered lists for clarity")

        if efficiency < 0.6:
            recommendations.append("Remove filler words and redundant phrases")
            recommendations.append("Be more concise and direct")

        # Specific recommendations based on issues
        if "Contains ambiguous language" in issues:
            recommendations.append("Replace ambiguous terms with specific instructions")

        if "Contains filler words" in issues:
            recommendations.append("Remove words like 'very', 'quite', 'really'")

        if "Too conversational" in issues:
            recommendations.append(
                "Use professional tone instead of conversational language"
            )

        if "Contains repeated words" in issues:
            recommendations.append("Remove repeated words and phrases")

        if "Lacks clear structure" in issues:
            recommendations.append("Organize prompt with clear sections and headings")

        return recommendations

    def get_metrics_summary(self, metrics: QualityMetrics) -> str:
        """Generate human-readable metrics summary."""
        return f"Clarity: {metrics.clarity_score:.2f} | Structure: {metrics.structure_score:.2f} | Efficiency: {metrics.efficiency_score:.2f} | Overall: {metrics.overall_score:.2f}"
