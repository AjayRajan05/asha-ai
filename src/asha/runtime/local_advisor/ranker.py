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

"""Two-axis ranking: hardware fit x workload fit x privacy x speed."""

from __future__ import annotations

from typing import List, Optional

from .catalog.fetcher import pick_best_variant
from .constants import FIT_MULTIPLIERS, MIN_SPEED_TOK_S
from .fit.compatibility import check_compatibility
from .fit.quantization import quant_quality_penalty
from .types import (
    CompatibilityResult,
    HardwareProfile,
    ModelInfo,
    ModelRecommendation,
    ScoreBreakdown,
    WorkloadProfile,
)


def _workload_fit(model: ModelInfo, workload: WorkloadProfile) -> float:
    if not model.specialization_tags:
        score = 0.5
    else:
        overlap = set(model.specialization_tags) & set(workload.specialization_tags)
        score = 0.4 + 0.6 * (len(overlap) / max(len(workload.specialization_tags), 1))

    params_b = model.parameter_count / 1e9
    if workload.avg_complexity > 0.7 and params_b >= 7:
        score += 0.15
    elif workload.avg_complexity < 0.4 and params_b <= 8:
        score += 0.1

    if (
        model.context_length
        and model.context_length >= workload.context_length_required
    ):
        score += 0.1
    elif (
        model.context_length and model.context_length < workload.context_length_required
    ):
        score -= 0.2

    dominant = {i.value for i in workload.dominant_intents}
    if "code" in dominant and "coding" in model.specialization_tags:
        score += 0.15
    if "analyze" in dominant and "analysis" in model.specialization_tags:
        score += 0.1

    return max(0.0, min(1.0, score))


def _privacy_compliance(model: ModelInfo, workload: WorkloadProfile) -> float:
    if not workload.requires_local:
        return 1.0
    base = model.instruction_following_score
    if workload.pii_rate > 0 and base < 0.8:
        return max(0.3, base)
    return max(0.5, base)


def _speed_usability(compat: CompatibilityResult) -> float:
    speed = compat.estimated_tok_per_sec
    if speed is None:
        return 0.5
    required = MIN_SPEED_TOK_S.get(compat.fit_type, 4.0)
    if speed < required:
        penalty = min(0.5, (required - speed) / required)
        return max(0.2, 1.0 - penalty)
    bonus = min(0.2, (speed - required) / max(required * 2, 1))
    return min(1.0, 0.85 + bonus)


def _hardware_fit(compat: CompatibilityResult) -> float:
    if not compat.can_run:
        return 0.0
    return FIT_MULTIPLIERS.get(compat.fit_type, 0.5)


def _build_reasoning(
    model: ModelInfo,
    compat: CompatibilityResult,
    workload: WorkloadProfile,
    scores: ScoreBreakdown,
) -> str:
    parts = [
        f"Fit: {compat.fit_type}",
        f"Workload tags {workload.specialization_tags} match {model.specialization_tags}",
    ]
    if workload.requires_local:
        parts.append("Local-only due to privacy/PII workload")
    if compat.estimated_tok_per_sec:
        parts.append(f"~{compat.estimated_tok_per_sec:.1f} tok/s estimated")
    if compat.warnings:
        parts.append(compat.warnings[0])
    return "; ".join(parts)


def rank_models(
    models: List[ModelInfo],
    hardware: HardwareProfile,
    workload: WorkloadProfile,
    *,
    top: int = 5,
    preferred_quant: Optional[str] = None,
) -> List[ModelRecommendation]:
    """Rank catalog models for hardware + compiled workload."""
    ranked: List[ModelRecommendation] = []

    for model in models:
        variant = pick_best_variant(model, preferred_quant)
        compat = check_compatibility(
            model, variant, hardware, workload.context_length_required
        )
        if not compat.can_run or not compat.context_fits:
            continue

        hw = _hardware_fit(compat)
        wl = _workload_fit(model, workload)
        priv = _privacy_compliance(model, workload)
        speed = _speed_usability(compat)
        quant_pen = quant_quality_penalty(model, variant)
        final = hw * wl * priv * speed * (1.0 - quant_pen) * 100.0

        scores = ScoreBreakdown(
            hardware_fit=hw,
            workload_fit=wl,
            privacy_compliance=priv,
            speed_usability=speed,
            final_score=final,
            quant_penalty=quant_pen,
        )
        confidence = min(0.95, 0.5 + (final / 200.0))
        if compat.fit_type != "full_gpu":
            confidence *= 0.85

        ranked.append(
            ModelRecommendation(
                model_id=model.id,
                quant=variant.quant_type if variant else "FP16",
                fit_type=compat.fit_type,
                estimated_tok_s=compat.estimated_tok_per_sec,
                scores=scores,
                reasoning=_build_reasoning(model, compat, workload, scores),
                confidence=round(confidence, 2),
                ollama_pull_name=model.ollama_name or model.name.lower(),
                vram_required_gb=round(compat.vram_required_bytes / (1024**3), 2),
            )
        )

    ranked.sort(key=lambda r: r.scores.final_score, reverse=True)
    return ranked[:top]
