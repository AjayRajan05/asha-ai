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

"""Optional Ollama micro-benchmark for top model candidates."""

from __future__ import annotations

import time
from typing import Any, List, Optional

import requests

from .types import ModelRecommendation, WorkloadProfile

_OLLAMA_URL = "http://localhost:11434/api/generate"


def _ollama_available() -> bool:
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def _probe_model(model_name: str, prompt: str, timeout: int = 60) -> Optional[float]:
    payload: dict[str, Any] = {"model": model_name, "prompt": prompt, "stream": False}
    start = time.perf_counter()
    try:
        r = requests.post(_OLLAMA_URL, json=payload, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        eval_count = data.get("eval_count") or 0
        eval_duration = data.get("eval_duration") or 0
        if eval_count > 0 and eval_duration > 0:
            return eval_count / (eval_duration / 1e9)
        elapsed = time.perf_counter() - start
        response = data.get("response") or ""
        tokens = max(1, len(response.split()))
        return tokens / elapsed
    except Exception:
        return None


def probe_recommendations(
    recommendations: List[ModelRecommendation],
    workload: WorkloadProfile,
    *,
    max_candidates: int = 3,
    measured_weight: float = 0.4,
) -> List[ModelRecommendation]:
    """Re-rank top candidates using measured Ollama throughput when available."""
    if not _ollama_available() or not recommendations:
        return recommendations

    prompts = workload.sample_compiled_prompts or ["Summarize this in one sentence."]
    updated: List[ModelRecommendation] = []

    for rec in recommendations[:max_candidates]:
        measured = None
        for prompt in prompts[:2]:
            measured = _probe_model(rec.ollama_pull_name or rec.model_id, prompt)
            if measured:
                break
        if measured and rec.estimated_tok_s:
            blended = (
                rec.estimated_tok_s * (1 - measured_weight) + measured * measured_weight
            )
            rec.measured_tok_s = round(measured, 2)
            rec.estimated_tok_s = round(blended, 2)
            rec.scores.speed_usability = min(
                1.0, rec.scores.speed_usability * 0.6 + (measured / 20.0) * 0.4
            )
            rec.scores.final_score = (
                rec.scores.hardware_fit
                * rec.scores.workload_fit
                * rec.scores.privacy_compliance
                * rec.scores.speed_usability
                * 100.0
            )
            rec.reasoning += f"; measured ~{measured:.1f} tok/s via Ollama"
        updated.append(rec)

    updated.extend(recommendations[max_candidates:])
    updated.sort(key=lambda r: r.scores.final_score, reverse=True)
    return updated
