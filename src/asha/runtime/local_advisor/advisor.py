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

"""AshaFit orchestrator - recommend_local_model entry point."""

from __future__ import annotations

from typing import Optional, Sequence

from asha.exceptions import warn_experimental

from .catalog.fetcher import get_catalog
from .hardware_profiler import detect_hardware
from .probe import probe_recommendations
from .ranker import rank_models
from .types import RecommendationReport
from .workload_profiler import load_prompts, profile_workload

# Module-level cache for wrap_llm / Agent auto-select
_last_report: Optional[RecommendationReport] = None


def get_last_recommendation() -> Optional[RecommendationReport]:
    """Return the most recent recommendation report (used by wrap_llm auto-select)."""
    return _last_report


def recommend_local_model(
    prompts: Sequence[str] | None = None,
    *,
    prompts_file: str | None = None,
    mode: str = "balanced",
    top: int = 5,
    gpu: Optional[str] = None,
    vram_gb: Optional[float] = None,
    cpu_only: bool = False,
    refresh_catalog: bool = False,
    preferred_quant: Optional[str] = None,
    probe: bool = False,
) -> RecommendationReport:
    """
    Recommend local LLM models for your compiled workload and hardware.

    .. warning::
        Experimental in 0.4.2; emits ``ASHAExperimentalWarning``.

    Args:
        prompts: Sample prompts representing your application workload.
        prompts_file: Path to JSON/JSONL file of prompts.
        mode: ASHA processing mode (balanced, strict, lite).
        top: Number of recommendations to return.
        gpu: Optional GPU name to simulate (e.g. "RTX 4090").
        vram_gb: Override VRAM when simulating GPU.
        cpu_only: Ignore GPU and rank for CPU inference.
        refresh_catalog: Bypass catalog cache and refetch from HuggingFace.
        preferred_quant: Prefer a specific quantization (e.g. Q4_K_M).
        probe: Run optional Ollama micro-benchmark on top candidates.

    Returns:
        RecommendationReport with ranked models and workload/hardware profiles.
    """
    global _last_report

    warn_experimental("recommend_local_model / AshaFit")
    prompt_list = load_prompts(prompts=prompts, prompts_file=prompts_file)
    workload = profile_workload(prompt_list, mode=mode)
    hardware = detect_hardware(gpu=gpu, vram_gb=vram_gb, cpu_only=cpu_only)
    catalog, source = get_catalog(refresh=refresh_catalog)

    ranked = rank_models(
        catalog,
        hardware,
        workload,
        top=top,
        preferred_quant=preferred_quant,
    )

    probed = False
    if probe and ranked:
        ranked = probe_recommendations(ranked, workload)
        probed = True

    report = RecommendationReport(
        top_models=ranked,
        workload=workload,
        hardware=hardware,
        catalog_source=source,
        probed=probed,
    )
    _last_report = report
    return report
