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

"""Compatibility checking: can a model run on given hardware?"""

from __future__ import annotations

from ..constants import MIN_COMPUTE_CAPABILITY_OLLAMA, _GiB
from ..types import (
    CompatibilityResult,
    GGUFVariant,
    GPUInfo,
    HardwareProfile,
    ModelInfo,
)
from .performance import estimate_tok_per_sec
from .quantization import estimate_weight_bytes
from .vram import estimate_vram


def _gpu_available_memory(gpu: GPUInfo, usable_ram: int) -> int:
    if gpu.shared_memory and gpu.vram_bytes < 2 * _GiB:
        return usable_ram
    return gpu.vram_bytes


def _uses_shared_pool(gpu: GPUInfo) -> bool:
    return gpu.shared_memory and gpu.vram_bytes < 2 * _GiB


def _fit_candidate_gpus(gpus: list[GPUInfo]) -> list[GPUInfo]:
    has_dedicated = any(not _uses_shared_pool(g) and g.vram_bytes > 0 for g in gpus)
    if not has_dedicated:
        return gpus
    return [g for g in gpus if not _uses_shared_pool(g)]


def check_compatibility(
    model: ModelInfo,
    variant: GGUFVariant | None,
    hardware: HardwareProfile,
    context_length: int = 4096,
) -> CompatibilityResult:
    warnings: list[str] = []
    vram_required = estimate_vram(model, variant, context_length)
    usable_ram = int(hardware.ram_bytes * 0.80)

    best_gpu: GPUInfo | None = None
    best_available = 0
    total_vram = 0
    candidate_gpus = _fit_candidate_gpus(hardware.gpus)
    for gpu in candidate_gpus:
        available = _gpu_available_memory(gpu, usable_ram)
        total_vram += available
        if best_gpu is None or available > best_available:
            best_gpu = gpu
            best_available = available

    vram_available = total_vram if total_vram > 0 else 0
    offload_ram = (
        0
        if best_gpu and (best_gpu.shared_memory or best_gpu.vendor == "apple")
        else usable_ram
    )

    if best_gpu and best_gpu.vendor == "nvidia" and best_gpu.compute_capability:
        if best_gpu.compute_capability < MIN_COMPUTE_CAPABILITY_OLLAMA:
            warnings.append("Compute capability below Ollama minimum")

    if vram_available >= vram_required:
        fit_type = "full_gpu"
        can_run = True
    elif vram_available > 0 and (vram_available + offload_ram) >= vram_required:
        fit_type = "partial_offload"
        can_run = True
        warnings.append("Partial CPU offload required")
    elif usable_ram >= vram_required:
        fit_type = "cpu_only"
        can_run = True
        warnings.append("CPU-only inference")
    else:
        fit_type = "cpu_only"
        can_run = False
        warnings.append("Insufficient memory")

    context_fits = not (
        model.context_length is not None and model.context_length < context_length
    )
    if not context_fits:
        warnings.append("Requested context exceeds model maximum")

    file_size = estimate_weight_bytes(model, variant)
    if hardware.disk_free_bytes > 0 and file_size > hardware.disk_free_bytes:
        warnings.append("Insufficient disk space")
        can_run = False

    speed = estimate_tok_per_sec(model, variant, best_gpu, fit_type)

    return CompatibilityResult(
        model=model,
        gguf_variant=variant,
        can_run=can_run,
        vram_required_bytes=vram_required,
        vram_available_bytes=vram_available,
        warnings=warnings,
        fit_type=fit_type,
        context_fits=context_fits,
        estimated_tok_per_sec=speed,
    )
