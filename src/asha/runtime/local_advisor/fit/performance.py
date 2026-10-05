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

"""Token generation speed estimation."""

from __future__ import annotations

from ..types import GGUFVariant, GPUInfo, ModelInfo
from .quantization import effective_quant_type, estimate_weight_bytes

_QUANT_EFFICIENCY = {
    "Q8_0": 0.45,
    "Q6_K": 0.50,
    "Q5_K_M": 0.52,
    "Q4_K_M": 0.55,
    "Q4_0": 0.53,
    "Q3_K_M": 0.50,
    "Q2_K": 0.45,
    "F16": 0.40,
    "FP16": 0.40,
}

_BACKEND_FACTOR = {
    "nvidia": 1.0,
    "amd": 0.78,
    "apple": 0.82,
    "intel": 0.65,
}

_FIT_FACTOR = {
    "full_gpu": 1.0,
    "partial_offload": 0.45,
    "cpu_only": 0.08,
}


def estimate_tok_per_sec(
    model: ModelInfo,
    variant: GGUFVariant | None,
    gpu: GPUInfo | None,
    fit_type: str,
) -> float | None:
    if fit_type == "cpu_only" or gpu is None:
        params_b = (model.parameter_count_active or model.parameter_count) / 1e9
        return max(0.5, params_b * 0.4)

    bandwidth = gpu.memory_bandwidth_gbps
    if not bandwidth:
        return None

    weight_bytes = estimate_weight_bytes(model, variant)
    quant = effective_quant_type(model, variant).upper()
    q_eff = _QUANT_EFFICIENCY.get(quant, 0.45)
    backend = _BACKEND_FACTOR.get(gpu.vendor, 0.7)
    fit = _FIT_FACTOR.get(fit_type, 0.5)

    if model.is_moe and model.parameter_count_active and model.parameter_count:
        active_ratio = max(model.parameter_count_active / model.parameter_count, 0.05)
        read_ratio = min(1.0, max(active_ratio, 0.25))
        weight_bytes = int(weight_bytes * read_ratio)

    bytes_per_token = weight_bytes / max(bandwidth * 1e9 / 8, 1)
    theoretical = 1.0 / max(bytes_per_token, 1e-9)
    return theoretical * q_eff * backend * fit
