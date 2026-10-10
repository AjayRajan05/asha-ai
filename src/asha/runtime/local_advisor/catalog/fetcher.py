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

"""HuggingFace model catalog fetcher with offline fallback."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, List, Optional

import requests

from ..constants import QUANT_BYTES_PER_WEIGHT, QUANT_PREFERENCE_ORDER
from ..types import GGUFVariant, ModelInfo
from .cache import load_cache, save_cache

logger = logging.getLogger(__name__)

HF_API_BASE = "https://huggingface.co/api"
_FALLBACK_PATH = Path(__file__).with_name("fallback.json")

_KNOWN_MOE_ACTIVE = {
    "Qwen/Qwen3-30B-A3B-GGUF": 3_000_000_000,
    "Qwen/Qwen3-235B-A22B": 22_000_000_000,
}


def _extract_size_hint(model_id: str) -> Optional[int]:
    matches = re.findall(r"(\d+(?:\.\d+)?)b(?:-a\d+(?:\.\d+)?b)?", model_id.lower())
    if not matches:
        return None
    return int(max(float(m) for m in matches) * 1e9)


def _extract_active_hint(model_id: str) -> Optional[int]:
    matches = re.findall(r"\d+(?:\.\d+)?b[-_]?a(\d+(?:\.\d+)?)b", model_id.lower())
    if not matches:
        return None
    return int(max(float(m) for m in matches) * 1e9)


def _extract_quant_type(filename: str) -> str:
    upper = filename.upper()
    for pattern in (
        r"[.-](Q\d+_K_[SMLA])",
        r"[.-](Q\d+_\d+)",
        r"[.-](Q\d+_K)",
        r"[.-](F16|FP16|BF16|F32)",
    ):
        m = re.search(pattern, upper)
        if m:
            return m.group(1)
    return "Q4_K_M"


def _infer_tags(model_id: str, architecture: str) -> List[str]:
    lower = model_id.lower()
    tags = ["general"]
    if any(k in lower for k in ("coder", "code", "codellama")):
        tags.append("coding")
    if any(k in lower for k in ("vision", "vl", "multimodal")):
        tags.append("vision")
    if any(k in lower for k in ("math", "r1", "reason")):
        tags.append("analysis")
    if "instruct" in lower or "chat" in lower:
        tags.append("analysis")
    if architecture.startswith("qwen"):
        tags.append("coding")
    return sorted(set(tags))


def _dict_to_model(data: dict[str, Any]) -> ModelInfo:
    variants = [
        GGUFVariant(
            filename=v["filename"],
            quant_type=v.get("quant_type", "Q4_K_M"),
            file_size_bytes=int(v.get("file_size_bytes", 0)),
        )
        for v in data.get("gguf_variants", [])
    ]
    model_id = data["id"]
    params = int(
        data.get("parameter_count") or _extract_size_hint(model_id) or 7_000_000_000
    )
    active = data.get("parameter_count_active")
    if active is None:
        active = _KNOWN_MOE_ACTIVE.get(model_id) or _extract_active_hint(model_id)
    arch = data.get("architecture") or model_id.split("/")[-1].split("-")[0].lower()
    return ModelInfo(
        id=model_id,
        family_id=data.get("family_id") or model_id.split("/")[0],
        name=data.get("name") or model_id.split("/")[-1],
        parameter_count=params,
        parameter_count_active=active,
        architecture=arch,
        is_moe=bool(data.get("is_moe") or (active and active < params)),
        context_length=data.get("context_length"),
        downloads=int(data.get("downloads", 0)),
        likes=int(data.get("likes", 0)),
        gguf_variants=variants,
        specialization_tags=data.get("specialization_tags")
        or _infer_tags(model_id, arch),
        instruction_following_score=float(
            data.get("instruction_following_score", 0.75)
        ),
        ollama_name=data.get("ollama_name"),
    )


def load_fallback_catalog() -> List[ModelInfo]:
    raw = json.loads(_FALLBACK_PATH.read_text(encoding="utf-8"))
    return [_dict_to_model(item) for item in raw]


def _parse_hf_model(item: dict[str, Any]) -> Optional[dict[str, Any]]:
    model_id = item.get("id") or item.get("modelId")
    if not model_id or not isinstance(model_id, str):
        return None
    if "gguf" not in model_id.lower() and "GGUF" not in model_id:
        tags = item.get("tags") or []
        if "gguf" not in [t.lower() for t in tags if isinstance(t, str)]:
            return None
    params = _extract_size_hint(model_id) or 7_000_000_000
    siblings = item.get("siblings") or []
    variants = []
    for s in siblings:
        fname = s.get("rfilename") or s.get("filename") or ""
        if not fname.lower().endswith(".gguf"):
            continue
        if "-of-" in fname.lower():
            continue
        quant = _extract_quant_type(fname)
        size = s.get("size")
        if not size:
            bpw = QUANT_BYTES_PER_WEIGHT.get(quant.upper(), 0.5625)
            size = int(params * bpw)
        variants.append(
            {"filename": fname, "quant_type": quant, "file_size_bytes": int(size)}
        )
    if not variants:
        return None
    arch = (item.get("config") or {}).get("model_type") or model_id.split("/")[-1][:8]
    return {
        "id": model_id,
        "name": model_id.split("/")[-1],
        "parameter_count": params,
        "parameter_count_active": _KNOWN_MOE_ACTIVE.get(model_id)
        or _extract_active_hint(model_id),
        "is_moe": bool(_extract_active_hint(model_id)),
        "context_length": 32768,
        "architecture": str(arch),
        "downloads": int(item.get("downloads") or 0),
        "likes": int(item.get("likes") or 0),
        "specialization_tags": _infer_tags(model_id, str(arch)),
        "instruction_following_score": 0.78,
        "gguf_variants": variants[:6],
    }


def _fetch_hf_page(params: dict[str, Any]) -> list[dict[str, Any]]:
    url = f"{HF_API_BASE}/models"
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data if isinstance(data, list) else []


def fetch_live_catalog(limit: int = 80) -> List[ModelInfo]:
    """Fetch GGUF models from HuggingFace API."""
    queries = [
        {"search": "GGUF", "sort": "downloads", "direction": -1, "limit": limit},
        {"filter": "gguf", "sort": "downloads", "direction": -1, "limit": limit // 2},
    ]
    seen: set[str] = set()
    parsed: list[dict[str, Any]] = []
    for params in queries:
        try:
            for item in _fetch_hf_page(params):
                model_id = item.get("id")
                if not model_id or model_id in seen:
                    continue
                detail_resp = requests.get(
                    f"{HF_API_BASE}/models/{model_id}", timeout=20
                )
                if detail_resp.status_code != 200:
                    continue
                detail = detail_resp.json()
                record = _parse_hf_model(detail)
                if record:
                    seen.add(model_id)
                    parsed.append(record)
        except Exception as exc:
            logger.warning("HF catalog fetch failed: %s", exc)
            break

    if not parsed:
        return load_fallback_catalog()
    return [_dict_to_model(r) for r in parsed]


def _model_to_dict(model: ModelInfo) -> dict[str, Any]:
    return {
        "id": model.id,
        "family_id": model.family_id,
        "name": model.name,
        "parameter_count": model.parameter_count,
        "parameter_count_active": model.parameter_count_active,
        "architecture": model.architecture,
        "is_moe": model.is_moe,
        "context_length": model.context_length,
        "downloads": model.downloads,
        "likes": model.likes,
        "specialization_tags": model.specialization_tags,
        "instruction_following_score": model.instruction_following_score,
        "ollama_name": model.ollama_name,
        "gguf_variants": [v.__dict__ for v in model.gguf_variants],
    }


def get_catalog(*, refresh: bool = False) -> tuple[List[ModelInfo], str]:
    """Return model catalog and source label."""
    if not refresh:
        cached = load_cache()
        if cached:
            return [_dict_to_model(m) for m in cached], "cache"

    try:
        models = fetch_live_catalog()
        if models:
            save_cache([_model_to_dict(m) for m in models])
            return models, "huggingface"
    except Exception as exc:
        logger.warning("Live catalog unavailable: %s", exc)

    return load_fallback_catalog(), "fallback"


def pick_best_variant(
    model: ModelInfo, preferred_quant: Optional[str] = None
) -> Optional[GGUFVariant]:
    if not model.gguf_variants:
        return None
    if preferred_quant:
        for v in model.gguf_variants:
            if v.quant_type.upper() == preferred_quant.upper():
                return v
    order = {q: i for i, q in enumerate(QUANT_PREFERENCE_ORDER)}
    return sorted(
        model.gguf_variants,
        key=lambda v: order.get(v.quant_type.upper(), 99),
    )[0]
