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

"""Build WorkloadProfile from prompt corpus via IR + compiled token metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Sequence

from ...core.engines import compile_prompt, optimize_tokens
from ...core._ir.ir_builder import IRBuilder
from ...core._ir.prompt_ir import ConstraintType, IntentType, PrivacyLevel
from .types import WorkloadProfile

_INTENT_SPECIALIZATION = {
    IntentType.CODE: "coding",
    IntentType.DEBUG: "coding",
    IntentType.ANALYZE: "analysis",
    IntentType.SUMMARIZE: "analysis",
    IntentType.GENERATE: "general",
    IntentType.CREATE: "general",
    IntentType.TRANSLATE: "general",
    IntentType.CLASSIFY: "general",
    IntentType.EXTRACT: "analysis",
    IntentType.COMPARE: "analysis",
    IntentType.EXPLAIN: "general",
    IntentType.MODIFY: "general",
    IntentType.VALIDATE: "analysis",
    IntentType.SEARCH: "general",
    IntentType.OPTIMIZE: "coding",
}


def _count_tokens(text: str) -> int:
    try:
        import tiktoken

        return len(tiktoken.get_encoding("cl100k_base").encode(text or ""))
    except Exception:
        return len((text or "").split())


def _percentile(values: Sequence[int], pct: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(len(ordered) * pct) - 1))
    return ordered[idx]


def load_prompts(
    prompts: Sequence[str] | None = None,
    prompts_file: str | Path | None = None,
) -> List[str]:
    """Load prompts from list or JSON/JSONL file."""
    if prompts:
        return [p for p in prompts if p and p.strip()]

    if not prompts_file:
        raise ValueError("Provide prompts or prompts_file")

    path = Path(prompts_file)
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        loaded: List[str] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            loaded.append(obj.get("prompt") or obj.get("text") or str(obj))
        return loaded

    data = json.loads(text)
    if isinstance(data, list):
        result: List[str] = []
        for item in data:
            if isinstance(item, str):
                result.append(item)
            elif isinstance(item, dict):
                result.append(item.get("prompt") or item.get("text") or "")
        return [p for p in result if p]
    if isinstance(data, dict) and "prompts" in data:
        return [str(p) for p in data["prompts"] if p]
    raise ValueError(f"Unsupported prompts file format: {path}")


def profile_workload(
    prompts: Iterable[str],
    mode: str = "balanced",
    *,
    max_samples: int = 50,
) -> WorkloadProfile:
    """Analyze prompts through ASHA pipeline and aggregate workload stats."""
    builder = IRBuilder()
    prompt_list = list(prompts)[:max_samples]
    if not prompt_list:
        raise ValueError("At least one prompt is required")

    intent_counts: dict[str, int] = {}
    compiled_tokens: List[int] = []
    complexities: List[float] = []
    pii_hits = 0
    dominant_intents: List[IntentType] = []
    specialization: set[str] = set()
    high_privacy = 0
    compiled_samples: List[str] = []

    security_level = {"balanced": "medium", "strict": "high", "lite": "low"}.get(
        mode, "medium"
    )

    for prompt in prompt_list:
        ir = builder.parse(prompt)
        intent_key = ir.intent.value
        intent_counts[intent_key] = intent_counts.get(intent_key, 0) + 1
        dominant_intents.append(ir.intent)

        if ir.complexity_score is not None:
            complexities.append(float(ir.complexity_score))

        if ir.privacy in (PrivacyLevel.CONFIDENTIAL, PrivacyLevel.RESTRICTED):
            high_privacy += 1

        tag = _INTENT_SPECIALIZATION.get(ir.intent, "general")
        specialization.add(tag)

        if ConstraintType.COMPLIANCE in ir.constraints:
            specialization.add("compliance")
        if ConstraintType.PRIVACY in ir.constraints:
            specialization.add("compliance")

        compiled = compile_prompt(prompt)
        opt_result = optimize_tokens(compiled)
        optimized = opt_result.output or prompt
        compiled_samples.append(optimized)
        compiled_tokens.append(_count_tokens(optimized))

        if "@" in prompt and "@" not in optimized:
            pii_hits += 1

    total = len(prompt_list)
    intent_distribution = {k: v / total for k, v in intent_counts.items()}
    pii_rate = pii_hits / total
    requires_local = pii_rate > 0 or high_privacy > 0 or mode == "strict"

    avg_tokens = int(sum(compiled_tokens) / len(compiled_tokens))
    p95_tokens = _percentile(compiled_tokens, 0.95)
    max_tokens = max(compiled_tokens)
    context_length = max(2048, int(p95_tokens * 1.2))

    avg_complexity = sum(complexities) / len(complexities) if complexities else 0.5

    intent_rank = sorted(intent_counts.items(), key=lambda x: x[1], reverse=True)
    top_intents: List[IntentType] = []
    for key, _ in intent_rank[:3]:
        try:
            top_intents.append(IntentType(key))
        except ValueError:
            continue

    return WorkloadProfile(
        intent_distribution=intent_distribution,
        avg_compiled_tokens=avg_tokens,
        p95_compiled_tokens=p95_tokens,
        max_compiled_tokens=max_tokens,
        avg_complexity=avg_complexity,
        pii_rate=pii_rate,
        requires_local=requires_local,
        dominant_intents=top_intents or dominant_intents[:3],
        context_length_required=context_length,
        specialization_tags=sorted(specialization) or ["general"],
        sample_compiled_prompts=compiled_samples[:5],
        mode=mode,
    )
