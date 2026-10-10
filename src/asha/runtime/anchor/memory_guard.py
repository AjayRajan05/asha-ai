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

"""Firewall for memory operations.

Protects runtime cognitive integrity against poisoning and secret leakage.

Detection stack (writes):
  1. Keyword fast-path: literal known-bad strings -> instant BLOCK
  2. Embedding poisoning classifier: paraphrases / multilingual / obfuscation
  3. Sensitive-keyword scan + Shannon-entropy secret scanner (parallel)

Decision bands come from ``config/thresholds.yaml`` (memory_poisoning /
secret_entropy). Uncertain mid-band scores route to REVIEW, not ALLOW.
"""

from __future__ import annotations

import uuid
import time
from typing import Any, List, Optional, Sequence

from .verdicts import MemoryVerdict, Verdict
from .types import MemoryEvent
from .contracts import MissionContract


def _hardened_poison_model_available() -> bool:
    from asha.core.ml.model_store import ensure_models, resolve_model_file

    ensure_models()
    return (
        resolve_model_file("memory_guard", "poisoning_rf.joblib", ensure=False)
        is not None
    )


def _embedding_stack_available() -> bool:
    """True when MiniLM deps import cleanly (not just present as broken modules)."""
    try:
        import torch  # noqa: F401
        import sentence_transformers  # noqa: F401
    except Exception:
        return False
    return True


class MemoryGuard:
    """Firewall for memory operations."""

    def __init__(
        self,
        *,
        poison_keywords: Optional[Sequence[str]] = None,
        sensitive_keywords: Optional[Sequence[str]] = None,
        allow_hash_fallback: bool = False,
    ) -> None:
        self.sensitive_keywords = list(
            sensitive_keywords
            or ("password", "secret", "token", "api_key", "credentials")
        )
        # Kept as fast-path only: not the sole poisoning signal.
        self.poison_keywords = list(
            poison_keywords
            or (
                "forget all previous instructions",
                "ignore previous",
                "you are now",
                "system prompt",
                "new instructions",
                "override agent memory",
                "override memory",
                "disable security checks",
                "disable security",
            )
        )
        self.allow_hash_fallback = allow_hash_fallback
        self._poisoning_clf: Optional[Any] = None
        self._last_poisoning_score: Optional[Any] = None
        self._last_secret_hits: List[object] = []

    def _classifier(self) -> Any:
        if self._poisoning_clf is None:
            from .poisoning_classifier import PoisoningClassifier

            # Prefer hardened MiniLM classifier when joblib exists; otherwise
            # still construct PoisoningClassifier (centroids); train artifacts
            # with ``python -m training.memory_guard.train`` for calibrated RF.
            # Explicit lite-only callers use LitePoisoningClassifier directly.
            if _hardened_poison_model_available() or _embedding_stack_available():
                self._poisoning_clf = PoisoningClassifier(
                    allow_hash_fallback=False,
                    poison_keywords=self.poison_keywords,
                )
            else:
                from .memory_guard_lite import (
                    LitePoisoningClassifier,
                    lite_artifact_available,
                )

                if not lite_artifact_available():
                    raise ImportError(
                        "Memory poisoning classifier requires asha[hardened] "
                        "(MiniLM + models/memory_guard/poisoning_rf.joblib) or "
                        "lite artifacts. Train with: "
                        "python -m training.memory_guard.train"
                    )
                self._poisoning_clf = LitePoisoningClassifier(
                    poison_keywords=self.poison_keywords,
                )
        return self._poisoning_clf

    def normalize_memory_event(
        self, operation: str, content: str, scope: str
    ) -> MemoryEvent:
        """Normalize a memory operation into a structured MemoryEvent."""
        return MemoryEvent(
            event_id=str(uuid.uuid4()),
            operation=operation,
            content_summary=content[:100] + "..." if len(content) > 100 else content,
            scope=scope,
            timestamp=time.time(),
        )

    def evaluate_memory(
        self, event: MemoryEvent, contract: MissionContract, full_content: str
    ) -> MemoryVerdict:
        """Evaluate memory operation against the contract and integrity rules."""
        from asha.core.ml.calibration import Verdict as MLVerdict
        from asha.core.ml.calibration import get_bands, load_thresholds
        from asha.core.ml.secret_entropy import scan_secrets

        verdict = Verdict.ALLOW
        reason = "Memory operation allowed."
        risk_score = 0.0

        content_lower = full_content.lower()

        # Check scope matching
        if (
            event.scope not in contract.allowed_memory_scopes
            and event.scope != "session"
        ):
            return MemoryVerdict(
                Verdict.BLOCK,
                f"Memory scope '{event.scope}' not permitted by mission contract.",
                1.0,
            )

        # High-risk writes
        if event.operation in ["write", "update", "insert"]:
            risk_score += 0.3

            poison = self._classifier().score(full_content)
            self._last_poisoning_score = poison

            if poison.verdict is MLVerdict.BLOCK or poison.keyword_hit:
                return MemoryVerdict(
                    Verdict.BLOCK,
                    (
                        "Detected potential memory poisoning or prompt injection"
                        + (
                            f" (keyword: {poison.matched_keyword})."
                            if poison.matched_keyword
                            else f" (p={poison.probability:.3f})."
                        )
                    ),
                    1.0,
                )

            if poison.verdict is MLVerdict.REVIEW:
                # Fail-closed: uncertain poisoning scores escalate to REVIEW,
                # never silent ALLOW.
                verdict = Verdict.REVIEW
                reason = (
                    "Possible memory poisoning: review required "
                    f"(p={poison.probability:.3f})."
                )
                risk_score = max(risk_score, poison.probability)

            # Untrusted content quarantine under LOW risk tolerance
            if contract.risk_tolerance == "LOW" and event.scope != "session":
                if verdict == Verdict.ALLOW:
                    verdict = Verdict.REVIEW
                    reason = "Persistent memory write requires review under LOW risk tolerance."
                    risk_score = max(risk_score, 0.6)

            # Secret leakage on write: keyword OR entropy/shape scanner
            keyword_sensitive = any(k in content_lower for k in self.sensitive_keywords)
            secret_hits = scan_secrets(full_content)
            self._last_secret_hits = list(secret_hits)
            max_secret_conf = secret_hits[0].confidence if secret_hits else 0.0

            try:
                secret_bands = get_bands("secret_entropy", thresholds=load_thresholds())
            except Exception:
                from asha.core.ml.calibration import ThresholdBands

                secret_bands = ThresholdBands(
                    safe_max=0.40, block_min=0.75, source="builtin"
                )

            if keyword_sensitive or max_secret_conf >= secret_bands.block_min:
                return MemoryVerdict(
                    Verdict.BLOCK,
                    "Attempted to write sensitive data to memory.",
                    1.0,
                )

            if max_secret_conf >= secret_bands.safe_max and verdict != Verdict.BLOCK:
                # Mid-band secret confidence → REVIEW (fail-closed).
                verdict = Verdict.REVIEW
                shape = secret_hits[0].shape if secret_hits else "unknown"
                reason = (
                    f"Possible secret in memory write (shape={shape}, "
                    f"p={max_secret_conf:.3f}): review required."
                )
                risk_score = max(risk_score, max_secret_conf)

        # Detect leakage on read via keyword (entropy less relevant on read)
        if any(k in content_lower for k in self.sensitive_keywords):
            if event.operation == "read":
                if verdict == Verdict.ALLOW:
                    verdict = Verdict.WARN
                    reason = "Reading sensitive data."
                risk_score = max(risk_score, 0.4)

        return MemoryVerdict(verdict, reason, min(risk_score, 1.0))
