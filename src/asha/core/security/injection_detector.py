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

"""Two-signal injection ensemble with calibrated SAFE / REVIEW / BLOCK bands.

Fusion strategy
---------------
Signal A (perplexity+length), Signal B (embedding classifier), and a cheap
regex-hit feature are combined with a **noisy-OR**:

    fused = 1 - (1 - p_ppl) * (1 - p_emb) * (1 - 0.5 * regex)

Why noisy-OR (not a pure mean)? A buried mid-prompt injection can produce a
strong windowed-perplexity *or* embedding hit while the other signal stays
moderate; averaging alone would dilute it below the block band. Noisy-OR
lets either independent detector drive the fused probability up, while still
rewarding agreement. Regex contributes a discounted boost (never the sole
gate: a lone regex hit without model support yields at most 0.5 before
banding).

When a trained meta-logistic artifact exists at
``models/injection/fusion_meta.joblib``, that replaces the blend above.

Decision bands always come from ``config/thresholds.yaml`` (injection.*),
never from inline magic numbers.
"""

from __future__ import annotations

import os
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from asha.core.ml.calibration import (
    ThresholdBands,
    Verdict,
    bucket_probability,
    get_bands,
    load_thresholds,
)

import logging
import warnings

logger = logging.getLogger(__name__)

# Shown whenever the ensemble path cannot run; keep the literal ``asha[hardened]``
# so install guidance cannot silently drift back to vague "transformers" wording.
HARDENED_INJECTION_FALLBACK_MSG = (
    "Hardened injection detection unavailable "
    "(sentence-transformers/scikit-learn not installed, or detector "
    "models could not be auto-downloaded); using lite path. "
    "Install with: pip install asha[hardened]. "
    "Models download automatically on first use unless "
    "ASHA_DISABLE_MODEL_DOWNLOAD=1."
)

_hardened_injection_warned = False


def warn_hardened_injection_unavailable(*, force: bool = False) -> str:
    """Emit the canonical hardened→lite warning once per process (unless force)."""
    global _hardened_injection_warned
    if force or not _hardened_injection_warned:
        _hardened_injection_warned = True
        warnings.warn(HARDENED_INJECTION_FALLBACK_MSG, UserWarning, stacklevel=2)
        logger.warning(HARDENED_INJECTION_FALLBACK_MSG)
    return HARDENED_INJECTION_FALLBACK_MSG


@dataclass(frozen=True)
class InjectionDetectionResult:
    """Calibrated injection detection outcome."""

    probability: float
    verdict: Verdict
    p_perplexity: float
    p_embedding: float
    regex_hit: bool
    regex_patterns_matched: Tuple[str, ...] = ()
    features: Dict[str, float] = field(default_factory=dict)
    mode: str = "lite"

    @property
    def is_block(self) -> bool:
        return self.verdict is Verdict.BLOCK

    @property
    def is_review(self) -> bool:
        return self.verdict is Verdict.REVIEW

    @property
    def is_safe(self) -> bool:
        return self.verdict is Verdict.SAFE


class InjectionDetector:
    """Injection detector (Base default: lite; opt into ensemble when hardened deps exist)."""

    def __init__(
        self,
        *,
        mode: str = "lite",
        patterns: Optional[Sequence[Dict[str, Any]]] = None,
        allow_hash_fallback: bool = False,
    ) -> None:
        """
        Args:
            mode: ``lite`` (Base default), ``ensemble`` (asha[hardened]), or
                ``regex_only`` (backward-compat opt-in).
            patterns: optional injection regex bank (from SecurityLayer).
            allow_hash_fallback: test-only; production must install asha[hardened].
                Hash vectors are not semantic; never enable silently in prod.
        """
        if mode not in ("ensemble", "regex_only", "lite"):
            raise ValueError(f"Unsupported injection mode: {mode}")
        self.mode = mode
        self._patterns = list(patterns or [])
        self.allow_hash_fallback = allow_hash_fallback
        self._ppl: Optional[Any] = None
        self._emb: Optional[Any] = None
        self._meta: Optional[Any] = None
        self._meta_ready = False
        self._lite: Optional[Any] = None
        self._lock = threading.Lock()

    def _perplexity(self) -> Any:
        if self._ppl is None:
            from .perplexity_scorer import PerplexityScorer

            self._ppl = PerplexityScorer()
        return self._ppl

    def _embedding(self) -> Any:
        if self._emb is None:
            from .embedding_classifier import EmbeddingClassifier

            self._emb = EmbeddingClassifier(
                allow_hash_fallback=self.allow_hash_fallback
            )
        return self._emb

    def _ensure_meta(self) -> None:
        if self._meta_ready:
            return
        with self._lock:
            from asha.core.ml.model_store import resolve_model_file

            path = resolve_model_file("injection", "fusion_meta.joblib", ensure=True)
            if path is not None:
                try:
                    from asha.core.ml.calibration import load_calibrator

                    self._meta = load_calibrator(path)
                except Exception:
                    self._meta = None
            self._meta_ready = True

    def regex_feature(self, text: str) -> Tuple[float, Tuple[str, ...]]:
        """Return (0/1 hit, matched pattern descriptions). Not a sole gate."""
        from asha.core.text.canonicalize import canonicalize

        canon = canonicalize(text)
        views = (text, canon) if canon != text else (text,)

        if not self._patterns:
            # Minimal built-in bank if caller did not supply patterns.
            bank = _builtin_regex_bank()
        else:
            bank = self._patterns
        matched: List[str] = []
        for info in bank:
            pattern = info.get("pattern")
            if not isinstance(pattern, str):
                continue
            try:
                for v in views:
                    if re.search(pattern, v, re.IGNORECASE):
                        desc = str(info.get("description") or pattern[:48])
                        if desc not in matched:
                            matched.append(desc)
                        break
            except re.error:
                continue
        return (1.0 if matched else 0.0, tuple(matched))

    def _lite_detector(self) -> Any:
        if self._lite is None:
            from .injection_lite import LiteInjectionDetector

            self._lite = LiteInjectionDetector()
        return self._lite

    def detect(self, text: str) -> InjectionDetectionResult:
        if self.mode == "lite":
            result = self._lite_detector().detect(text)
            if isinstance(result, InjectionDetectionResult):
                return result
            raise TypeError("Lite injection detector returned an invalid result.")

        bands = _injection_bands()

        if self.mode == "regex_only":
            hit, matched = self.regex_feature(text)
            # Legacy behavior approximation: regex hit → BLOCK, else SAFE.
            # Still routed through bands so thresholds stay centralized.
            probability = 0.95 if hit else 0.05
            verdict = bucket_probability(probability, bands, fail_closed=True)
            # Preserve old "miss = allow" for explicit regex_only opt-in:
            if not hit:
                verdict = Verdict.SAFE
                probability = 0.05
            else:
                verdict = Verdict.BLOCK
                probability = max(probability, bands.block_min)
            return InjectionDetectionResult(
                probability=probability,
                verdict=verdict,
                p_perplexity=0.0,
                p_embedding=0.0,
                regex_hit=bool(hit),
                regex_patterns_matched=matched,
                features={"regex": hit},
                mode=self.mode,
            )

        p_ppl, feats = self._perplexity().score_probability(text)
        p_emb = self._embedding().score_probability(text)
        regex_hit, matched = self.regex_feature(text)
        probability = self._fuse(p_ppl, p_emb, regex_hit)
        verdict = bucket_probability(probability, bands, fail_closed=True)

        return InjectionDetectionResult(
            probability=probability,
            verdict=verdict,
            p_perplexity=p_ppl,
            p_embedding=p_emb,
            regex_hit=bool(regex_hit),
            regex_patterns_matched=matched,
            features={
                "log_ppl": feats.log_ppl,
                "token_length": float(feats.token_length),
                "ppl_length_ratio": feats.ppl_length_ratio,
                "max_window_log_ppl": feats.max_window_log_ppl,
                "regex": float(regex_hit),
            },
            mode=self.mode,
        )

    def _fuse(self, p_ppl: float, p_emb: float, regex_hit: float) -> float:
        self._ensure_meta()
        if self._meta is not None:
            try:
                import numpy as np
                from asha.core.ml.calibration import calibrated_predict_proba

                X = np.asarray([[p_ppl, p_emb, float(regex_hit)]], dtype=np.float64)
                return max(
                    0.0, min(1.0, float(calibrated_predict_proba(self._meta, X)[0]))
                )
            except Exception:
                pass
        # Documented noisy-OR fusion (see module docstring).
        # Regex is discounted (0.5×) so it cannot sole-gate a BLOCK.
        fused = 1.0 - (1.0 - p_ppl) * (1.0 - p_emb) * (1.0 - 0.5 * float(regex_hit))
        return max(0.0, min(1.0, fused))


_detector_singleton: Optional[InjectionDetector] = None
_singleton_lock = threading.Lock()


def _injection_artifact_available() -> bool:
    """True when at least one hardened injection joblib artifact is on disk."""
    from asha.core.ml.model_store import ensure_models, resolve_model_file

    ensure_models()
    return (
        resolve_model_file("injection", "fusion_meta.joblib", ensure=False) is not None
        or resolve_model_file("injection", "embedding_rf.joblib", ensure=False)
        is not None
    )


def hardened_injection_available() -> bool:
    """True when asha[hardened] deps import, MiniLM loads, and artifacts exist.

    Artifacts are auto-downloaded into the user cache on first probe unless
    ``ASHA_DISABLE_MODEL_DOWNLOAD=1``.
    """
    global _hardened_probe_cache
    if _hardened_probe_cache is True:
        return True
    try:
        import sklearn  # noqa: F401
        import joblib  # noqa: F401
        import torch  # noqa: F401
        import sentence_transformers  # noqa: F401
        from asha.core.ml.embeddings import minilm_loadable

        if not minilm_loadable():
            # Do not permanently cache False: MiniLM may load later in-session.
            return False
    except Exception:
        return False
    ok = _injection_artifact_available()
    if ok:
        _hardened_probe_cache = True
    return ok


_hardened_probe_cache: Optional[bool] = None


def resolve_injection_mode(explicit: Optional[str] = None) -> str:
    """Resolve injection mode with a uniform auto-upgrade policy.

    Priority:
      1. Explicit argument (non-empty)
      2. ``ASHA_INJECTION_MODE`` env
      3. ``"ensemble"`` when hardened deps + artifacts are available
      4. ``"lite"`` (Base default)

    Explicit ``"lite"`` / ``"ensemble"`` / ``"regex_only"`` always wins over auto-upgrade.
    """
    requested_ensemble = False
    if explicit is not None and str(explicit).strip():
        mode = str(explicit).strip().lower()
        requested_ensemble = mode == "ensemble"
    else:
        env = os.environ.get("ASHA_INJECTION_MODE")
        if env is not None and str(env).strip():
            mode = str(env).strip().lower()
            requested_ensemble = mode == "ensemble"
        elif hardened_injection_available():
            mode = "ensemble"
        else:
            mode = "lite"
    if mode not in ("ensemble", "regex_only", "lite"):
        raise ValueError(f"Unsupported injection mode: {mode}")
    if requested_ensemble and not hardened_injection_available():
        warn_hardened_injection_unavailable()
    return mode


def get_injection_detector(
    *,
    mode: Optional[str] = None,
    patterns: Optional[Sequence[Dict[str, Any]]] = None,
    reset: bool = False,
) -> InjectionDetector:
    """Process-wide detector; mode from :func:`resolve_injection_mode` when unset."""
    global _detector_singleton
    resolved_mode = resolve_injection_mode(mode)
    if (
        reset
        or _detector_singleton is None
        or (_detector_singleton.mode != resolved_mode)
    ):
        with _singleton_lock:
            _detector_singleton = InjectionDetector(
                mode=resolved_mode,
                patterns=patterns,
            )
    elif patterns is not None:
        _detector_singleton._patterns = list(patterns)
    return _detector_singleton


def detect_injection(
    text: str,
    *,
    mode: Optional[str] = None,
    patterns: Optional[Sequence[Dict[str, Any]]] = None,
) -> InjectionDetectionResult:
    """Convenience wrapper around :class:`InjectionDetector`."""
    return get_injection_detector(mode=mode, patterns=patterns).detect(text)


def _injection_bands() -> ThresholdBands:
    try:
        return get_bands("injection", thresholds=load_thresholds())
    except Exception:
        return ThresholdBands(safe_max=0.15, block_min=0.85, source="builtin")


def _builtin_regex_bank() -> List[Dict[str, Any]]:
    return [
        {
            "pattern": r"(?i)(ignore|forget|disregard)\s+((?:all|any|the)\s+)?(previous|all|above|earlier|prior)\s+(instructions|prompts|commands|directives)",
            "description": "Direct instruction override attempt",
        },
        {
            "pattern": r"(?i)(you\s+are\s+now|act\s+as|pretend\s+to\s+be).*?(jailbreak|uncensored|unrestricted|DAN)",
            "description": "Jailbreak attempt",
        },
        {
            "pattern": r"(?i)(bypass|override|circumvent)\s+(security|restrictions|limitations|filters)",
            "description": "Security bypass attempt",
        },
        {
            "pattern": r"(?i)(drop\s+table|union\s+select|;\s*delete)",
            "description": "SQL injection pattern",
        },
        {
            "pattern": r"(?i)reveal\s+(your\s+)?(system|root|hidden)\s+(prompt|message|instructions)",
            "description": "System prompt disclosure attempt",
        },
    ]
