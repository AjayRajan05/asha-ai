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

"""Perplexity + length features for injection detection (Signal A).

Uses KenLM when available; otherwise a pure-Python character/token n-gram
language model. Never calls a generative LLM.
"""

from __future__ import annotations

import math
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

_TOKEN_RE = re.compile(r"\S+")
_DEFAULT_WINDOW = 8

# Tiny English-ish reference corpus for the offline n-gram LM (benign prose).
_REFERENCE_CORPUS = (
    "summarize the quarterly sales report in three bullet points "
    "write a python function that checks if a string is a palindrome "
    "draft a polite email declining a meeting invite next tuesday "
    "explain how photosynthesis works to a ten year old "
    "what is the difference between rest and graphql apis "
    "translate good morning how are you into tamil "
    "help me plan a three day itinerary for chennai "
    "system requirements for the new staffing dashboard "
    "encode hello world in base64 for a config file "
    "the weather today is sunny with a chance of rain later "
)


@dataclass(frozen=True)
class PerplexityFeatures:
    """Feature vector for the perplexity-side classifier."""

    log_ppl: float
    token_length: int
    ppl_length_ratio: float
    max_window_log_ppl: float

    def as_vector(self) -> List[float]:
        return [
            self.log_ppl,
            float(self.token_length),
            self.ppl_length_ratio,
            self.max_window_log_ppl,
        ]


class CharNgramLM:
    """Simple character n-gram LM with add-k smoothing (KenLM fallback)."""

    def __init__(self, order: int = 3, k: float = 0.1) -> None:
        self.order = order
        self.k = k
        self._counts: Dict[str, Dict[str, int]] = {}
        self._context_totals: Dict[str, int] = {}
        self._vocab: set[str] = set()

    def fit(self, texts: Sequence[str]) -> "CharNgramLM":
        for text in texts:
            padded = ("^" * (self.order - 1)) + text.lower() + "$"
            for i in range(self.order - 1, len(padded)):
                ctx = padded[i - self.order + 1 : i]
                ch = padded[i]
                self._vocab.add(ch)
                bucket = self._counts.setdefault(ctx, {})
                bucket[ch] = bucket.get(ch, 0) + 1
                self._context_totals[ctx] = self._context_totals.get(ctx, 0) + 1
        return self

    def log_perplexity(self, text: str) -> float:
        if not text:
            return 0.0
        padded = ("^" * (self.order - 1)) + text.lower() + "$"
        total = 0.0
        n = 0
        v = max(len(self._vocab), 1)
        for i in range(self.order - 1, len(padded)):
            ctx = padded[i - self.order + 1 : i]
            ch = padded[i]
            count = self._counts.get(ctx, {}).get(ch, 0)
            ctx_total = self._context_totals.get(ctx, 0)
            prob = (count + self.k) / (ctx_total + self.k * v)
            total += -math.log(max(prob, 1e-12))
            n += 1
        if n == 0:
            return 0.0
        # Cross-entropy in nats → report as log-perplexity (avg NLL).
        return total / n


class KenLMWrapper:
    """Optional KenLM backend; raises ImportError if kenlm is unavailable."""

    def __init__(self, model_path: Union[str, Path]) -> None:
        import kenlm

        self._model = kenlm.Model(str(model_path))

    def log_perplexity(self, text: str) -> float:
        if not text.strip():
            return 0.0
        # kenlm returns log10 probability of the sentence.
        log10_prob = self._model.score(text, bos=True, eos=True)
        tokens = max(len(_TOKEN_RE.findall(text)), 1)
        # Convert to average NLL in nats for consistency with CharNgramLM.
        nll = -log10_prob * math.log(10) / tokens
        return float(nll)


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text)


class PerplexityScorer:
    """Compute perplexity features and optional calibrated GB classifier score."""

    def __init__(
        self,
        *,
        window_size: int = _DEFAULT_WINDOW,
        kenlm_path: Optional[Union[str, Path]] = None,
        classifier_path: Optional[Union[str, Path]] = None,
    ) -> None:
        self.window_size = window_size
        self._kenlm_path = Path(kenlm_path) if kenlm_path else None
        self._classifier_path = Path(classifier_path) if classifier_path else None
        self._lm: Any = None
        self._classifier: Any = None
        self._lock = threading.Lock()
        self._lm_ready = False
        self._clf_ready = False

    def _ensure_lm(self) -> None:
        if self._lm_ready:
            return
        with self._lock:
            if self._kenlm_path and self._kenlm_path.is_file():
                try:
                    self._lm = KenLMWrapper(self._kenlm_path)
                    self._lm_ready = True
                    return
                except ImportError:
                    pass
            self._lm = CharNgramLM(order=3).fit([_REFERENCE_CORPUS])
            self._lm_ready = True

    def _ensure_classifier(self) -> None:
        if self._clf_ready:
            return
        with self._lock:
            path = self._classifier_path or _default_model_path("perplexity_gb.joblib")
            if path is not None and path.is_file():
                try:
                    from asha.core.ml.calibration import load_calibrator

                    self._classifier = load_calibrator(path)
                except Exception:
                    self._classifier = None
            self._clf_ready = True

    def extract_features(self, text: str) -> PerplexityFeatures:
        self._ensure_lm()
        tokens = tokenize(text)
        token_length = len(tokens)
        log_ppl = float(self._lm.log_perplexity(text))
        max_window = log_ppl
        if token_length >= self.window_size:
            for i in range(0, token_length - self.window_size + 1):
                window = " ".join(tokens[i : i + self.window_size])
                max_window = max(max_window, float(self._lm.log_perplexity(window)))
        ratio = log_ppl / max(token_length, 1)
        return PerplexityFeatures(
            log_ppl=log_ppl,
            token_length=token_length,
            ppl_length_ratio=ratio,
            max_window_log_ppl=max_window,
        )

    def score_probability(self, text: str) -> Tuple[float, PerplexityFeatures]:
        """Return (calibrated P(injection), features).

        When no trained classifier is present, use a heuristic mapped into
        ``[0, 1]`` from windowed perplexity. Callers bucket the score via
        ``config/thresholds.yaml`` rather than hardcoded decision thresholds.
        """
        feats = self.extract_features(text)
        self._ensure_classifier()
        if self._classifier is not None:
            try:
                import numpy as np
                from asha.core.ml.calibration import calibrated_predict_proba

                X = np.asarray([feats.as_vector()], dtype=np.float64)
                proba = float(calibrated_predict_proba(self._classifier, X)[0])
                return max(0.0, min(1.0, proba)), feats
            except Exception:
                pass
        return _heuristic_ppl_probability(feats), feats


def _heuristic_ppl_probability(feats: PerplexityFeatures) -> float:
    """Soft mapping from features → probability without a trained model.

    Uses a logistic over windowed perplexity centered *above* typical
    English cross-entropy so ordinary prompts stay near 0. Decision bands
    still come from thresholds.yaml (fail-closed REVIEW in the middle).
    """
    z = (feats.max_window_log_ppl - 4.0) / 0.55
    logistic = 1.0 / (1.0 + math.exp(-z))
    return max(0.0, min(1.0, 0.9 * logistic))


def _default_model_path(filename: str) -> Optional[Path]:
    candidates = [
        Path.cwd() / "models" / "injection" / filename,
        Path(__file__).resolve().parents[4] / "models" / "injection" / filename,
        Path(__file__).resolve().parents[3] / "models" / "injection" / filename,
    ]
    try:
        from asha.core.ml.model_store import candidate_model_paths

        candidates = candidate_model_paths("injection", filename) + candidates
    except Exception:
        pass
    env = __import__("os").environ.get("ASHA_MODELS_DIR")
    if env:
        candidates.insert(0, Path(env) / "injection" / filename)
    for path in candidates:
        if path.is_file():
            return path
    return candidates[0] if candidates else None
