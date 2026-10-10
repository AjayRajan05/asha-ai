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

"""Compact character n-gram language model (stdlib-only).

Loads a small JSON table of ``{trigram: log_probability}`` plus a
smoothing constant for unseen n-grams. Provides whole-text and
windowed-max perplexity (the latter catches injections buried mid-prompt).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Union


class CharNgramLM:
    """Char n-gram LM with Laplace-style unseen log-prob fallback."""

    def __init__(
        self,
        log_probs: Mapping[str, float],
        *,
        n: int = 3,
        unseen_log_prob: float = -10.0,
        alphabet_hint: int = 64,
    ) -> None:
        if n < 2 or n > 6:
            raise ValueError("n must be in [2, 6]")
        self.n = n
        self.log_probs: Dict[str, float] = {
            str(k): float(v) for k, v in log_probs.items()
        }
        # Unseen: log(1 / (V^n)) style fallback; caller may override.
        self.unseen_log_prob = float(unseen_log_prob)
        self.alphabet_hint = int(alphabet_hint)

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "CharNgramLM":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "CharNgramLM":
        table = data.get("log_probs") or data.get("trigrams") or {}
        if not isinstance(table, dict):
            raise TypeError("log_probs must be a dict")
        log_probs = {str(key): float(value) for key, value in table.items()}
        n = int(str(data.get("n", 3)))
        unseen = float(str(data.get("unseen_log_prob", -10.0)))
        alphabet = int(str(data.get("alphabet_hint", 64)))
        return cls(log_probs, n=n, unseen_log_prob=unseen, alphabet_hint=alphabet)

    @classmethod
    def from_counts(
        cls,
        texts: Iterable[str],
        *,
        n: int = 3,
        top_k: int = 8000,
        add_k: float = 0.5,
    ) -> "CharNgramLM":
        """Build a compact LM from raw training strings (training helper)."""
        from collections import Counter

        counts: Counter[str] = Counter()
        for text in texts:
            padded = f" {' '.join((text or '').lower().split())} "
            for i in range(max(0, len(padded) - n + 1)):
                counts[padded[i : i + n]] += 1
        if not counts:
            return cls({}, n=n, unseen_log_prob=-10.0)
        most = counts.most_common(top_k)
        vocab = len(most)
        total = sum(c for _, c in most) + add_k * vocab
        log_probs = {gram: math.log((c + add_k) / total) for gram, c in most}
        unseen = math.log(add_k / (total + add_k))
        return cls(
            log_probs, n=n, unseen_log_prob=unseen, alphabet_hint=max(32, vocab // 10)
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "n": self.n,
            "unseen_log_prob": self.unseen_log_prob,
            "alphabet_hint": self.alphabet_hint,
            "log_probs": self.log_probs,
        }

    def _ngrams(self, text: str) -> List[str]:
        padded = f" {' '.join((text or '').lower().split())} "
        n = self.n
        if len(padded) < n:
            return [padded] if padded.strip() else []
        return [padded[i : i + n] for i in range(len(padded) - n + 1)]

    def average_nll(self, text: str) -> float:
        grams = self._ngrams(text)
        if not grams:
            return abs(self.unseen_log_prob)
        total = 0.0
        for g in grams:
            total += -self.log_probs.get(g, self.unseen_log_prob)
        return total / len(grams)

    def perplexity(self, text: str) -> float:
        """exp(average negative log-likelihood)."""
        nll = self.average_nll(text)
        # Clamp to avoid overflow on pathological inputs.
        nll = min(nll, 50.0)
        return math.exp(nll)

    def windowed_max_perplexity(self, text: str, window: int = 8) -> float:
        """Max perplexity over sliding windows of ``window`` n-grams.

        Catches high-surprise spans buried in otherwise benign text.
        """
        grams = self._ngrams(text)
        if not grams:
            return self.perplexity(text)
        if len(grams) <= window:
            return self.perplexity(text)
        best = 0.0
        for i in range(len(grams) - window + 1):
            chunk = grams[i : i + window]
            nll = (
                sum(-self.log_probs.get(g, self.unseen_log_prob) for g in chunk)
                / window
            )
            nll = min(nll, 50.0)
            best = max(best, math.exp(nll))
        return best
