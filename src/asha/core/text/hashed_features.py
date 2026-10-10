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

"""Hashing trick + tiny linear classifier (stdlib-only at runtime).

Feature hashing: Weinberger et al., ICML 2009.
Runtime loads flat weight arrays (JSON / array.array) produced by
``training/*/train_lite.py`` without importing sklearn/numpy at inference.
"""

from __future__ import annotations

import array
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union


def _murmurhash3_32(key: bytes, seed: int = 0) -> int:
    """MurmurHash3 x86_32 (public-domain algorithm), pure Python."""
    length = len(key)
    nblocks = length // 4
    h1 = seed & 0xFFFFFFFF
    c1 = 0xCC9E2D51
    c2 = 0x1B873593

    for block_start in range(0, nblocks * 4, 4):
        k1 = (
            key[block_start]
            | (key[block_start + 1] << 8)
            | (key[block_start + 2] << 16)
            | (key[block_start + 3] << 24)
        )
        k1 = (k1 * c1) & 0xFFFFFFFF
        k1 = ((k1 << 15) | (k1 >> 17)) & 0xFFFFFFFF
        k1 = (k1 * c2) & 0xFFFFFFFF
        h1 ^= k1
        h1 = ((h1 << 13) | (h1 >> 19)) & 0xFFFFFFFF
        h1 = (h1 * 5 + 0xE6546B64) & 0xFFFFFFFF

    tail_index = nblocks * 4
    k1 = 0
    tail_size = length & 3
    if tail_size >= 3:
        k1 ^= key[tail_index + 2] << 16
    if tail_size >= 2:
        k1 ^= key[tail_index + 1] << 8
    if tail_size >= 1:
        k1 ^= key[tail_index]
        k1 = (k1 * c1) & 0xFFFFFFFF
        k1 = ((k1 << 15) | (k1 >> 17)) & 0xFFFFFFFF
        k1 = (k1 * c2) & 0xFFFFFFFF
        h1 ^= k1

    h1 ^= length
    h1 ^= h1 >> 16
    h1 = (h1 * 0x85EBCA6B) & 0xFFFFFFFF
    h1 ^= h1 >> 13
    h1 = (h1 * 0xC2B2AE35) & 0xFFFFFFFF
    h1 ^= h1 >> 16
    return h1


def _char_ngrams(text: str, ngram_range: Tuple[int, int]) -> Iterable[str]:
    lo, hi = ngram_range
    # Pad lightly so short strings still produce features.
    padded = f" {text.strip().lower()} "
    n = len(padded)
    for size in range(lo, hi + 1):
        if n < size:
            continue
        for i in range(n - size + 1):
            yield padded[i : i + size]


def hash_features(
    text: str,
    n_bits: int = 16,
    ngram_range: Tuple[int, int] = (3, 5),
    *,
    seed: int = 0,
) -> Dict[int, float]:
    """Map text to a sparse hashed feature vector of size ``2**n_bits``.

    Uses the sign-hash trick (ξ ∈ {-1,+1}) to reduce collision bias.
    """
    if n_bits < 8 or n_bits > 24:
        raise ValueError("n_bits must be in [8, 24]")
    dim = 1 << n_bits
    mask = dim - 1
    feats: Dict[int, float] = {}
    for gram in _char_ngrams(text or "", ngram_range):
        raw = gram.encode("utf-8", errors="replace")
        h = _murmurhash3_32(raw, seed=seed)
        idx = h & mask
        # Sign-hash trick (Weinberger): ξ ∈ {-1,+1} from high bit.
        sign = 1.0 if (h >> 31) == 0 else -1.0
        feats[idx] = feats.get(idx, 0.0) + sign
    return feats


def _sigmoid(x: float) -> float:
    # Numerically stable sigmoid.
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


class HashedLinearClassifier:
    """Linear model over hashed features: score = bias + w·x, then sigmoid."""

    def __init__(
        self,
        weights: Sequence[float],
        bias: float = 0.0,
        *,
        n_bits: Optional[int] = None,
        ngram_range: Tuple[int, int] = (3, 5),
        calibration: Optional[Sequence[Tuple[float, float]]] = None,
    ) -> None:
        self.weights = array.array("f", weights)
        self.bias = float(bias)
        self.n_bits = (
            n_bits if n_bits is not None else int(math.log2(len(self.weights)))
        )
        if (1 << self.n_bits) != len(self.weights):
            raise ValueError(
                f"weights length {len(self.weights)} is not 2**n_bits "
                f"(n_bits={self.n_bits})"
            )
        self.ngram_range = ngram_range
        # Sorted (raw_score_or_prob, calibrated_prob) breakpoints for isotonic-
        # style interpolation. Optional.
        self.calibration: List[Tuple[float, float]] = (
            [(float(a), float(b)) for a, b in calibration] if calibration else []
        )

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "HashedLinearClassifier":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "HashedLinearClassifier":
        import base64

        if "weights_b64" in data:
            raw = base64.b64decode(str(data["weights_b64"]))
            weights_arr = array.array("f")
            weights_arr.frombytes(raw)
            weight_list: Sequence[float] = weights_arr
        else:
            weights_obj = data.get("weights")
            if not isinstance(weights_obj, list):
                raise TypeError("weights must be a list of floats or weights_b64")
            weight_list = [float(value) for value in weights_obj]
        bias = float(str(data.get("bias", 0.0)))
        n_bits = data.get("n_bits")
        ngram = data.get("ngram_range", [3, 5])
        if not isinstance(ngram, (list, tuple)) or len(ngram) != 2:
            ngram_range = (3, 5)
        else:
            ngram_range = (int(ngram[0]), int(ngram[1]))
        calib_raw = data.get("calibration") or []
        calibration: List[Tuple[float, float]] = []
        if isinstance(calib_raw, list):
            for pair in calib_raw:
                if isinstance(pair, (list, tuple)) and len(pair) == 2:
                    calibration.append((float(pair[0]), float(pair[1])))
        return cls(
            weight_list,
            bias,
            n_bits=int(str(n_bits)) if n_bits is not None else None,
            ngram_range=ngram_range,
            calibration=calibration,
        )

    def to_dict(self) -> Dict[str, object]:
        import base64

        # Compact float32 payload keeps OvR artifacts in the 50-500 KB band.
        payload = base64.b64encode(self.weights.tobytes()).decode("ascii")
        return {
            "weights_b64": payload,
            "weights_dtype": "f32",
            "bias": round(self.bias, 6),
            "n_bits": self.n_bits,
            "ngram_range": list(self.ngram_range),
            "calibration": [[round(a, 6), round(b, 6)] for a, b in self.calibration],
        }

    def decision_function(self, text: str) -> float:
        feats = hash_features(text, n_bits=self.n_bits, ngram_range=self.ngram_range)
        score = self.bias
        w = self.weights
        for idx, val in feats.items():
            score += w[idx] * val
        return float(score)

    def _calibrate(self, probability: float) -> float:
        if not self.calibration:
            return probability
        pts = self.calibration
        if probability <= pts[0][0]:
            return pts[0][1]
        if probability >= pts[-1][0]:
            return pts[-1][1]
        # Linear interpolation via bisect
        import bisect

        xs = [p[0] for p in pts]
        i = bisect.bisect_right(xs, probability)
        x0, y0 = pts[i - 1]
        x1, y1 = pts[i]
        if x1 == x0:
            return y0
        t = (probability - x0) / (x1 - x0)
        return y0 + t * (y1 - y0)

    def predict_proba(self, text: str) -> float:
        """P(positive class) in [0, 1], optionally isotonic-calibrated."""
        raw = _sigmoid(self.decision_function(text))
        return float(self._calibrate(raw))


class HashedOvRClassifier:
    """One-vs-rest hashed linear classifiers for multi-class lite inference."""

    def __init__(
        self,
        labels: Sequence[str],
        models: Mapping[str, HashedLinearClassifier],
        *,
        n_bits: Optional[int] = None,
        ngram_range: Tuple[int, int] = (3, 5),
    ) -> None:
        self.labels = list(labels)
        self.models = dict(models)
        if not self.labels:
            raise ValueError("labels must be non-empty")
        sample = next(iter(self.models.values()), None)
        self.n_bits = (
            n_bits if n_bits is not None else (sample.n_bits if sample else 14)
        )
        self.ngram_range = ngram_range

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "HashedOvRClassifier":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "HashedOvRClassifier":
        labels_raw = data.get("labels") or []
        if not isinstance(labels_raw, list):
            raise TypeError("labels must be a list")
        labels = [str(x) for x in labels_raw]
        ngram = data.get("ngram_range", [3, 5])
        if isinstance(ngram, (list, tuple)) and len(ngram) == 2:
            ngram_range = (int(ngram[0]), int(ngram[1]))
        else:
            ngram_range = (3, 5)
        n_bits = data.get("n_bits")
        models_raw = data.get("models") or {}
        if not isinstance(models_raw, dict):
            raise TypeError("models must be a dict")
        models: Dict[str, HashedLinearClassifier] = {}
        for label, payload in models_raw.items():
            if isinstance(payload, Mapping):
                models[str(label)] = HashedLinearClassifier.from_dict(payload)
        return cls(
            labels,
            models,
            n_bits=int(str(n_bits)) if n_bits is not None else None,
            ngram_range=ngram_range,
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "labels": list(self.labels),
            "n_bits": self.n_bits,
            "ngram_range": list(self.ngram_range),
            "models": {k: v.to_dict() for k, v in self.models.items()},
        }

    def predict_proba_all(self, text: str) -> Dict[str, float]:
        scores: Dict[str, float] = {}
        for label in self.labels:
            model = self.models.get(label)
            if model is None:
                scores[label] = 0.0
            else:
                scores[label] = model.predict_proba(text)
        return scores

    def predict(self, text: str) -> Tuple[str, float]:
        scores = self.predict_proba_all(text)
        if not scores:
            return "unknown", 0.0
        best_label = max(scores, key=scores.get)  # type: ignore[arg-type]
        return best_label, float(scores[best_label])
