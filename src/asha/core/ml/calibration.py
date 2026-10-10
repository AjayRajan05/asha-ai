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

"""Calibration helpers and threshold-band loading.

Thresholds live in ``config/thresholds.yaml`` (or the packaged copy under
``asha.config``). Call sites must not hardcode magic cutoffs; load bands
from config and bucket calibrated probabilities instead.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

# ---------------------------------------------------------------------------
# Verdicts / bands
# ---------------------------------------------------------------------------


class Verdict(str, Enum):
    """Fail-closed decision buckets for calibrated detectors."""

    SAFE = "SAFE"
    REVIEW = "REVIEW"
    BLOCK = "BLOCK"
    # Alignment evaluator also uses WARN between SAFE and REVIEW.
    WARN = "WARN"
    ALLOW = "ALLOW"  # alias of SAFE for evaluator-facing APIs


@dataclass(frozen=True)
class ThresholdBands:
    """Calibrated probability boundaries for a detector.

    Interpretation for a three-band detector (injection, memory, …):
      p < safe_max            → SAFE / ALLOW
      safe_max ≤ p < block_min → REVIEW (fail-closed middle band)
      p ≥ block_min           → BLOCK

    For four-band alignment scoring, optionally set ``warn_max``.

    For rarity / transition probabilities (``higher_is_safer=True``):
      p ≥ safe_max            → SAFE
      block_min ≤ p < safe_max → REVIEW
      p < block_min           → BLOCK
    """

    safe_max: float
    block_min: float
    warn_max: Optional[float] = None
    target_fpr: Optional[float] = None
    source: str = "config"
    higher_is_safer: bool = False

    def __post_init__(self) -> None:
        if not (0.0 <= self.safe_max <= 1.0):
            raise ValueError(f"safe_max out of range: {self.safe_max}")
        if not (0.0 <= self.block_min <= 1.0):
            raise ValueError(f"block_min out of range: {self.block_min}")
        if self.higher_is_safer:
            if self.safe_max < self.block_min:
                raise ValueError(
                    f"higher_is_safer requires safe_max ({self.safe_max}) "
                    f"≥ block_min ({self.block_min})"
                )
        elif self.safe_max > self.block_min:
            raise ValueError(
                f"safe_max ({self.safe_max}) must be ≤ block_min ({self.block_min})"
            )
        if self.warn_max is not None and not self.higher_is_safer:
            if not (self.safe_max <= self.warn_max <= self.block_min):
                raise ValueError(
                    "warn_max must satisfy safe_max ≤ warn_max ≤ block_min"
                )


def bucket_probability(
    probability: float,
    bands: ThresholdBands,
    *,
    fail_closed: bool = True,
) -> Verdict:
    """Map a calibrated probability into SAFE/WARN/REVIEW/BLOCK.

    Uncertain / NaN inputs route to REVIEW when ``fail_closed`` is True
    (default), never to SAFE/ALLOW.
    """
    try:
        p = float(probability)
    except (TypeError, ValueError):
        return Verdict.REVIEW if fail_closed else Verdict.SAFE

    if p != p or p < 0.0 or p > 1.0:  # NaN or out of range
        return Verdict.REVIEW if fail_closed else Verdict.SAFE

    if p >= bands.block_min:
        return Verdict.BLOCK
    if bands.warn_max is not None and p >= bands.warn_max:
        return Verdict.REVIEW
    if p >= bands.safe_max:
        if bands.warn_max is not None:
            return Verdict.WARN
        return Verdict.REVIEW
    return Verdict.SAFE


# ---------------------------------------------------------------------------
# Threshold config loading
# ---------------------------------------------------------------------------

_THRESHOLDS_CACHE: Optional[Dict[str, Any]] = None


def _candidate_threshold_paths() -> List[Path]:
    env = os.environ.get("ASHA_THRESHOLDS_PATH")
    paths: List[Path] = []
    if env:
        paths.append(Path(env))
    # Repo-root / cwd config (dev + training scripts).
    paths.append(Path.cwd() / "config" / "thresholds.yaml")
    # Walk up from this file to find a repo-level config/.
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "config" / "thresholds.yaml"
        if candidate not in paths:
            paths.append(candidate)
    # Packaged default shipped with the wheel.
    pkg_cfg = (
        Path(__file__).resolve().parent.parent.parent / "config" / "thresholds.yaml"
    )
    if pkg_cfg not in paths:
        paths.append(pkg_cfg)
    return paths


def load_thresholds(
    path: Optional[Union[str, Path]] = None,
    *,
    reload: bool = False,
) -> Dict[str, Any]:
    """Load the centralized thresholds YAML.

    Resolution order when ``path`` is omitted:
      1. ``ASHA_THRESHOLDS_PATH`` env var
      2. ``./config/thresholds.yaml``
      3. nearest ancestor ``config/thresholds.yaml``
      4. packaged ``asha/config/thresholds.yaml``
    """
    global _THRESHOLDS_CACHE
    if _THRESHOLDS_CACHE is not None and not reload and path is None:
        return _THRESHOLDS_CACHE

    if path is not None:
        resolved = Path(path)
        data = _read_yaml(resolved)
        return data

    for candidate in _candidate_threshold_paths():
        if candidate.is_file():
            data = _read_yaml(candidate)
            _THRESHOLDS_CACHE = data
            return data

    # Absolute last resort: built-in fail-closed defaults (no magic production
    # cutoffs: these only apply when the YAML file is missing entirely).
    data = _builtin_fail_closed_defaults()
    _THRESHOLDS_CACHE = data
    return data


def get_bands(
    detector_key: str,
    *,
    thresholds: Optional[Mapping[str, Any]] = None,
) -> ThresholdBands:
    """Return :class:`ThresholdBands` for a named detector section."""
    cfg = dict(thresholds) if thresholds is not None else load_thresholds()
    section = cfg.get(detector_key)
    if not isinstance(section, Mapping):
        raise KeyError(
            f"No threshold section '{detector_key}' in thresholds config. "
            f"Known keys: {sorted(k for k in cfg if not k.startswith('_'))}"
        )
    return ThresholdBands(
        safe_max=float(section["safe_max"]),
        block_min=float(section["block_min"]),
        warn_max=(
            float(section["warn_max"]) if section.get("warn_max") is not None else None
        ),
        target_fpr=(
            float(section["target_fpr"])
            if section.get("target_fpr") is not None
            else None
        ),
        source=str(section.get("source", "config")),
        higher_is_safer=bool(section.get("higher_is_safer", False)),
    )


def _read_yaml(path: Path) -> Dict[str, Any]:
    try:
        import importlib

        yaml: Any = importlib.import_module("yaml")
    except ImportError:
        # Minimal subset parser for our flat YAML (no PyYAML required in lite).
        return _parse_simple_yaml(path.read_text(encoding="utf-8"))
    with path.open("r", encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh)
    if not isinstance(loaded, dict):
        raise ValueError(f"Thresholds file must be a mapping: {path}")
    return loaded


def _parse_simple_yaml(text: str) -> Dict[str, Any]:
    """Tiny YAML subset reader for thresholds files without PyYAML.

    Supports nested maps of scalars (float/int/bool/str/null) only.
    """
    root: Dict[str, Any] = {}
    stack: List[Tuple[int, Dict[str, Any]]] = [(-1, root)]

    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()
        if ":" not in line:
            continue
        key, _, rest = line.partition(":")
        key = key.strip()
        rest = rest.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if rest == "" or rest == "|" or rest == ">":
            child: Dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_scalar(rest)
    return root


def _parse_scalar(value: str) -> Any:
    if value in ("null", "Null", "NULL", "~"):
        return None
    if value in ("true", "True", "TRUE"):
        return True
    if value in ("false", "False", "FALSE"):
        return False
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    try:
        if "." in value or "e" in value.lower():
            return float(value)
        return int(value)
    except ValueError:
        return value


def _builtin_fail_closed_defaults() -> Dict[str, Any]:
    """Conservative defaults used only when no thresholds.yaml is found.

    Safe band is intentionally narrow; most mass routes to REVIEW.
    """
    return {
        "_meta": {
            "note": "builtin fail-closed defaults; replace with calibrated YAML",
            "version": 0,
        },
        "injection": {
            "safe_max": 0.15,
            "block_min": 0.85,
            "target_fpr": 0.01,
            "source": "builtin",
        },
        "injection_lite": {
            "safe_max": 0.15,
            "block_min": 0.85,
            "target_fpr": 0.01,
            "source": "builtin",
        },
        "memory_poisoning": {
            "safe_max": 0.15,
            "block_min": 0.85,
            "target_fpr": 0.01,
            "source": "builtin",
        },
        "chain_transition": {
            "safe_max": 0.05,
            "block_min": 0.01,
            "target_fpr": 0.01,
            "higher_is_safer": True,
            "source": "builtin",
        },
        "mission_domain": {
            "safe_max": 0.55,
            "block_min": 0.85,
            "target_fpr": 0.05,
            "higher_is_safer": False,
            "source": "builtin",
        },
        "mission_domain_lite": {
            "safe_max": 0.55,
            "block_min": 0.85,
            "target_fpr": 0.05,
            "higher_is_safer": False,
            "source": "builtin",
        },
        "intent_extraction": {
            "safe_max": 0.55,
            "block_min": 0.85,
            "target_fpr": 0.05,
            "higher_is_safer": False,
            "source": "builtin",
        },
        "intent_extraction_lite": {
            "safe_max": 0.55,
            "block_min": 0.85,
            "target_fpr": 0.05,
            "higher_is_safer": False,
            "source": "builtin",
        },
        "alignment": {
            "safe_max": 0.80,
            "warn_max": 0.50,
            "block_min": 0.30,
            "target_fpr": 0.01,
            "higher_is_safer": True,
            "source": "builtin",
        },
        "optimizer_similarity": {
            "safe_max": 0.70,
            "block_min": 0.70,
            "target_fpr": 0.02,
            "higher_is_safer": True,
            "source": "builtin",
        },
        "optimizer_similarity_lite": {
            "safe_max": 0.70,
            "block_min": 0.70,
            "target_fpr": 0.02,
            "higher_is_safer": True,
            "source": "builtin",
        },
        "secret_entropy": {
            "safe_max": 0.40,
            "block_min": 0.75,
            "target_fpr": 0.01,
            "source": "builtin",
        },
    }


# ---------------------------------------------------------------------------
# Calibrator fit / load / predict
# ---------------------------------------------------------------------------


def fit_and_serialize_calibrator(
    estimator: Any,
    X: Any,
    y: Sequence[int],
    path: Union[str, Path],
    *,
    method: str = "isotonic",
    cv: int = 3,
    meta: Optional[Mapping[str, Any]] = None,
) -> Any:
    """Fit ``CalibratedClassifierCV`` and serialize model + metadata.

    Requires ``sklearn`` and ``joblib`` (provided by ``asha[hardened]``).
    """
    from sklearn.calibration import CalibratedClassifierCV

    try:
        import joblib
    except ImportError as exc:
        raise ImportError(
            "joblib is required to serialize calibrators. "
            "Install with: pip install asha[hardened]"
        ) from exc

    calibrator = CalibratedClassifierCV(estimator, method=method, cv=cv)
    calibrator.fit(X, y)

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "calibrator": calibrator,
        "method": method,
        "cv": cv,
        "meta": dict(meta or {}),
    }
    joblib.dump(payload, out)
    # Sidecar JSON for humans / non-sklearn tooling.
    sidecar = out.with_suffix(out.suffix + ".meta.json")
    sidecar.write_text(
        json.dumps(
            {"method": method, "cv": cv, "meta": dict(meta or {}), "path": str(out)},
            indent=2,
        ),
        encoding="utf-8",
    )
    return calibrator


def load_calibrator(path: Union[str, Path]) -> Any:
    """Load a calibrator previously written by :func:`fit_and_serialize_calibrator`."""
    try:
        import joblib
    except ImportError as exc:
        raise ImportError(
            "joblib is required to load calibrators. "
            "Install with: pip install asha[hardened]"
        ) from exc
    payload = joblib.load(Path(path))
    if isinstance(payload, dict) and "calibrator" in payload:
        return payload["calibrator"]
    return payload


def calibrated_predict_proba(calibrator: Any, X: Any) -> Any:
    """Return positive-class calibrated probabilities as a 1-D array."""
    import numpy as np

    proba = calibrator.predict_proba(X)
    arr = np.asarray(proba, dtype=np.float64)
    if arr.ndim == 2 and arr.shape[1] >= 2:
        return arr[:, 1]
    return arr.reshape(-1)


def derive_threshold_at_fpr(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    target_fpr: float,
) -> float:
    """Choose the lowest score threshold whose empirical FPR ≤ ``target_fpr``.

    Returns 1.0 if no threshold can satisfy the FPR constraint (fail-closed:
    everything becomes REVIEW/BLOCK rather than falsely ALLOW).
    """
    import numpy as np

    if not (0.0 < target_fpr < 1.0):
        raise ValueError(f"target_fpr must be in (0, 1), got {target_fpr}")

    y = np.asarray(list(y_true), dtype=int)
    scores = np.asarray(list(y_prob), dtype=float)
    if y.shape != scores.shape:
        raise ValueError("y_true and y_prob must have the same shape")

    negatives = y == 0
    n_neg = int(negatives.sum())
    if n_neg == 0:
        return 1.0

    # Sort unique scores descending; evaluate FPR at each cut.
    order = np.argsort(-scores)
    sorted_scores = scores[order]
    sorted_neg = negatives[order]

    best = 1.0
    fp = 0
    for i, score in enumerate(sorted_scores):
        if sorted_neg[i]:
            fp += 1
        fpr = fp / n_neg
        if fpr <= target_fpr:
            best = float(score)
        else:
            # Further lowering the threshold only increases FPR.
            break
    return best


@dataclass
class CalibrationReport:
    """Summary produced by training scripts after fitting bands."""

    detector: str
    method: str
    target_fpr: float
    derived_safe_max: float
    derived_block_min: float
    n_samples: int
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_bands(self) -> ThresholdBands:
        return ThresholdBands(
            safe_max=self.derived_safe_max,
            block_min=self.derived_block_min,
            target_fpr=self.target_fpr,
            source="calibrated",
        )
