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

"""Calibrated alignment score → ALLOW/WARN/REVIEW/BLOCK banding.

Bands come from ``config/thresholds.yaml`` (alignment.*), optionally
refit by ``training/alignment/train.py`` via isotonic regression against
``asha_training_data_generators/.../labeled_actions.jsonl``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union

from asha.core.ml.calibration import ThresholdBands, get_bands, load_thresholds
from asha.runtime.anchor.verdicts import Verdict

# Phrases / substrings in risk triggers that force BLOCK regardless of score.
HARD_BLOCK_TRIGGER_MARKERS: Tuple[str, ...] = (
    "explicitly forbidden",
    "forbidden capability",
    "resource scope violation",
    "external or destructive side-effects",
    "high-risk tool",
    "not in allowed tools",
    "network-egress",
    "blocked under the current mission",
)


def alignment_bands() -> ThresholdBands:
    """Load alignment bands (higher alignment score = safer)."""
    try:
        bands = get_bands("alignment", thresholds=load_thresholds())
        if not bands.higher_is_safer:
            return ThresholdBands(
                safe_max=bands.safe_max,
                block_min=bands.block_min,
                warn_max=bands.warn_max,
                target_fpr=bands.target_fpr,
                source=bands.source,
                higher_is_safer=True,
            )
        return bands
    except Exception:
        # Provisional calibrated defaults matching historical 0.8/0.5/0.3 intent
        # until training/alignment/train.py rewrites thresholds.yaml.
        return ThresholdBands(
            safe_max=0.80,
            warn_max=0.50,
            block_min=0.30,
            source="builtin",
            higher_is_safer=True,
        )


def score_to_verdict(score: float, bands: Optional[ThresholdBands] = None) -> Verdict:
    """Map alignment score to verdict using calibrated bands.

    With ``higher_is_safer``:
      score >= safe_max             → ALLOW
      warn_max <= score < safe_max  → WARN
      block_min <= score < warn_max → REVIEW
      score < block_min             → BLOCK
    """
    bands = bands or alignment_bands()
    s = float(score)
    if s != s:  # NaN → fail-closed REVIEW
        return Verdict.REVIEW

    if bands.higher_is_safer:
        if s >= bands.safe_max:
            return Verdict.ALLOW
        warn = bands.warn_max if bands.warn_max is not None else bands.block_min
        if s >= warn:
            return Verdict.WARN
        if s >= bands.block_min:
            return Verdict.REVIEW
        return Verdict.BLOCK

    # Risk-style fallback (should not apply to alignment).
    if s >= bands.block_min:
        return Verdict.BLOCK
    if bands.warn_max is not None and s >= bands.warn_max:
        return Verdict.REVIEW
    if s >= bands.safe_max:
        return Verdict.WARN
    return Verdict.ALLOW


def has_hard_block_trigger(triggers: Sequence[str], explanation: str = "") -> bool:
    """True if any independent signal requires BLOCK regardless of aggregate score."""
    blob = " | ".join(list(triggers) + [explanation]).lower()
    return any(marker in blob for marker in HARD_BLOCK_TRIGGER_MARKERS)


def load_isotonic_calibrator(
    path: Optional[Union[str, Path]] = None,
) -> Any:
    """Load optional isotonic calibrator artifact (maps raw → calibrated score)."""
    bp = load_isotonic_breakpoints(path=path)
    if bp:
        return bp
    candidates: List[Path] = []
    if path:
        candidates.append(Path(path))
    try:
        from asha.core.ml.model_store import candidate_model_paths, resolve_model_file

        found = resolve_model_file("alignment", "isotonic.joblib", ensure=True)
        if found is not None:
            candidates.insert(0, found)
        candidates.extend(candidate_model_paths("alignment", "isotonic.joblib"))
    except Exception:
        env = os.environ.get("ASHA_MODELS_DIR")
        if env:
            candidates.append(Path(env) / "alignment" / "isotonic.joblib")
        candidates.append(Path.cwd() / "models" / "alignment" / "isotonic.joblib")
    seen = set()
    for cand in candidates:
        key = str(cand)
        if key in seen:
            continue
        seen.add(key)
        if cand.is_file():
            try:
                import joblib

                payload = joblib.load(cand)
                if isinstance(payload, dict):
                    return payload.get("calibrator", payload)
                return payload
            except Exception:
                return None
    return None


def load_isotonic_breakpoints(
    path: Optional[Union[str, Path]] = None,
) -> Optional[List[Tuple[float, float]]]:
    """Load stdlib JSON isotonic breakpoints (no joblib)."""
    candidates: List[Path] = []
    if path:
        candidates.append(Path(path))
    try:
        from asha.core.ml.model_store import candidate_model_paths, resolve_model_file

        found = resolve_model_file(
            "alignment", "isotonic_breakpoints.json", ensure=True
        )
        if found is not None:
            candidates.insert(0, found)
        candidates.extend(
            candidate_model_paths("alignment", "isotonic_breakpoints.json")
        )
    except Exception:
        env = os.environ.get("ASHA_MODELS_DIR")
        if env:
            candidates.append(Path(env) / "alignment" / "isotonic_breakpoints.json")
        candidates.append(
            Path.cwd() / "models" / "alignment" / "isotonic_breakpoints.json"
        )
        repo = (
            Path(__file__).resolve().parents[4]
            / "models"
            / "alignment"
            / "isotonic_breakpoints.json"
        )
        candidates.append(repo)
    seen = set()
    for cand in candidates:
        key = str(cand)
        if key in seen:
            continue
        seen.add(key)
        if cand.is_file():
            try:
                data = json.loads(cand.read_text(encoding="utf-8"))
                raw = data.get("breakpoints") or data
                if isinstance(raw, list):
                    pts: List[Tuple[float, float]] = []
                    for pair in raw:
                        if isinstance(pair, (list, tuple)) and len(pair) == 2:
                            pts.append((float(pair[0]), float(pair[1])))
                    if pts:
                        return pts
            except Exception:
                return None
    return None


def _interpolate_breakpoints(
    raw: float,
    breakpoints: Sequence[Tuple[float, float]],
) -> float:
    pts = breakpoints
    if raw <= pts[0][0]:
        return pts[0][1]
    if raw >= pts[-1][0]:
        return pts[-1][1]
    import bisect

    xs = [p[0] for p in pts]
    i = bisect.bisect_right(xs, raw)
    x0, y0 = pts[i - 1]
    x1, y1 = pts[i]
    if x1 == x0:
        return y0
    t = (raw - x0) / (x1 - x0)
    return y0 + t * (y1 - y0)


def calibrate_score(raw_score: float, calibrator: Any = None) -> float:
    """Apply isotonic calibrator if present; else identity.

    Extreme scores (exact 0 / 1) are preserved so hard-block and perfect
    alignment are not softened by a sparse calibration fit.
    """
    raw = max(0.0, min(1.0, float(raw_score)))
    if raw <= 0.0 or raw >= 1.0:
        return raw
    if calibrator is None:
        calibrator = load_isotonic_calibrator()
    if calibrator is None:
        return raw
    if isinstance(calibrator, list):
        try:
            return float(max(0.0, min(1.0, _interpolate_breakpoints(raw, calibrator))))
        except Exception:
            return raw
    try:
        import numpy as np

        pred = calibrator.predict(np.asarray([[raw]]))
        return float(max(0.0, min(1.0, pred[0])))
    except Exception:
        return raw
