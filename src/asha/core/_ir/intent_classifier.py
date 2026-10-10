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

"""Fast TF-IDF intent classifier for Prompt IR extraction.

Abstains when max calibrated confidence is below ``intent_extraction.safe_max``.
sklearn is lazy-imported; regex verb-bank fallback abstains on ties / zero score.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from asha.core.ml.calibration import ThresholdBands, get_bands, load_thresholds

from .prompt_ir import IntentType

# Teaching / docs framing that embeds SQL should not be rewritten as ANALYZE.
_DOC_SQL_EXAMPLE = re.compile(
    r"(?is)\b(documentation|tutorial|walkthrough|sample|as an example|for example|e\.g\.)\b"
    r".{0,120}\b(select|insert|update|delete|from|where|join)\b"
    r"|"
    r"\b(select|insert|update|delete)\b"
    r".{0,120}\b(as an example|for example|documentation|tutorial|sample)\b"
)

# Verb banks mirrored from IRBuilder patterns (lite fallback).
_VERB_BANKS: Dict[IntentType, Sequence[str]] = {
    IntentType.ANALYZE: (
        "analyze",
        "examine",
        "investigate",
        "study",
        "review",
        "inspect",
        "evaluate",
        "assess",
    ),
    IntentType.GENERATE: (
        "generate",
        "create",
        "produce",
        "write",
        "make",
        "build",
        "develop",
        "compose",
    ),
    IntentType.SUMMARIZE: (
        "summarize",
        "summarise",
        "condense",
        "brief",
        "shorten",
        "abstract",
        "outline",
        "recap",
    ),
    IntentType.TRANSLATE: (
        "translate",
        "convert",
        "transform",
        "adapt",
        "localize",
        "transcribe",
    ),
    IntentType.CLASSIFY: (
        "classify",
        "categorize",
        "categorise",
        "group",
        "sort",
        "label",
        "tag",
        "organize",
    ),
    IntentType.EXTRACT: (
        "extract",
        "pull",
        "retrieve",
        "obtain",
        "find",
        "locate",
        "identify",
    ),
    IntentType.COMPARE: (
        "compare",
        "contrast",
        "diff",
        "versus",
        "against",
        "difference",
    ),
    IntentType.EXPLAIN: (
        "explain",
        "describe",
        "detail",
        "elaborate",
        "clarify",
        "define",
        "interpret",
    ),
    IntentType.CREATE: (
        "create",
        "design",
        "build",
        "construct",
        "craft",
        "form",
    ),
    IntentType.MODIFY: (
        "modify",
        "change",
        "adjust",
        "alter",
        "edit",
        "update",
        "revise",
        "tweak",
    ),
    IntentType.VALIDATE: (
        "validate",
        "verify",
        "check",
        "confirm",
        "test",
        "ensure",
        "guarantee",
    ),
    IntentType.SEARCH: (
        "search",
        "look for",
        "query",
        "seek",
        "hunt",
    ),
    IntentType.DEBUG: (
        "debug",
        "fix",
        "troubleshoot",
        "repair",
        "solve",
        "resolve",
        "correct",
    ),
    IntentType.OPTIMIZE: (
        "optimize",
        "improve",
        "enhance",
        "boost",
        "streamline",
        "fine-tune",
        "refine",
    ),
}

_AST_HINT_BOOST = {
    "analyze": IntentType.ANALYZE,
    "create": IntentType.GENERATE,
    "compare": IntentType.COMPARE,
    "generate": IntentType.GENERATE,
}

_HEURISTIC_CONFIDENCE_CAP = 0.40


def _unpack_intent_payload(payload: Any) -> Tuple[Any, List[str]]:
    """Accept legacy ``calibrator`` dumps and current ``pipeline`` train dumps."""
    if not isinstance(payload, dict):
        return payload, []
    model = payload.get("pipeline") or payload.get("calibrator") or payload.get("model")
    meta = payload.get("meta") or {}
    labels = (
        payload.get("classes")
        or payload.get("mlb_classes")
        or meta.get("labels")
        or meta.get("classes")
        or []
    )
    return model, [str(x) for x in labels]


@dataclass(frozen=True)
class IntentPrediction:
    intent: IntentType
    confidence: float
    abstained: bool


class IntentClassifier:
    """Intent label predictor with abstain below configured confidence floor."""

    def __init__(self, *, classifier_path: Optional[Union[str, Path]] = None) -> None:
        self._classifier_path = Path(classifier_path) if classifier_path else None
        self._calibrator: Any = None
        self._lite_clf: Any = None
        self._labels: List[str] = []
        self._backend: str = "heuristic"
        self._ready = False
        self._lock = threading.Lock()

    def predict(
        self,
        text: str,
        *,
        ast_hint: Optional[str] = None,
    ) -> IntentPrediction:
        # Documentation / teaching examples that embed SQL are not task prompts.
        if text and _DOC_SQL_EXAMPLE.search(text):
            return IntentPrediction(
                intent=IntentType.ABSTAIN,
                confidence=0.0,
                abstained=True,
            )
        intent, confidence = self._predict_core(text, ast_hint=ast_hint)
        bands = _intent_bands(self._backend)
        if confidence < bands.safe_max:
            return IntentPrediction(
                intent=IntentType.ABSTAIN,
                confidence=confidence,
                abstained=True,
            )
        return IntentPrediction(intent=intent, confidence=confidence, abstained=False)

    def _predict_core(
        self,
        text: str,
        *,
        ast_hint: Optional[str] = None,
    ) -> Tuple[IntentType, float]:
        self._ensure_model()
        if self._lite_clf is not None:
            intent, confidence = self._predict_lite(text)
        elif self._calibrator is not None:
            intent, confidence = self._predict_ml(text)
        else:
            intent, confidence = self._predict_heuristic(text)

        if ast_hint and not intent == IntentType.ABSTAIN:
            mapped = _AST_HINT_BOOST.get(str(ast_hint).lower())
            if mapped is not None and mapped == intent:
                confidence = min(1.0, confidence + 0.08)
            elif mapped is not None and mapped != intent:
                confidence = max(0.0, confidence - 0.05)
        return intent, confidence

    def _predict_lite(self, text: str) -> Tuple[IntentType, float]:
        from asha.core.text.canonicalize import canonicalize

        try:
            label, confidence = self._lite_clf.predict(canonicalize(text))
            try:
                return IntentType(label), confidence
            except ValueError:
                return IntentType.ABSTAIN, confidence
        except Exception:
            return self._predict_heuristic(text)

    def _ensure_model(self) -> None:
        if self._ready:
            return
        with self._lock:
            path = self._classifier_path or _default_model_path("intent_clf.joblib")
            if path is not None and Path(path).is_file():
                try:
                    import joblib

                    payload = joblib.load(path)
                    model, labels = _unpack_intent_payload(payload)
                    if model is not None:
                        self._calibrator = model
                        if labels:
                            self._labels = list(labels)
                        self._backend = "hardened"
                        self._ready = True
                        return
                except Exception:
                    self._calibrator = None
            lite_path = _default_lite_model_path("intent_clf.json")
            if lite_path is not None and Path(lite_path).is_file():
                try:
                    from asha.core.text.hashed_features import HashedOvRClassifier

                    self._lite_clf = HashedOvRClassifier.from_json(lite_path)
                    import json

                    data = json.loads(Path(lite_path).read_text(encoding="utf-8"))
                    meta = data.get("meta") or {}
                    labels = meta.get("labels") or data.get("labels")
                    if labels:
                        self._labels = list(labels)
                    self._backend = "lite"
                except Exception:
                    self._lite_clf = None
                    self._backend = "heuristic"
            else:
                self._backend = "heuristic"
            self._ready = True

    def _predict_ml(self, text: str) -> Tuple[IntentType, float]:
        try:
            import numpy as np

            proba = self._calibrator.predict_proba([text])
            arr = np.asarray(proba, dtype=np.float64)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            idx = int(arr.argmax(axis=1)[0])
            confidence = float(arr[0, idx])
            classes = list(self._labels)
            if not classes:
                raw_classes = getattr(self._calibrator, "classes_", None)
                if raw_classes is not None:
                    classes = [str(c) for c in list(raw_classes)]
                else:
                    step = getattr(self._calibrator, "named_steps", {}).get("clf")
                    raw_classes = (
                        getattr(step, "classes_", None) if step is not None else None
                    )
                    if raw_classes is not None:
                        classes = [str(c) for c in list(raw_classes)]
            label = str(classes[idx]) if idx < len(classes) else "abstain"
            try:
                intent = IntentType(label)
            except ValueError:
                intent = IntentType.ABSTAIN
            return intent, confidence
        except Exception:
            return self._predict_heuristic(text)

    def _predict_heuristic(self, text: str) -> Tuple[IntentType, float]:
        lowered = text.lower()
        scores: Dict[IntentType, int] = {}
        for intent, verbs in _VERB_BANKS.items():
            score = 0
            for verb in verbs:
                if re.search(rf"\b{re.escape(verb)}\b", lowered):
                    score += 1
            if score:
                scores[intent] = score
        if not scores:
            return IntentType.ABSTAIN, 0.0

        best = max(scores.values())
        leaders = [i for i, s in scores.items() if s == best]
        if len(leaders) > 1:
            return IntentType.ABSTAIN, 0.0

        intent = leaders[0]
        confidence = min(_HEURISTIC_CONFIDENCE_CAP, 0.12 + 0.06 * best)
        return intent, confidence


def _intent_bands(backend: str = "hardened") -> ThresholdBands:
    key = (
        "intent_extraction_lite"
        if backend in ("lite", "heuristic")
        else "intent_extraction"
    )
    try:
        return get_bands(key, thresholds=load_thresholds())
    except Exception:
        return ThresholdBands(safe_max=0.55, block_min=0.85, source="builtin")


def _default_lite_model_path(filename: str) -> Optional[Path]:
    from asha.core.ml.model_store import resolve_model_file

    return resolve_model_file("intent", "lite", filename, ensure=True)


def _default_model_path(filename: str) -> Optional[Path]:
    from asha.core.ml.model_store import candidate_model_paths, resolve_model_file

    found = resolve_model_file("intent", filename, ensure=True)
    if found is not None:
        return found
    cands = candidate_model_paths("intent", filename)
    return cands[0] if cands else None


_default_classifier: Optional[IntentClassifier] = None
_default_lock = threading.Lock()


def get_intent_classifier() -> IntentClassifier:
    global _default_classifier
    if _default_classifier is None:
        with _default_lock:
            if _default_classifier is None:
                _default_classifier = IntentClassifier()
    return _default_classifier


def predict_intent(
    text: str,
    *,
    ast_hint: Optional[str] = None,
) -> IntentPrediction:
    return get_intent_classifier().predict(text, ast_hint=ast_hint)
