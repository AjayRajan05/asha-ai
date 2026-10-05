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

"""Mission domain classifier for MissionCompiler.

TF-IDF + calibrated logistic model when ``models/mission/domain_clf.joblib``
exists; otherwise lightweight keyword heuristics with intentionally low
confidence (fail-closed).  sklearn is lazy-imported only on predict/load.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from asha.core.ml.calibration import ThresholdBands, get_bands, load_thresholds

# Domain → mission contract hints (allowed domains / extra actions / scopes).
_DOMAIN_POLICY: Dict[str, Dict[str, Any]] = {
    "analytics_reporting": {
        "allowed_domains": ["analytics", "spreadsheets", "analytics_reporting"],
        "allowed_actions": [],
        "memory_scopes": ["reporting"],
    },
    "communication": {
        "allowed_domains": ["communication"],
        "allowed_actions": ["send"],
        "memory_scopes": [],
    },
    "payments_escrow": {
        "allowed_domains": ["payments", "escrow", "payments_escrow"],
        "allowed_actions": [],
        "memory_scopes": ["payments"],
    },
    "identity_verification": {
        "allowed_domains": ["identity", "verification", "identity_verification"],
        "allowed_actions": [],
        "memory_scopes": ["identity"],
    },
    "healthcare_clinical": {
        "allowed_domains": ["healthcare", "clinical", "healthcare_clinical"],
        "allowed_actions": [],
        "memory_scopes": ["clinical"],
    },
    "law_enforcement_intel": {
        "allowed_domains": ["law_enforcement", "intel", "law_enforcement_intel"],
        "allowed_actions": [],
        "memory_scopes": ["intel"],
    },
    "file_document_ops": {
        "allowed_domains": ["file_ops", "documents", "file_document_ops"],
        "allowed_actions": ["write"],
        "memory_scopes": ["documents"],
    },
    # Gemini-augmented mission domains (models/mission/domain_clf.joblib)
    "marketplace_ops": {
        "allowed_domains": ["marketplace", "marketplace_ops", "gigs"],
        "allowed_actions": [],
        "memory_scopes": ["marketplace"],
    },
    "account_support": {
        "allowed_domains": ["support", "account_support"],
        "allowed_actions": ["send"],
        "memory_scopes": ["support"],
    },
    "healthcare_ops": {
        "allowed_domains": ["healthcare", "healthcare_ops", "clinical"],
        "allowed_actions": [],
        "memory_scopes": ["clinical"],
    },
    "civic_safety": {
        "allowed_domains": ["civic", "safety", "civic_safety"],
        "allowed_actions": [],
        "memory_scopes": ["civic"],
    },
    "traffic_infra": {
        "allowed_domains": ["traffic", "infrastructure", "traffic_infra"],
        "allowed_actions": [],
        "memory_scopes": ["traffic"],
    },
    "off_topic": {
        "allowed_domains": ["off_topic"],
        "allowed_actions": [],
        "memory_scopes": [],
    },
}

_DEFAULT_FORBIDDEN_BY_DOMAIN: Dict[str, List[str]] = {
    "analytics_reporting": [
        "send_email",
        "make_payment",
        "delete_file",
        "network_egress",
        "email",
        "payments",
    ],
    "communication": [
        "delete_file",
        "make_payment",
        "modify_database_schema",
        "payments",
        "credentials",
        "database_write",
    ],
    "payments_escrow": [
        "send_email_externally",
        "delete_file",
        "modify_user_permissions",
        "email",
        "credentials",
    ],
    "identity_verification": [
        "make_payment",
        "send_bulk_email",
        "modify_other_users_data",
        "payments",
    ],
    "healthcare_clinical": [
        "make_payment",
        "delete_patient_record",
        "send_data_externally",
        "payments",
    ],
    "law_enforcement_intel": [
        "send_data_externally",
        "make_payment",
        "modify_case_records",
        "payments",
    ],
    "file_document_ops": [
        "make_payment",
        "send_bulk_email",
        "network_egress",
        "payments",
        "email",
    ],
    "marketplace_ops": [
        "make_payment",
        "delete_file",
        "network_egress",
        "send_bulk_email",
        "payments",
        "credentials",
    ],
    "account_support": [
        "make_payment",
        "delete_file",
        "modify_user_permissions",
        "payments",
        "credentials",
    ],
    "healthcare_ops": [
        "make_payment",
        "delete_patient_record",
        "send_data_externally",
        "network_egress",
        "payments",
    ],
    "civic_safety": [
        "make_payment",
        "send_data_externally",
        "delete_file",
        "payments",
    ],
    "traffic_infra": [
        "make_payment",
        "send_bulk_email",
        "delete_file",
        "payments",
        "credentials",
    ],
    "off_topic": [
        "send_email",
        "make_payment",
        "delete_file",
        "network_egress",
        "email",
        "payments",
        "credentials",
    ],
}

# Keyword heuristics: confidence capped below typical safe_max.
_HEURISTIC_KEYWORDS: Dict[str, Sequence[str]] = {
    "analytics_reporting": (
        "report",
        "analytics",
        "dashboard",
        "chart",
        "trend",
        "analyze",
        "compare",
    ),
    "communication": (
        "email",
        "message",
        "notify",
        "reply",
        "whatsapp",
        "send a",
    ),
    "payments_escrow": (
        "payment",
        "escrow",
        "payout",
        "refund",
        "commission",
    ),
    "identity_verification": (
        "verify",
        "kyc",
        "aadhaar",
        "identity",
        "registration",
    ),
    "healthcare_clinical": (
        "patient",
        "clinical",
        "healthcare",
        "abdm",
        "abha",
    ),
    "law_enforcement_intel": (
        "crime",
        "incident",
        "hotspot",
        "sql query",
        "scrb",
    ),
    "file_document_ops": (
        "pdf",
        "document",
        "pitch deck",
        "export",
        "slides",
    ),
}

_HEURISTIC_CONFIDENCE_CAP = 0.40


def _unpack_domain_payload(
    payload: Any,
) -> Tuple[Any, List[str], Optional[Dict[str, Any]]]:
    """Accept legacy ``calibrator`` dumps and current ``pipeline`` train dumps."""
    if not isinstance(payload, dict):
        return payload, [], None
    model = payload.get("pipeline") or payload.get("calibrator") or payload.get("model")
    meta = payload.get("meta") or {}
    labels = (
        payload.get("mlb_classes")
        or payload.get("classes")
        or meta.get("labels")
        or meta.get("classes")
        or []
    )
    fmap = (
        meta.get("domain_forbidden")
        if isinstance(meta.get("domain_forbidden"), dict)
        else None
    )
    return model, [str(x) for x in labels], fmap


@dataclass(frozen=True)
class DomainPrediction:
    domain: str
    confidence: float
    suggested_forbidden: Tuple[str, ...]
    low_confidence: bool


class DomainClassifier:
    """Multi-class mission domain classifier with fail-closed fallback."""

    def __init__(
        self,
        *,
        classifier_path: Optional[Union[str, Path]] = None,
        domain_forbidden_map: Optional[Mapping[str, Sequence[str]]] = None,
    ) -> None:
        self._classifier_path = Path(classifier_path) if classifier_path else None
        self._domain_forbidden = dict(
            domain_forbidden_map or _DEFAULT_FORBIDDEN_BY_DOMAIN
        )
        self._calibrator: Any = None
        self._lite_clf: Any = None
        self._labels: List[str] = []
        self._backend: str = "heuristic"
        self._ready = False
        self._lock = threading.Lock()

    def predict(self, text: str) -> DomainPrediction:
        """Return domain label, confidence, and suggested forbidden actions."""
        domain, confidence, forbidden = self._predict_core(text)
        bands = _mission_bands(self._backend)
        low = confidence < bands.safe_max
        return DomainPrediction(
            domain=domain,
            confidence=confidence,
            suggested_forbidden=tuple(forbidden),
            low_confidence=low,
        )

    def policy_for(self, domain: str) -> Dict[str, Any]:
        """Return allowed_domains / actions / scopes for a domain label."""
        return dict(_DOMAIN_POLICY.get(domain, {}))

    def _predict_core(self, text: str) -> Tuple[str, float, List[str]]:
        self._ensure_model()
        if self._lite_clf is not None:
            return self._predict_lite(text)
        if self._calibrator is not None:
            return self._predict_ml(text)
        return self._predict_heuristic(text)

    def _predict_lite(self, text: str) -> Tuple[str, float, List[str]]:
        from asha.core.text.canonicalize import canonicalize

        try:
            domain, confidence = self._lite_clf.predict(canonicalize(text))
            forbidden = list(
                self._domain_forbidden.get(domain) or _restrictive_forbidden()
            )
            return domain, confidence, forbidden
        except Exception:
            return self._predict_heuristic(text)

    def _ensure_model(self) -> None:
        if self._ready:
            return
        with self._lock:
            path = self._classifier_path or _default_model_path("domain_clf.joblib")
            if path is not None and Path(path).is_file():
                try:
                    import joblib

                    payload = joblib.load(path)
                    model, labels, fmap = _unpack_domain_payload(payload)
                    if model is not None:
                        self._calibrator = model
                        if labels:
                            self._labels = list(labels)
                        if isinstance(fmap, dict):
                            self._domain_forbidden.update(fmap)
                        self._backend = "hardened"
                        self._ready = True
                        return
                except Exception:
                    self._calibrator = None
            lite_path = _default_lite_model_path("domain_clf.json")
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
                    fmap = meta.get("domain_forbidden")
                    if isinstance(fmap, dict):
                        self._domain_forbidden.update(fmap)
                    self._backend = "lite"
                except Exception:
                    self._lite_clf = None
                    self._backend = "heuristic"
            else:
                self._backend = "heuristic"
            self._ready = True

    def _predict_ml(self, text: str) -> Tuple[str, float, List[str]]:
        try:
            import numpy as np

            proba = self._calibrator.predict_proba([text])
            # OneVsRest / multi-label may return a list of per-class arrays.
            if isinstance(proba, list):
                arr = np.asarray(
                    [float(np.asarray(p, dtype=np.float64).ravel()[-1]) for p in proba],
                    dtype=np.float64,
                ).reshape(1, -1)
            else:
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
            domain = str(classes[idx]) if idx < len(classes) else "unknown"
            forbidden = list(
                self._domain_forbidden.get(domain) or _restrictive_forbidden()
            )
            return domain, confidence, forbidden
        except Exception:
            return self._predict_heuristic(text)

    def _predict_heuristic(self, text: str) -> Tuple[str, float, List[str]]:
        lowered = text.lower()
        scores: Dict[str, int] = {}
        for domain, keywords in _HEURISTIC_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in lowered)
            if score:
                scores[domain] = score
        if not scores:
            return "unknown", 0.15, list(_restrictive_forbidden())

        best_score = max(scores.values())
        leaders = [d for d, s in scores.items() if s == best_score]
        if len(leaders) > 1:
            return "unknown", 0.20, list(_restrictive_forbidden())

        domain = leaders[0]
        # Scale confidence down: heuristics must not clear safe_max.
        raw = min(0.35, 0.15 + 0.05 * best_score)
        confidence = min(raw, _HEURISTIC_CONFIDENCE_CAP)
        forbidden = list(self._domain_forbidden.get(domain, ()))
        return domain, confidence, forbidden


def _restrictive_forbidden() -> List[str]:
    return [
        "delete",
        "drop",
        "remove",
        "destroy",
        "reboot",
        "email",
        "payments",
        "credentials",
        "send",
        "write",
        "network",
        "network_egress",
        "make_payment",
        "send_email",
        "send_bulk_email",
        "database_write",
    ]


def restrictive_forbidden() -> List[str]:
    """Public accessor for fail-closed mission defaults."""
    return _restrictive_forbidden()


def _mission_bands(backend: str = "hardened") -> ThresholdBands:
    key = (
        "mission_domain_lite" if backend in ("lite", "heuristic") else "mission_domain"
    )
    try:
        return get_bands(key, thresholds=load_thresholds())
    except Exception:
        return ThresholdBands(safe_max=0.55, block_min=0.85, source="builtin")


def _default_lite_model_path(filename: str) -> Optional[Path]:
    from asha.core.ml.model_store import resolve_model_file

    return resolve_model_file("mission", "lite", filename, ensure=True)


def _default_model_path(filename: str) -> Optional[Path]:
    from asha.core.ml.model_store import candidate_model_paths, resolve_model_file

    found = resolve_model_file("mission", filename, ensure=True)
    if found is not None:
        return found
    # Preserve prior "first candidate" fallback for callers that open paths.
    cands = candidate_model_paths("mission", filename)
    return cands[0] if cands else None


_default_classifier: Optional[DomainClassifier] = None
_default_lock = threading.Lock()


def get_domain_classifier() -> DomainClassifier:
    global _default_classifier
    if _default_classifier is None:
        with _default_lock:
            if _default_classifier is None:
                _default_classifier = DomainClassifier()
    return _default_classifier


def predict_domain(text: str) -> DomainPrediction:
    """Module-level convenience wrapper."""
    return get_domain_classifier().predict(text)
