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

"""Optional spaCy NER recognizer for names/orgs/locations (Layer 2).

Lazy-imports spaCy. When unavailable, ``detect`` returns an empty list so
the hybrid pipeline still works offline with regex + context layers.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

# spaCy label → PII type
_LABEL_MAP = {
    "PERSON": "name",
    "PER": "name",
    "ORG": "organization",
    "GPE": "location",
    "LOC": "location",
    "FAC": "address",
}

# Common English tokens spaCy often mislabels as PERSON; never mask these.
_NER_STOP = frozenset(
    {
        "please",
        "help",
        "me",
        "my",
        "with",
        "the",
        "a",
        "an",
        "and",
        "or",
        "to",
        "for",
        "of",
        "in",
        "on",
        "is",
        "are",
        "was",
        "were",
        "be",
        "this",
        "that",
        "it",
        "you",
        "we",
        "they",
        "i",
        "call",
        "send",
        "email",
        "phone",
        "address",
        "card",
        "number",
        "subscription",
        "cancellation",
        "customer",
        "support",
        "billing",
        "issue",
        "explain",
        "difference",
        "between",
        "supervised",
        "unsupervised",
        "learning",
        "need",
        "analyze",
        "summarize",
        "generate",
        "write",
        "read",
        "data",
        "report",
        "today",
        "tomorrow",
        "yesterday",
    }
)


class SpacyNERDetector:
    """Layer-2 contextual NER recognizer (discriminative, not generative)."""

    def __init__(self, model_name: Optional[str] = None) -> None:
        self.model_name = model_name or os.environ.get(
            "ASHA_SPACY_MODEL", "en_core_web_sm"
        )
        self._nlp: Any = None
        self._load_attempted = False
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        self._ensure_loaded()
        return self._nlp is not None

    def _ensure_loaded(self) -> None:
        if self._load_attempted:
            return
        with self._lock:
            self._load_attempted = True
            if os.environ.get("ASHA_DISABLE_ML", "").lower() in ("1", "true", "yes"):
                return
            try:
                import spacy

                try:
                    self._nlp = spacy.load(self.model_name)
                except OSError:
                    # Prefer small model; try blank English with no vectors.
                    try:
                        self._nlp = spacy.load("en_core_web_sm")
                    except OSError:
                        self._nlp = None
            except ImportError:
                self._nlp = None

    def detect(self, text: str) -> List[Any]:
        """Return PIIEntity-like dicts compatible with DetectionStage."""
        from asha.core.pii_pipeline.stages.base_stage import PIIEntity

        self._ensure_loaded()
        if self._nlp is None or not text.strip():
            return []

        doc = self._nlp(text)
        entities: List[PIIEntity] = []
        for ent in doc.ents:
            pii_type = _LABEL_MAP.get(ent.label_)
            if not pii_type:
                continue
            span = ent.text.strip()
            # Skip very short spans (high FP).
            if len(span) < 2:
                continue
            if not _looks_like_named_entity(span, pii_type):
                continue
            entities.append(
                PIIEntity(
                    text=ent.text,
                    start=ent.start_char,
                    end=ent.end_char,
                    pii_type=pii_type,
                    confidence=0.65,
                    context=text[max(0, ent.start_char - 30) : ent.end_char + 30],
                    metadata={
                        "detector": "spacy_ner",
                        "spacy_label": ent.label_,
                        "model": self.model_name,
                    },
                )
            )
        return entities


def _looks_like_named_entity(span: str, pii_type: str) -> bool:
    """Reject spaCy FPs that would mangle ordinary prompt language."""
    tokens = span.split()
    lower = span.lower()
    if lower in _NER_STOP:
        return False
    if all(tok.lower() in _NER_STOP for tok in tokens):
        return False
    # Single-token PERSON must be Title/UPPER case and not a stopword.
    if pii_type == "name" and len(tokens) == 1:
        tok = tokens[0]
        if tok.lower() in _NER_STOP:
            return False
        if not (tok[:1].isupper() and any(c.isalpha() for c in tok)):
            return False
        # Reject all-lowercase false names ("please", "subscription").
        if tok == tok.lower():
            return False
    return True
