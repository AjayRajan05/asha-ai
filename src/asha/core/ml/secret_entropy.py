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

"""Shannon-entropy + shape-matcher secret scanner.

Port of the detect-secrets / truffleHog style approach: high-entropy
substrings combined with known key-shape regexes. Pure Python (no
generative models, no network calls).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import List, Optional, Pattern, Sequence, Tuple


@dataclass(frozen=True)
class SecretHit:
    """A candidate secret span found in text."""

    value: str
    start: int
    end: int
    entropy: float
    shape: str
    confidence: float

    @property
    def span(self) -> Tuple[int, int]:
        return (self.start, self.end)


# Shape matchers: (name, compiled regex, base confidence boost).
# Confidence is later fused with entropy; these are not hard gates alone.
_SHAPE_PATTERNS: Tuple[Tuple[str, Pattern[str], float], ...] = (
    (
        "openai_sk",
        re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}\b"),
        0.35,
    ),
    (
        "google_api",
        re.compile(r"\bAIza[0-9A-Za-z_\-]{20,}\b"),
        0.35,
    ),
    (
        "aws_access_key",
        re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
        0.40,
    ),
    (
        "aws_secret_key",
        re.compile(r"(?<![A-Za-z0-9/+=])[A-Za-z0-9/+=]{40}(?![A-Za-z0-9/+=])"),
        0.15,  # high false-positive rate alone; needs entropy
    ),
    (
        "jwt",
        re.compile(
            r"\beyJ[A-Za-z0-9_\-]+=*\.eyJ[A-Za-z0-9_\-]+=*\.[A-Za-z0-9_\-+=/]*\b"
        ),
        0.40,
    ),
    (
        "github_pat",
        re.compile(r"\bghp_[A-Za-z0-9]{36}\b"),
        0.40,
    ),
    (
        "github_fine_grained",
        re.compile(r"\bgithub_pat_[A-Za-z0-9_]{22,}\b"),
        0.40,
    ),
    (
        "slack_token",
        re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]{10,}\b"),
        0.35,
    ),
    (
        "generic_bearer",
        re.compile(
            r"(?i)\b(?:bearer|api[_-]?key|secret|token|password)\s*[:=]\s*['\"]?"
            r"([A-Za-z0-9_\-+/=.]{12,})"
        ),
        0.25,
    ),
)

# Candidate token extractor for entropy-only pass (no labeled keyword nearby).
_TOKEN_RE = re.compile(r"[A-Za-z0-9_\-+/=.]{16,}")

# Characters considered for Shannon entropy (printable ASCII subset).
_ENTROPY_ALPHABET_SIZE = 64  # approximate for base64-ish tokens


def shannon_entropy(text: str, *, alphabet_size: Optional[int] = None) -> float:
    """Shannon entropy of ``text`` in bits per character.

    Uses observed character frequencies. When ``alphabet_size`` is set,
    returns normalized entropy in ``[0, 1]`` (H / log2(alphabet_size)).
    """
    if not text:
        return 0.0
    counts: dict[str, int] = {}
    for ch in text:
        counts[ch] = counts.get(ch, 0) + 1
    length = len(text)
    entropy = 0.0
    for count in counts.values():
        p = count / length
        entropy -= p * math.log2(p)
    if alphabet_size is not None and alphabet_size > 1:
        return entropy / math.log2(alphabet_size)
    return entropy


def _entropy_confidence(entropy: float, length: int) -> float:
    """Map raw entropy + length into a [0, 1] confidence contribution.

    Tuned to flag high-entropy secrets while ignoring short/low-entropy
    identifiers. Absolute cutoffs are NOT decision thresholds; they feed
    a continuous score that callers bucket via ``config/thresholds.yaml``.
    """
    # Typical English around 4.0-4.5 bits/char; secrets often >= 4.5-5.5.
    # Use a soft logistic around 4.5 bits/char, boosted by length.
    centered = (entropy - 4.5) / 0.6
    logistic = 1.0 / (1.0 + math.exp(-centered))
    length_factor = min(1.0, max(0.0, (length - 12) / 28.0))
    return max(0.0, min(1.0, 0.65 * logistic + 0.35 * length_factor))


def _combine_confidence(shape_boost: float, entropy_conf: float) -> float:
    # Noisy-OR style fusion so either strong signal can boost confidence,
    # without requiring both.
    combined = 1.0 - (1.0 - shape_boost) * (1.0 - entropy_conf)
    return max(0.0, min(1.0, combined))


def scan_secrets(
    text: str,
    *,
    min_token_length: int = 16,
    extra_shapes: Optional[Sequence[Tuple[str, Pattern[str], float]]] = None,
) -> List[SecretHit]:
    """Scan ``text`` for secret-like substrings.

    Returns hits sorted by descending confidence. Overlapping spans are
    kept when they come from different shape families; identical spans
    keep the higher-confidence hit.
    """
    if not text:
        return []

    shapes = list(_SHAPE_PATTERNS)
    if extra_shapes:
        shapes.extend(extra_shapes)

    hits: List[SecretHit] = []
    seen_spans: dict[Tuple[int, int], float] = {}

    def _add(hit: SecretHit) -> None:
        key = (hit.start, hit.end)
        prev = seen_spans.get(key)
        if prev is not None and prev >= hit.confidence:
            return
        seen_spans[key] = hit.confidence
        # Replace any existing hit with same span.
        for i, existing in enumerate(hits):
            if (existing.start, existing.end) == key:
                hits[i] = hit
                return
        hits.append(hit)

    # Pass 1: shape matchers.
    for name, pattern, boost in shapes:
        for match in pattern.finditer(text):
            # Prefer a captured group if present (e.g. generic_bearer).
            if match.lastindex:
                value = match.group(1)
                start, end = match.span(1)
            else:
                value = match.group(0)
                start, end = match.span(0)
            if len(value) < min_token_length and name != "aws_access_key":
                # AKIA keys are fixed 20 chars; allow shorter only for known shapes.
                if name not in {"aws_access_key"}:
                    continue
            ent = shannon_entropy(value)
            ent_conf = _entropy_confidence(ent, len(value))
            conf = _combine_confidence(boost, ent_conf)
            _add(
                SecretHit(
                    value=value,
                    start=start,
                    end=end,
                    entropy=ent,
                    shape=name,
                    confidence=conf,
                )
            )

    # Pass 2: entropy-only over opaque tokens not already covered.
    for match in _TOKEN_RE.finditer(text):
        value = match.group(0)
        start, end = match.span(0)
        if len(value) < min_token_length:
            continue
        # Skip if fully covered by an existing hit.
        covered = any(h.start <= start and end <= h.end for h in hits)
        if covered:
            continue
        ent = shannon_entropy(value)
        ent_conf = _entropy_confidence(ent, len(value))
        # Require stronger entropy when no shape matched.
        if ent_conf < 0.55:
            continue
        _add(
            SecretHit(
                value=value,
                start=start,
                end=end,
                entropy=ent,
                shape="high_entropy",
                confidence=ent_conf,
            )
        )

    hits.sort(key=lambda h: h.confidence, reverse=True)
    return hits


def max_secret_confidence(text: str, **kwargs: object) -> float:
    """Convenience: highest secret confidence in ``text``, or 0.0."""
    hits = scan_secrets(text, **kwargs)  # type: ignore[arg-type]
    if not hits:
        return 0.0
    return float(hits[0].confidence)
