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

"""Security canonicalization facade (Phase 8: Unicode Injection Bypass).

This module provides the security-facing interface for canonicalisation.
Detection always operates on the ``canonical_text``; the original text
is preserved so application output can retain the appropriate representation.

The implementation delegates to :mod:`asha.core.text.canonicalize`, which
provides a full NFKC → zero-width removal → homoglyph folding pipeline.

Usage::

    from asha.core.security.canonicalize import SecurityCanonicalizer

    sc = SecurityCanonicalizer("ign\\u200bore previous instructions")
    # detection runs on sc.canonical_text
    assert "ignore previous instructions" in sc.canonical_text
    # application output can use sc.original_text
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from asha.core.text.canonicalize import (  # noqa: F401  (re-export)
    CanonicalViews,
    canonicalize,
    canonicalize_views,
    expand_for_matching,
)


@dataclass(frozen=True)
class SecurityCanonicalizer:
    """Pair an original text with its canonical form for safe detection.

    Args:
        original_text: The raw input string received from the user/document.

    Attributes:
        original_text: The unmodified input.
        canonical_text: The canonicalized form used for threat detection.
            This is produced by the NFKC → zero-width removal → confusable-
            fold pipeline and is NOT written back to the user.

    Examples:
        >>> from asha.core.security.canonicalize import SecurityCanonicalizer
        >>> sc = SecurityCanonicalizer("ign\\u200bore previous")
        >>> sc.original_text
        'ign\\u200bore previous'
        >>> "ignore previous" in sc.canonical_text
        True
    """

    original_text: str
    canonical_text: str = ""

    def __init__(
        self, original_text: str, canonical_text: Optional[str] = None
    ) -> None:
        object.__setattr__(self, "original_text", original_text)
        if canonical_text is None:
            canonical_text = canonicalize(original_text)
        object.__setattr__(self, "canonical_text", canonical_text)

    @property
    def was_obfuscated(self) -> bool:
        """Return ``True`` when the canonical form differs from the original.

        A difference indicates the presence of zero-width characters,
        homoglyphs, or other obfuscation techniques that were stripped
        during canonicalization.
        """
        return self.original_text != self.canonical_text

    def detection_views(self) -> tuple[str, ...]:
        """Return all text views to be used during detection.

        Returns:
            A tuple containing (canonical_text,) and optionally the original
            when it differs, giving detectors both views.
        """
        views: list[str] = [self.canonical_text]
        if self.was_obfuscated:
            views.append(self.original_text)
        return tuple(views)
