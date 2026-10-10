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

"""
BaseScanner and PatternScanner: Foundation for custom PII & sensitive token scanning.

Provides an open-source plugin protocol enabling developers and enterprise teams
to extend ASHA's security engine with custom domain-specific pattern detectors
(e.g., employee badge IDs, internal ticket numbers, patient record identifiers).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import re
from typing import Callable, List, Optional, Pattern, Union

from ..core.security.pii_detector import PIIMatch


class BaseScanner(ABC):
    r"""
    Abstract base class for custom ASHA scanners.

    Implement this protocol to define custom detection logic for specialized
    PII, secrets, internal identifiers, or sensitive entity types.

    Example:
        >>> from asha.scanners import BaseScanner, PIIMatch, register_scanner
        >>> @register_scanner
        ... class PatientIdScanner(BaseScanner):
        ...     @property
        ...     def name(self) -> str:
        ...         return "patient_id"
        ...
        ...     def scan(self, text: str) -> list[PIIMatch]:
        ...         matches = []
        ...         for m in re.finditer(r"MRN-\d{6}", text):
        ...             matches.append(PIIMatch(
        ...                 text=m.group(),
        ...                 pii_type="patient_id",
        ...                 start=m.start(),
        ...                 end=m.end(),
        ...                 confidence=0.98,
        ...             ))
        ...         return matches
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this scanner plugin."""
        ...

    @abstractmethod
    def scan(self, text: str) -> List[PIIMatch]:
        """
        Scan input text and return detected sensitive matches.

        Args:
            text: Raw input string to analyze.

        Returns:
            List[PIIMatch]: Detected entities with span offsets and confidence.
        """
        ...

    @property
    def pii_types(self) -> List[str]:
        """List of PII / entity type tags emitted by this scanner."""
        return [self.name]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"


class PatternScanner(BaseScanner):
    """
    Ready-to-use regex scanner for custom PII or sensitive tokens.

    Simplifies registering custom regular expressions with optional
    algorithmic validation (e.g., checksum, length check).

    Args:
        name: Unique scanner name.
        pattern: Regex string or compiled ``re.Pattern``.
        pii_type: PII entity label (e.g., ``"employee_id"``, ``"mrn"``).
            Defaults to ``name`` if not provided.
        confidence: Detection confidence between 0.0 and 1.0 (default: 0.9).
        validator: Optional callable taking the matched string and returning
            True if valid, False to discard.

    Examples:
        >>> from asha.scanners import PatternScanner, register_scanner
        >>> badge_scanner = PatternScanner(
        ...     name="corp_badge",
        ...     pattern=r"\\bBADGE-[A-Z]{2}-\\d{4}\\b",
        ...     pii_type="badge_number",
        ...     confidence=0.95,
        ... )
        >>> register_scanner(badge_scanner)
    """

    def __init__(
        self,
        name: str,
        pattern: Union[str, Pattern[str]],
        *,
        pii_type: Optional[str] = None,
        confidence: float = 0.9,
        validator: Optional[Callable[[str], bool]] = None,
    ) -> None:
        if not name:
            raise ValueError("Scanner name cannot be empty")
        self._name = str(name)
        self._pii_type = str(pii_type or name)
        self._confidence = max(0.0, min(1.0, float(confidence)))
        self._validator = validator

        if isinstance(pattern, str):
            self._pattern: Pattern[str] = re.compile(pattern)
        else:
            self._pattern = pattern

    @property
    def name(self) -> str:
        return self._name

    @property
    def pii_types(self) -> List[str]:
        return [self._pii_type]

    def scan(self, text: str) -> List[PIIMatch]:
        """Scan text using the configured regex pattern."""
        if not text:
            return []

        results: List[PIIMatch] = []
        for match in self._pattern.finditer(text):
            matched_str = match.group()
            if self._validator is not None:
                try:
                    if not self._validator(matched_str):
                        continue
                except Exception:
                    continue

            results.append(
                PIIMatch(
                    text=matched_str,
                    pii_type=self._pii_type,
                    start=match.start(),
                    end=match.end(),
                    confidence=self._confidence,
                )
            )

        return results

    def __repr__(self) -> str:
        return (
            f"PatternScanner(name={self._name!r}, pii_type={self._pii_type!r}, "
            f"pattern={self._pattern.pattern!r})"
        )
