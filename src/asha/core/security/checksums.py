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

"""Checksum / structural validators for PII recognizers.

Used by both the lite (PIIDetector) and hybrid (RegexDetector) paths.
A regex hit that fails checksum validation should keep a *lower* confidence,
not be dropped entirely.
"""

from __future__ import annotations

import re
import string
from typing import Tuple

# ---------------------------------------------------------------------------
# Luhn (credit / debit cards)
# ---------------------------------------------------------------------------


def luhn_validate(number: str) -> bool:
    digits = re.sub(r"\D", "", number)
    if not digits.isdigit() or len(digits) not in (13, 14, 15, 16, 19):
        return False
    total = 0
    for i, digit in enumerate(reversed(digits)):
        n = int(digit)
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


# ---------------------------------------------------------------------------
# Verhoeff (Aadhaar)
# ---------------------------------------------------------------------------

_VERHOEFF_D = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
    [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
    [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
    [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
]
_VERHOEFF_P = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
    [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
    [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
    [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
]
_VERHOEFF_INV = [0, 4, 3, 2, 1, 5, 6, 7, 8, 9]


def verhoeff_checksum_digit(number_without_check: str) -> str:
    c = 0
    digits = [int(d) for d in reversed(number_without_check)]
    for i, digit in enumerate(digits, start=1):
        c = _VERHOEFF_D[c][_VERHOEFF_P[i % 8][digit]]
    return str(_VERHOEFF_INV[c])


def verhoeff_validate(number: str) -> bool:
    digits_only = re.sub(r"\D", "", number)
    if not digits_only.isdigit() or len(digits_only) != 12:
        return False
    # Aadhaar must not start with 0 or 1.
    if digits_only[0] in ("0", "1"):
        return False
    c = 0
    for i, digit in enumerate(reversed([int(d) for d in digits_only])):
        c = _VERHOEFF_D[c][_VERHOEFF_P[i % 8][digit]]
    return c == 0


def normalize_aadhaar(text: str) -> str:
    return re.sub(r"\D", "", text)


# ---------------------------------------------------------------------------
# PAN (structural, no public checksum)
# ---------------------------------------------------------------------------

_PAN_RE = re.compile(r"^[A-Z]{5}[0-9]{4}[A-Z]$")


def pan_validate(text: str) -> bool:
    """Structural PAN check: 5 letters + 4 digits + 1 letter."""
    cleaned = re.sub(r"[\s-]", "", text).upper()
    return bool(_PAN_RE.match(cleaned))


# ---------------------------------------------------------------------------
# GSTIN (base-36 weighted check digit, best-effort documented algorithm)
# ---------------------------------------------------------------------------

_GST_CHARSET = string.digits + string.ascii_uppercase
_GSTIN_RE = re.compile(r"^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z][0-9A-Z]Z[0-9A-Z]$")


def gstin_check_digit(gstin_14: str) -> str:
    total = 0
    factor = 1
    for ch in gstin_14.upper():
        value = _GST_CHARSET.index(ch)
        product = value * factor
        total += (product // 36) + (product % 36)
        factor = 2 if factor == 1 else 1
    remainder = total % 36
    check_val = (36 - remainder) % 36
    return _GST_CHARSET[check_val]


def gstin_validate(text: str) -> bool:
    cleaned = re.sub(r"[\s-]", "", text).upper()
    if not _GSTIN_RE.match(cleaned) or len(cleaned) != 15:
        return False
    expected = gstin_check_digit(cleaned[:14])
    return cleaned[14] == expected


def validate_pii_checksum(pii_type: str, value: str) -> Tuple[bool, str]:
    """Return (passed, algorithm_name). Types without checksums return (True, 'none')."""
    if pii_type == "credit_card":
        return luhn_validate(value), "luhn"
    if pii_type in ("aadhaar", "in_aadhaar"):
        return verhoeff_validate(value), "verhoeff"
    if pii_type in ("pan", "in_pan"):
        return pan_validate(value), "pan_structure"
    if pii_type in ("gstin", "in_gstin"):
        return gstin_validate(value), "gstin_base36"
    return True, "none"
