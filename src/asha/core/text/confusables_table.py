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

"""Common Unicode confusables → ASCII/Latin folds.

Primary source (Unicode Consortium, UTS #39 / security data):
  https://www.unicode.org/Public/security/latest/confusables.txt
  Index: https://www.unicode.org/Public/security/

This module keeps a **curated high-signal subset** (Cyrillic / Greek /
fullwidth Latin look-alikes common in prompt-injection obfuscation), not the
full multi-hundred-KB table. Spot-check (2026-07-17): downloaded confusables.txt
(~671 KB, 9994 lines); 85/128 of our ``CONFUSABLES`` source codepoints appear in
that file; 77/128 match the Unicode preferred target exactly (remaining entries
are intentional security-oriented folds or near-homoglyphs). Separate
``LEETSPEAK`` map is ASCII-only and is **not** from Unicode confusables.
"""

from __future__ import annotations

# Map: confusable code point → preferred ASCII/Latin replacement (1 char).
# Sourced from Unicode confusables; curated to the high-signal subset for
# security text matching (not the full multi-MB table).
CONFUSABLES: dict[str, str] = {
    # --- Cyrillic look-alikes ---
    "а": "a",  # U+0430 CYRILLIC SMALL LETTER A
    "А": "A",  # U+0410
    "е": "e",  # U+0435
    "Е": "E",  # U+0415
    "о": "o",  # U+043E
    "О": "O",  # U+041E
    "р": "p",  # U+0440
    "Р": "P",  # U+0420
    "с": "c",  # U+0441
    "С": "C",  # U+0421
    "у": "y",  # U+0443
    "х": "x",  # U+0445
    "Х": "X",  # U+0425
    "і": "i",  # U+0456 Ukrainian/Belarusian
    "І": "I",  # U+0406
    "ј": "j",  # U+0458
    "ѕ": "s",  # U+0455
    "ԁ": "d",  # U+0501
    "ԛ": "q",  # U+051B
    "ԝ": "w",  # U+051D
    "һ": "h",  # U+04BB
    "Ӏ": "l",  # U+04C0 PALOCHKA (often used as I/l)
    "ɡ": "g",  # U+0261
    # --- Greek look-alikes ---
    "Α": "A",  # U+0391
    "Β": "B",  # U+0392
    "Ε": "E",  # U+0395
    "Ζ": "Z",  # U+0396
    "Η": "H",  # U+0397
    "Ι": "I",  # U+0399
    "Κ": "K",  # U+039A
    "Μ": "M",  # U+039C
    "Ν": "N",  # U+039D
    "Ο": "O",  # U+039F
    "Ρ": "P",  # U+03A1
    "Τ": "T",  # U+03A4
    "Υ": "Y",  # U+03A5
    "Χ": "X",  # U+03A7
    "α": "a",  # U+03B1
    "ο": "o",  # U+03BF
    "ν": "v",  # U+03BD
    "ρ": "p",  # U+03C1
    "τ": "t",  # U+03C4
    "υ": "u",  # U+03C5
    "χ": "x",  # U+03C7
    "ι": "i",  # U+03B9
    "κ": "k",  # U+03BA
    # --- Fullwidth Latin / digits (FF01-FF5E range core) ---
    "Ａ": "A",
    "Ｂ": "B",
    "Ｃ": "C",
    "Ｄ": "D",
    "Ｅ": "E",
    "Ｆ": "F",
    "Ｇ": "G",
    "Ｈ": "H",
    "Ｉ": "I",
    "Ｊ": "J",
    "Ｋ": "K",
    "Ｌ": "L",
    "Ｍ": "M",
    "Ｎ": "N",
    "Ｏ": "O",
    "Ｐ": "P",
    "Ｑ": "Q",
    "Ｒ": "R",
    "Ｓ": "S",
    "Ｔ": "T",
    "Ｕ": "U",
    "Ｖ": "V",
    "Ｗ": "W",
    "Ｘ": "X",
    "Ｙ": "Y",
    "Ｚ": "Z",
    "ａ": "a",
    "ｂ": "b",
    "ｃ": "c",
    "ｄ": "d",
    "ｅ": "e",
    "ｆ": "f",
    "ｇ": "g",
    "ｈ": "h",
    "ｉ": "i",
    "ｊ": "j",
    "ｋ": "k",
    "ｌ": "l",
    "ｍ": "m",
    "ｎ": "n",
    "ｏ": "o",
    "ｐ": "p",
    "ｑ": "q",
    "ｒ": "r",
    "ｓ": "s",
    "ｔ": "t",
    "ｕ": "u",
    "ｖ": "v",
    "ｗ": "w",
    "ｘ": "x",
    "ｙ": "y",
    "ｚ": "z",
    "０": "0",
    "１": "1",
    "２": "2",
    "３": "3",
    "４": "4",
    "５": "5",
    "６": "6",
    "７": "7",
    "８": "8",
    "９": "9",
    # --- Misc high-signal confusables ---
    "／": "/",
    "－": "-",
    "＿": "_",
    "．": ".",
    "＠": "@",
    "：": ":",
    "；": ";",
    "！": "!",
    "？": "?",
    "（": "(",
    "）": ")",
    "［": "[",
    "］": "]",
    "｛": "{",
    "｝": "}",
    "　": " ",  # ideographic space
    " ": " ",  # narrow no-break space
    " ": " ",  # thin space
    " ": " ",  # figure space
    " ": " ",  # punctuation space
}


# Leetspeak / keyboard-substitution folds. Applied only as an *additional*
# view (lossy, so callers keep the non-leet canonical form too).
LEETSPEAK: dict[str, str] = {
    "0": "o",
    "1": "i",
    "3": "e",
    "4": "a",
    "5": "s",
    "7": "t",
    "@": "a",
    "$": "s",
    "!": "i",
    "|": "l",
    "+": "t",
}
