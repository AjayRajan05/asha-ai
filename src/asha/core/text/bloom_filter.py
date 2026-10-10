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

"""Bloom filter (Bloom, 1970): stdlib-only bit-array membership test.

Sizing (standard optimal formulas; not arbitrary constants):

  m = ceil( -n * ln(p) / (ln 2)^2 )   # bit-array length
  k = round( (m / n) * ln 2 )         # hash count, clamped to [1, 16]

where ``n = expected_items`` and ``p = false_positive_rate`` (default 0.01).
``m`` is then rounded up to a whole number of bytes. See Broder & Mitzenmacher
survey / Bloom 1970; implemented in :class:`BloomFilter.__init__`.
"""

from __future__ import annotations

import math
import struct
from typing import Iterable, Optional

from .hashed_features import _murmurhash3_32


class BloomFilter:
    """Standard k-hash Bloom filter with automatic sizing."""

    def __init__(
        self,
        expected_items: int,
        false_positive_rate: float = 0.01,
        *,
        bit_array: Optional[bytearray] = None,
        num_hashes: Optional[int] = None,
    ) -> None:
        if expected_items < 1:
            raise ValueError("expected_items must be >= 1")
        if not (0.0 < false_positive_rate < 1.0):
            raise ValueError("false_positive_rate must be in (0, 1)")

        if bit_array is not None and num_hashes is not None:
            self._bits = bytearray(bit_array)
            self.num_bits = len(self._bits) * 8
            self.num_hashes = int(num_hashes)
        else:
            # m = -n ln(p) / (ln 2)^2
            m = int(
                math.ceil(
                    -expected_items * math.log(false_positive_rate) / (math.log(2) ** 2)
                )
            )
            m = max(m, 64)
            # Round up to byte boundary
            m = (m + 7) // 8 * 8
            # k = (m/n) ln 2
            k = int(round((m / expected_items) * math.log(2)))
            k = max(1, min(k, 16))
            self.num_bits = m
            self.num_hashes = k
            self._bits = bytearray(m // 8)

        self.expected_items = expected_items
        self.false_positive_rate = false_positive_rate
        self._count = 0

    @staticmethod
    def _hashes(item: str, k: int, m: int) -> Iterable[int]:
        raw = item.encode("utf-8", errors="replace")
        h1 = _murmurhash3_32(raw, seed=0x9747B28C)
        h2 = _murmurhash3_32(raw, seed=0xC2B2AE35) or 0x01000193
        for i in range(k):
            yield (h1 + i * h2) % m

    def add(self, item: str) -> None:
        for idx in self._hashes(item, self.num_hashes, self.num_bits):
            byte_i, bit_i = divmod(idx, 8)
            self._bits[byte_i] |= 1 << bit_i
        self._count += 1

    def __contains__(self, item: object) -> bool:
        if not isinstance(item, str):
            return False
        for idx in self._hashes(item, self.num_hashes, self.num_bits):
            byte_i, bit_i = divmod(idx, 8)
            if (self._bits[byte_i] & (1 << bit_i)) == 0:
                return False
        return True

    def to_bytes(self) -> bytes:
        header = struct.pack(
            ">IIIdI",
            self.num_bits,
            self.num_hashes,
            self.expected_items,
            self.false_positive_rate,
            self._count,
        )
        return header + bytes(self._bits)

    @classmethod
    def from_bytes(cls, data: bytes) -> "BloomFilter":
        if len(data) < 24:
            raise ValueError("bloom payload too short")
        num_bits, num_hashes, expected, fpr, count = struct.unpack(">IIIdI", data[:24])
        body = data[24:]
        expected_len = num_bits // 8
        if len(body) != expected_len:
            raise ValueError(
                f"bit array length mismatch: got {len(body)}, want {expected_len}"
            )
        bf = cls(
            expected_items=max(1, expected),
            false_positive_rate=fpr if 0 < fpr < 1 else 0.01,
            bit_array=bytearray(body),
            num_hashes=num_hashes,
        )
        bf._count = count
        return bf

    @property
    def count(self) -> int:
        return self._count
