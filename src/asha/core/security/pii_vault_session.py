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
PIIVault: Thread-safe, session-scoped, reversible PII masking.

Provides bidirectional placeholder replacement so LLM providers only ever
see tokens like ``<EMAIL_1>`` while the calling application transparently
receives the original PII values back when re-hydrating completions.

    >>> vault = PIIVault()
    >>> masked = vault.mask("Contact alice@example.com or bob@example.com")
    >>> assert "alice@example.com" not in masked
    >>> response = vault.unmask("Sent to <EMAIL_1> and <EMAIL_2>")
    >>> assert "alice@example.com" in response

Thread Safety:
    ``PIIVault`` is thread-safe. Each instance maintains its own lock so
    concurrent requests can use separate vault instances without contention.
    The recommended pattern for multi-turn conversations is one vault per
    session/user, not one global vault for the process.
"""

from __future__ import annotations

import re
import threading
import time
from typing import Dict, List, Tuple

# Ordered regex patterns for PII detection: most specific first
_PII_PATTERNS: List[Tuple[str, str]] = [
    # API keys and tokens
    ("api_key", r"\bsk-[a-zA-Z0-9]{20,}\b"),
    ("jwt_token", r"\beyJ[a-zA-Z0-9_-]+\.eyJ[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+\b"),
    ("bearer_token", r"\b(?:Bearer|Token)\s+[a-zA-Z0-9\-._~+/]{20,}\b"),
    # Financial identifiers
    ("credit_card", r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
    ("ssn", r"\b\d{3}-\d{2}-\d{4}\b"),
    # Contact information
    ("email", r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"),
    ("phone", r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"),
    # Location
    (
        "address",
        r"\b\d+\s+[A-Za-z0-9\s]{2,40}\b"
        r"(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Way)\b",
    ),
]

# Compiled patterns for performance
_COMPILED_PATTERNS: List[Tuple[str, re.Pattern[str]]] = [
    (name, re.compile(pattern)) for name, pattern in _PII_PATTERNS
]

# Maximum age of a vault entry in seconds before TTL cleanup
_DEFAULT_TTL_SECONDS = 3600  # 1 hour


class PIIVault:
    """
    Session-scoped, reversible PII masking vault.

    Replaces detected PII values with deterministic indexed placeholder
    tokens (e.g. ``<EMAIL_1>``) and supports bidirectional re-hydration.
    Designed for use in multi-turn LLM conversations where the application
    must remain aware of real PII values while the model provider never
    sees them.

    Attributes:
        ttl_seconds: Maximum age in seconds before vault entries are
            eligible for cleanup. Defaults to 3600 (one hour).

    Examples:
        Single-turn masking and re-hydration::

            vault = PIIVault()
            prompt = "Please email alice@corp.com the Q3 report."
            masked_prompt = vault.mask(prompt)
            # masked_prompt → "Please email <EMAIL_1> the Q3 report."

            raw_response = llm.call(masked_prompt)
            # raw_response  → "I've queued the email to <EMAIL_1>."
            final_response = vault.unmask(raw_response)
            # final_response → "I've queued the email to alice@corp.com."

        Integration with :class:`SecureLLM`::

            vault = PIIVault()
            client = SecureLLM(openai_client, vault=vault, rehydrate_pii=True)
    """

    def __init__(
        self,
        *,
        ttl_seconds: int = _DEFAULT_TTL_SECONDS,
        session_id: Optional[str] = None,
        custom_scanners: Optional[List[Any]] = None,
    ) -> None:
        """
        Initialise a new PIIVault.

        Args:
            ttl_seconds: How long (in seconds) entries survive before TTL
                cleanup. Cleanup is triggered lazily on ``mask()`` calls.
                Defaults to 3600 (one hour).
            session_id: Optional identifier for correlating vault instances with
                session or conversation contexts.
            custom_scanners: Optional list of :class:`~asha.scanners.BaseScanner` plugins.
        """
        self.ttl_seconds = ttl_seconds
        self.session_id = session_id
        self.custom_scanners = list(custom_scanners or [])
        self._lock = threading.Lock()
        # Mapping: placeholder token → (original_value, timestamp, pii_type)
        self._vault: Dict[str, Tuple[str, float, str]] = {}
        # Counter per PII type for indexed tokens (e.g. EMAIL → 3 means EMAIL_3 was last used)
        self._counters: Dict[str, int] = {}

    @property
    def has_pii(self) -> bool:
        """Whether the vault currently contains any tracked PII entries."""
        with self._lock:
            return bool(self._vault)

    def __len__(self) -> int:
        """Total number of tracked PII tokens in the vault."""
        with self._lock:
            return len(self._vault)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def mask(self, text: str) -> str:
        """
        Detect and mask all PII in ``text``, storing originals in the vault.

        Each unique PII value is assigned a stable deterministic placeholder
        token within this vault instance. Calling ``mask()`` with the same
        PII value across multiple calls returns the same token, enabling
        consistent multi-turn masking.

        Args:
            text: Input text potentially containing PII.

        Returns:
            str: The input text with PII replaced by indexed placeholder tokens
                (e.g. ``alice@example.com`` → ``<EMAIL_1>``).

        Examples:
            >>> vault = PIIVault()
            >>> vault.mask("Call 555-123-4567 or email you@example.com")
            'Call <PHONE_1> or email <EMAIL_1>'
        """
        self._maybe_cleanup()
        with self._lock:
            return self._mask_text(text)

    def unmask(self, text: str) -> str:
        """
        Replace all placeholder tokens in ``text`` with their original values.

        Only tokens that were created by this vault instance are re-hydrated.
        Unknown tokens are left unchanged.

        Args:
            text: LLM response text that may contain placeholder tokens.

        Returns:
            str: The text with placeholder tokens replaced by their original
                PII values. Tokens not known to this vault are preserved as-is.

        Examples:
            >>> vault = PIIVault()
            >>> _ = vault.mask("my email is alice@x.com")
            >>> vault.unmask("Replied to <EMAIL_1>")
            'Replied to alice@x.com'
        """
        with self._lock:
            return self._unmask_text(text)

    def clear(self) -> None:
        """
        Remove all stored PII mappings from the vault.

        After calling ``clear()``, all placeholder tokens become opaque;
        ``unmask()`` will return them unchanged.

        Examples:
            >>> vault = PIIVault()
            >>> _ = vault.mask("alice@x.com")
            >>> vault.clear()
            >>> vault.unmask("<EMAIL_1>")
            '<EMAIL_1>'
        """
        with self._lock:
            self._vault.clear()
            self._counters.clear()

    def summary(self) -> Dict[str, int]:
        """
        Return a count of stored PII entries by type.

        Returns:
            dict: Mapping of PII type name to count of masked values.
                Example: ``{'email': 2, 'phone': 1}``.
        """
        with self._lock:
            counts: Dict[str, int] = {}
            for _token, (_value, _ts, pii_type) in self._vault.items():
                counts[pii_type] = counts.get(pii_type, 0) + 1
            return counts

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _mask_text(self, text: str) -> str:
        """Apply all PII patterns to text and replace with vault tokens."""
        # We process text left-to-right, replacing as we go. To avoid
        # overlapping replacement issues, we collect all matches sorted
        # by position, then apply them right-to-left.
        all_matches: List[Tuple[int, int, str, str, str]] = []  # start, end, value, type, token

        # Pre-build reverse map: original_value → token (dedup per type)
        value_to_token: Dict[str, str] = {
            v: tok for tok, (v, _ts, _t) in self._vault.items()
        }

        for pii_type, pattern in _COMPILED_PATTERNS:
            for match in pattern.finditer(text):
                value = match.group(0)
                # Reuse existing token if this value is already vaulted
                if value in value_to_token:
                    token = value_to_token[value]
                else:
                    # Mint a new indexed token
                    self._counters[pii_type] = self._counters.get(pii_type, 0) + 1
                    token = f"<{pii_type.upper()}_{self._counters[pii_type]}>"
                    self._vault[token] = (value, time.monotonic(), pii_type)
                    value_to_token[value] = token

                all_matches.append((match.start(), match.end(), value, pii_type, token))

        # Check registered custom scanner plugins
        try:
            from ...scanners.registry import ScannerRegistry

            active_scanners = list(ScannerRegistry.get_all())
            for cs in (self.custom_scanners or []):
                if cs not in active_scanners:
                    active_scanners.append(cs)

            for scanner in active_scanners:
                try:
                    for custom_match in scanner.scan(text):
                        val = custom_match.text
                        p_type = custom_match.pii_type
                        if val in value_to_token:
                            tok = value_to_token[val]
                        else:
                            self._counters[p_type] = self._counters.get(p_type, 0) + 1
                            tok = f"<{p_type.upper()}_{self._counters[p_type]}>"
                            self._vault[tok] = (val, time.monotonic(), p_type)
                            value_to_token[val] = tok

                        all_matches.append(
                            (custom_match.start, custom_match.end, val, p_type, tok)
                        )
                except Exception:
                    pass
        except Exception:
            pass

        if not all_matches:
            return text

        # Sort by start position descending so right-to-left replacement is safe
        all_matches.sort(key=lambda x: x[0], reverse=True)
        result = text
        seen_spans: List[Tuple[int, int]] = []
        for start, end, _value, _pii_type, token in all_matches:
            # Skip if span overlaps an already-replaced region
            if any(s <= start < e or s < end <= e for s, e in seen_spans):
                continue
            result = result[:start] + token + result[end:]
            seen_spans.append((start, end))

        return result

    def _unmask_text(self, text: str) -> str:
        """Replace all known vault tokens in text with their original values."""
        for token, (original, _ts, _pii_type) in self._vault.items():
            text = text.replace(token, original)
        return text

    def _maybe_cleanup(self) -> None:
        """Lazily remove expired vault entries based on TTL."""
        now = time.monotonic()
        with self._lock:
            expired = [
                tok
                for tok, (_v, ts, _t) in self._vault.items()
                if now - ts > self.ttl_seconds
            ]
            for tok in expired:
                del self._vault[tok]
