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

"""Real-world use-case tests: Scanner plugin architecture (BaseScanner & PatternScanner)."""

from __future__ import annotations

import re
import pytest

import asha
from asha import PIIVault, Profile, SecureLLM, process
from asha.scanners import (
    BaseScanner,
    PatternScanner,
    clear_registered_scanners,
    get_registered_scanners,
    register_scanner,
    unregister_scanner,
)
from asha.core.security.pii_detector import PIIDetector, PIIMatch


@pytest.fixture(autouse=True)
def clean_scanner_registry():
    """Ensure scanner registry is clean before and after each test."""
    clear_registered_scanners()
    yield
    clear_registered_scanners()


def test_pattern_scanner_registers_and_masks_in_dropin_process():
    """A developer creates a PatternScanner for employee IDs and uses it in process()."""
    # 1. Register a custom PatternScanner
    badge_scanner = PatternScanner(
        name="employee_badge",
        pattern=r"\bEMP-\d{5}\b",
        pii_type="employee_id",
        confidence=0.96,
    )
    register_scanner(badge_scanner)

    assert badge_scanner in get_registered_scanners()

    # 2. Run drop-in process with standard profile
    raw_prompt = "Transfer access rights to employee EMP-59218 for server admin."
    result = process(raw_prompt, profile=Profile.BALANCED())

    # 3. EMP-59218 is scrubbed from the processed output
    assert "EMP-59218" not in result.output
    pii_types = [str(e).lower() for e in (result.security.pii_detected if result.security else [])]
    assert "employee_id" in pii_types or "EMPLOYEE_ID" in result.output


def test_custom_basescanner_subclass_with_validation_logic():
    """A developer writes a custom BaseScanner with algorithmic validation logic."""

    @register_scanner
    class MedicalRecordNumberScanner(BaseScanner):
        @property
        def name(self) -> str:
            return "medical_record_number"

        def scan(self, text: str) -> list[PIIMatch]:
            matches = []
            for m in re.finditer(r"\bMRN-(\d{6})\b", text):
                digits = m.group(1)
                # Validation rule: first digit cannot be 0
                if digits.startswith("0"):
                    continue
                matches.append(
                    PIIMatch(
                        text=m.group(0),
                        pii_type="mrn",
                        start=m.start(),
                        end=m.end(),
                        confidence=0.99,
                    )
                )
            return matches

    detector = PIIDetector()

    # Valid MRN should be detected
    valid_text = "Admit patient with MRN-847291 to ICU"
    types = detector.detect_pii_types(valid_text)
    assert "mrn" in types
    masked = detector.mask(valid_text)
    assert "MRN-847291" not in masked
    assert "MRN" in masked

    # Invalid MRN starting with 0 should be ignored by the scanner logic
    invalid_text = "Archive record MRN-012345"
    types_invalid = detector.detect_pii_types(invalid_text)
    assert "mrn" not in types_invalid


def test_custom_scanner_pii_vault_and_secure_llm_rehydration():
    """Custom scanner tokens are masked by PIIVault and rehydrated transparently by SecureLLM."""
    scanner = PatternScanner(
        name="internal_ticket",
        pattern=r"\bTICKET-[A-Z]{3}-\d{4}\b",
        pii_type="support_ticket",
        confidence=0.95,
    )
    register_scanner(scanner)

    vault = PIIVault()
    prompt = "Please escalate TICKET-OPS-9941 to high priority."
    masked = vault.mask(prompt)

    # Underlying model only sees placeholder token
    assert "TICKET-OPS-9941" not in masked
    assert "<SUPPORT_TICKET_1>" in masked

    # Rehydration restores original secret value
    llm_echo_reply = f"Acknowledged resolution for <SUPPORT_TICKET_1>."
    unmasked = vault.unmask(llm_echo_reply)
    assert "TICKET-OPS-9941" in unmasked
    assert "<SUPPORT_TICKET_1>" not in unmasked


def test_scanner_registry_lifecycle():
    """Scanners can be dynamically registered, inspected, and unregistered."""
    s1 = PatternScanner("s1", r"FOO-\d+", pii_type="foo")
    s2 = PatternScanner("s2", r"BAR-\d+", pii_type="bar")

    register_scanner(s1)
    register_scanner(s2)
    assert len(get_registered_scanners()) == 2

    # Unregister one
    success = unregister_scanner("s1")
    assert success is True
    assert len(get_registered_scanners()) == 1
    assert get_registered_scanners()[0].name == "s2"

    # Clear all
    clear_registered_scanners()
    assert len(get_registered_scanners()) == 0
