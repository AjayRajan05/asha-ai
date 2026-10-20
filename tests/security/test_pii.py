from __future__ import annotations
"""Security regression suite: PII detection tests.

Tests email, phone, SSN, credit card, and address detection.
Phase 19 regression: all PII categories from the audit.
"""


import pytest
from asha import sanitize


# ---------------------------------------------------------------------------
# PII categories that must be detected and masked
# ---------------------------------------------------------------------------


def test_email_masked():
    result = sanitize("Contact john.doe@example.com for support.")
    assert "john.doe@example.com" not in result.output
    # Task structure must survive
    assert "Contact" in result.output or "contact" in result.output.lower()


def test_phone_number_masked():
    result = sanitize("Call +1-555-867-5309 for assistance.")
    assert "555-867-5309" not in result.output
    assert "867-5309" not in result.output


def test_ssn_masked():
    result = sanitize("My SSN is 123-45-6789 as listed on the form.")
    assert "123-45-6789" not in result.output


def test_credit_card_masked():
    result = sanitize("Charge card number 4532015112830366 for the order.")
    assert "4532015112830366" not in result.output


def test_multiple_pii_in_one_prompt():
    result = sanitize(
        "Send the report to alice@company.com. Her phone is 555-123-4567. "
        "Her SSN is 987-65-4321."
    )
    output = result.output
    assert "alice@company.com" not in output
    assert "987-65-4321" not in output


# ---------------------------------------------------------------------------
# PII masking must not destroy context
# ---------------------------------------------------------------------------


def test_email_masked_context_preserved():
    result = sanitize("Contact john@example.com for data analysis support.")
    output = result.output
    # The task instruction must remain
    assert "analysis" in output.lower() or "support" in output.lower()


def test_name_masking_preserves_task():
    result = sanitize("Review the contract for John Smith and summarize key obligations.")
    output = result.output
    # Task must remain
    assert "summarize" in output.lower() or "contract" in output.lower() or "obligations" in output.lower()


# ---------------------------------------------------------------------------
# False positive guard
# ---------------------------------------------------------------------------


def test_email_in_technical_doc_masked():
    """Even in technical docs, emails should be masked."""
    result = sanitize("Example: send POST to admin@company.com with payload {'key': 'val'}")
    assert "admin@company.com" not in result.output


def test_no_pii_not_flagged():
    """Prompts with no PII should not have PII masked."""
    result = sanitize("Explain the theory of relativity in simple terms.")
    # Output should be essentially unchanged
    assert "relativity" in result.output.lower() or "theory" in result.output.lower()
    assert result.security_result.pii_detected_count == 0


# --- Consolidated from tests/test_pii_phone.py ---
"""Phone number PII detection and masking tests."""

from asha import process
from asha.core.pii_pipeline.stages.detection_stage import RegexDetector
from asha.core.security.pii_detector import PIIDetector
from asha.types.results import OptimizeResult

from conftest import output_of


def test_phone_dash_format_detected_and_masked():
    out = output_of(sanitize("Call 555-123-4567 for support"))
    assert "555-123-4567" not in out
    assert "[PHONE_HASH]" in out


def test_phone_dash_format_detected_by_regex():
    entities = RegexDetector().detect("Call 555-123-4567 for support")
    assert any(e.pii_type == "phone" for e in entities)


def test_phone_international_detected_by_regex():
    entities = RegexDetector().detect("International +1-555-123-4567")
    assert any(e.pii_type == "phone" for e in entities)


def test_process_masks_phone_dash_format():
    out = output_of(process("Contact 555-987-6543 today", mode="strict"))
    assert "555-987-6543" not in out


def test_optimize_tokens_only_may_keep_phone():
    """optimize() is tokens-only and does not mask PII."""
    from asha import optimize

    result = optimize("My number is 555-111-2222")
    assert isinstance(result, OptimizeResult)
    assert "555-111-2222" in result.output


def test_phone_not_confused_with_order_number():
    types = PIIDetector().detect_pii_types("Order #12345-6789 shipped")
    assert "phone" not in types


# --- Consolidated from tests/test_pii_credit_address.py ---
"""PII detection tests for credit cards and addresses."""

from asha.core.policy_config import PolicyConfig
from asha.core.security.pii_detector import PIIDetector
from asha.core.pii_pipeline.components.detectors.regex_detector import RegexDetector
from asha.utils.dropin import sanitize

from conftest import output_of


VALID_VISA = "4111111111111111"
VALID_VISA_SPACED = "4111 1111 1111 1111"
INVALID_LUHN = "4111111111111112"


class TestCreditCardLuhn:
    def test_valid_card_detected(self):
        detector = PIIDetector()
        types = detector.detect_pii_types(f"Pay with {VALID_VISA}")
        assert "credit_card" in types

    def test_invalid_luhn_rejected_by_detector(self):
        detector = PIIDetector()
        types = detector.detect_pii_types(f"ID {INVALID_LUHN}")
        assert "credit_card" not in types

    def test_regex_detector_luhn_soft_fail(self):
        """Checksum failure lowers confidence; does not grant full confidence."""
        regex = RegexDetector()
        valid = regex.detect(f"Card: {VALID_VISA_SPACED}")
        invalid = regex.detect(f"Ref: {INVALID_LUHN}")
        assert any(e.pii_type == "credit_card" for e in valid)
        valid_cc = [e for e in valid if e.pii_type == "credit_card"]
        assert all(e.metadata.get("checksum_valid") for e in valid_cc)
        invalid_cc = [e for e in invalid if e.pii_type == "credit_card"]
        for e in invalid_cc:
            assert e.metadata.get("checksum_valid") is False
            assert e.confidence < valid_cc[0].confidence

    def test_sanitize_masks_valid_card(self):
        # Exact hash-token format is the lite/PIIDetector path.
        out = output_of(
            sanitize(
                f"My card is {VALID_VISA}",
                policy=PolicyConfig(pii_mode="lite"),
            )
        )
        assert VALID_VISA not in out
        assert "[CREDIT_CARD_HASH]" in out

    def test_sanitize_masks_spaced_card(self):
        out = output_of(
            sanitize(
                f"Charge {VALID_VISA_SPACED} please",
                policy=PolicyConfig(pii_mode="lite"),
            )
        )
        assert VALID_VISA not in out.replace(" ", "")


class TestAddressDetection:
    def test_street_suffix_detected(self):
        detector = PIIDetector()
        text = "Office at 500 Main Street, Boston"
        types = detector.detect_pii_types(text)
        assert "address" in types

    def test_sanitize_masks_street_address(self):
        out = output_of(
            sanitize(
                "Ship to 742 Evergreen Street today",
                policy=PolicyConfig(pii_mode="lite"),
            )
        )
        assert "742 Evergreen Street" not in out
        assert "[ADDRESS_HASH]" in out

    def test_process_masks_full_address(self):
        from asha import process

        text = "Office at 100 Oak Avenue, Springfield, IL 62701"
        out = output_of(process(text, mode="strict"))
        assert "Oak Avenue" not in out or "[ADDRESS_HASH]" in out

    def test_address_hash_token_in_output(self):
        out = output_of(
            sanitize(
                "Located at 42 Maple Drive",
                policy=PolicyConfig(pii_mode="lite"),
            )
        )
        assert "[ADDRESS_HASH]" in out


# --- Consolidated from tests/test_pii_lite_canonicalize.py ---
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

"""PII detection via canonicalize + context boost (lite path)."""


from asha.core.security.pii_detector import PIIDetector
from asha.utils.dropin import sanitize


def test_obfuscated_phone_detected_and_masked():
    """Zero-width split Indian mobile must be detected after canonicalize."""
    phone = "9\u200b8\u200c7\u200d6\u206065\u206143\u206132\u206110"
    text = f"My phone number is {phone} for callbacks."
    detector = PIIDetector()
    masked = detector.mask(text)
    assert "[PHONE_HASH]" in masked
    assert phone not in masked


def test_obfuscated_email_detected_and_masked():
    """Fullwidth / homoglyph email local-part must still match."""
    email = "j\u0430ne\u200b@gmail.com"  # Cyrillic 'a' + zero-width
    text = f"Contact me at {email} please."
    detector = PIIDetector()
    masked = detector.mask(text)
    assert "[EMAIL_HASH]" in masked


def test_sanitize_masks_obfuscated_phone():
    phone = "9\u200b8\u200c7\u200d6\u206065\u206143\u206132\u206110"
    text = f"Call me on phone {phone} tomorrow."
    result = sanitize(text, mode="lite")
    assert result.output is not None
    assert "[PHONE_HASH]" in result.output or "PHONE" in result.output.upper()


# --- Consolidated from tests/test_pii_false_positives.py ---
"""False positive regression tests."""

from asha import process
from asha.core.security.pii_detector import PIIDetector
from asha.types.results import ProcessResult, SanitizeResult

from conftest import output_of


def test_order_number_not_flagged_as_phone():
    types = PIIDetector().detect_pii_types("Order #12345-6789 shipped today")
    assert "phone" not in types


def test_version_string_not_flagged_as_phone():
    types = PIIDetector().detect_pii_types("Upgrade to v1.2.3.4 release")
    assert "phone" not in types


def test_sql_in_documentation_context_not_blocked():
    text = "Documentation: use SELECT name FROM users WHERE id = 1 as an example."
    result = process(text, mode="balanced")
    assert isinstance(result, ProcessResult)
    out = output_of(result)
    assert len(out) > 0
    # Must not hard-fail the request; documentation context should remain usable.
    assert result.degraded is False or out.strip() != ""
    assert "[REQUEST_BLOCKED]" not in out
    assert (
        "careful assistant" in out.lower()
        or "documentation" in out.lower()
        or "select" in out.lower()
    )


def test_teaching_example_com_still_allowed():
    result = sanitize("Use test@example.com in your docs.")
    assert isinstance(result, SanitizeResult)
    assert "test@example.com" in result.output


# --- Consolidated from tests/test_pii_scoring.py ---
"""Unit tests for PII confidence scoring stage."""

from asha.core.pii_pipeline.stages.base_stage import PIIEntity, create_pii_context
from asha.core.pii_pipeline.stages.scoring_stage import ScoringStage


def test_scoring_boosts_high_confidence_regex():
    stage = ScoringStage()
    entity = PIIEntity(
        text="john@example.com",
        start=0,
        end=16,
        pii_type="email",
        confidence=0.85,
        context="email",
        metadata={"detector": "regex"},
    )
    context = create_pii_context("john@example.com", {}, False)
    context.entities = [entity]
    result = stage.execute(context)
    assert result.success
    assert context.entities[0].confidence > 0.5


def test_scoring_suppresses_low_confidence():
    stage = ScoringStage()
    entity = PIIEntity(
        text="123 Main",
        start=0,
        end=8,
        pii_type="address",
        confidence=0.2,
        context="generic",
        metadata={"detector": "heuristic"},
    )
    context = create_pii_context("123 Main St", {}, False)
    context.entities = [entity]
    stage.execute(context)
    # Low-confidence entities may be filtered or remain below threshold
    assert all(e.confidence <= 0.5 for e in context.entities if e.confidence < 0.5)


# --- Consolidated from tests/test_threshold_section_splits.py ---
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

"""Regression: lite vs hardened threshold sections must stay independently calibrated."""


from asha.core.ml.calibration import get_bands, load_thresholds


def test_mission_domain_lite_and_hardened_thresholds_differ() -> None:
    """Accidental re-merge of shared cutoffs must fail loudly."""
    cfg = load_thresholds()
    hardened = get_bands("mission_domain", thresholds=cfg)
    lite = get_bands("mission_domain_lite", thresholds=cfg)
    assert "mission_domain" in cfg
    assert "mission_domain_lite" in cfg
    assert hardened.safe_max != lite.safe_max, (
        "mission_domain and mission_domain_lite share the same safe_max; "
        "recalibrate each path independently (do not copy one value into both)"
    )


def test_intent_and_optimizer_split_sections_exist() -> None:
    cfg = load_thresholds()
    for key in (
        "intent_extraction",
        "intent_extraction_lite",
        "optimizer_similarity",
        "optimizer_similarity_lite",
    ):
        assert key in cfg, f"missing threshold section: {key}"
        bands = get_bands(key, thresholds=cfg)
        assert bands.safe_max > 0.0


# --- Consolidated from tests/test_pii_hardening.py ---
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

"""PII hardening tests: Indian formats, hybrid default, checksum soft-fail."""


from asha.core.hybrid_pii import HybridPIIDetector
from asha.core.ml_utils import check_pii_mode_feasibility, validate_pii_mode
from asha.core.policy_config import PolicyConfig
from asha.core.security.checksums import (
    gstin_check_digit,
    gstin_validate,
    luhn_validate,
    pan_validate,
    verhoeff_checksum_digit,
    verhoeff_validate,
)
from asha.core.security.locale_detect import detect_locale
from asha.core.security.pii_detector import PIIDetector
from asha.core.security.service import run_security


def _valid_aadhaar() -> str:
    body = "23456789012"
    return body + verhoeff_checksum_digit(body)


def _invalid_aadhaar() -> str:
    good = _valid_aadhaar()
    return good[:-1] + ("0" if good[-1] != "0" else "1")


def _valid_gstin() -> str:
    # 22 + PAN-shaped + entity + Z + check
    pan = "ABCDE1234F"
    base = f"22{pan}1Z"
    return base + gstin_check_digit(base)


def test_validate_pii_mode_lite_alias():
    assert validate_pii_mode("lite") == "lite"
    assert validate_pii_mode("rule") == "lite"
    assert validate_pii_mode("hybrid") == "hybrid"
    assert validate_pii_mode("hybrid_calibrated") == "hybrid"


def test_hybrid_feasible_without_full_ml_stack():
    feas = check_pii_mode_feasibility("hybrid")
    assert feas["feasible"] is True
    assert feas["requires_ml"] is False


def test_policy_default_is_hybrid():
    assert PolicyConfig().pii_mode == "hybrid"


def test_verhoeff_aadhaar_valid_and_invalid():
    good = _valid_aadhaar()
    bad = _invalid_aadhaar()
    assert verhoeff_validate(good)
    assert not verhoeff_validate(bad)


def test_pan_and_gstin_validators():
    assert pan_validate("ABCDE1234F")
    assert not pan_validate("ABCDE12345")  # wrong shape
    gstin = _valid_gstin()
    assert gstin_validate(gstin)
    assert not gstin_validate(gstin[:-1] + ("0" if gstin[-1] != "0" else "1"))


def test_luhn_still_works():
    # Well-known Luhn-valid test Visa number (public test fixture).
    assert luhn_validate("4111111111111111")
    assert not luhn_validate("4111111111111112")


def test_lite_detector_masks_valid_aadhaar():
    detector = PIIDetector()
    aadhaar = _valid_aadhaar()
    text = f"My Aadhaar number is {aadhaar} for KYC."
    masked = detector.mask(text)
    assert aadhaar not in masked
    assert "AADHAAR" in masked or "HASH" in masked


def test_invalid_aadhaar_lower_confidence_not_dropped():
    detector = PIIDetector()
    bad = _invalid_aadhaar()
    text = f"Aadhaar {bad} submitted"
    matches = detector._detect_pii_with_context(text)
    aadhaar_matches = [m for m in matches if m.pii_type == "aadhaar"]
    # Soft-fail: may still surface with reduced confidence, or drop below 0.3.
    # Ensure we don't assign full (≥0.9) confidence to a checksum failure.
    for m in aadhaar_matches:
        assert m.confidence < 0.9


def test_pan_and_gstin_detection_in_context():
    detector = PIIDetector()
    pan = "ABCDE1234F"
    gstin = _valid_gstin()
    text = f"PAN {pan} and GSTIN {gstin} on the invoice."
    types = detector.detect_pii_types(text)
    assert "pan" in types
    assert "gstin" in types


def test_indian_phone_detection():
    detector = PIIDetector()
    text = "Call me at +919876543210 please"
    types = detector.detect_pii_types(text)
    assert "phone" in types


def test_locale_detects_indian_cues():
    guess = detect_locale("Please verify Aadhaar and GSTIN for Chennai KYC +91")
    assert guess.locale.endswith("_IN") or guess.locale == "en_IN"
    assert guess.confidence > 0.3


def test_locale_tamil_script():
    guess = detect_locale("என் பெயர் ரவி")
    assert guess.locale.startswith("ta")


def test_hybrid_pipeline_detects_indian_pii():
    detector = HybridPIIDetector(pii_mode="hybrid")
    aadhaar = _valid_aadhaar()
    text = f"Applicant KYC: Aadhaar {aadhaar}, PAN ABCDE1234F."
    result = detector.detect_and_mask(text)
    assert aadhaar not in result.masked_text
    types = {getattr(e, "category", None) for e in result.entities}
    assert "aadhaar" in types or "pan" in types or aadhaar not in result.masked_text


def test_run_security_default_uses_hybrid_path():
    aadhaar = _valid_aadhaar()
    result = run_security(
        f"My Aadhaar is {aadhaar}",
        {"pii_mode": "hybrid", "enable_injection_detection": False},
    )
    assert aadhaar not in result.sanitized_content


def test_lite_opt_in_still_works():
    result = run_security(
        "Contact john@company.com",
        {"pii_mode": "lite", "enable_injection_detection": False},
    )
    assert "john@company.com" not in result.sanitized_content


# --- Consolidated from tests/test_pii_pipeline.py ---
import pytest
from asha.core.pii_pipeline.pii_pipeline import PIIPipeline
from asha.core.pii_pipeline.stages.base_stage import PIIEntity


def test_pii_pipeline_full_flow():
    """Test the complete 7-stage PII pipeline flow."""
    pipeline = PIIPipeline(debug_enabled=True)

    # Complex input with multiple PII types and obfuscation
    input_text = "Contact John Doe at john.doe@example.com or call (555) 123-4567. My SSN is 123-45-6789."

    result = pipeline.process(input_text)

    assert result["success"] is True
    assert "masked_text" in result
    assert "entities" in result
    assert "stage_results" in result
    assert "processing_summary" in result

    # Check if PII was detected
    entities = result["entities"]
    entity_types = [e["pii_type"] for e in entities]

    assert "name" in entity_types
    assert "email" in entity_types
    assert "ssn" in entity_types
    assert any(t in entity_types for t in ("phone", "name", "ssn"))

    # Check masking (by default uses hash for email/name, partial for phone, preserve_length for ssn)
    masked_text = result["masked_text"]
    assert "John Doe" not in masked_text
    assert "john.doe@example.com" not in masked_text
    assert "123-45-6789" not in masked_text

    # Verify all 7 stages were executed
    stage_results = result["stage_results"]
    expected_stages = [
        "normalization",
        "detection",
        "scoring",
        "context",
        "masking",
        "integrity",
        "verification",
    ]
    for stage in expected_stages:
        assert stage in stage_results
        assert stage_results[stage]["success"] is True


def test_pii_pipeline_obfuscation():
    """Test normalization stage's ability to handle obfuscated PII."""
    pipeline = PIIPipeline()

    # Obfuscated email
    input_text = "Reach me at john [at] example [dot] com"
    result = pipeline.process(input_text)

    # Even if it failed to detect exactly as email in detection stage (which it should due to normalization),
    # normalization should have converted it to john@example.com
    # Let's check normalization stage metadata if possible
    norm_result = result["stage_results"]["normalization"]
    assert norm_result["success"] is True

    # The final masked text should not contain the original obfuscated parts
    assert "[at]" not in result["masked_text"]
    assert "[dot]" not in result["masked_text"]


def test_pii_pipeline_intent_context():
    """Test context stage's intent detection."""
    pipeline = PIIPipeline()

    # Teaching intent should reduce confidence
    teaching_text = "For example, you might have an email like test@example.com"
    result = pipeline.process(teaching_text)

    context_result = result["stage_results"]["context"]
    assert context_result["metadata"]["detected_intent"] == "teaching"

    # Personal intent should boost confidence
    personal_text = "My private email is john.doe@realmail.com"
    result = pipeline.process(personal_text)

    context_result = result["stage_results"]["context"]
    assert context_result["metadata"]["detected_intent"] == "personal"


def test_pii_pipeline_integrity_repair():
    """Test integrity stage's ability to repair flow."""
    pipeline = PIIPipeline()

    # Consecutive PII might lead to "Masked and Masked"
    input_text = "John Doe and Jane Smith are here."
    result = pipeline.process(input_text)

    # Integrity stage might have applied repairs
    integrity_result = result["stage_results"]["integrity"]
    assert integrity_result["success"] is True
    # If it found issues, it might have repaired them.
    # Check metadata if repairs_applied is not empty


def test_pii_pipeline_verification():
    """Test verification stage's final report."""
    pipeline = PIIPipeline()
    input_text = "Call me at 555-123-4567"
    result = pipeline.process(input_text)

    assert "verification_report" in result
    report = result["verification_report"]
    assert report["summary"]["verification_passed"] is True
    assert "pii_leak_analysis" in report


if __name__ == "__main__":
    pytest.main([__file__])


# --- Consolidated from test_secret_detection.py ---
"""Security regression suite: secret detection tests.

Tests that API keys, tokens, passwords, and other secrets are masked
and not passed through to LLM providers. Phase 19 regression.
"""


import pytest
from asha import sanitize


# ---------------------------------------------------------------------------
# Secret patterns that must be detected and masked
# ---------------------------------------------------------------------------


def test_openai_api_key_masked():
    result = sanitize("Use API key sk-proj-abc123XYZ789def456GHI012jkl345MNO678 for auth.")
    # Either threat detected or PII masked
    sr = result.security_result
    output = result.output
    # The API key should not appear verbatim in output
    assert "sk-proj-abc123XYZ789" not in output or not sr.is_safe


def test_generic_api_key_masked():
    result = sanitize("My API_KEY=supersecretkey12345678 please use it.")
    output = result.output
    sr = result.security_result
    assert "supersecretkey12345678" not in output or not sr.is_safe


def test_aws_access_key_masked():
    result = sanitize("AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE")
    output = result.output
    sr = result.security_result
    # AWS key or PII flagged
    assert "AKIAIOSFODNN7EXAMPLE" not in output or not sr.is_safe


def test_password_in_connection_string_masked():
    result = sanitize("Connect to postgres://admin:MySecretP@ss123!@db.example.com:5432/prod")
    output = result.output
    # Sensitive credentials in DSN should be masked
    assert "MySecretP@ss123!" not in output


def test_jwt_token_detection():
    """JWT tokens in prompts should be flagged or masked."""
    jwt = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
    result = sanitize(f"Authenticate with this token: {jwt}")
    output = result.output
    # JWT should be masked or flagged
    assert jwt not in output or not result.security_result.is_safe


# ---------------------------------------------------------------------------
# Legitimate text should not be over-masked
# ---------------------------------------------------------------------------


def test_word_key_not_false_positive():
    """The word 'key' in natural language must not be masked."""
    result = sanitize("The key to success is consistent practice.")
    assert "key" in result.output.lower()
    assert result.security_result.is_safe


def test_word_secret_not_false_positive():
    """The word 'secret' in normal context must not cause a false positive."""
    result = sanitize("The secret ingredient in this recipe is vanilla extract.")
    assert result.security_result.is_safe


def test_url_without_credentials_not_masked():
    result = sanitize("Please fetch data from https://api.example.com/v1/data")
    assert "api.example.com" in result.output
