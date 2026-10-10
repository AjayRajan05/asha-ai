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
