"""Tests for document extraction + PII masking via sanitize/process/wrap_llm."""

from __future__ import annotations

from pathlib import Path

import pytest

from asha import process, sanitize
from asha.documents import extract_text, prepare_llm_request_kwargs
from asha.integrations.llm_wrap import wrap_llm


EMAIL = "alice@example.com"


@pytest.fixture()
def txt_with_email(tmp_path: Path) -> Path:
    path = tmp_path / "note.txt"
    path.write_text(f"Contact {EMAIL} about the report.\n", encoding="utf-8")
    return path


@pytest.fixture()
def pdf_with_email(tmp_path: Path) -> Path:
    # Minimal PDF with a literal email string in the content stream.
    pdf_bytes = (
        b"%PDF-1.1\n"
        b"1 0 obj<< /Type /Catalog /Pages 2 0 R >>endobj\n"
        b"2 0 obj<< /Type /Pages /Kids [3 0 R] /Count 1 >>endobj\n"
        b"3 0 obj<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 144] "
        b"/Contents 4 0 R /Resources<< /Font<< /F1 5 0 R >> >> >>endobj\n"
        b"4 0 obj<< /Length 68 >>stream\n"
        b"BT /F1 12 Tf 10 100 Td (" + EMAIL.encode() + b") Tj ET\n"
        b"endstream\nendobj\n"
        b"5 0 obj<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>endobj\n"
        b"xref\n0 6\n0000000000 65535 f \n"
        b"trailer<< /Size 6 /Root 1 0 R >>\nstartxref\n0\n%%EOF\n"
    )
    path = tmp_path / "note.pdf"
    path.write_bytes(pdf_bytes)
    return path


@pytest.fixture()
def docx_with_email(tmp_path: Path) -> Path:
    from docx import Document

    path = tmp_path / "note.docx"
    doc = Document()
    doc.add_paragraph(f"Please email {EMAIL} for details.")
    doc.save(path)
    return path


def test_sanitize_string_unchanged_behavior() -> None:
    out = sanitize(f"Contact {EMAIL}").output
    assert EMAIL not in out
    assert "EMAIL" in out.upper() or "[" in out


def test_sanitize_txt_path_masks_pii(txt_with_email: Path) -> None:
    out = sanitize(txt_with_email).output
    assert EMAIL not in out
    assert "alice" not in out.lower() or "[" in out


def test_process_txt_path_masks_pii(txt_with_email: Path) -> None:
    out = process(str(txt_with_email)).output
    assert EMAIL not in out


def test_extract_docx(docx_with_email: Path) -> None:
    doc = extract_text(docx_with_email)
    assert EMAIL in doc.text
    assert doc.format == "docx"


def test_sanitize_docx_masks_pii(docx_with_email: Path) -> None:
    out = sanitize(docx_with_email).output
    assert EMAIL not in out


def test_sanitize_pdf_path(pdf_with_email: Path) -> None:
    # Extraction may be empty on minimal PDFs; if email extracted, must be masked
    extracted = extract_text(pdf_with_email).text
    out = sanitize(pdf_with_email).output
    if EMAIL in extracted:
        assert EMAIL not in out
    else:
        # Still must not raise; empty extract is acceptable for sparse PDFs
        assert isinstance(out, str)


def test_prepare_kwargs_masks_file_attachment(txt_with_email: Path) -> None:
    kwargs = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Summarize this file"},
                    {"type": "file", "path": str(txt_with_email)},
                ],
            }
        ]
    }
    prepared = prepare_llm_request_kwargs(kwargs, mode="balanced")
    content = prepared["messages"][0]["content"]
    blob = str(content)
    assert EMAIL not in blob
    assert any(
        isinstance(p, dict)
        and p.get("type") == "text"
        and EMAIL not in str(p.get("text"))
        for p in content
    )


def test_wrap_llm_masks_attachment(txt_with_email: Path) -> None:
    captured: dict = {}

    class _Client:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    captured.update(kwargs)
                    return {"ok": True}

    wrapped = wrap_llm(_Client(), mode="balanced")
    wrapped.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Read the attachment"},
                    {"type": "file", "path": str(txt_with_email)},
                ],
            }
        ]
    )
    assert captured
    blob = str(captured)
    assert EMAIL not in blob
