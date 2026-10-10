# Copyright 2026 Ajay Rajan
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Extractor registry and extract_text entry point."""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO, Optional, Union

from ..types import ExtractedDocument

TEXT_EXTENSIONS = frozenset({".txt", ".md", ".markdown", ".csv", ".html", ".htm"})
PDF_EXTENSIONS = frozenset({".pdf"})
DOCX_EXTENSIONS = frozenset({".docx"})
KNOWN_EXTENSIONS = TEXT_EXTENSIONS | PDF_EXTENSIONS | DOCX_EXTENSIONS

# PDF magic / ZIP (docx is a zip)
_PDF_MAGIC = b"%PDF"
_ZIP_MAGIC = b"PK"


def _ext_of(name: Optional[str]) -> str:
    if not name:
        return ""
    return Path(name).suffix.lower()


def _read_bytes(source: Union[Path, bytes, BinaryIO]) -> bytes:
    if isinstance(source, bytes):
        return source
    if isinstance(source, Path):
        return source.read_bytes()
    # Typed as bytes for BinaryIO, but some text-mode file objects return str.
    data: object = source.read()
    if isinstance(data, str):
        return data.encode("utf-8", errors="replace")
    if isinstance(data, (bytes, bytearray)):
        return bytes(data)
    raise TypeError(f"Unsupported file payload type: {type(data)!r}")


def _detect_format(data: bytes, filename: Optional[str] = None) -> str:
    ext = _ext_of(filename)
    if ext in TEXT_EXTENSIONS:
        return "plaintext"
    if ext in PDF_EXTENSIONS or data.startswith(_PDF_MAGIC):
        return "pdf"
    if ext in DOCX_EXTENSIONS or data.startswith(_ZIP_MAGIC):
        # Heuristic: treat zip as docx when extension says so; otherwise try docx
        if ext in DOCX_EXTENSIONS or b"word/" in data[:8000]:
            return "docx"
        if ext == ".docx":
            return "docx"
    if ext in KNOWN_EXTENSIONS:
        return ext.lstrip(".")
    # Default: try plaintext decode
    return "plaintext"


def extract_text(
    source: Union[str, Path, bytes, BinaryIO],
    *,
    filename: Optional[str] = None,
) -> ExtractedDocument:
    """
    Extract plain text from a supported document.

    Supported: ``.txt``, ``.md``, ``.csv``, ``.html``, ``.pdf``, ``.docx``.
    No OCR: scanned/image-only PDFs return empty or sparse text.
    """
    path: Optional[Path] = None
    if isinstance(source, str):
        path = Path(source)
        if path.is_file():
            data = path.read_bytes()
            filename = filename or path.name
        else:
            raise FileNotFoundError(f"Document not found: {source}")
    elif isinstance(source, Path):
        path = source
        if not path.is_file():
            raise FileNotFoundError(f"Document not found: {source}")
        data = path.read_bytes()
        filename = filename or path.name
    else:
        data = _read_bytes(source)

    fmt = _detect_format(data, filename)
    source_label = str(path) if path is not None else filename

    if fmt == "pdf":
        from .pdf import extract_pdf

        return extract_pdf(data, source=source_label)
    if fmt == "docx":
        from .docx import extract_docx

        return extract_docx(data, source=source_label)

    from .plaintext import extract_plaintext

    return extract_plaintext(data, source=source_label, format_hint=fmt)
