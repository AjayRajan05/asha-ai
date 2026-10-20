# Copyright 2026 Ajay Rajan
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Scan LLM request kwargs for file attachments and PII-mask them."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List, Optional

from .coerce import coerce_prompt_input
from .extractors.base import KNOWN_EXTENSIONS


def _mask_text(text: str, mode: str, *, token_budget: int = 1200) -> str:
    from ..utils.dropin import process, _coerce_process_output
    from ..exceptions import ASHAProcessingError

    if mode == "off":
        # Still extract/coerce only; no PII pipeline.
        return text
    processed = process(
        text,
        mode=mode,
        token_budget=token_budget if mode != "strict" else min(token_budget, 800),
    )
    if mode == "strict" and ((processed.security and not processed.security.safe) or not processed.is_safe):
        raise ASHAProcessingError(
            f"Prompt blocked by strict mode security policy: {processed.security.threats if processed.security else 'threat detected'}"
        )
    return _coerce_process_output(processed, text)


def _sanitize_only(text: str, mode: str) -> str:
    from ..utils.dropin import sanitize

    if mode == "off":
        return text
    return sanitize(text, mode=mode).output


def _extract_and_mask(
    source: Any,
    *,
    filename: Optional[str],
    mode: str,
    token_budget: int,
    vault: Optional[Any] = None,
) -> str:
    text = coerce_prompt_input(source, filename=filename)
    if vault is not None:
        masked = vault.mask(text)
        return _mask_text(masked, mode, token_budget=token_budget)
    # Attachments: security-first (PII) via sanitize; keep process for wrap string prompts.
    return _sanitize_only(text, mode)


def _is_path_string(value: str) -> bool:
    try:
        p = Path(value)
        return p.suffix.lower() in KNOWN_EXTENSIONS and p.is_file()
    except OSError:
        return False


def _mask_content_value(
    content: Any,
    *,
    mode: str,
    token_budget: int,
) -> Any:
    """Mask a message content field (str or list of parts)."""
    if isinstance(content, str):
        if _is_path_string(content):
            return _extract_and_mask(
                content,
                filename=Path(content).name,
                mode=mode,
                token_budget=token_budget,
            )
        return _mask_text(content, mode, token_budget=token_budget)

    if isinstance(content, list):
        out: List[Any] = []
        for part in content:
            if not isinstance(part, dict):
                out.append(part)
                continue
            part = dict(part)
            ptype = str(part.get("type") or "").lower()

            # OpenAI-style input_file / file parts with local path or bytes
            if ptype in ("file", "input_file", "document"):
                file_value = part.get("file")
                file_obj: Dict[str, Any] = (
                    file_value if isinstance(file_value, dict) else {}
                )
                filename = (
                    part.get("filename") or part.get("name") or file_obj.get("filename")
                )
                filename_str = str(filename) if filename is not None else None
                file_data = (
                    part.get("data")
                    or part.get("content")
                    or part.get("file_data")
                    or file_obj.get("data")
                )
                path = part.get("path") or part.get("file_path")
                if isinstance(path, str) and _is_path_string(path):
                    masked = _extract_and_mask(
                        path,
                        filename=filename_str or Path(path).name,
                        mode=mode,
                        token_budget=token_budget,
                    )
                    out.append({"type": "text", "text": masked})
                    continue
                if isinstance(file_data, (bytes, bytearray)):
                    masked = _extract_and_mask(
                        bytes(file_data),
                        filename=filename_str,
                        mode=mode,
                        token_budget=token_budget,
                    )
                    out.append({"type": "text", "text": masked})
                    continue
                if isinstance(file_data, str) and _is_path_string(file_data):
                    masked = _extract_and_mask(
                        file_data,
                        filename=filename_str or Path(file_data).name,
                        mode=mode,
                        token_budget=token_budget,
                    )
                    out.append({"type": "text", "text": masked})
                    continue
                # Opaque file_id: cannot read; leave as-is (documented limitation)
                out.append(part)
                continue

            if ptype == "text" or "text" in part:
                text_val = part.get("text")
                if isinstance(text_val, str):
                    if _is_path_string(text_val):
                        part["text"] = _extract_and_mask(
                            text_val,
                            filename=Path(text_val).name,
                            mode=mode,
                            token_budget=token_budget,
                        )
                    else:
                        part["text"] = _mask_text(
                            text_val, mode, token_budget=token_budget
                        )
                out.append(part)
                continue

            out.append(part)
        return out

    return content


def prepare_llm_request_kwargs(
    kwargs: Dict[str, Any],
    *,
    mode: str = "balanced",
    token_budget: int = 1200,
    vault: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Return a copy of LLM call kwargs with prompts and file attachments PII-masked.

    Replaces local file path / bytes attachments with masked text parts so raw
    document bytes are not forwarded to the provider.
    """
    new_kwargs = copy.deepcopy(kwargs)

    # Top-level document-ish keys
    for key in ("file", "files", "document", "documents", "attachments"):
        if key not in new_kwargs:
            continue
        value = new_kwargs[key]
        items = value if isinstance(value, list) else [value]
        masked_texts: List[str] = []
        for item in items:
            if isinstance(item, (str, Path)) or isinstance(item, (bytes, bytearray)):
                name = Path(item).name if isinstance(item, (str, Path)) else None
                masked_texts.append(
                    _extract_and_mask(
                        item,
                        filename=name,
                        mode=mode,
                        token_budget=token_budget,
                        vault=vault,
                    )
                )
            elif isinstance(item, dict):
                path = item.get("path") or item.get("file_path") or item.get("filename")
                data = item.get("data") or item.get("content")
                if isinstance(path, str) and _is_path_string(path):
                    masked_texts.append(
                        _extract_and_mask(
                            path,
                            filename=Path(path).name,
                            mode=mode,
                            token_budget=token_budget,
                            vault=vault,
                        )
                    )
                elif isinstance(data, (bytes, bytearray)):
                    masked_texts.append(
                        _extract_and_mask(
                            bytes(data),
                            filename=str(item.get("filename") or item.get("name") or "")
                            or None,
                            mode=mode,
                            token_budget=token_budget,
                            vault=vault,
                        )
                    )
        if masked_texts:
            # Fold into messages rather than sending raw files
            new_kwargs.pop(key, None)
            doc_block = "\n\n".join(masked_texts)
            if "messages" in new_kwargs and isinstance(new_kwargs["messages"], list):
                new_kwargs["messages"] = list(new_kwargs["messages"]) + [
                    {"role": "user", "content": doc_block}
                ]
            elif "prompt" in new_kwargs and isinstance(new_kwargs["prompt"], str):
                new_kwargs["prompt"] = new_kwargs["prompt"] + "\n\n" + doc_block
            else:
                new_kwargs["prompt"] = doc_block

    if "messages" in new_kwargs and isinstance(new_kwargs["messages"], list):
        messages = []
        for msg in new_kwargs["messages"]:
            if isinstance(msg, dict) and "content" in msg:
                msg = dict(msg)
                if vault is not None:
                    # In vault mode, message is already vault-masked; check strict policy if needed
                    if mode == "strict" and isinstance(msg["content"], str):
                        _mask_text(msg["content"], mode="strict", token_budget=token_budget)
                else:
                    msg["content"] = _mask_content_value(
                        msg["content"], mode=mode, token_budget=token_budget
                    )
            messages.append(msg)
        new_kwargs["messages"] = messages

    for key in ("prompt", "input", "text", "query", "content", "contents"):
        if key in new_kwargs and isinstance(
            new_kwargs[key], (str, Path, bytes, bytearray)
        ):
            val = new_kwargs[key]
            if isinstance(val, str) and not _is_path_string(val):
                if vault is not None:
                    if mode == "strict":
                        _mask_text(val, mode="strict", token_budget=token_budget)
                else:
                    new_kwargs[key] = _mask_text(val, mode, token_budget=token_budget)
            else:
                name = Path(val).name if isinstance(val, (str, Path)) else None
                new_kwargs[key] = _extract_and_mask(
                    val,
                    filename=name,
                    mode=mode,
                    token_budget=token_budget,
                    vault=vault,
                )

    return new_kwargs
