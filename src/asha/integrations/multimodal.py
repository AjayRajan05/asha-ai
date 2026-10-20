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
Multimodal content inspection and preprocessing for LLM requests.

Provides inspection hooks and text extraction for image, audio, and file
parts across OpenAI, Anthropic, and Gemini multimodal request payloads.
"""

from __future__ import annotations

import base64
import io
import logging
import urllib.parse
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("asha.multimodal")

# Global registry of custom multimodal preprocessing hooks
# Each hook has signature: hook(part_dict: Dict[str, Any]) -> Optional[str]
_MULTIMODAL_HOOKS: List[Callable[[Dict[str, Any]], Optional[str]]] = []


def register_multimodal_hook(hook: Callable[[Dict[str, Any]], Optional[str]]) -> None:
    """
    Register a custom multimodal extraction or inspection hook.

    Args:
        hook: Callable receiving a multimodal content part dict and returning
            extracted text (or None if unhandled).
    """
    if hook not in _MULTIMODAL_HOOKS:
        _MULTIMODAL_HOOKS.append(hook)


def clear_multimodal_hooks() -> None:
    """Clear all registered multimodal inspection hooks."""
    _MULTIMODAL_HOOKS.clear()


def get_multimodal_hooks() -> List[Callable[[Dict[str, Any]], Optional[str]]]:
    """Return all currently registered multimodal hooks."""
    return list(_MULTIMODAL_HOOKS)


def extract_text_from_multimodal_part(part: Dict[str, Any]) -> Optional[str]:
    """
    Extract text or embedded instructions from a multimodal content block.

    Supports:
        1. Custom registered hooks via ``register_multimodal_hook()``.
        2. Data URIs with embedded image OCR (if Pillow + pytesseract installed).
        3. URL query parameter payloads in ``image_url``.
        4. Base64 encoded image buffers.
    """
    # 1. Check custom registered hooks first
    for hook in _MULTIMODAL_HOOKS:
        try:
            res = hook(part)
            if res and isinstance(res, str):
                return res
        except Exception as exc:
            logger.debug("Custom multimodal hook failed: %s", exc)

    p_type = part.get("type", "")

    # 2. OpenAI-style image_url
    if p_type == "image_url":
        image_url_obj = part.get("image_url", {})
        url = image_url_obj.get("url", "") if isinstance(image_url_obj, dict) else str(image_url_obj)

        # 2a. Check for data URI (base64 image)
        if url.startswith("data:image/"):
            return _ocr_base64_image(url.split(",", 1)[-1])

        # 2b. Check URL query parameters for embedded injection payloads
        try:
            parsed = urllib.parse.urlparse(url)
            params = urllib.parse.parse_qs(parsed.query)
            extracted_params = []
            for k, vals in params.items():
                for v in vals:
                    if len(v) > 10:
                        extracted_params.append(f"{k}: {v}")
            if extracted_params:
                return " ".join(extracted_params)
        except Exception:
            pass

    # 3. Anthropic-style image source: {"type": "image", "source": {"type": "base64", "data": "..."}}
    elif p_type == "image":
        source = part.get("source", {})
        if isinstance(source, dict) and source.get("type") == "base64":
            b64_data = source.get("data", "")
            return _ocr_base64_image(b64_data)

    # 4. OpenAI-style input_audio: {"type": "input_audio", "input_audio": {"data": "...", "format": "wav"}}
    elif p_type == "input_audio":
        audio_obj = part.get("input_audio", {})
        if isinstance(audio_obj, dict):
            # If transcription is available or provided in transcript field
            if "transcript" in audio_obj and isinstance(audio_obj["transcript"], str):
                return audio_obj["transcript"]

    # 5. Gemini inline_data: {"inline_data": {"mime_type": "...", "data": "..."}}
    elif "inline_data" in part:
        inline = part["inline_data"]
        if isinstance(inline, dict) and inline.get("data"):
            return _ocr_base64_image(inline["data"])

    return None


def _ocr_base64_image(b64_data: str) -> Optional[str]:
    """Perform OCR on base64 image data if Pillow and pytesseract are available."""
    try:
        from PIL import Image
        import pytesseract  # type: ignore

        image_bytes = base64.b64decode(b64_data)
        img = Image.open(io.BytesIO(image_bytes))
        text = pytesseract.image_to_string(img)
        return text.strip() if text else None
    except ImportError:
        # Optional [multimodal] dependencies not installed
        return None
    except Exception as exc:
        logger.debug("OCR extraction failed: %s", exc)
        return None


def inspect_and_sanitize_multimodal_part(
    part: Any,
    mode: str = "balanced",
    vault: Optional[Any] = None,
) -> Any:
    """
    Inspect a multimodal content part, validating it against prompt injection and policy.

    Args:
        part: Content part (dict) from request messages or contents.
        mode: Operating mode ('strict', 'balanced', 'lite', 'off').
        vault: Optional session PIIVault.

    Returns:
        Processed part (or raises ASHAProcessingError in strict mode if malicious).
    """
    if not isinstance(part, dict) or mode == "off":
        return part

    extracted_text = extract_text_from_multimodal_part(part)
    if not extracted_text:
        return part

    # Check extracted text for prompt injection and policy violations
    from ..utils.dropin import process
    from ..exceptions import ASHAProcessingError

    processed = process(extracted_text, mode=mode)

    # In strict mode, if threat detected in image/audio text, fail closed
    if mode == "strict" and processed.security and not processed.security.safe:
        threats = processed.security.threats or ["Multimodal injection detected"]
        raise ASHAProcessingError(
            f"Adversarial instruction detected in multimodal payload: {threats}"
        )

    # In balanced mode with vault, if PII is detected in extracted text, record or sanitize
    if vault is not None and getattr(vault, "has_pii", False):
        if hasattr(vault, "mask"):
            vault.mask(extracted_text)

    return part
