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
Universal Google Gemini Adapter for ASHA.

Supports all Google Gemini models across both the modern Google GenAI SDK
(`google-genai`) and legacy `google-generativeai`.

Supports:
- All Google Gemini models (Gemini 1.5 Flash/Pro, Gemini 2.0 Flash/Pro, Gemini 2.5,
  Gemini 1.0 Pro, Gemini Ultra, experimental versions, and tuned models)
- Modern Google GenAI SDK (google-genai, `from google import genai`)
- Backwards-compatible fallback to legacy google-generativeai
- System instructions, generation configuration
- Safe response handling (safety blocks do not crash on .text)
- Usage metadata extraction
- Async generation (generate_async)
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from .base import BaseAdapter


class GeminiAdapter(BaseAdapter):
    """
    Universal Google Gemini Adapter supporting modern google-genai and legacy google-generativeai.

    Works with any Gemini model family (1.0, 1.5, 2.0, 2.5, Pro, Flash,
    Ultra, experimental, and fine-tuned models).
    """

    def __init__(
        self,
        model: Optional[str] = None,
        *,
        api_key: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> None:
        """Initialize Gemini adapter for any Google Gemini model.

        Args:
            model: Any Gemini model identifier (e.g., 'gemini-1.5-flash',
                'gemini-1.5-pro', 'gemini-2.0-flash', 'gemini-2.5-flash',
                'gemini-pro', or custom/tuned models).
                If omitted, reads GEMINI_MODEL from the environment, defaulting
                to 'gemini-1.5-flash'.
            api_key: Optional API key. If not provided, reads GOOGLE_API_KEY
                or GEMINI_API_KEY from the environment.
            timeout: Optional request timeout in seconds.
        """
        raw_model = model or os.getenv("GEMINI_MODEL") or "gemini-1.5-flash"
        # Normalize model identifier if passed with 'models/' prefix
        self.model = (
            raw_model.removeprefix("models/")
            if raw_model.startswith("models/")
            else raw_model
        )
        self.timeout = timeout
        self.api_key = (
            api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        )
        if not self.api_key:
            raise ValueError(
                "GOOGLE_API_KEY or GEMINI_API_KEY environment variable is not set. "
                "Set it with: export GOOGLE_API_KEY='your-key'"
            )

        self._last_usage: Optional[Dict[str, int]] = None
        self._is_genai_sdk = False
        self._client = self._init_client()
        self._adapter = type("_AdapterCompat", (), {"client": self._client})()

    def _init_client(self) -> Any:
        """Initialize modern google-genai client or fallback to google.generativeai."""
        try:
            import importlib

            genai = importlib.import_module("google.genai")
            client = genai.Client(api_key=self.api_key)
            self._is_genai_sdk = True
            return client
        except ImportError:
            pass

        try:
            import google.generativeai as genai_legacy

            genai_legacy.configure(api_key=self.api_key)
            self._is_genai_sdk = False
            return genai_legacy
        except ImportError:
            raise ImportError(
                "Google GenAI library not installed. Install with: pip install google-genai"
            )

    @property
    def is_modern_sdk(self) -> bool:
        """Return True if using modern google-genai SDK."""
        return self._is_genai_sdk

    def generate(
        self,
        prompt: str,
        *,
        system_instruction: Optional[str] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        """Generate response using Gemini model.

        Args:
            prompt: Input prompt text.
            system_instruction: Optional system instruction for the model.
            temperature: Sampling temperature.
            max_output_tokens: Maximum tokens in response.
            **kwargs: Extra model generation arguments.

        Returns:
            Generated response text, or a safety notice if blocked by provider filters.
        """
        if self._is_genai_sdk:
            return self._generate_modern(
                prompt,
                system_instruction=system_instruction,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                **kwargs,
            )
        return self._generate_legacy(
            prompt,
            system_instruction=system_instruction,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            **kwargs,
        )

    def _generate_modern(
        self,
        prompt: str,
        *,
        system_instruction: Optional[str] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        config: Dict[str, Any] = dict(kwargs)
        if system_instruction is not None:
            config["system_instruction"] = system_instruction
        if temperature is not None:
            config["temperature"] = temperature
        if max_output_tokens is not None:
            config["max_output_tokens"] = max_output_tokens

        try:
            response = self._client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=config if config else None,
            )
        except Exception as exc:
            raise RuntimeError(f"Gemini generation failed: {exc}") from exc

        self._extract_modern_usage(response)
        return self._safe_extract_text(response)

    def _generate_legacy(
        self,
        prompt: str,
        *,
        system_instruction: Optional[str] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        model_kwargs: Dict[str, Any] = {}
        if system_instruction is not None:
            model_kwargs["system_instruction"] = system_instruction

        client = getattr(self._adapter, "client", self._client)
        model = client.GenerativeModel(self.model, **model_kwargs)
        gen_config: Dict[str, Any] = dict(kwargs)
        if temperature is not None:
            gen_config["temperature"] = temperature
        if max_output_tokens is not None:
            gen_config["max_output_tokens"] = max_output_tokens

        try:
            response = model.generate_content(
                prompt,
                generation_config=gen_config if gen_config else None,
            )
        except Exception as exc:
            raise RuntimeError(f"Gemini generation failed: {exc}") from exc

        self._extract_legacy_usage(response)
        return self._safe_extract_text(response)

    async def generate_async(
        self,
        prompt: str,
        *,
        system_instruction: Optional[str] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        """Asynchronously generate response using Gemini model."""
        if self._is_genai_sdk:
            config: Dict[str, Any] = dict(kwargs)
            if system_instruction is not None:
                config["system_instruction"] = system_instruction
            if temperature is not None:
                config["temperature"] = temperature
            if max_output_tokens is not None:
                config["max_output_tokens"] = max_output_tokens

            try:
                response = await self._client.aio.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=config if config else None,
                )
            except Exception as exc:
                raise RuntimeError(f"Gemini async generation failed: {exc}") from exc

            self._extract_modern_usage(response)
            return self._safe_extract_text(response)

        # Legacy async
        model_kwargs: Dict[str, Any] = {}
        if system_instruction is not None:
            model_kwargs["system_instruction"] = system_instruction
        model = self._client.GenerativeModel(self.model, **model_kwargs)
        gen_config: Dict[str, Any] = dict(kwargs)
        if temperature is not None:
            gen_config["temperature"] = temperature
        if max_output_tokens is not None:
            gen_config["max_output_tokens"] = max_output_tokens

        try:
            response = await model.generate_content_async(
                prompt,
                generation_config=gen_config if gen_config else None,
            )
        except Exception as exc:
            raise RuntimeError(f"Gemini async generation failed: {exc}") from exc

        self._extract_legacy_usage(response)
        return self._safe_extract_text(response)

    def _safe_extract_text(self, response: Any) -> str:
        """Safely extract text from response without crashing on safety filters."""
        # Check standard .text attribute
        try:
            text = getattr(response, "text", None)
            if text is not None:
                return str(text)
        except (ValueError, AttributeError):
            pass

        # Inspect candidates for finish reason / safety block
        candidates = getattr(response, "candidates", None) or []
        if candidates:
            c0 = candidates[0]
            finish_reason = getattr(c0, "finish_reason", None)
            if str(finish_reason).upper() in ("SAFETY", "2", "FINISH_REASON_SAFETY"):
                return "[BLOCKED_BY_SAFETY_FILTERS]"
            content = getattr(c0, "content", None)
            if content and hasattr(content, "parts"):
                parts_text = [
                    p.text for p in content.parts if hasattr(p, "text") and p.text
                ]
                if parts_text:
                    return "".join(parts_text)

        # Inspect prompt feedback
        feedback = getattr(response, "prompt_feedback", None)
        if feedback and getattr(feedback, "block_reason", None):
            return "[BLOCKED_BY_SAFETY_FILTERS]"

        return ""

    def _extract_modern_usage(self, response: Any) -> None:
        usage = getattr(response, "usage_metadata", None)
        if usage is not None:
            self._last_usage = {
                "prompt_tokens": getattr(usage, "prompt_token_count", 0) or 0,
                "candidates_tokens": getattr(usage, "candidates_token_count", 0) or 0,
                "total_tokens": getattr(usage, "total_token_count", 0) or 0,
            }
        else:
            self._last_usage = None

    def _extract_legacy_usage(self, response: Any) -> None:
        usage = getattr(response, "usage_metadata", None)
        if usage is not None:
            self._last_usage = {
                "prompt_tokens": getattr(usage, "prompt_token_count", 0) or 0,
                "candidates_tokens": getattr(usage, "candidates_token_count", 0) or 0,
                "total_tokens": getattr(usage, "total_token_count", 0) or 0,
            }
        else:
            self._last_usage = None

    def get_last_usage(self) -> Optional[Dict[str, int]]:
        """Return token usage metadata from the most recent request."""
        return self._last_usage

    def generate_with_fallback(
        self, prompt: str, fallback_models: Optional[List[BaseAdapter]] = None
    ) -> str:
        """Generate with fallback support."""
        try:
            return self.generate(prompt)
        except Exception:
            if fallback_models:
                for fallback in fallback_models:
                    try:
                        return fallback.generate(prompt)
                    except Exception:
                        continue
            raise

    def get_provider_info(self) -> Dict[str, Any]:
        """Get provider information."""
        return {
            "provider": "gemini",
            "model": self.model,
            "sdk": "google-genai" if self._is_genai_sdk else "google-generativeai",
            "supports_async": True,
            "supports_system_instruction": True,
            "supports_usage_metadata": True,
        }
