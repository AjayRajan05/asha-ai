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

import os
import time
import logging
from typing import List, Dict, Any, Optional
from .base import BaseAdapter


class UniversalModelAdapter(BaseAdapter):
    """
    Universal Model Gateway that supports multiple LLM providers
    with retry logic and fallback capabilities.
    """

    def __init__(
        self,
        provider: str,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = 10,
        retries: int = 3,
        base_url: Optional[str] = None,
    ) -> None:
        """
        Initialize universal adapter with provider-specific configuration.

        Args:
            provider: Provider name ("openai", "anthropic", "grok", "huggingface")
            api_key: API key (if None, will try environment variables)
            model: Model name
            timeout: Request timeout in seconds
            retries: Number of retry attempts
            base_url: Optional base URL for OpenAI-compatible endpoints (e.g. llama.cpp)
        """
        self.provider = provider.lower()
        self.model = model
        self.timeout = timeout
        self.retries = retries
        self.base_url = base_url
        self.logger = logging.getLogger(__name__)

        # Initialize provider-specific client
        self.client = self._initialize_client(api_key)
        self._fallback_adapters: Optional[List[BaseAdapter]] = None

        # Validate configuration
        self._validate_config()

    def _initialize_client(self, api_key: Optional[str] = None) -> Any:
        """Initialize provider-specific client."""
        if self.provider == "openai":
            return self._init_openai_client(api_key)
        elif self.provider == "anthropic":
            return self._init_anthropic_client(api_key)
        elif self.provider == "grok":
            return self._init_grok_client(api_key)
        elif self.provider == "gemini":
            return self._init_gemini_client(api_key)
        elif self.provider == "huggingface":
            return self._init_hf_client(api_key)
        elif self.provider == "mock":
            return self._init_mock_client()
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _init_openai_client(self, api_key: Optional[str] = None) -> Any:
        """Initialize OpenAI client."""
        try:
            from openai import OpenAI

            key = (
                api_key
                or os.getenv("OPENAI_API_KEY")
                or ("llama-cpp" if self.base_url else None)
            )
            if not key:
                raise ValueError("OpenAI API key not found")
            client_kwargs: Dict[str, Any] = {"api_key": key}
            if self.base_url:
                client_kwargs["base_url"] = self.base_url
            return OpenAI(**client_kwargs)
        except ImportError:
            raise ImportError(
                "OpenAI library not installed. Install with: pip install openai"
            )

    def _init_anthropic_client(self, api_key: Optional[str] = None) -> Any:
        """Initialize Anthropic client."""
        try:
            import anthropic

            key = api_key or os.getenv("ANTHROPIC_API_KEY")
            if not key:
                raise ValueError("Anthropic API key not found")
            return anthropic.Anthropic(api_key=key)
        except ImportError:
            raise ImportError(
                "Anthropic library not installed. Install with: pip install anthropic"
            )

    def _init_grok_client(self, api_key: Optional[str] = None) -> Any:
        """Initialize Grok (xAI) client using OpenAI-compatible interface."""
        try:
            from openai import OpenAI

            key = api_key or os.getenv("GROK_API_KEY")
            if not key:
                raise ValueError("Grok API key not found")
            return OpenAI(api_key=key, base_url="https://api.x.ai/v1")
        except ImportError:
            raise ImportError(
                "OpenAI library not installed. Install with: pip install openai"
            )

    def _init_gemini_client(self, api_key: Optional[str] = None) -> Any:
        """Initialize Google Gemini client."""
        key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not key:
            raise ValueError("Google API key not found")
        try:
            from google import genai

            return genai.Client(api_key=key)
        except ImportError:
            pass

        try:
            import google.generativeai as genai_legacy

            genai_legacy.configure(api_key=key)
            return genai_legacy
        except ImportError:
            raise ImportError(
                "Google Generative AI library not installed. Install with: pip install google-genai"
            )

    def _init_hf_client(self, api_key: Optional[str] = None) -> Any:
        """Initialize HuggingFace client."""
        try:
            from transformers import pipeline

            key = api_key or os.getenv("HUGGINGFACE_API_KEY")
            # For HF, we'll use the pipeline interface
            return {"pipeline": pipeline, "api_key": key}
        except ImportError:
            raise ImportError(
                "Transformers library not installed. Install with: pip install transformers"
            )

    def _init_mock_client(self) -> Dict[str, str]:
        """Initialize mock client for testing."""
        return {"type": "mock", "client": "mock"}

    def _validate_config(self) -> None:
        """Validate provider and model configuration."""
        if self.provider == "openai" and not self.model:
            self.model = "gpt-4o-mini"
        elif self.provider == "anthropic" and not self.model:
            self.model = "claude-3-sonnet-20240229"
        elif self.provider == "grok" and not self.model:
            self.model = "grok-beta"
        elif self.provider == "gemini" and not self.model:
            self.model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        elif self.provider == "huggingface" and not self.model:
            self.model = "microsoft/DialoGPT-medium"
        elif self.provider == "mock" and not self.model:
            self.model = "mock"

    def generate(self, prompt: str) -> str:
        """
        Generate response with retry logic.

        Args:
            prompt: Input prompt

        Returns:
            Generated response text
        """
        for attempt in range(self.retries + 1):
            try:
                return self._generate_with_provider(prompt)
            except Exception as e:
                if attempt == self.retries:
                    self.logger.error(
                        f"All {self.retries + 1} attempts failed for {self.provider}: {e}"
                    )
                    raise
                wait_time = 2**attempt  # Exponential backoff
                self.logger.warning(
                    f"Attempt {attempt + 1} failed, retrying in {wait_time}s: {e}"
                )
                time.sleep(wait_time)
        raise RuntimeError("All retry attempts failed")

    def _generate_with_provider(self, prompt: str) -> str:
        """Generate response using provider-specific implementation."""
        if self.provider == "openai":
            return self._openai_generate(prompt)
        elif self.provider == "anthropic":
            return self._anthropic_generate(prompt)
        elif self.provider == "grok":
            return self._grok_generate(prompt)
        elif self.provider == "gemini":
            return self._gemini_generate(prompt)
        elif self.provider == "huggingface":
            return self._hf_generate(prompt)
        elif self.provider == "mock":
            return self._mock_generate(prompt)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _openai_generate(self, prompt: str) -> str:
        """Generate using OpenAI."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            timeout=self.timeout,
        )
        content = response.choices[0].message.content
        return str(content) if content is not None else ""

    def _anthropic_generate(self, prompt: str) -> str:
        """Generate using Anthropic Claude."""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
            timeout=self.timeout,
        )
        return str(response.content[0].text)

    def _grok_generate(self, prompt: str) -> str:
        """Generate using Grok (xAI) - OpenAI compatible."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            timeout=self.timeout,
        )
        content = response.choices[0].message.content
        return str(content) if content is not None else ""

    def _gemini_generate(self, prompt: str) -> str:
        """Generate using Google Gemini."""
        if hasattr(self.client, "models"):
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
            )
            try:
                return str(response.text)
            except Exception:
                return ""
        model = self.client.GenerativeModel(self.model)
        response = model.generate_content(prompt)
        try:
            return str(response.text)
        except Exception:
            return ""

    def _hf_generate(self, prompt: str) -> str:
        """Generate using HuggingFace pipeline."""
        # Note: This is a simplified implementation
        # In production, you'd want to handle different model types appropriately
        try:
            generator = self.client["pipeline"]("text-generation", model=self.model)
            result = generator(prompt, max_length=100, num_return_sequences=1)
            return str(result[0]["generated_text"])
        except Exception as e:
            raise RuntimeError(f"HuggingFace generation failed: {e}")

    def _mock_generate(self, prompt: str) -> str:
        """Generate using mock for testing."""
        return f"Mock response to: {prompt[:50]}{'...' if len(prompt) > 50 else ''}"

    def generate_with_fallback(
        self, prompt: str, fallback_models: Optional[List[BaseAdapter]] = None
    ) -> str:
        """
        Generate with fallback to alternative models.

        Args:
            prompt: Input prompt
            fallback_models: List of fallback adapters

        Returns:
            Generated response text
        """
        try:
            return self.generate(prompt)
        except Exception as primary_error:
            self.logger.warning(
                f"Primary provider {self.provider} failed: {primary_error}"
            )

            if fallback_models:
                for i, fallback in enumerate(fallback_models):
                    try:
                        self.logger.info(f"Trying fallback model {i+1}")
                        return fallback.generate(prompt)
                    except Exception as fallback_error:
                        self.logger.warning(f"Fallback {i+1} failed: {fallback_error}")
                        continue

            raise Exception("All providers failed")

    def get_provider_info(self) -> Dict[str, Any]:
        """Get provider information."""
        return {
            "provider": self.provider,
            "model": self.model,
            "timeout": self.timeout,
            "retries": self.retries,
        }
