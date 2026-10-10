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
from typing import List, Optional

from .base import BaseAdapter
from .universal_adapter import UniversalModelAdapter


class OpenAIAdapter(BaseAdapter):
    """
    OpenAI Adapter - Backward compatible wrapper using UniversalModelAdapter.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 10,
    ) -> None:
        """Initialize OpenAI adapter with specified model.

        Args:
            model: OpenAI model name (default: 'gpt-4o-mini')
            api_key: Optional API key. Defaults to OPENAI_API_KEY env var.
            base_url: Optional base URL for OpenAI-compatible endpoints (e.g. llama.cpp server).
            timeout: Request timeout in seconds.
        """
        key = (
            api_key
            or os.getenv("OPENAI_API_KEY")
            or ("llama-cpp" if base_url else None)
        )
        if not key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is not set. "
                "Set it with: export OPENAI_API_KEY='your-key'"
            )

        self.model = model
        self.base_url = base_url
        self._adapter = UniversalModelAdapter(
            provider="openai",
            model=model,
            api_key=key,
            base_url=base_url,
            timeout=timeout,
        )
        self.model = model

    def generate(self, prompt: str) -> str:
        """Generate response using OpenAI model.

        Args:
            prompt: Input prompt for OpenAI model

        Returns:
            Generated response text

        Raises:
            Exception: If API call fails
            ValueError: If API key not set
        """
        return self._adapter.generate(prompt)

    # Additional v2 methods exposed for compatibility
    def generate_with_fallback(
        self, prompt: str, fallback_models: Optional[List[BaseAdapter]] = None
    ) -> str:
        """Generate with fallback support.

        Args:
            prompt: Input prompt
            fallback_models: List of fallback model configurations

        Returns:
            Generated response with fallback if primary fails
        """
        return self._adapter.generate_with_fallback(prompt, fallback_models)

    def get_provider_info(self) -> dict:
        """Get provider information.

        Returns:
            Dictionary with provider details including model, capabilities, etc.
        """
        return self._adapter.get_provider_info()
