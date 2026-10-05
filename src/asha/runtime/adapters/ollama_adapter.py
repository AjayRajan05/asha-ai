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
Ollama Adapter for ASHA.

Supports:
- Parameterized host / base_url (via constructor and OLLAMA_HOST env)
- Configurable timeouts
- Model selection
- System prompt configuration
- Clean connection failure diagnostics
- Streaming generation generator
"""

from __future__ import annotations

import os
from typing import Any, Dict, Iterator, Optional

import requests

from .base import BaseAdapter


def _normalize_ollama_base_url(base_url: Optional[str] = None) -> str:
    """Normalize Ollama base URL from argument or OLLAMA_HOST environment."""
    url = base_url or os.getenv("OLLAMA_HOST") or "http://localhost:11434"
    url = url.strip()
    if not url.startswith("http://") and not url.startswith("https://"):
        url = f"http://{url}"
    return url.rstrip("/")


class OllamaAdapter(BaseAdapter):
    """
    Ollama adapter for local model inference with configurable endpoint and timeout.
    """

    def __init__(
        self,
        model: str = "llama3",
        *,
        base_url: Optional[str] = None,
        timeout: int = 30,
        system: Optional[str] = None,
    ) -> None:
        """Initialize Ollama adapter.

        Args:
            model: Ollama model name (default: 'llama3').
            base_url: Base URL of Ollama server. Defaults to OLLAMA_HOST env var
                or 'http://localhost:11434'.
            timeout: Request timeout in seconds (default: 30).
            system: Optional default system prompt.
        """
        self.model = model
        self.base_url = _normalize_ollama_base_url(base_url)
        self.timeout = timeout
        self.system = system

    def generate(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        stream: bool = False,
        temperature: Optional[float] = None,
        **kwargs: Any,
    ) -> str:
        """Generate response using Ollama model.

        Args:
            prompt: Input prompt text.
            system: Optional system instruction overriding instance default.
            stream: Whether to stream the response (if True, consumes stream and joins).
            temperature: Optional sampling temperature.
            **kwargs: Extra parameters passed to Ollama API.

        Returns:
            Generated response string.

        Raises:
            requests.RequestException: If connection to Ollama fails.
            KeyError: If response format is invalid.
        """
        url = f"{self.base_url}/api/generate"

        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": stream,
        }
        active_system = system if system is not None else self.system
        if active_system is not None:
            payload["system"] = active_system

        options: Dict[str, Any] = dict(kwargs.get("options", {}))
        if temperature is not None:
            options["temperature"] = temperature
        if options:
            payload["options"] = options

        for k, v in kwargs.items():
            if k not in ("options",):
                payload[k] = v

        try:
            r = requests.post(url, json=payload, timeout=self.timeout)
            r.raise_for_status()
            if not stream:
                data = r.json()
                return str(data.get("response", ""))
            # Consolidate streamed NDJSON chunks
            chunks = []
            for line in r.iter_lines():
                if line:
                    import json

                    chunk = json.loads(line)
                    chunks.append(chunk.get("response", ""))
            return "".join(chunks)
        except requests.RequestException as e:
            raise requests.RequestException(
                f"Failed to connect to Ollama server at {self.base_url}: {e}. "
                "Make sure Ollama is running (e.g., 'ollama serve') and accessible."
            ) from e
        except KeyError as e:
            raise KeyError(
                f"Unexpected response format from Ollama at {self.base_url}: {e}. "
                "Check Ollama server version and compatibility."
            ) from e

    def generate_stream(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """Stream token chunks from Ollama model."""
        url = f"{self.base_url}/api/generate"
        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
        }
        active_system = system if system is not None else self.system
        if active_system is not None:
            payload["system"] = active_system
        if temperature is not None:
            payload.setdefault("options", {})["temperature"] = temperature

        for k, v in kwargs.items():
            if k != "options":
                payload[k] = v

        try:
            r = requests.post(
                url, json=payload, timeout=self.timeout, stream=True
            )
            r.raise_for_status()
            for line in r.iter_lines():
                if line:
                    import json

                    chunk = json.loads(line)
                    if "response" in chunk:
                        yield chunk["response"]
        except requests.RequestException as e:
            raise requests.RequestException(
                f"Failed to stream from Ollama server at {self.base_url}: {e}"
            ) from e

    def get_provider_info(self) -> Dict[str, Any]:
        """Get provider information."""
        return {
            "provider": "ollama",
            "model": self.model,
            "base_url": self.base_url,
            "timeout": self.timeout,
            "supports_streaming": True,
            "supports_system_instruction": True,
        }
