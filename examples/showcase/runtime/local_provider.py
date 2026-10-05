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

"""Deterministic local demo provider for offline/no-key usage in ASHA Control Center."""

from __future__ import annotations

import time
from typing import Any, Dict, Optional


class LocalDemoProvider:
    """
    Offline deterministic demonstration provider.

    IMPORTANT:
    This provider is explicitly labeled as '[LOCAL / DEMO PROVIDER]' and does not
    simulate or pretend to be OpenAI, Claude, or Gemini. It allows developers to
    test and experience the real ASHA security and optimization pipeline without
    requiring external commercial cloud credentials.
    """

    def __init__(self, model: str = "asha-local-deterministic-v1") -> None:
        self.model = model
        self.provider_name = "LOCAL / DEMO PROVIDER"

    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Generate a deterministic completion based on the sanitized/protected input."""
        start_t = time.perf_counter()
        # Clean deterministic response summarizing reception
        length = len(prompt)
        words = len(prompt.split())

        preview = prompt[:120] + ("..." if len(prompt) > 120 else "")
        response_text = (
            f"[LOCAL / DEMO PROVIDER]\n\n"
            f"Processed prompt received by mock downstream inference engine ({words} words, {length} chars):\n"
            f"\"{preview}\"\n\n"
            f"Summary response: Request received successfully via ASHA protection layer. "
            f"No external API keys were transmitted."
        )

        latency_ms = (time.perf_counter() - start_t) * 1000.0

        return {
            "text": response_text,
            "provider": self.provider_name,
            "model": self.model,
            "latency_ms": max(0.5, latency_ms),
            "usage": {
                "prompt_tokens": words,
                "completion_tokens": len(response_text.split()),
                "total_tokens": words + len(response_text.split()),
            },
            "system_instruction": system_instruction,
            "offline": True,
        }
