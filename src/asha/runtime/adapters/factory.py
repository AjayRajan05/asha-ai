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

from .universal_adapter import UniversalModelAdapter
from .base import BaseAdapter
from typing import Any, Dict, List, Optional, cast
import os


class AdapterFactory:
    """
    Complete Adapter Factory with v1 compatibility and v2 enhanced functionality.

    This factory provides both legacy v1 methods and advanced v2 features
    in a single unified interface.
    """

    @staticmethod
    def create(model_name: str) -> Any:
        """
        Create adapter based on model name pattern (v1 compatibility).

        Args:
            model_name: Model identifier or provider prefix

        Returns:
            Appropriate adapter instance

        Raises:
            ValueError: If provider is not supported
            ImportError: If required dependencies are missing

        Examples:
            - "mock" → MockAdapter
            - "gpt-4", "gpt-3.5-turbo" → OpenAIAdapter
            - "claude-3-sonnet", "claude-3-opus" → ClaudeAdapter
            - "mistralai/Mistral-7B", "microsoft/DialoGPT" → HuggingFaceAdapter
            - "llama2", "codellama" → OllamaAdapter
        """

        if model_name.lower() == "mock":
            from .mock_adapter import MockAdapter

            return MockAdapter(model_name)
        elif model_name.lower().startswith("gpt"):
            from .openai_adapter import OpenAIAdapter

            return OpenAIAdapter(model_name)
        elif model_name.lower().startswith("claude"):
            from .claude_adapter import ClaudeAdapter

            return ClaudeAdapter(model_name)
        elif model_name.lower().startswith("grok"):
            from .grok_adapter import GrokAdapter

            return GrokAdapter(model_name)
        elif (
            model_name.lower().startswith("gemini")
            or "gemini" in model_name.lower()
        ):
            from .gemini_adapter import GeminiAdapter

            return GeminiAdapter(model_name)
        elif model_name.lower().startswith("mistralai") or "/" in model_name:
            from .hf_adapter import HuggingFaceAdapter

            return HuggingFaceAdapter(model_name)
        elif (
            model_name.lower().startswith("llama.cpp")
            or model_name.lower().startswith("llama-cpp")
            or model_name.lower().startswith("llamacpp")
        ):
            from .openai_adapter import OpenAIAdapter

            base_url = os.getenv("LLAMA_CPP_BASE_URL", "http://localhost:8080/v1")
            return OpenAIAdapter(model=model_name, base_url=base_url)
        elif model_name.lower().startswith("llama") or model_name.lower().startswith(
            "codellama"
        ):
            from .ollama_adapter import OllamaAdapter

            return OllamaAdapter(model_name)
        else:
            raise ValueError(
                f"Unsupported provider/model: {model_name}. Supported providers: mock, gpt, claude, grok, gemini, mistralai, llama, llama.cpp, codellama"
            )

    # Enhanced v2 methods (consolidated from enhanced_factory.py)
    @staticmethod
    def create_universal(
        provider: str,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: int = 10,
        retries: int = 3,
    ) -> UniversalModelAdapter:
        """
        Create a universal adapter for specified provider.

        Args:
            provider: Provider name ("openai", "anthropic", "grok", "huggingface", "mock")
            model: Model name (will use defaults if None)
            api_key: API key (will use environment if None)
            timeout: Request timeout
            retries: Number of retry attempts

        Returns:
            UniversalModelAdapter instance
        """
        return UniversalModelAdapter(
            provider=provider,
            api_key=api_key,
            model=model,
            timeout=timeout,
            retries=retries,
        )

    @staticmethod
    def create_with_fallbacks(
        primary_provider: str,
        primary_model: Optional[str] = None,
        fallback_providers: Optional[List[Dict[str, Any]]] = None,
    ) -> UniversalModelAdapter:
        """
        Create adapter with fallback configuration.

        Args:
            primary_provider: Primary provider name
            primary_model: Primary model name
            fallback_providers: List of fallback configs [{"provider": "openai", "model": "gpt-3.5-turbo"}]

        Returns:
            UniversalModelAdapter with fallback adapters
        """
        primary = AdapterFactory.create_universal(primary_provider, primary_model)

        if fallback_providers:
            fallback_adapters = []
            for fallback_config in fallback_providers:
                fallback = AdapterFactory.create_universal(
                    provider=fallback_config["provider"],
                    model=fallback_config.get("model"),
                )
                fallback_adapters.append(fallback)

            # Store fallbacks on the primary adapter
            primary._fallback_adapters = cast(List[BaseAdapter], fallback_adapters)

        return primary

    @staticmethod
    def create_from_env() -> UniversalModelAdapter:
        """
        Create adapter from environment configuration.

        Environment variables:
        - ASHA_PROVIDER: Provider name
        - ASHA_MODEL: Model name
        - ASHA_TIMEOUT: Timeout (default 10)
        - ASHA_RETRIES: Retries (default 3)

        Returns:
            UniversalModelAdapter configured from environment
        """
        provider = os.getenv("ASHA_PROVIDER", "openai")
        model = os.getenv("ASHA_MODEL")
        timeout = int(os.getenv("ASHA_TIMEOUT", "10"))
        retries = int(os.getenv("ASHA_RETRIES", "3"))

        return AdapterFactory.create_universal(
            provider=provider, model=model, timeout=timeout, retries=retries
        )

    @staticmethod
    def create_smart_routing(routing_config: dict) -> "SmartRoutingAdapter":
        """
        Create smart routing adapter based on task types.

        Args:
            routing_config: Dict mapping task types to models
                         {"analysis": "claude-3", "chat": "gpt-4o-mini", "bulk": "llama3"}

        Returns:
            SmartRoutingAdapter instance
        """
        return SmartRoutingAdapter(routing_config)


class SmartRoutingAdapter:
    """
    Smart routing adapter that automatically selects the best model
    based on task type, cost, latency, and complexity.

    Experimental in 0.4.2; emits ``ASHAExperimentalWarning``.
    """

    def __init__(self, routing_config: Dict[str, str]) -> None:
        from asha.exceptions import warn_experimental

        warn_experimental("SmartRoutingAdapter / Agent(routing_config=...)")
        self.routing_config = routing_config
        self.adapters: Dict[str, UniversalModelAdapter] = {}

        # Initialize adapters for each model in routing config
        for task_type, model_name in routing_config.items():
            provider = self._infer_provider(model_name)
            self.adapters[task_type] = AdapterFactory.create_universal(
                provider=provider, model=model_name
            )

    def _infer_provider(self, model_name: str) -> str:
        """Infer provider from model name."""
        if model_name == "mock":
            return "mock"
        elif model_name.startswith("gpt"):
            return "openai"
        elif model_name.startswith("claude"):
            return "anthropic"
        elif model_name.startswith("grok"):
            return "grok"
        elif "/" in model_name:
            return "huggingface"
        else:
            return "ollama"

    def route(self, task_type: str, prompt: str) -> str:
        """
        Route prompt to appropriate model based on task type.

        Args:
            task_type: Type of task ("analysis", "chat", "bulk", etc.)
            prompt: Input prompt

        Returns:
            Generated response
        """
        adapter = self.adapters.get(task_type)
        if not adapter:
            # Fallback to first available adapter
            adapter = list(self.adapters.values())[0]

        return adapter.generate(prompt)

    def generate(self, prompt: str, task_type: str = "chat") -> str:
        """
        Generate response (for BaseAdapter compatibility).

        Args:
            prompt: Input prompt
            task_type: Task type for routing

        Returns:
            Generated response
        """
        return self.route(task_type, prompt)
