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
from typing import Any, Dict, List, Optional, Union, cast

from ..runtime.adapters.factory import AdapterFactory
from ..runtime.adapters.base import BaseAdapter
from .processor import PromptProcessor
from ..types.results import AgentResult, ProcessResult


def _get_compiled_prompt_from_result(result: ProcessResult) -> str:
    """Extract compiled prompt from Processor result."""
    return result.output or ""


def _count_tokens(text: str) -> int:
    """Count tokens using tiktoken with word-split fallback."""
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text or ""))
    except Exception:
        return int(len((text or "").split()) * 1.3)


class Agent:
    """
    ASHA Agent - Complete v2 functionality with v1 compatibility.

    This class provides the same simple API as v1 but with all v2 capabilities
    built directly in. No separate enhanced files needed.
    """

    def __init__(
        self,
        model: Optional[str] = "gpt-4o-mini",
        privacy: bool = True,
        token_budget: int = 1200,
        provider: Optional[str] = None,
        fallback_providers: Optional[List[Dict[str, Any]]] = None,
        routing_config: Optional[Dict[str, str]] = None,
        timeout: int = 10,
        retries: int = 3,
        api_key: Optional[str] = None,
        local_model: Optional[str] = None,
        sample_prompts: Optional[List[str]] = None,
        tools: Optional[List[Any]] = None,
    ) -> None:
        """
        Initialize Agent with full v2 capabilities.

        Args:
            model: Model name (defaults to gpt-4o-mini for better default experience)
            privacy: Enable privacy features
            token_budget: Token budget for optimization
            provider: Provider override (auto-detected from model if None)
            fallback_providers: List of fallback provider configurations
            routing_config: Smart routing configuration for task-based model selection
            timeout: Request timeout in seconds
            retries: Number of retry attempts
            api_key: API key (uses environment if None)
            local_model: Set to "auto" to pick local model via AshaFit
            sample_prompts: Prompt corpus for AshaFit when local_model="auto"
        """
        self.privacy = privacy
        self.token_budget = token_budget
        self.sample_prompts = sample_prompts or []
        self.tools = tools or []
        self.max_tool_iterations = 8

        if local_model == "auto":
            from ..runtime.local_advisor.advisor import recommend_local_model

            prompts = self.sample_prompts or ["Summarize this text."]
            report = recommend_local_model(prompts=prompts, mode="balanced", top=1)
            if report.top_pick and report.top_pick.ollama_pull_name:
                model = report.top_pick.ollama_pull_name
                provider = provider or "ollama"
            else:
                model = model if model != "auto" else "llama3"
                provider = provider or "ollama"
        else:
            model = model

        self.model = model

        self.processor = PromptProcessor()
        self._processor_kwargs: Dict[str, Any] = {
            "mode": "strict" if privacy else "off",
            "token_budget": token_budget,
        }

        # Determine provider from model if not specified
        if not provider:
            provider = self._infer_provider(model)

        self.adapter: Any

        # Initialize adapter based on configuration
        if routing_config:
            # Smart routing mode
            self.adapter = AdapterFactory.create_smart_routing(routing_config)
            self.mode = "routing"
        elif fallback_providers:
            # Fallback mode
            self.adapter = AdapterFactory.create_with_fallbacks(
                primary_provider=provider,
                primary_model=model,
                fallback_providers=fallback_providers,
            )
            self.mode = "fallback"
        else:
            # Universal adapter mode (default)
            self.adapter = AdapterFactory.create_universal(
                provider=provider,
                model=model,
                api_key=api_key,
                timeout=timeout,
                retries=retries,
            )
            self.mode = "universal"

    def _infer_provider(self, model: Optional[str]) -> str:
        """Infer provider from model name."""
        if model == "mock":
            return "mock"
        if model is None:
            raise ValueError(
                "ASHA Agent: 'model' must be set when 'provider' is not explicitly given. "
                "Pass model='gpt-4o-mini' (or any supported model string) or set the "
                "ASHA_MODEL environment variable."
            )
        elif model.startswith("gpt"):
            return "openai"
        elif model.startswith("claude"):
            return "anthropic"
        elif model.startswith("grok"):
            return "grok"
        elif model.startswith("gemini"):
            return "gemini"
        elif "/" in model:
            return "huggingface"
        else:
            return "ollama"

    def _generate(self, prompt: str, task_type: str = "chat") -> str:
        if self.mode == "routing" and hasattr(self.adapter, "route"):
            return str(self.adapter.route(task_type, prompt))
        return str(self.adapter.generate(prompt))

    def run(
        self, prompt: str, trace: bool = False, task_type: str = "chat"
    ) -> Union[str, AgentResult, Dict]:
        """Run prompt through the enhanced pipeline.

        When tools are configured, runs a governed ReAct tool loop before returning.
        """
        proc_result = self.processor.run(prompt, **self._processor_kwargs)
        compiled = _get_compiled_prompt_from_result(proc_result)

        if self.tools:
            from .agent_tools import run_tool_loop

            response = run_tool_loop(
                generate=lambda p: self._generate(p, task_type=task_type),
                tools=self.tools,
                prompt=compiled,
                max_iterations=self.max_tool_iterations,
            )
        else:
            response = self._generate(compiled, task_type=task_type)

        if trace:
            return AgentResult.from_process(
                proc_result,
                response,
                adapter_info=self._get_adapter_info(),
                mode=self.mode,
            )

        return cast(Union[str, AgentResult, Dict[str, Any]], response)

    def run_with_fallback(
        self,
        prompt: str,
        trace: bool = False,
        fallback_adapters: Optional[List[BaseAdapter]] = None,
    ) -> Union[str, AgentResult, Dict[str, Any]]:
        """
        Run prompt with explicit fallback adapters.

        Args:
            prompt: Input prompt
            trace: Return full pipeline trace
            fallback_adapters: Additional fallback adapters

        Returns:
            Response text or full trace dict
        """
        proc_result = self.processor.run(prompt, **self._processor_kwargs)
        compiled = _get_compiled_prompt_from_result(proc_result)

        # Try primary adapter first
        try:
            response = self.adapter.generate(compiled)
        except Exception as primary_error:
            if hasattr(self.adapter, "generate_with_fallback"):
                # Use built-in fallback
                response = self.adapter.generate_with_fallback(
                    compiled, fallback_adapters
                )
            else:
                # Use provided fallbacks
                if fallback_adapters:
                    for fallback in fallback_adapters:
                        try:
                            response = fallback.generate(compiled)
                            break
                        except Exception:
                            continue
                    else:
                        raise Exception("All adapters failed")
                else:
                    raise primary_error

        if trace:
            return AgentResult.from_process(
                proc_result,
                response,
                adapter_info=self._get_adapter_info(),
            )

        return cast(Union[str, AgentResult, Dict[str, Any]], response)

    def _get_adapter_info(self) -> Dict[str, Any]:
        """Get adapter information for tracing."""
        if hasattr(self.adapter, "get_provider_info"):
            return cast(Dict[str, Any], self.adapter.get_provider_info())
        elif hasattr(self.adapter, "routing_config"):
            return {"routing_config": self.adapter.routing_config}
        else:
            return {"type": type(self.adapter).__name__}

    def get_token_metrics(self, prompt: str) -> Dict[str, int]:
        """
        Get token optimization metrics for a prompt.

        Args:
            prompt: Input prompt

        Returns:
            Token usage metrics
        """
        proc_result = self.processor.run(prompt, **self._processor_kwargs)
        raw = proc_result.original
        optimized = proc_result.output
        compiled = proc_result.output

        raw_tokens = _count_tokens(raw)
        optimized_tokens = _count_tokens(optimized)
        compiled_tokens = _count_tokens(compiled)

        return {
            "raw_tokens": raw_tokens,
            "optimized_tokens": optimized_tokens,
            "compiled_tokens": compiled_tokens,
            "token_reduction_percentage": (
                int((1 - optimized_tokens / raw_tokens) * 100) if raw_tokens > 0 else 0
            ),
        }

    def get_debug_trace(
        self, session_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Get debug trace for session."""
        # For now, return adapter info - could be enhanced with full tracing
        return self._get_adapter_info()

    def print_debug_trace(
        self, session_id: Optional[str] = None, format_type: str = "readable"
    ) -> str:
        """Print debug trace in specified format."""
        info = self._get_adapter_info()
        return f"Debug Trace ({format_type}): {info}"

    def switch_provider(self, provider: str, model: Optional[str] = None) -> None:
        """Switch to a different provider."""
        self.adapter = AdapterFactory.create_universal(provider=provider, model=model)
        self.mode = "universal"
        self.model = model or self.model

    def add_fallback(self, provider: str, model: Optional[str] = None) -> None:
        """Add a fallback provider."""
        if not hasattr(self.adapter, "_fallback_adapters"):
            self.adapter._fallback_adapters = []

        fallback = AdapterFactory.create_universal(provider, model)
        self.adapter._fallback_adapters.append(fallback)

    def get_adapter_info(self) -> Dict[str, Any]:
        """Get adapter information."""
        return self._get_adapter_info()

    def get_pipeline_statistics(self) -> Dict[str, Any]:
        """Get pipeline performance statistics."""
        return {
            "mode": self.mode,
            "model": self.model,
            "privacy": self.privacy,
            "token_budget": self.token_budget,
            "adapter_info": self._get_adapter_info(),
        }

    # Backward compatibility methods
    def get_token_savings(self, prompt: str) -> Dict[str, Any]:
        """Get token savings (backward compatibility)."""
        metrics = self.get_token_metrics(prompt)
        return {
            "original_tokens": metrics.get("raw_tokens", 0),
            "optimized_tokens": metrics.get("optimized_tokens", 0),
            "savings": metrics.get("raw_tokens", 0)
            - metrics.get("optimized_tokens", 0),
            "savings_percentage": metrics.get("token_reduction_percentage", 0),
        }

    @classmethod
    def from_env(cls) -> "Agent":
        """
        Create agent from environment configuration.

        Environment variables:
        - ASHA_PROVIDER: Provider name
        - ASHA_MODEL: Model name
        - ASHA_PRIVACY: Enable privacy (default true)
        - ASHA_TOKEN_BUDGET: Token budget (default 1200)
        - ASHA_TIMEOUT: Timeout (default 10)
        - ASHA_RETRIES: Retries (default 3)

        Returns:
            Agent configured from environment
        """
        return cls(
            provider=os.getenv("ASHA_PROVIDER"),
            model=os.getenv("ASHA_MODEL"),
            privacy=os.getenv("ASHA_PRIVACY", "true").lower() == "true",
            token_budget=int(os.getenv("ASHA_TOKEN_BUDGET", "1200")),
            timeout=int(os.getenv("ASHA_TIMEOUT", "10")),
            retries=int(os.getenv("ASHA_RETRIES", "3")),
        )

    # Class methods for different creation patterns
    @classmethod
    def create_simple(
        cls, provider: str = "openai", model: Optional[str] = None
    ) -> "Agent":
        """Create simple agent with single provider."""
        return cls(provider=provider, model=model)

    @classmethod
    def create_advanced(
        cls,
        provider: str = "openai",
        model: Optional[str] = None,
        fallback_providers: Optional[List[Dict[str, Any]]] = None,
    ) -> "Agent":
        """Create advanced agent with fallback providers."""
        if fallback_providers is None:
            fallback_providers = [
                {"provider": "anthropic", "model": "claude-3-haiku"},
                {"provider": "grok", "model": "grok-beta"},
            ]
        return cls(
            provider=provider, model=model, fallback_providers=fallback_providers
        )

    @classmethod
    def create_smart_routing(
        cls, routing_config: Optional[Dict[str, str]] = None
    ) -> "Agent":
        """Create agent with smart routing configuration."""
        if routing_config is None:
            routing_config = {
                "analysis": "claude-3-sonnet",
                "chat": "gpt-4o-mini",
                "coding": "codellama-7b",
                "bulk": "llama3-8b",
            }
        return cls(routing_config=routing_config)
