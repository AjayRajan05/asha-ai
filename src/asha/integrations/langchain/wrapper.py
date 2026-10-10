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
LangChain Integration for ASHA

This module provides direct integration with LangChain components,
allowing ASHA to secure and optimize prompts within LangChain pipelines.
"""

from typing import Any, Dict, List, Optional, TYPE_CHECKING

# Prefer langchain_core (modern extras); fall back to legacy monolith paths.
# Catch Exception: a polluted torch/numpy process can raise RuntimeError during
# transitive imports and must not abort the wrapper module load.
_LANGCHAIN_IMPORT_ERROR: Optional[BaseException] = None
try:
    from langchain_core.prompts.base import BasePromptTemplate
    from langchain_core.output_parsers import BaseOutputParser
    from langchain_core.runnables import Runnable
    from langchain_core.messages import BaseMessage, HumanMessage
    from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
    from langchain_core.language_models.llms import BaseLLM
    from langchain_core.language_models.chat_models import BaseChatModel

    try:
        from langchain.chains import LLMChain  # type: ignore
    except Exception:  # pragma: no cover - optional legacy helper

        class LLMChain:  # type: ignore[no-redef]
            llm: Any
            prompt: Any

    LANGCHAIN_AVAILABLE = True
except Exception as _core_exc:
    try:
        from langchain.schema import BasePromptTemplate, BaseOutputParser
        from langchain.schema.runnable import Runnable
        from langchain.schema.messages import BaseMessage, HumanMessage
        from langchain.chains import LLMChain
        from langchain.prompts import PromptTemplate, ChatPromptTemplate
        from langchain.llms.base import BaseLLM
        from langchain.chat_models.base import BaseChatModel

        LANGCHAIN_AVAILABLE = True
    except Exception as _legacy_exc:
        _LANGCHAIN_IMPORT_ERROR = _legacy_exc or _core_exc
        LANGCHAIN_AVAILABLE = False

        class BasePromptTemplate:
            def __init__(self, input_variables: List[str], **kwargs: Any) -> None:
                self.input_variables = input_variables

        class BaseOutputParser:
            pass

        class Runnable:
            def invoke(
                self, input: Any, config: Optional[Dict[str, Any]] = None
            ) -> Any:
                return input

        class BaseMessage:
            pass

        class HumanMessage:
            pass

        class LLMChain:
            llm: Any
            prompt: Any

        class PromptTemplate:
            pass

        class ChatPromptTemplate:
            pass

        class BaseLLM:
            def __call__(self, prompt: str, stop: Optional[List[str]] = None) -> str:
                return prompt

            @property
            def _llm_type(self) -> str:
                return "base"

        class BaseChatModel:
            pass


if TYPE_CHECKING:
    LANGCHAIN_AVAILABLE = True

from ...utils.dropin import process, _coerce_process_output


def _optimize_prompt_text(
    text: str,
    *,
    privacy: bool,
    token_budget: int,
    debug_metrics: bool,
) -> tuple[str, Optional[Dict[str, Any]]]:
    """Run process() and return optimized text plus optional metrics."""
    mode = "balanced" if privacy else "off"
    result = process(text, mode=mode, token_budget=token_budget)
    optimized = _coerce_process_output(result, text)
    if debug_metrics:
        return optimized, result.to_dict()
    return optimized, None


class ASHAPromptTemplate(PromptTemplate):
    """
    LangChain PromptTemplate wrapper that applies ASHA optimization.

    This wrapper automatically processes prompts through ASHA's
    security and optimization pipeline before passing them to the LLM.
    """

    privacy: bool = True
    token_budget: int = 1200
    debug_metrics: bool = False

    def __init__(
        self,
        input_variables: List[str],
        template: str,
        privacy: bool = True,
        token_budget: int = 1200,
        debug_metrics: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Initialize ASHA-enhanced prompt template.

        Args:
            input_variables: Variables to substitute in template
            template: Prompt template string
            privacy: Enable privacy features
            token_budget: Token budget for optimization
            debug_metrics: Return optimization metrics
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "LangChain is not installed. Install with: pip install asha[langchain]\n"
                "Or use the standalone ASHA functions instead."
            )

        super().__init__(
            input_variables=input_variables,
            template=template,
            **kwargs,
        )
        # Prefer object.__setattr__ so pydantic/core models accept extra state.
        object.__setattr__(self, "privacy", privacy)
        object.__setattr__(self, "token_budget", token_budget)
        object.__setattr__(self, "debug_metrics", debug_metrics)
        object.__setattr__(self, "_last_metrics", None)

    def format(self, **kwargs: Any) -> str:
        """
        Format the template with input variables and apply ASHA optimization.

        Args:
            **kwargs: Input variable values

        Returns:
            Optimized prompt string
        """
        formatted_prompt = super().format(**kwargs)

        optimized, metrics = _optimize_prompt_text(
            formatted_prompt,
            privacy=bool(getattr(self, "privacy", True)),
            token_budget=int(getattr(self, "token_budget", 1200)),
            debug_metrics=bool(getattr(self, "debug_metrics", False)),
        )
        if metrics is not None:
            object.__setattr__(self, "_last_metrics", metrics)
        return optimized

    def format_prompt(self, **kwargs: Any) -> Any:
        """Return a PromptValue for LangChain Runnable / LCEL pipelines."""
        try:
            from langchain_core.prompt_values import StringPromptValue
        except ImportError:  # pragma: no cover - legacy langchain
            from langchain.schema import StringPromptValue  # type: ignore

        return StringPromptValue(text=self.format(**kwargs))

    def get_last_metrics(self) -> Optional[Dict[str, Any]]:
        """Get metrics from the last prompt processing."""
        return getattr(self, "_last_metrics", None)


class ASHALLMWrapper(BaseLLM):
    """
    LangChain LLM wrapper that applies ASHA security and optimization.

    This wrapper intercepts all prompts sent to the LLM and processes them
    through ASHA's pipeline before forwarding to the actual LLM.
    """

    def __init__(
        self,
        llm: BaseLLM,
        privacy: bool = True,
        token_budget: int = 1200,
        debug_metrics: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Initialize ASHA-enhanced LLM wrapper.

        Args:
            llm: Base LLM to wrap
            privacy: Enable privacy features
            token_budget: Token budget for optimization
            debug_metrics: Return optimization metrics
        """
        super().__init__(**kwargs)
        self.llm = llm
        self.privacy = privacy
        self.token_budget = token_budget
        self.debug_metrics = debug_metrics
        self._last_metrics: Optional[Dict[str, Any]] = None

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """
        Process prompt through ASHA and forward to LLM.

        Args:
            prompt: Input prompt
            stop: Stop sequences

        Returns:
            LLM response
        """
        optimized_prompt, metrics = _optimize_prompt_text(
            prompt,
            privacy=self.privacy,
            token_budget=self.token_budget,
            debug_metrics=self.debug_metrics,
        )
        if metrics is not None:
            self._last_metrics = metrics

        # Forward to actual LLM
        return str(self.llm(optimized_prompt, stop=stop))

    @property
    def _llm_type(self) -> str:
        """Return LLM type identifier."""
        return f"asha_{self.llm._llm_type}"

    def get_last_metrics(self) -> Optional[Dict[str, Any]]:
        """Get metrics from the last prompt processing."""
        return self._last_metrics


class ASHARunnable(Runnable):
    """
    LangChain Runnable wrapper that applies ASHA optimization.

    This wrapper can be used with any LangChain Runnable component
    to automatically optimize inputs before processing.
    """

    def __init__(
        self,
        runnable: Runnable,
        privacy: bool = True,
        token_budget: int = 1200,
        debug_metrics: bool = False,
        input_key: str = "input",
    ) -> None:
        """
        Initialize ASHA-enhanced Runnable wrapper.

        Args:
            runnable: Base Runnable to wrap
            privacy: Enable privacy features
            token_budget: Token budget for optimization
            debug_metrics: Return optimization metrics
            input_key: Key to extract text input from input dict
        """
        self.runnable = runnable
        self.privacy = privacy
        self.token_budget = token_budget
        self.debug_metrics = debug_metrics
        self.input_key = input_key
        self._last_metrics: Optional[Dict[str, Any]] = None

    def invoke(self, input: Any, config: Optional[Dict[str, Any]] = None) -> Any:
        """
        Process input through ASHA and forward to Runnable.

        Args:
            input: Input data
            config: Configuration

        Returns:
            Runnable output
        """
        # Extract text input
        if isinstance(input, dict) and self.input_key in input:
            text_input = input[self.input_key]
        elif isinstance(input, str):
            text_input = input
        else:
            # If we can't extract text, pass through unchanged
            return self.runnable.invoke(input, config)

        optimized_input, metrics = _optimize_prompt_text(
            text_input,
            privacy=self.privacy,
            token_budget=self.token_budget,
            debug_metrics=self.debug_metrics,
        )
        if metrics is not None:
            self._last_metrics = metrics

        # Update input with optimized text
        if isinstance(input, dict) and self.input_key in input:
            optimized_input_dict = input.copy()
            optimized_input_dict[self.input_key] = optimized_input
            return self.runnable.invoke(optimized_input_dict, config)
        return self.runnable.invoke(optimized_input, config)

    def get_last_metrics(self) -> Optional[Dict[str, Any]]:
        """Get metrics from the last input processing."""
        return self._last_metrics


# Convenience functions for easy integration
def wrap_prompt_template(
    template: str,
    input_variables: List[str],
    privacy: bool = True,
    token_budget: int = 1200,
    debug_metrics: bool = False,
) -> ASHAPromptTemplate:
    """
    Create a ASHA-enhanced prompt template.

    Args:
        template: Prompt template string
        input_variables: Variables to substitute
        privacy: Enable privacy features
        token_budget: Token budget for optimization
        debug_metrics: Return optimization metrics

    Returns:
        ASHA-enhanced prompt template
    """
    return ASHAPromptTemplate(
        input_variables=input_variables,
        template=template,
        privacy=privacy,
        token_budget=token_budget,
        debug_metrics=debug_metrics,
    )


def wrap_llm_chain(
    chain: LLMChain,
    privacy: bool = True,
    token_budget: int = 1200,
    debug_metrics: bool = False,
) -> LLMChain:
    """
    Wrap an existing LLMChain with ASHA optimization.

    Args:
        chain: Existing LLMChain
        privacy: Enable privacy features
        token_budget: Token budget for optimization
        debug_metrics: Return optimization metrics

    Returns:
        ASHA-enhanced LLMChain
    """
    # Wrap the LLM
    if hasattr(chain, "llm"):
        chain.llm = ASHALLMWrapper(
            chain.llm,
            privacy=privacy,
            token_budget=token_budget,
            debug_metrics=debug_metrics,
        )

    # Wrap the prompt template
    if hasattr(chain, "prompt"):
        if isinstance(chain.prompt, BasePromptTemplate):
            chain.prompt = ASHAPromptTemplate(
                input_variables=chain.prompt.input_variables,
                template=chain.prompt.template,
                privacy=privacy,
                token_budget=token_budget,
                debug_metrics=debug_metrics,
            )

    return chain


def wrap_runnable(
    runnable: Runnable,
    privacy: bool = True,
    token_budget: int = 1200,
    debug_metrics: bool = False,
    input_key: str = "input",
) -> ASHARunnable:
    """
    Wrap any LangChain Runnable with ASHA optimization.

    Args:
        runnable: LangChain Runnable to wrap
        privacy: Enable privacy features
        token_budget: Token budget for optimization
        debug_metrics: Return optimization metrics
        input_key: Key to extract text input

    Returns:
        ASHA-enhanced Runnable
    """
    return ASHARunnable(
        runnable=runnable,
        privacy=privacy,
        token_budget=token_budget,
        debug_metrics=debug_metrics,
        input_key=input_key,
    )
