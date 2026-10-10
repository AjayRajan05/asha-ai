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
Composition Strategy - Phase 4 Key Differentiator

Makes ASHA work WITH other tools instead of competing:
# With Instructor
secure_prompt = process(prompt)
result = instructor_client.create(...)

# With LangChain
secure_chain = ASHATemplate(llm)
chain.run(prompt)

We are not competing - we are complementing and enhancing.
"""

import json
from typing import (
    Dict,
    List,
    Any,
    Optional,
    Callable,
    Type,
    TypeVar,
    cast,
)
from dataclasses import dataclass
from enum import Enum
import time

# Import ASHA components
from ..utils.dropin import process, _coerce_process_output

_F = TypeVar("_F", bound=Callable[..., Any])


class CompositionMode(Enum):
    """Composition strategy modes."""

    PREPROCESS = "preprocess"  # Process before other tools
    POSTPROCESS = "postprocess"  # Process after other tools
    ENHANCE = "enhance"  # Enhance existing tool output
    VALIDATE = "validate"  # Validate tool results


@dataclass
class CompositionConfig:
    """Configuration for composition strategy."""

    mode: CompositionMode = CompositionMode.PREPROCESS
    enable_pii_masking: bool = True
    enable_optimization: bool = True
    enable_safety_check: bool = True
    preserve_original: bool = False
    merge_results: bool = True
    error_handling: str = "continue"  # "continue", "stop", "fallback"
    policy_mode: str = "balanced"


class ToolChain:
    """Chain of tools with ASHA integration."""

    def __init__(self, config: Optional[CompositionConfig] = None):
        """Initialize tool chain."""
        self.config = config or CompositionConfig()
        self.tools: List[Dict[str, Any]] = []
        self.results: List[Dict[str, Any]] = []

    def add_tool(
        self, tool: Callable[..., Any], name: str, **kwargs: Any
    ) -> "ToolChain":
        """Add tool to chain."""
        self.tools.append({"tool": tool, "name": name, "kwargs": kwargs})
        return self

    def run(self, input_data: Any) -> Dict[str, Any]:
        """Run the tool chain."""
        current_data = input_data
        chain_results = {}

        for i, tool_info in enumerate(self.tools):
            tool = tool_info["tool"]
            name = tool_info["name"]
            kwargs = tool_info["kwargs"]

            try:
                # Preprocess with ASHA if configured
                if self.config.mode == CompositionMode.PREPROCESS:
                    if isinstance(current_data, str):
                        current_data = self._preprocess(current_data)

                # Run the tool
                start_time = time.time()
                if kwargs:
                    result = tool(current_data, **kwargs)
                else:
                    result = tool(current_data)
                processing_time = (time.time() - start_time) * 1000

                # Postprocess with ASHA if configured
                if self.config.mode == CompositionMode.POSTPROCESS:
                    if isinstance(result, str):
                        result = self._postprocess(result)

                # Store result
                chain_results[name] = {
                    "result": result,
                    "processing_time_ms": processing_time,
                    "success": True,
                }

                # Update current data for next tool
                current_data = result

            except Exception as e:
                chain_results[name] = {"error": str(e), "success": False}

                if self.config.error_handling == "stop":
                    break
                elif self.config.error_handling == "fallback":
                    current_data = self._fallback_processing(current_data)

        return {
            "final_result": current_data,
            "chain_results": chain_results,
            "config": self.config.__dict__,
        }

    def _preprocess(self, text: str) -> str:
        """Preprocess text with ASHA."""
        return _coerce_process_output(
            process(text, mode=self.config.policy_mode),
            text,
        )

    def _postprocess(self, text: str) -> str:
        """Postprocess text with ASHA."""
        return _coerce_process_output(
            process(text, mode=self.config.policy_mode),
            text,
        )

    def _fallback_processing(self, data: Any) -> Any:
        """Fallback processing if tool fails."""
        if isinstance(data, str):
            return f"[FALLBACK] {data}"
        return data


class ASHAInstructorComposer:
    """
    Compose ASHA with Instructor for structured outputs.

    Example:
        composer = ASHAInstructorComposer()
        result = composer.create_with_asha(
            prompt="Extract info from john@email.com",
            response_model=UserInfo,
            client=instructor_client
        )
    """

    def __init__(self, config: Optional[CompositionConfig] = None):
        """Initialize Instructor composer."""
        self.config = config or CompositionConfig()

    def create_with_asha(
        self,
        prompt: str,
        response_model: Type[Any],
        client: Any,
        **kwargs: Any,
    ) -> Any:
        """Create structured output with ASHA preprocessing."""
        # Preprocess prompt with ASHA
        secure_prompt = _coerce_process_output(
            process(prompt, mode=self.config.policy_mode),
            prompt,
        )

        # Create with Instructor
        result = client.chat.completions.create(
            response_model=response_model,
            messages=[{"role": "user", "content": secure_prompt}],
            **kwargs,
        )

        return result

    def validate_with_asha(
        self, result: Any, schema: Type[Any], client: Any
    ) -> Dict[str, Any]:
        """Validate Instructor result with ASHA."""
        validation_result = {
            "original_result": result,
            "is_valid": True,
            "asha_processed": False,
        }

        try:
            # Convert result to dict/string for processing
            if hasattr(result, "dict"):
                result_text = json.dumps(result.dict())
            elif hasattr(result, "__dict__"):
                result_text = json.dumps(result.__dict__)
            else:
                result_text = str(result)

            # Process with ASHA
            processed_text = _coerce_process_output(
                process(result_text, mode=self.config.policy_mode),
                result_text,
            )

            validation_result["asha_processed"] = True
            validation_result["processed_text"] = processed_text

        except Exception as e:
            validation_result["error"] = str(e)
            validation_result["is_valid"] = False

        return validation_result


class ASHALangChainComposer:
    """
    Compose ASHA with LangChain for enhanced chains.

    Example:
        composer = ASHALangChainComposer()
        chain = composer.create_asha_chain(llm)
        result = chain.run("Analyze data with john@email.com")
    """

    def __init__(self, config: Optional[CompositionConfig] = None):
        """Initialize LangChain composer."""
        self.config = config or CompositionConfig()

    def create_asha_chain(self, llm: Any, template: Optional[str] = None) -> Any:
        """Create LangChain with ASHA integration."""
        try:
            from langchain.chains import LLMChain
            from ..integrations.framework_adapters import (
                ASHAPromptTemplate,
                ASHALLM,
            )
        except ImportError:
            raise ImportError("LangChain not available")

        # Wrap LLM with ASHA
        asha_llm = ASHALLM(
            llm,
            (
                self.config.to_policy_config()
                if hasattr(self.config, "to_policy_config")
                else None
            ),
        )

        # Create or use template
        if template:
            asha_template = ASHAPromptTemplate(
                input_variables=["input"],
                template=template,
                config=cast(Any, self.config),
            )
        else:
            asha_template = ASHAPromptTemplate(
                input_variables=["input"],
                template="Process: {input}",
                config=cast(Any, self.config),
            )

        # Create chain
        chain = LLMChain(llm=asha_llm, prompt=asha_template)

        return chain

    def enhance_existing_chain(self, chain: Any) -> Any:
        """Enhance existing LangChain with ASHA."""
        try:
            from ..integrations.framework_adapters import ASHALLM
        except ImportError:
            raise ImportError("LangChain not available")

        # Wrap the chain's LLM
        if hasattr(chain, "llm"):
            original_llm = chain.llm
            asha_llm = ASHALLM(original_llm)
            chain.llm = asha_llm

        return chain


class OpenAIComposer:
    """
    Compose ASHA with OpenAI for enhanced API calls.

    Example:
        composer = OpenAIComposer()
        client = composer.create_asha_client()
        response = client.chat.completions.create(...)
    """

    def __init__(self, config: Optional[CompositionConfig] = None):
        """Initialize OpenAI composer."""
        self.config = config or CompositionConfig()

    def create_asha_client(self, **kwargs: Any) -> Any:
        """Create OpenAI client with ASHA integration."""
        try:
            from ..integrations.framework_adapters import OpenAIWrapper
        except ImportError:
            raise ImportError("OpenAI not available")

        return OpenAIWrapper(config=cast(Any, self.config), **kwargs)

    def enhance_api_call(
        self, api_function: Callable[..., Any], **api_kwargs: Any
    ) -> Callable[..., Any]:
        """Enhance OpenAI API function with ASHA."""

        def enhanced_function(**kwargs: Any) -> Any:
            # Process messages if present
            if "messages" in kwargs:
                processed_messages = []
                for message in kwargs["messages"]:
                    if isinstance(message, dict) and "content" in message:
                        processed_content = _coerce_process_output(
                            process(
                                message["content"],
                                mode=self.config.policy_mode,
                            ),
                            message["content"],
                        )
                        processed_messages.append(
                            {**message, "content": processed_content}
                        )
                    else:
                        processed_messages.append(message)
                kwargs["messages"] = processed_messages

            # Call original API function
            return api_function(**kwargs)

        return enhanced_function


class UniversalComposer:
    """
    Universal composer that works with any tool.

    This is the key to making ASHA unavoidable - it can enhance
    ANY existing tool or workflow.
    """

    def __init__(self, config: Optional[CompositionConfig] = None):
        """Initialize universal composer."""
        self.config = config or CompositionConfig()
        self.composers = {
            "instructor": ASHAInstructorComposer(config),
            "langchain": ASHALangChainComposer(config),
            "openai": OpenAIComposer(config),
        }

    def compose(self, tool: Any, tool_type: str = "universal", **kwargs: Any) -> Any:
        """
        Compose ASHA with any tool.

        Args:
            tool: The tool to compose with
            tool_type: Type of tool for specialized handling
            **kwargs: Additional arguments for composition

        Returns:
            Enhanced tool or composition result
        """
        if tool_type in self.composers:
            return cast(Any, self.composers[tool_type]).compose(tool, **kwargs)
        else:
            return self._universal_compose(tool, **kwargs)

    def _universal_compose(self, tool: Any, **kwargs: Any) -> Callable[..., Any]:
        """Universal composition for any tool."""

        def enhanced_tool(*args: Any, **tool_kwargs: Any) -> Any:
            # Process string arguments
            processed_args = []
            for arg in args:
                if isinstance(arg, str):
                    processed_arg = _coerce_process_output(
                        process(arg, mode=self.config.policy_mode),
                        arg,
                    )
                    processed_args.append(processed_arg)
                else:
                    processed_args.append(arg)

            # Process string keyword arguments
            processed_kwargs = {}
            for key, value in tool_kwargs.items():
                if isinstance(value, str):
                    processed_value = _coerce_process_output(
                        process(
                            value,
                            mode=self.config.policy_mode,
                        ),
                        value,
                    )
                    processed_kwargs[key] = processed_value
                else:
                    processed_kwargs[key] = value

            # Call original tool
            return tool(*processed_args, **processed_kwargs)

        return enhanced_tool

    def create_pipeline(self, tools: List[Any]) -> ToolChain:
        """Create a pipeline of tools with ASHA integration."""
        chain = ToolChain(self.config)

        for i, tool in enumerate(tools):
            tool_name = f"tool_{i}"
            enhanced_tool = self.compose(tool)
            chain.add_tool(enhanced_tool, tool_name)

        return chain

    def get_composition_info(self) -> Dict[str, Any]:
        """Get composition capabilities."""
        return {
            "config": self.config.__dict__,
            "supported_tools": list(self.composers.keys()),
            "universal_composition": True,
            "modes": [mode.value for mode in CompositionMode],
            "capabilities": {
                "preprocessing": True,
                "postprocessing": True,
                "enhancement": True,
                "validation": True,
                "error_handling": True,
            },
        }


# Convenience functions for easy composition
def compose_with_instructor(
    client: Any = None, config: Optional[CompositionConfig] = None
) -> ASHAInstructorComposer:
    """Create Instructor composer.

    ``client`` is accepted for call-site compatibility with Instructor wrappers;
    composition config is what the composer currently uses.
    """
    _ = client
    return ASHAInstructorComposer(config)


def compose_with_langchain(
    llm: Any, config: Optional[CompositionConfig] = None
) -> ASHALangChainComposer:
    """Create LangChain composer."""
    return ASHALangChainComposer(config)


def compose_with_openai(**kwargs: Any) -> OpenAIComposer:
    """Create OpenAI composer."""
    return OpenAIComposer(**kwargs)


def compose_universal(tool: Any, config: Optional[CompositionConfig] = None) -> Any:
    """Compose with any tool universally."""
    composer = UniversalComposer(config)
    return composer.compose(tool)


# Decorator for automatic composition
def asha_compose(
    tool_type: str = "universal", config: Optional[CompositionConfig] = None
) -> Callable[[_F], _F]:
    """Decorator to automatically compose any function with ASHA."""

    def decorator(func: _F) -> _F:
        composer = UniversalComposer(config)

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            enhanced_func = composer.compose(func, tool_type)
            return enhanced_func(*args, **kwargs)

        return cast(_F, wrapper)

    return decorator


# Quick test function
def test_composition_strategy() -> None:
    """Test composition strategy."""
    print("Testing Composition Strategy:")
    print("=" * 50)

    # Test universal composer
    composer = UniversalComposer()

    # Test with a simple function
    def test_tool(text: str) -> str:
        return f"Tool processed: {text}"

    enhanced_tool = compose_universal(test_tool)
    result = enhanced_tool("Contact john@email.com")
    print(f"✅ Universal composition: {result}")

    # Test pipeline
    def tool1(text: str) -> str:
        return f"Step 1: {text}"

    def tool2(text: str) -> str:
        return f"Step 2: {text}"

    pipeline = composer.create_pipeline([tool1, tool2])
    pipeline_result = pipeline.run("User data with john@email.com")
    print(f"✅ Pipeline result: {pipeline_result['final_result']}")

    # Test composition info
    info = composer.get_composition_info()
    print(f"✅ Supported tools: {info['supported_tools']}")
    print(f"✅ Composition modes: {info['modes']}")

    # Test decorator
    @asha_compose()
    def decorated_tool(prompt: str) -> str:
        return f"Decorated: {prompt}"

    decorated_result = decorated_tool("Email to admin@company.com")
    print(f"✅ Decorated tool: {decorated_result}")


if __name__ == "__main__":
    test_composition_strategy()
