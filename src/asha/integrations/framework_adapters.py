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
Framework Integration Adapters - Phase 4 Ecosystem Domination

Priority Order for Integration:
1. FastAPI middleware ✅ (production-ready)
2. LangChain wrapper
3. Instructor integration
4. OpenAI/Gemini wrappers

Goal: Works in <3 lines everywhere with no breaking changes.
"""

import json
from typing import (
    Dict,
    List,
    Any,
    Optional,
    Union,
    Callable,
    Type,
    TypeVar,
    TYPE_CHECKING,
    cast,
)
from dataclasses import dataclass, replace
from functools import wraps

# Framework imports from our local source
from .fastapi.middleware import BaseHTTPMiddleware, FASTAPI_AVAILABLE

if TYPE_CHECKING:
    from fastapi import FastAPI, Request, Response
    from fastapi.responses import JSONResponse
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain
    from langchain.llms.base import LLM
else:
    try:
        from fastapi import FastAPI, Request, Response
        from fastapi.responses import JSONResponse
    except ImportError:

        class FastAPI:
            pass

        class Request:
            pass

        class Response:
            pass

        class JSONResponse:
            pass

    try:
        from langchain.prompts import PromptTemplate
        from langchain.chains import LLMChain
        from langchain.llms.base import LLM
    except ImportError:

        class PromptTemplate:
            pass

        class LLMChain:
            pass

        class LLM:
            def _call(self, *args: Any, **kwargs: Any) -> str:
                return ""

            @property
            def _llm_type(self) -> str:
                return "llm"


_F = TypeVar("_F", bound=Callable[..., Any])

# LangChain imports from our local source
from .langchain.wrapper import BasePromptTemplate, LANGCHAIN_AVAILABLE

# LLM imports from our local source
from .langchain.wrapper import BaseLLM as LangChainLLM

try:
    import instructor

    INSTRUCTOR_AVAILABLE = True
except ImportError:
    INSTRUCTOR_AVAILABLE = False

try:
    import openai

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Import ASHA entrypoint (integrations delegate here - no direct orchestration)
from ..core.policy_config import PolicyConfig, PolicyMode
from ..core.schema_validation import SchemaValidationMode
from ..utils.dropin import process, _coerce_process_output


def _integration_process_text(text: str, config: "IntegrationConfig") -> str:
    """
    Process text via the public process() entrypoint.

    Integrations pass behavior as configuration instead of composing core services.
    """
    if not config.enable_pii_detection and not config.enable_optimization:
        return text

    policy = replace(
        config.to_policy_config(),
        security_level="high" if config.enable_safety_check else "medium",
    )
    processed = process(
        text,
        mode=config.policy_mode if config.enable_pii_detection else "off",
        token_budget=1200 if config.enable_optimization else 99999,
        policy=policy,
    )
    return _coerce_process_output(processed, text)


@dataclass
class IntegrationConfig:
    """Configuration for framework integrations."""

    enable_pii_detection: bool = True
    enable_optimization: bool = True
    enable_safety_check: bool = True
    enable_schema_validation: bool = False
    policy_mode: str = "balanced"
    optimization_level: str = "balanced"
    schema: Optional[Union[Dict[str, Any], Type]] = None
    debug_mode: bool = False

    def to_policy_config(self) -> PolicyConfig:
        """Convert to ASHA PolicyConfig."""
        return PolicyConfig.from_mode(
            PolicyMode(self.policy_mode),
            pii_masking=self.enable_pii_detection,
            enable_optimization=self.enable_optimization,
            debug_diff=self.debug_mode,
        )


class FastAPIMiddleware(BaseHTTPMiddleware):
    """
    FastAPI Middleware for automatic ASHA processing.

    Usage:
        app = FastAPI()
        app.add_middleware(ASHAMiddleware, config=config)
    """

    def __init__(
        self, app: FastAPI, config: Optional[IntegrationConfig] = None
    ) -> None:
        """Initialize FastAPI middleware."""
        if not FASTAPI_AVAILABLE:
            raise ImportError(
                "FastAPI not available. Install with: pip install fastapi"
            )

        super().__init__(app)
        self.config = config or IntegrationConfig()
        self.policy_config = self.config.to_policy_config()
        self.schema_validator = (
            SchemaValidationMode() if self.config.enable_schema_validation else None
        )

    async def dispatch(
        self, request: Request, call_next: Callable[..., Any]
    ) -> Response:
        """Process request through ASHA pipeline."""
        # Only process POST/PUT/PATCH requests with JSON body
        if request.method not in ["POST", "PUT", "PATCH"]:
            return cast(Response, await call_next(request))

        try:
            # Get request body
            body = await request.body()
            if not body:
                return cast(Response, await call_next(request))

            # Parse JSON
            try:
                data = json.loads(body.decode())
            except json.JSONDecodeError:
                return cast(Response, await call_next(request))

            # Process through ASHA
            processed_data = await self._process_data(data)

            # Update request body
            processed_body = json.dumps(processed_data).encode()

            # Create new request with processed body
            request._body = processed_body

            return cast(Response, await call_next(request))

        except Exception as e:
            if self.config.debug_mode:
                return cast(
                    Response,
                    JSONResponse(
                        status_code=500,
                        content={"error": f"ASHA middleware error: {str(e)}"},
                    ),
                )
            return cast(Response, await call_next(request))

    async def _process_data(self, data: Any) -> Any:
        """Process data through ASHA components."""
        if isinstance(data, dict):
            processed = {}
            for key, value in data.items():
                if isinstance(value, str):
                    processed[key] = await self._process_text(value)
                elif isinstance(value, (dict, list)):
                    processed[key] = await self._process_data(value)
                else:
                    processed[key] = value
            return processed
        elif isinstance(data, list):
            return [await self._process_data(item) for item in data]
        else:
            return data

    async def _process_text(self, text: str) -> str:
        """Process text through ASHA via the public process() entrypoint."""
        processed = _integration_process_text(text, self.config)

        if self.schema_validator and self.config.schema:
            validation_result = self.schema_validator.process_with_validation(
                processed, self.config.schema
            )
            if validation_result.repaired_content:
                processed = validation_result.repaired_content

        return processed


class LangChainPromptTemplate(BasePromptTemplate):
    """
    LangChain PromptTemplate with ASHA integration.

    Usage:
        template = ASHAPromptTemplate(
            input_variables=["query"],
            template="Analyze: {query}",
            config=config
        )
    """

    def __init__(
        self,
        input_variables: List[str],
        template: str,
        config: Optional[IntegrationConfig] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize LangChain prompt template."""
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "LangChain not available. Install with: pip install langchain"
            )

        super().__init__(input_variables=input_variables, **kwargs)
        self.template = template
        self.config = config or IntegrationConfig()
        self.policy_config = self.config.to_policy_config()

    def format(self, **kwargs: Any) -> str:
        """Format prompt with ASHA processing."""
        # Format the template
        formatted = self.template.format(**kwargs)

        # Process through ASHA
        return self._process_prompt(formatted)

    def _process_prompt(self, prompt: str) -> str:
        """Process prompt through ASHA via process()."""
        return _integration_process_text(prompt, self.config)

    def _validate_input_variables(self, kwargs: Dict[str, Any]) -> None:
        """Validate input variables."""
        missing_vars = set(self.input_variables) - set(kwargs.keys())
        if missing_vars:
            raise ValueError(f"Missing required input variables: {missing_vars}")


class ASHALLM(LangChainLLM):
    """
    LangChain LLM wrapper with ASHA integration.

    Usage:
        llm = ASHALLM(
            base_llm=OpenAI(),
            config=config
        )
    """

    def __init__(
        self, base_llm: LLM, config: Optional[IntegrationConfig] = None
    ) -> None:
        """Initialize ASHA LLM wrapper."""
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "LangChain not available. Install with: pip install langchain"
            )

        super().__init__()
        self.base_llm = base_llm
        self.config = config or IntegrationConfig()
        self.policy_config = self.config.to_policy_config()

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> str:
        """Call LLM with ASHA processing."""
        # Process input prompt
        processed_prompt = self._process_prompt(prompt)

        # Call base LLM
        response = self.base_llm._call(
            processed_prompt, stop=stop, run_manager=run_manager, **kwargs
        )

        # Process response if needed
        processed_response = self._process_response(response)

        return processed_response

    def _process_prompt(self, prompt: str) -> str:
        """Process prompt through ASHA via process()."""
        return _integration_process_text(prompt, self.config)

    def _process_response(self, response: str) -> str:
        """Process response through ASHA via process()."""
        if not self.config.enable_safety_check:
            return response
        return _integration_process_text(response, self.config)

    @property
    def _llm_type(self) -> str:
        """Return LLM type."""
        return f"asha_{self.base_llm._llm_type}"


class ASHAInstructorClient:
    """
    Instructor integration with ASHA.

    Usage:
        client = ASHAInstructor(
            OpenAI(),
            config=config
        )
    """

    def __init__(
        self, base_client: Any, config: Optional[IntegrationConfig] = None
    ) -> None:
        """Initialize Instructor client."""
        if not INSTRUCTOR_AVAILABLE:
            raise ImportError(
                "Instructor not available. Install with: pip install instructor"
            )

        self.base_client = base_client
        self.config = config or IntegrationConfig()
        self.policy_config = self.config.to_policy_config()
        self.schema_validator = (
            SchemaValidationMode() if self.config.enable_schema_validation else None
        )

        # Wrap with instructor
        self.client = instructor.patch(base_client)

    def create(
        self,
        response_model: Type[Any],
        messages: Optional[List[Dict[str, str]]] = None,
        **kwargs: Any,
    ) -> Any:
        """Create structured output with ASHA processing."""
        # Process messages
        processed_messages = []
        if messages:
            for message in messages:
                if isinstance(message, dict) and "content" in message:
                    processed_content = self._process_text(message["content"])
                    processed_messages.append({**message, "content": processed_content})
                else:
                    processed_messages.append(message)

        # Create with instructor
        result = self.client.chat.completions.create(
            response_model=response_model, messages=processed_messages, **kwargs
        )

        return result

    def _process_text(self, text: str) -> str:
        """Process text through ASHA via process()."""
        return _integration_process_text(text, self.config)


class OpenAIWrapper:
    """
    OpenAI client wrapper with ASHA integration.

    Usage:
        client = ASHAOpenAI(
            api_key="your-key",
            config=config
        )
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        config: Optional[IntegrationConfig] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize OpenAI wrapper."""
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI not available. Install with: pip install openai")

        self.config = config or IntegrationConfig()
        self.policy_config = self.config.to_policy_config()

        # Initialize OpenAI client
        self.client = openai.OpenAI(api_key=api_key, **kwargs)

    def chat_completions_create(self, **kwargs: Any) -> Any:
        """Create chat completion with ASHA processing."""
        # Process messages
        if "messages" in kwargs:
            processed_messages = []
            for message in kwargs["messages"]:
                if isinstance(message, dict) and "content" in message:
                    processed_content = self._process_text(message["content"])
                    processed_messages.append({**message, "content": processed_content})
                else:
                    processed_messages.append(message)
            kwargs["messages"] = processed_messages

        # Create completion
        response = self.client.chat.completions.create(**kwargs)

        return response

    def _process_text(self, text: str) -> str:
        """Process text through ASHA via process()."""
        return _integration_process_text(text, self.config)


# Backward-compatible alias used by composition_strategy
ASHAPromptTemplate = LangChainPromptTemplate


# Convenience functions for easy integration
def add_asha_to_fastapi(
    app: FastAPI, config: Optional[IntegrationConfig] = None
) -> None:
    """Add ASHA middleware to FastAPI app."""
    if not FASTAPI_AVAILABLE:
        raise ImportError("FastAPI not available")

    app.add_middleware(FastAPIMiddleware, config=config)


def wrap_langchain_llm(llm: LLM, config: Optional[IntegrationConfig] = None) -> LLM:
    """Wrap LangChain LLM with ASHA."""
    if not LANGCHAIN_AVAILABLE:
        raise ImportError("LangChain not available")

    return ASHALLM(llm, config)


def wrap_instructor_client(
    client: Any, config: Optional[IntegrationConfig] = None
) -> ASHAInstructorClient:
    """Wrap Instructor client with ASHA."""
    if not INSTRUCTOR_AVAILABLE:
        raise ImportError("Instructor not available")

    return ASHAInstructorClient(client, config)


def wrap_openai_client(
    api_key: Optional[str] = None,
    config: Optional[IntegrationConfig] = None,
    **kwargs: Any,
) -> OpenAIWrapper:
    """Wrap OpenAI client with ASHA."""
    if not OPENAI_AVAILABLE:
        raise ImportError("OpenAI not available")

    return OpenAIWrapper(api_key, config, **kwargs)


# Decorator for easy integration
def asha_process(
    config: Optional[IntegrationConfig] = None,
) -> Callable[[_F], _F]:
    """Decorator to add ASHA processing to any function."""

    def decorator(func: _F) -> _F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            policy_mode = config.policy_mode if config else "balanced"
            # Process string arguments
            processed_args: List[Any] = []
            for arg in args:
                if isinstance(arg, str):
                    processed_args.append(
                        _coerce_process_output(process(arg, mode=policy_mode), arg)
                    )
                else:
                    processed_args.append(arg)

            # Process string keyword arguments
            processed_kwargs: Dict[str, Any] = {}
            for key, value in kwargs.items():
                if isinstance(value, str):
                    processed_kwargs[key] = _coerce_process_output(
                        process(value, mode=policy_mode), value
                    )
                else:
                    processed_kwargs[key] = value

            return func(*processed_args, **processed_kwargs)

        return cast(_F, wrapper)

    return decorator


# Quick test function
def test_framework_integrations() -> None:
    """Test framework integrations."""
    print("Testing Framework Integrations:")
    print("=" * 50)

    # Test FastAPI middleware
    if FASTAPI_AVAILABLE:
        print("✅ FastAPI integration available")
        config = IntegrationConfig(
            enable_pii_detection=True, enable_optimization=True, policy_mode="balanced"
        )
        print(f"   - PII detection: {config.enable_pii_detection}")
        print(f"   - Optimization: {config.enable_optimization}")
        print(f"   - Policy mode: {config.policy_mode}")
    else:
        print("❌ FastAPI not available")

    # Test LangChain
    if LANGCHAIN_AVAILABLE:
        print("✅ LangChain integration available")
    else:
        print("❌ LangChain not available")

    # Test Instructor
    if INSTRUCTOR_AVAILABLE:
        print("✅ Instructor integration available")
    else:
        print("❌ Instructor not available")

    # Test OpenAI
    if OPENAI_AVAILABLE:
        print("✅ OpenAI integration available")
    else:
        print("❌ OpenAI not available")

    # Test decorator
    @asha_process()
    def test_function(prompt: str) -> str:
        return f"Processed: {prompt}"

    result = test_function("Contact john@email.com for support")
    print(f"✅ Decorator test: {result}")


if __name__ == "__main__":
    test_framework_integrations()
