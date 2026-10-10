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
Streaming compatibility layer for ASHA.

Ensures ASHA never blocks or buffers streaming responses.
"""

from typing import Any, AsyncGenerator, Dict, Generator, Optional, cast
import inspect


def handle_streaming_response(
    response: Any, expected_format: Optional[str] = None
) -> Generator[Any, None, None]:
    """
    Handle streaming responses with pass-through behavior.

    Rule: NEVER block or buffer streams - just pass through safely.

    Args:
        response: Streaming response from LLM
        expected_format: Expected response format (for validation)

    Returns:
        Generator yielding chunks unchanged
    """
    try:
        # Check if response is iterable (streaming)
        if hasattr(response, "__iter__") and not isinstance(
            response, (str, bytes, dict)
        ):
            for chunk in response:
                yield chunk
        else:
            # Not actually streaming, return as single chunk
            yield response
    except Exception:
        # Fail-safe: yield original response
        yield response


def is_streaming_response(response: Any) -> bool:
    """
    Check if response is a streaming response.

    Args:
        response: Response to check

    Returns:
        True if streaming, False otherwise
    """
    # Check for common streaming indicators
    if hasattr(response, "__iter__") and not isinstance(response, (str, bytes, dict)):
        return True

    # Check for async generator
    if inspect.isasyncgen(response):
        return True

    return False


async def handle_async_streaming_response(
    response: Any, expected_format: Optional[str] = None
) -> AsyncGenerator[Any, None]:
    """
    Handle async streaming responses.

    Args:
        response: Async streaming response from LLM
        expected_format: Expected response format

    Yields:
        Chunks from async stream
    """
    try:
        if inspect.isasyncgen(response):
            async for chunk in response:
                yield chunk
        else:
            # Not actually async streaming, yield as single chunk
            yield response
    except Exception:
        # Fail-safe: yield original response
        yield response


def extract_prompt_from_kwargs(kwargs: Dict[str, Any]) -> str:
    """
    Extract prompt from various SDK parameter formats.

    Args:
        kwargs: Function keyword arguments

    Returns:
        Extracted prompt string
    """
    # Common prompt parameter names across SDKs
    prompt_keys = ["prompt", "messages", "input", "text", "query", "content"]

    for key in prompt_keys:
        if key in kwargs:
            value = kwargs[key]

            # Handle different prompt formats
            if isinstance(value, str):
                return value
            elif isinstance(value, list) and value:
                # Handle message format (OpenAI, Anthropic)
                if isinstance(value[0], dict) and "content" in value[0]:
                    return cast(str, value[0]["content"])
                elif isinstance(value[0], str):
                    return value[0]
            elif isinstance(value, dict) and "content" in value:
                return cast(str, value["content"])

    # Fallback: return empty string
    return ""


def replace_prompt_in_kwargs(kwargs: Dict[str, Any], new_prompt: str) -> Dict[str, Any]:
    """
    Replace prompt in kwargs with processed prompt.

    Args:
        kwargs: Original function arguments
        new_prompt: Processed prompt to insert

    Returns:
        Modified kwargs with new prompt
    """
    # Create a copy to avoid modifying original
    new_kwargs = kwargs.copy()

    # Common prompt parameter names
    prompt_keys = ["prompt", "messages", "input", "text", "query", "content"]

    for key in prompt_keys:
        if key in new_kwargs:
            value = new_kwargs[key]

            # Handle different prompt formats
            if isinstance(value, str):
                new_kwargs[key] = new_prompt
            elif isinstance(value, list) and value:
                # Handle message format
                if isinstance(value[0], dict) and "content" in value[0]:
                    new_kwargs[key] = [
                        {"content": new_prompt, "role": value[0].get("role", "user")}
                    ]
                elif isinstance(value[0], str):
                    new_kwargs[key] = [new_prompt]
            elif isinstance(value, dict) and "content" in value:
                new_kwargs[key] = {"content": new_prompt}

    return new_kwargs


class StreamingGuard:
    """
    Guard class for streaming operations.

    Ensures streaming operations are safe and non-blocking.
    """

    def __init__(self, max_chunk_size: Optional[int] = None) -> None:
        """Initialize streaming guard."""
        self.max_chunk_size = max_chunk_size

    def guard_stream(self, response: Any) -> Generator[Any, None, None]:
        """
        Guard a streaming response.

        Args:
            response: Streaming response to guard

        Yields:
            Safe chunks from the stream
        """
        try:
            for chunk in handle_streaming_response(response):
                if self.max_chunk_size and len(str(chunk)) > self.max_chunk_size:
                    # Split large chunks if needed
                    chunk_str = str(chunk)
                    for i in range(0, len(chunk_str), self.max_chunk_size):
                        yield chunk_str[i : i + self.max_chunk_size]
                else:
                    yield chunk
        except Exception:
            # Fail-safe: yield empty chunk
            yield ""

    async def guard_async_stream(self, response: Any) -> AsyncGenerator[Any, None]:
        """
        Guard an async streaming response.

        Args:
            response: Async streaming response to guard

        Yields:
            Safe chunks from the async stream
        """
        try:
            async for chunk in handle_async_streaming_response(response):
                if self.max_chunk_size and len(str(chunk)) > self.max_chunk_size:
                    # Split large chunks if needed
                    chunk_str = str(chunk)
                    for i in range(0, len(chunk_str), self.max_chunk_size):
                        yield chunk_str[i : i + self.max_chunk_size]
                else:
                    yield chunk
        except Exception:
            # Fail-safe: yield empty chunk
            yield ""


# Convenience functions for streaming operations
def safe_stream_passthrough(response: Any) -> Generator[Any, None, None]:
    """
    Safely pass through streaming response.

    Args:
        response: Streaming response

    Returns:
        Generator yielding chunks safely
    """
    return handle_streaming_response(response)


async def safe_async_stream_passthrough(
    response: Any,
) -> AsyncGenerator[Any, None]:
    """
    Safely pass through async streaming response.

    Args:
        response: Async streaming response

    Returns:
        Async generator yielding chunks safely
    """
    async for chunk in handle_async_streaming_response(response):
        yield chunk
