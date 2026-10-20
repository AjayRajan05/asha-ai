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
Unified wrapper for ASHA with async and streaming support.

Provides drop-in wrapping for any LLM client with fail-safe execution.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, Optional
from functools import wraps

from ..core.streaming import (
    handle_streaming_response,
    handle_async_streaming_response,
    is_streaming_response,
)
from ..core.output_guard import OutputGuard
from ..exceptions import ASHAProcessingError


def _handle_wrap_processing_error(mode: str, exc: Exception) -> None:
    """Fail-closed when processing is enabled - never send raw prompts silently."""
    if mode != "off":
        raise ASHAProcessingError(
            "ASHA prompt processing failed; refusing to send unprocessed prompt."
        ) from exc
    raise exc


def _process_prompt_for_wrap(
    prompt: str,
    mode: str,
    *,
    token_budget: int = 1200,
) -> str:
    """Run process() and return a prompt string for wrapped clients."""
    from ..utils.dropin import process, _coerce_process_output

    if mode == "off":
        processed = process(prompt, mode="off")
        return _coerce_process_output(processed, prompt)

    processed = process(
        prompt,
        mode=mode,
        token_budget=token_budget if mode != "strict" else min(token_budget, 800),
    )
    if mode == "strict" and ((processed.security and not processed.security.safe) or not processed.is_safe):
        raise ASHAProcessingError(
            f"Prompt blocked by strict mode security policy: {processed.security.threats if processed.security else 'threat detected'}"
        )
    return _coerce_process_output(processed, prompt)


def _prepare_kwargs_for_llm(
    kwargs: dict[str, Any],
    mode: str,
    *,
    token_budget: int = 1200,
) -> dict[str, Any]:
    """Mask prompts and extract+mask file attachments before the LLM call."""
    from ..documents.attachments import prepare_llm_request_kwargs

    return prepare_llm_request_kwargs(kwargs, mode=mode, token_budget=token_budget)


def _prepare_args_for_llm(
    args: tuple[Any, ...],
    mode: str,
    *,
    token_budget: int = 1200,
) -> tuple[Any, ...]:
    """Process positional prompt args (Gemini generate_content(prompt), etc.)."""
    if not args:
        return args
    first = args[0]
    if not isinstance(first, str):
        return args
    return (_process_prompt_for_wrap(first, mode, token_budget=token_budget),) + args[
        1:
    ]


def _evaluate_anchor(response: Any, kwargs: dict[str, Any]) -> None:
    try:
        from ..runtime.anchor.runtime import current_anchor_runtime
        from ..runtime.anchor.llm_guard import evaluate_llm_tool_calls
    except ImportError:
        return

    runtime = current_anchor_runtime.get()
    if not runtime:
        return

    evaluate_llm_tool_calls(response, runtime, raise_on_block=False, record=True)


def _guard_response(
    output_guard: OutputGuard, response: Any, kwargs: dict[str, Any]
) -> Any:
    """Validate LLM response with optional format constraint."""
    _evaluate_anchor(response, kwargs)
    response_format = kwargs.get("response_format")
    fmt: Optional[str] = response_format if isinstance(response_format, str) else None
    return output_guard.guard_output(response, fmt)


def _wrap_nested_chat_completions(
    client: Any,
    mode: str,
    *,
    token_budget: int = 1200,
) -> bool:
    """
    Wrap OpenAI-style client.chat.completions.create if present.

    Returns True if nested wrapping was applied.
    """
    chat = getattr(client, "chat", None)
    if chat is None:
        return False
    completions = getattr(chat, "completions", None)
    if completions is None:
        return False
    original_create = getattr(completions, "create", None)
    if not callable(original_create):
        return False

    def wrapped_create(*args: Any, **kwargs: Any) -> Any:
        try:
            kwargs = _prepare_kwargs_for_llm(kwargs, mode, token_budget=token_budget)
            response = original_create(*args, **kwargs)
            if kwargs.get("stream", False) and is_streaming_response(response):
                return handle_streaming_response(response)
            output_guard = OutputGuard()
            return _guard_response(output_guard, response, kwargs)
        except Exception as exc:
            _handle_wrap_processing_error(mode, exc)

    completions.create = wrapped_create
    return True


def wrap_llm(
    client: Any,
    mode: str = "balanced",
    *,
    token_budget: int = 1200,
    rehydrate_pii: Optional[bool] = None,
    profile: Optional[Any] = None,
) -> Any:
    """
    Wrap any LLM client with ASHA security and optimization without mutating the client.

    Returns a SecureLLM proxy that intercepts LLM calls while leaving the original
    client untouched. Works with both sync and async clients, handling streaming
    automatically.
    """
    if client is None:
        raise ValueError("Client cannot be None")

    if isinstance(client, SecureLLM):
        return client

    generation_method = _find_generation_method(client)
    has_chat_completions = hasattr(client, "chat") and hasattr(
        getattr(client, "chat"), "completions"
    )
    has_messages = hasattr(client, "messages")

    if not (generation_method or has_chat_completions or has_messages):
        raise ValueError(
            "Could not find compatible generation method on client. "
            "Supported clients: OpenAI, Anthropic, Gemini, HuggingFace, Ollama. "
            "Make sure your client has a generate/create method."
        )

    return SecureLLM(
        client,
        profile=profile,
        mode=mode,
        token_budget=token_budget,
        rehydrate_pii=rehydrate_pii,
    )


def wrap_function(
    func: Callable[..., Any], mode: str = "balanced"
) -> Callable[..., Any]:
    """
    Wrap a function that takes a prompt as first argument.

    Args:
        func: Function to wrap
        mode: Processing mode

    Returns:
        Wrapped function with ASHA protection
    """
    if inspect.iscoroutinefunction(func):

        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                if args:
                    # Process first positional argument (prompt)
                    prompt = str(args[0])
                    from ..utils.dropin import process, _coerce_process_output

                    processed = process(prompt, mode=mode)
                    optimized_prompt = _coerce_process_output(processed, prompt)
                    new_args = (optimized_prompt,) + args[1:]
                    return await func(*new_args, **kwargs)

                return await func(*args, **kwargs)

            except Exception as exc:
                _handle_wrap_processing_error(mode, exc)

        return async_wrapper

    else:

        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                if args:
                    # Process first positional argument (prompt)
                    prompt = str(args[0])
                    from ..utils.dropin import process, _coerce_process_output

                    processed = process(prompt, mode=mode)
                    optimized_prompt = _coerce_process_output(processed, prompt)
                    new_args = (optimized_prompt,) + args[1:]
                    return func(*new_args, **kwargs)

                return func(*args, **kwargs)

            except Exception as exc:
                _handle_wrap_processing_error(mode, exc)

        return sync_wrapper


def _find_generation_method(client: Any) -> Optional[str]:
    """
    Find the main generation method on an LLM client.

    Args:
        client: LLM client instance (OpenAI, Anthropic, Gemini, etc.)

    Returns:
        Method name string if found, None otherwise

    Supported clients:
        - OpenAI: 'create', 'generate', 'chat.completions.create'
        - Anthropic: 'messages.create', 'complete'
        - Gemini: 'generate_content', 'send_message'
        - HuggingFace: 'generate', '__call__'
        - Ollama: 'generate', 'chat'
    """
    # Prefer provider-specific names first so Gemini is not missed
    # (GenerativeModel exposes generate_content, not create/generate).
    method_names = [
        "generate_content",  # Google Generative AI / Gemini
        "send_message",  # Gemini chat sessions
        "create",  # OpenAI
        "generate",  # Anthropic, HuggingFace, Ollama
        "chat",  # Some custom clients
        "completion",  # Some older clients
        "predict",  # Some ML clients
        "run",  # Generic
    ]

    for method_name in method_names:
        if hasattr(client, method_name):
            method = getattr(client, method_name)
            if callable(method):
                return method_name

    return None


class UniversalWrapper:
    """
    Universal wrapper that can adapt to any LLM interface.

    Provides automatic method detection and wrapping with fail-safe execution.
    """

    def __init__(self, mode: str = "balanced", auto_detect: bool = True) -> None:
        """
        Initialize universal wrapper.

        Args:
            mode: Processing mode
            auto_detect: Whether to auto-detect generation methods
        """
        self.mode = mode
        self.auto_detect = auto_detect
        self.output_guard = OutputGuard()

    def wrap_client(self, client: Any) -> Any:
        """
        Wrap an entire LLM client.

        Args:
            client: LLM client to wrap

        Returns:
            Wrapped client
        """
        if self.auto_detect:
            return wrap_llm(client, self.mode)
        else:
            # Manual wrapping for specific methods
            return self._manual_wrap(client)

    def wrap_method(self, client: Any, method_name: str) -> Any:
        """
        Wrap a specific method on a client.

        Args:
            client: LLM client
            method_name: Name of method to wrap

        Returns:
            Client with wrapped method
        """
        if not hasattr(client, method_name):
            raise ValueError(
                f"Method '{method_name}' not found on client. "
                "Available methods depend on client type."
            )

        original_method = getattr(client, method_name)

        if inspect.iscoroutinefunction(original_method):

            async def async_method_wrapper(*args: Any, **kwargs: Any) -> Any:
                try:
                    kwargs = _prepare_kwargs_for_llm(kwargs, self.mode)

                    # Call original method
                    response = await original_method(*args, **kwargs)

                    # Handle streaming and validation
                    if kwargs.get("stream", False) and is_streaming_response(response):
                        return handle_async_streaming_response(response)

                    return _guard_response(self.output_guard, response, kwargs)

                except Exception as exc:
                    _handle_wrap_processing_error(self.mode, exc)

            setattr(client, method_name, async_method_wrapper)

        else:

            def sync_method_wrapper(*args: Any, **kwargs: Any) -> Any:
                try:
                    kwargs = _prepare_kwargs_for_llm(kwargs, self.mode)

                    # Call original method
                    response = original_method(*args, **kwargs)

                    # Handle streaming and validation
                    if kwargs.get("stream", False) and is_streaming_response(response):
                        return handle_streaming_response(response)

                    return _guard_response(self.output_guard, response, kwargs)

                except Exception as exc:
                    _handle_wrap_processing_error(self.mode, exc)

            setattr(client, method_name, sync_method_wrapper)

        return client

    def _manual_wrap(self, client: Any) -> Any:
        """
        Manual wrapping when auto-detect is disabled.

        Args:
            client: LLM client to wrap

        Returns:
            Wrapped client
        """
        # Try common methods manually
        methods_to_try = ["create", "generate", "chat"]

        for method_name in methods_to_try:
            if hasattr(client, method_name):
                return self.wrap_method(client, method_name)

        raise ValueError("No compatible generation method found")


# Convenience functions for quick wrapping
def wrap_llm_safe(
    client: Any,
    mode: str = "balanced",
    *,
    token_budget: int = 1200,
) -> Any:
    """Wrap client; raises :class:`ASHAWrapError` on setup failure."""
    from ..exceptions import ASHAWrapError

    try:
        return wrap_llm(
            client,
            mode=mode,
            token_budget=token_budget,
        )
    except Exception as exc:
        raise ASHAWrapError(
            f"Failed to wrap LLM client: {exc}. "
            "Privacy protection is not active - fix the error or disable privacy."
        ) from exc


def safe_wrap(client: Any, mode: str = "balanced") -> Any:
    """Quick safe wrapper with maximum compatibility."""
    return wrap_llm_safe(client, mode=mode)


def auto_wrap(*clients: Any, mode: str = "balanced") -> list[Any]:
    """
    Wrap multiple clients automatically.
    """
    return [safe_wrap(client, mode=mode) for client in clients]


class SecureLLM:
    """
    Single-line, transparent proxy that adds ASHA security and privacy to any LLM client.

    ``SecureLLM`` wraps your existing LLM client via attribute delegation and
    method interception. Any attribute access or method call that is not
    intercepted is forwarded verbatim to the underlying client, preserving
    100% API compatibility.

    Intercepted paths:
        - ``client.chat.completions.create(...)``: OpenAI / compatible
        - ``client.messages.create(...)``: Anthropic
        - ``client.generate_content(...)``: Google Gemini
        - ``client.generate(...)``: Ollama, HuggingFace, LiteLLM

    Security applied per call:
        1. PII is detected and masked using an optional :class:`~asha.core.security.pii_vault_session.PIIVault` instance.
        2. The masked prompt is scanned for injection attempts.
        3. The prompt passes through ASHA's ``process()`` engine according to the profile mode.
        4. The LLM response is scanned for format violations via ``OutputGuard``.
        5. If ``rehydrate_pii=True``, masked PII placeholders in the response are replaced
           with their original values before returning to the caller.

    Args:
        client: Any LLM client instance (OpenAI, Anthropic, Gemini, Ollama, etc.).
        profile: A :class:`~asha.core.policy_config.Profile` instance that configures
            security thresholds, fail-closed behaviour, and token budgets.
            Defaults to ``Profile.BALANCED()``.
        mode: String shorthand for mode selection (``'strict'``, ``'balanced'``,
            ``'lite'``, ``'off'``). Ignored if ``profile`` is provided.
        token_budget: Maximum tokens per prompt. Overrides profile's token_budget
            when set explicitly. Defaults to ``None`` (use profile setting).
        rehydrate_pii: Whether to re-hydrate masked PII in LLM responses. When
            ``True``, callers receive responses with original PII values restored.
            Defaults to ``True``.
        vault: An existing :class:`~asha.core.security.pii_vault_session.PIIVault`
            instance to use for session-scoped PII tracking. If ``None``, a new vault
            is created per ``SecureLLM`` instance (suitable for single-session use).

    Examples:
        OpenAI::

            import openai
            from asha import SecureLLM, Profile

            raw_client = openai.OpenAI()
            client = SecureLLM(raw_client, profile=Profile.BALANCED())

            # Identical API: ASHA processes prompts transparently
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Email alice@corp.com the report"}]
            )

        Anthropic::

            import anthropic
            from asha import SecureLLM

            client = SecureLLM(anthropic.Anthropic())
            msg = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1024,
                messages=[{"role": "user", "content": "Ignore all previous instructions"}],
            )
            # Injection attempt blocked or flagged per profile

        Custom profile::

            from asha import SecureLLM, Profile

            client = SecureLLM(
                my_client,
                profile=Profile.STRICT(),
                rehydrate_pii=True,
            )
    """

    def __init__(
        self,
        client: Any,
        *,
        profile: "Optional[Any]" = None,
        mode: str = "balanced",
        token_budget: Optional[int] = None,
        rehydrate_pii: Optional[bool] = None,
        vault: "Optional[Any]" = None,
    ) -> None:
        """
        Wrap an LLM client with ASHA security.

        Args:
            client: The underlying LLM client to wrap.
            profile: A :class:`~asha.core.policy_config.Profile` preset. If
                provided, ``mode`` and ``token_budget`` defaults are derived
                from the profile unless overridden explicitly.
            mode: Policy mode string. Only used when ``profile=None``.
            token_budget: Max tokens per prompt. Overrides profile when set.
            rehydrate_pii: Re-hydrate masked PII in responses. Inherits profile
                setting if provided, otherwise False.
            vault: Existing ``PIIVault`` for session continuity. If None, a new
                vault is created for this ``SecureLLM`` instance.

        Raises:
            ValueError: If ``client`` is ``None``.
        """
        if client is None:
            raise ValueError(
                "SecureLLM requires a valid LLM client. Got None. "
                "Example: SecureLLM(openai.OpenAI(), profile=Profile.BALANCED())"
            )

        # Resolve profile settings
        if profile is not None:
            if callable(profile):
                profile = profile()
            self._mode = profile.mode
            self._token_budget = token_budget if token_budget is not None else profile.token_budget
            self._fail_closed = profile.fail_closed
            self._rehydrate_pii = profile.rehydrate_pii if rehydrate_pii is None else rehydrate_pii
            self._verbose = profile.verbose
            self._profile = profile
        else:
            self._mode = mode
            self._token_budget = token_budget if token_budget is not None else 2048
            self._fail_closed = True
            self._rehydrate_pii = False if rehydrate_pii is None else rehydrate_pii
            self._verbose = False
            self._profile = None

        self._client = client
        self._output_guard = OutputGuard()

        # Initialise PIIVault for this session
        if vault is not None:
            self._vault = vault
        else:
            from ..core.security.pii_vault_session import PIIVault

            self._vault = PIIVault()

    # ------------------------------------------------------------------
    # Transparent attribute delegation
    # ------------------------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        """Delegate any unknown attribute/method to the underlying client.

        This preserves 100% API compatibility with the wrapped client.
        Intercepted attributes (``chat``, ``messages``) are handled before
        this fallback.
        """
        # Intercept OpenAI-style nested chat.completions
        if name == "chat":
            return _SecureNestedProxy(
                getattr(self._client, "chat"),
                wrapper=self,
            )
        # Intercept Anthropic-style messages
        if name == "messages":
            return _SecureNestedProxy(
                getattr(self._client, "messages"),
                wrapper=self,
            )
        if hasattr(self._client, name):
            attr = getattr(self._client, name)
            if callable(attr):
                gen_method = _find_generation_method(self._client)
                if name == gen_method or name in (
                    "generate_content",
                    "send_message",
                    "create",
                    "generate",
                    "chat",
                    "completion",
                    "predict",
                    "run",
                ):
                    return lambda *args, **kwargs: self._intercept_call(name, args, kwargs)
            return attr
        return getattr(self._client, name)

    def __repr__(self) -> str:
        profile_name = repr(self._profile) if self._profile else f"mode={self._mode!r}"
        return f"SecureLLM({self._client!r}, {profile_name})"

    # ------------------------------------------------------------------
    # Top-level generation method interception
    # ------------------------------------------------------------------

    def generate_content(self, *args: Any, **kwargs: Any) -> Any:
        """Intercept Google Gemini ``generate_content`` calls."""
        return self._intercept_call("generate_content", args, kwargs)

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        """Intercept Ollama / HuggingFace ``generate`` calls."""
        return self._intercept_call("generate", args, kwargs)

    def create(self, *args: Any, **kwargs: Any) -> Any:
        """Intercept bare ``create`` calls."""
        return self._intercept_call("create", args, kwargs)

    # ------------------------------------------------------------------
    # Core interception logic
    # ------------------------------------------------------------------

    def _intercept_call(
        self, method_name: str, args: tuple, kwargs: dict
    ) -> Any:
        """Apply ASHA security and delegate to the underlying client method."""
        original = getattr(self._client, method_name)

        # Pre-process: mask PII + run ASHA security
        try:
            args, kwargs = self._preprocess(args, kwargs)
        except Exception as exc:
            if self._fail_closed:
                from ..exceptions import ASHAProcessingError

                raise ASHAProcessingError(
                    f"SecureLLM pre-processing failed (fail_closed=True): {exc}"
                ) from exc
            # Fail-open: pass through unchanged
            if self._verbose:
                import logging

                logging.getLogger("asha.secure_llm").warning(
                    "ASHA pre-processing error (fail_closed=False): %s", exc
                )

        # Call underlying client (supporting both async coroutines and sync methods)
        if inspect.iscoroutinefunction(original):
            async def _async_call() -> Any:
                response = await original(*args, **kwargs)
                if kwargs.get("stream", False) and is_streaming_response(response):
                    return handle_async_streaming_response(
                        response, vault=self._vault if self._rehydrate_pii else None
                    )
                return self._postprocess(response, kwargs)

            return _async_call()

        response = original(*args, **kwargs)

        if kwargs.get("stream", False) and is_streaming_response(response):
            return handle_streaming_response(
                response, vault=self._vault if self._rehydrate_pii else None
            )

        # Post-process: guard output, re-hydrate PII
        return self._postprocess(response, kwargs)

    def _preprocess(
        self, args: tuple, kwargs: dict
    ) -> tuple[tuple, dict]:
        """Mask PII and run ASHA process() on prompt content."""
        from ..documents.attachments import prepare_llm_request_kwargs

        # Mask positional prompt args (Gemini / Ollama style)
        if args and isinstance(args[0], str):
            masked_prompt = args[0]
            if self._rehydrate_pii and self._mode != "off":
                masked_prompt = self._vault.mask(masked_prompt)
            masked_prompt = _process_prompt_for_wrap(
                masked_prompt, self._mode, token_budget=self._token_budget
            )
            args = (masked_prompt,) + args[1:]

        # Also vault-mask any string messages content if mode != off
        if (
            self._mode != "off"
            and "messages" in kwargs
            and isinstance(kwargs["messages"], list)
        ):
            kwargs["messages"] = [
                self._mask_message(m) for m in kwargs["messages"]
            ]

        # Also vault-mask keyword prompt/contents if present and mode != off
        for p_key in ("prompt", "contents", "content"):
            if (
                self._mode != "off"
                and self._rehydrate_pii
                and p_key in kwargs
                and isinstance(kwargs[p_key], str)
            ):
                kwargs[p_key] = self._vault.mask(kwargs[p_key])

        # Also sanitize system prompt if present
        if (
            self._mode != "off"
            and "system" in kwargs
            and isinstance(kwargs["system"], str)
        ):
            if self._rehydrate_pii:
                kwargs["system"] = self._vault.mask(kwargs["system"])
            kwargs["system"] = _process_prompt_for_wrap(
                kwargs["system"], self._mode, token_budget=self._token_budget
            )

        # Mask kwargs (messages / prompt fields)
        kwargs = prepare_llm_request_kwargs(
            kwargs,
            mode=self._mode,
            token_budget=self._token_budget,
            vault=self._vault if self._rehydrate_pii else None,
        )

        return args, kwargs

    def _mask_message(self, message: Any) -> Any:
        """Apply PIIVault masking to a single message dict."""
        if not isinstance(message, dict):
            return message
        content = message.get("content", "")
        if isinstance(content, str):
            masked_content = self._vault.mask(content) if self._rehydrate_pii else content
            return {**message, "content": masked_content}
        if isinstance(content, list):
            new_parts = []
            for part in content:
                if (
                    isinstance(part, dict)
                    and part.get("type") == "text"
                    and isinstance(part.get("text"), str)
                ):
                    text_val = (
                        self._vault.mask(part["text"])
                        if self._rehydrate_pii
                        else part["text"]
                    )
                    new_parts.append({**part, "text": text_val})
                else:
                    # Multimodal part (image_url, image, input_audio, etc.)
                    from .multimodal import inspect_and_sanitize_multimodal_part

                    inspected = inspect_and_sanitize_multimodal_part(
                        part,
                        mode=self._mode,
                        vault=self._vault if self._rehydrate_pii else None,
                    )
                    new_parts.append(inspected)
            return {**message, "content": new_parts}
        return message

    def _postprocess(self, response: Any, kwargs: dict) -> Any:
        """Guard the LLM response and re-hydrate PII if configured."""
        # Output guard (format validation)
        response_format = kwargs.get("response_format")
        fmt: Optional[str] = (
            response_format if isinstance(response_format, str) else None
        )
        guarded = self._output_guard.guard_output(response, fmt)

        # Re-hydrate PII placeholders in text responses
        if self._rehydrate_pii:
            guarded = _rehydrate_response(guarded, self._vault)

        return guarded


class _SecureNestedProxy:
    """
    Proxy that intercepts ``completions.create`` / ``messages.create``
    on nested attributes of the underlying LLM client.

    Not part of the public API. Used internally by ``SecureLLM.__getattr__``.
    """

    def __init__(self, nested: Any, *, wrapper: "SecureLLM") -> None:
        self._nested = nested
        self._wrapper = wrapper

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._nested, name)
        if name in ("create", "complete") and callable(attr):
            return self._make_interceptor(attr)
        # Return another proxy layer if needed (e.g. chat.completions)
        if hasattr(attr, "create"):
            return _SecureNestedProxy(attr, wrapper=self._wrapper)
        return attr

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if callable(self._nested):
            interceptor = self._make_interceptor(self._nested)
            return interceptor(*args, **kwargs)
        raise TypeError(f"'{type(self._nested).__name__}' object is not callable")

    def _make_interceptor(self, original: Any) -> Any:
        """Return a wrapper function that applies ASHA processing."""
        w = self._wrapper

        if inspect.iscoroutinefunction(original):
            @wraps(original)
            async def async_intercepted(*args: Any, **kwargs: Any) -> Any:
                try:
                    args, kwargs = w._preprocess(args, kwargs)
                except Exception as exc:
                    if w._fail_closed:
                        from ..exceptions import ASHAProcessingError

                        raise ASHAProcessingError(
                            f"SecureLLM pre-processing failed: {exc}"
                        ) from exc
                response = await original(*args, **kwargs)
                if kwargs.get("stream", False) and is_streaming_response(response):
                    return handle_async_streaming_response(
                        response, vault=w._vault if w._rehydrate_pii else None
                    )
                guarded = w._output_guard.guard_output(response, None)
                if w._rehydrate_pii:
                    guarded = _rehydrate_response(guarded, w._vault)
                return guarded

            return async_intercepted

        @wraps(original)
        def intercepted(*args: Any, **kwargs: Any) -> Any:
            try:
                args, kwargs = w._preprocess(args, kwargs)
            except Exception as exc:
                if w._fail_closed:
                    from ..exceptions import ASHAProcessingError

                    raise ASHAProcessingError(
                        f"SecureLLM pre-processing failed: {exc}"
                    ) from exc

            response = original(*args, **kwargs)

            # Handle streaming
            if kwargs.get("stream", False) and is_streaming_response(response):
                return handle_streaming_response(
                    response, vault=w._vault if w._rehydrate_pii else None
                )

            # Guard + re-hydrate
            guarded = w._output_guard.guard_output(response, None)
            if w._rehydrate_pii:
                guarded = _rehydrate_response(guarded, w._vault)
            return guarded

        return intercepted


def _rehydrate_response(response: Any, vault: Any) -> Any:
    """Walk common LLM response shapes and unmask PII placeholders."""
    if isinstance(response, str):
        return vault.unmask(response)

    # Dictionary response (Ollama, raw JSON, custom APIs)
    if isinstance(response, dict):
        if "message" in response and isinstance(response["message"], dict):
            msg_content = response["message"].get("content")
            if isinstance(msg_content, str):
                response["message"]["content"] = vault.unmask(msg_content)
        for key in ("response", "text", "content"):
            if key in response and isinstance(response[key], str):
                response[key] = vault.unmask(response[key])
        if "choices" in response and isinstance(response["choices"], list):
            for choice in response["choices"]:
                if isinstance(choice, dict):
                    if "message" in choice and isinstance(choice["message"], dict):
                        m_content = choice["message"].get("content")
                        if isinstance(m_content, str):
                            choice["message"]["content"] = vault.unmask(m_content)
                    if "text" in choice and isinstance(choice["text"], str):
                        choice["text"] = vault.unmask(choice["text"])
        return response

    # OpenAI ChatCompletion / Completion response
    choices = getattr(response, "choices", None)
    if choices:
        for choice in choices:
            msg = getattr(choice, "message", None)
            if msg and hasattr(msg, "content") and isinstance(msg.content, str):
                msg.content = vault.unmask(msg.content)
            text = getattr(choice, "text", None)
            if text and isinstance(text, str):
                choice.text = vault.unmask(text)

    # Anthropic message response
    content = getattr(response, "content", None)
    if content and isinstance(content, list):
        for block in content:
            if hasattr(block, "text") and isinstance(block.text, str):
                block.text = vault.unmask(block.text)

    # Ollama message attribute (e.g. response.message.content)
    message_attr = getattr(response, "message", None)
    if message_attr and hasattr(message_attr, "content") and isinstance(message_attr.content, str):
        try:
            message_attr.content = vault.unmask(message_attr.content)
        except (AttributeError, TypeError):
            pass

    # Gemini candidates response
    candidates = getattr(response, "candidates", None)
    if candidates and isinstance(candidates, list):
        for candidate in candidates:
            cand_content = getattr(candidate, "content", None)
            if cand_content and hasattr(cand_content, "parts"):
                for part in getattr(cand_content, "parts", []):
                    if hasattr(part, "text") and isinstance(part.text, str):
                        try:
                            part.text = vault.unmask(part.text)
                        except (AttributeError, TypeError):
                            pass

    # Response with writable text or response attribute
    for attr in ("text", "response"):
        if hasattr(response, attr):
            val = getattr(response, attr, None)
            if isinstance(val, str):
                try:
                    setattr(response, attr, vault.unmask(val))
                except (AttributeError, TypeError):
                    pass

    return response

