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
Flask Middleware Integration for ASHA

This module provides Flask middleware that automatically processes requests
containing prompts through ASHA's security and optimization pipeline.
"""

from typing import Dict, Any, Optional, List, Callable, TypeVar, TYPE_CHECKING, cast

import json
import time

if TYPE_CHECKING:
    from flask import Flask, request, Response, g
    from werkzeug.local import LocalProxy

    FLASK_AVAILABLE = True
else:
    try:
        from flask import Flask, request, Response, g
        from werkzeug.local import LocalProxy

        FLASK_AVAILABLE = True
    except ImportError:
        FLASK_AVAILABLE = False

        class Flask:
            pass

        class request:
            pass

        class Response:
            headers: Dict[str, str]

        class g:
            pass

        class LocalProxy:
            def __init__(self, func: Callable[[], Any]) -> None:
                self.func = func

            def __getattr__(self, name: str) -> Any:
                return getattr(self.func(), name)


_F = TypeVar("_F", bound=Callable[..., Any])

from ...utils.dropin import process


class ASHAMiddleware:
    """
    Flask middleware that automatically processes prompts through ASHA.

    This middleware intercepts requests containing prompts, processes them
    through ASHA's security and optimization pipeline, and forwards the
    optimized prompts to the actual Flask endpoints.

    Usage:
        from asha.integrations.flask import ASHAMiddleware

        app = Flask(__name__)
        ASHAMiddleware(app)

    Configuration:
        ASHAMiddleware(app,
                          privacy=True,
                          token_budget=1200,
                          endpoints=["/chat", "/completions"],
                          prompt_fields=["prompt", "messages", "input"],
                          debug_mode=False)
    """

    def __init__(
        self,
        app: Any,
        privacy: bool = True,
        token_budget: int = 1200,
        endpoints: Optional[List[str]] = None,
        prompt_fields: Optional[List[str]] = None,
        debug_mode: bool = False,
    ) -> None:
        """
        Initialize ASHA middleware for Flask.

        Args:
            app: Flask application
            privacy: Enable privacy features
            token_budget: Token budget for optimization
            endpoints: List of endpoints to process (default: common LLM endpoints)
            prompt_fields: List of fields that contain prompts (default: common field names)
            debug_mode: Enable debug mode for detailed logging
        """
        if not FLASK_AVAILABLE:
            raise ImportError(
                "Flask is not installed. Install with: pip install flask\n"
                "Or use the standalone ASHA functions instead."
            )

        self.app = app
        self.privacy = privacy
        self.token_budget = token_budget
        self.debug_mode = debug_mode

        # Default endpoints that typically contain LLM prompts
        self.endpoints = endpoints or [
            "/chat/completions",
            "/completions",
            "/chat",
            "/generate",
            "/ask",
            "/query",
            "/api/chat",
            "/api/completions",
            "/v1/chat/completions",
            "/v1/completions",
        ]

        # Default field names that typically contain prompts
        self.prompt_fields = prompt_fields or [
            "prompt",
            "messages",
            "input",
            "query",
            "text",
            "content",
            "question",
            "instruction",
        ]

        # Register middleware
        self._register_middleware()

    def _register_middleware(self) -> None:
        """Register the middleware with Flask."""

        before_request_hook = cast(
            Callable[[Callable[..., Any]], Callable[..., Any]],
            self.app.before_request,
        )
        after_request_hook = cast(
            Callable[[Callable[..., Any]], Callable[..., Any]],
            self.app.after_request,
        )

        @before_request_hook
        def before_request() -> None:
            """Process request before it reaches the endpoint."""
            # Store original request data
            g.asha_original_data = None
            g.asha_processing_info = None
            g.asha_start_time = time.time()

            # Check if this endpoint should be processed
            if not self._should_process_endpoint(request.endpoint or request.path):
                return

            # Process the request
            try:
                # Get request data
                request_data = self._get_request_data()
                if not request_data:
                    return

                # Store original data
                g.asha_original_data = request_data.copy()

                # Extract and process prompts
                processed_data, processing_info = self._process_prompts(request_data)

                if self.debug_mode:
                    print(
                        f"ASHA Debug: Processed {processing_info['prompts_processed']} prompts"
                    )
                    print(
                        f"ASHA Debug: Token reduction: {processing_info.get('avg_token_reduction', 0)}%"
                    )
                    print(
                        f"ASHA Debug: Security threats: {processing_info.get('security_threats_detected', 0)}"
                    )

                # Store processing info
                g.asha_processing_info = processing_info

                # Replace request data with processed data
                self._replace_request_data(processed_data)

            except Exception as e:
                if self.debug_mode:
                    print(f"ASHA Error: {str(e)}")
                # Fail gracefully - continue with original request

        @after_request_hook
        def after_request(response: Response) -> Response:
            """Add processing info to response headers."""
            if hasattr(g, "asha_processing_info") and g.asha_processing_info:
                processing_info = g.asha_processing_info

                if processing_info["prompts_processed"] > 0:
                    # Add processing headers
                    response.headers["X-ASHA-Processed"] = "true"
                    response.headers["X-ASHA-Prompts"] = str(
                        processing_info["prompts_processed"]
                    )
                    response.headers["X-ASHA-Token-Reduction"] = str(
                        processing_info.get("avg_token_reduction", 0)
                    )
                    response.headers["X-ASHA-Threats-Blocked"] = str(
                        processing_info.get("security_threats_detected", 0)
                    )

                    if self.debug_mode:
                        processing_time = (time.time() - g.asha_start_time) * 1000
                        response.headers["X-ASHA-Processing-Time"] = (
                            f"{processing_time:.2f}ms"
                        )

            return response

    def _should_process_endpoint(self, endpoint_path: str) -> bool:
        """Check if the endpoint should be processed by ASHA."""
        # Check if endpoint path matches any of our target endpoints
        return any(endpoint in endpoint_path for endpoint in self.endpoints)

    def _get_request_data(self) -> Optional[Dict[str, Any]]:
        """Extract and parse request data."""
        try:
            if request.method in ["POST", "PUT", "PATCH"]:
                content_type = request.content_type or ""

                if "application/json" in content_type:
                    return cast(Dict[str, Any], request.get_json() or {})
                elif "application/x-www-form-urlencoded" in content_type:
                    return cast(Dict[str, Any], request.form.to_dict())
                elif "multipart/form-data" in content_type:
                    return cast(
                        Dict[str, Any],
                        {key: value for key, value in request.form.items()},
                    )

            return None

        except Exception:
            return None

    def _process_prompts(
        self, data: Dict[str, Any]
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Process all prompts in the request data.

        Args:
            data: Request data dictionary

        Returns:
            Tuple of (processed_data, processing_info)
        """
        processed_data = data.copy()
        processing_info: Dict[str, Any] = {
            "prompts_processed": 0,
            "total_token_reduction": 0,
            "avg_token_reduction": 0,
            "security_threats_detected": 0,
            "pii_masked": 0,
        }

        # Find and process prompts
        prompts_found = self._find_prompts(data)

        for field_path, prompt in prompts_found:
            try:
                # Process prompt through ASHA
                mode = "balanced" if self.privacy else "off"
                result = process(
                    prompt,
                    mode=mode,
                    token_budget=self.token_budget,
                ).to_dict()

                # Update the field in the processed data
                self._update_field(processed_data, field_path, str(result["optimized"]))

                # Track processing info
                processing_info["prompts_processed"] += 1
                processing_info["total_token_reduction"] += result.get(
                    "token_reduction", 0
                )

                security_result = cast(
                    Dict[str, Any], result.get("security_result", {})
                )
                if not security_result.get("is_safe", True):
                    processing_info["security_threats_detected"] += 1

                processing_info["pii_masked"] += len(
                    security_result.get("masked_entities", [])
                )

                if self.debug_mode:
                    print(f"ASHA Debug: Processed field '{field_path}'")
                    print(f"ASHA Debug: Original: {prompt[:100]}...")
                    print(f"ASHA Debug: Optimized: {result['optimized'][:100]}...")
                    print(
                        f"ASHA Debug: Token reduction: {result.get('token_reduction', 0)}%"
                    )

            except Exception as e:
                if self.debug_mode:
                    print(f"ASHA Error processing field '{field_path}': {str(e)}")
                # Continue with original prompt if processing fails
                continue

        # Calculate averages
        if processing_info["prompts_processed"] > 0:
            processing_info["avg_token_reduction"] = (
                processing_info["total_token_reduction"]
                / processing_info["prompts_processed"]
            )

        return processed_data, processing_info

    def _find_prompts(
        self, data: Dict[str, Any], path: str = ""
    ) -> List[tuple[str, str]]:
        """
        Recursively find all prompts in the request data.

        Args:
            data: Data to search
            path: Current field path (for nested objects)

        Returns:
            List of (field_path, prompt) tuples
        """
        prompts = []

        for key, value in data.items():
            current_path = f"{path}.{key}" if path else key

            if isinstance(value, str):
                # Check if this field name suggests it contains a prompt
                if any(field in key.lower() for field in self.prompt_fields):
                    prompts.append((current_path, value))

            elif isinstance(value, list):
                # Handle lists (e.g., OpenAI messages format)
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        # Check for message content
                        if "content" in item and isinstance(item["content"], str):
                            prompts.append(
                                (f"{current_path}[{i}].content", item["content"])
                            )
                        # Recursively search nested dicts
                        prompts.extend(self._find_prompts(item, f"{current_path}[{i}]"))

            elif isinstance(value, dict):
                # Recursively search nested dicts
                prompts.extend(self._find_prompts(value, current_path))

        return prompts

    def _update_field(
        self, data: Dict[str, Any], field_path: str, new_value: str
    ) -> None:
        """
        Update a field in the data using the field path.

        Args:
            data: Data to update
            field_path: Field path (e.g., "messages[0].content")
            new_value: New value to set
        """
        try:
            # Parse the field path
            parts = field_path.split(".")
            current = data

            # Navigate to the parent of the target field
            for part in parts[:-1]:
                if "[" in part and "]" in part:
                    # Handle array access (e.g., "messages[0]")
                    field_name = part.split("[")[0]
                    index = int(part.split("[")[1].split("]")[0])
                    current = current[field_name][index]
                else:
                    current = current[part]

            # Set the final field
            final_field = parts[-1]
            if "[" in final_field and "]" in final_field:
                # Handle array access for the final field
                field_name = final_field.split("[")[0]
                index = int(final_field.split("[")[1].split("]")[0])
                current[field_name][index] = new_value
            else:
                current[final_field] = new_value

        except Exception:
            # If field update fails, skip it
            pass

    def _replace_request_data(self, processed_data: Dict[str, Any]) -> None:
        """Replace the request data with processed data."""
        # Store processed data in Flask's g context
        g.asha_processed_data = processed_data

        # For JSON requests, we need to modify the request
        if request.content_type and "application/json" in request.content_type:
            # Store the processed data for the endpoint to use
            request._asha_processed_json = processed_data


# Flask-specific convenience functions
def add_asha_middleware(
    app: Flask,
    privacy: bool = True,
    token_budget: int = 1200,
    endpoints: Optional[List[str]] = None,
    prompt_fields: Optional[List[str]] = None,
    debug_mode: bool = False,
) -> None:
    """
    Add ASHA middleware to Flask app.

    Args:
        app: Flask application
        privacy: Enable privacy features
        token_budget: Token budget for optimization
        endpoints: List of endpoints to process
        prompt_fields: List of prompt field names to process
        debug_mode: Enable debug mode

    Examples:
        from asha.integrations.flask import add_asha_middleware

        app = Flask(__name__)
        add_asha_middleware(app, privacy=True, debug_mode=True)
    """
    ASHAMiddleware(
        app=app,
        privacy=privacy,
        token_budget=token_budget,
        endpoints=endpoints,
        prompt_fields=prompt_fields,
        debug_mode=debug_mode,
    )


# Flask context proxy for accessing ASHA data
asha_metrics = LocalProxy(lambda: getattr(g, "asha_processing_info", None))
asha_original_data = LocalProxy(lambda: getattr(g, "asha_original_data", None))
asha_processed_data = LocalProxy(lambda: getattr(g, "asha_processed_data", None))


# Flask decorator for endpoint-specific processing
def asha_endpoint(
    privacy: bool = True, token_budget: int = 1200, debug_metrics: bool = False
) -> Callable[[_F], _F]:
    """
    Decorator to apply ASHA processing to specific Flask endpoints.

    Args:
        privacy: Enable privacy features
        token_budget: Token budget for optimization
        debug_metrics: Return optimization metrics

    Usage:
        @app.route('/chat')
        @asha_endpoint(privacy=True, debug_metrics=True)
        def chat_endpoint():
            data = request.get_json()
            # Data is already processed by ASHA
            return jsonify(response)
    """

    def decorator(f: _F) -> _F:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Get request data
            data = request.get_json() or {}

            # Process through ASHA
            try:
                result = process(
                    json.dumps(data),
                    mode="balanced" if privacy else "off",
                    token_budget=token_budget,
                )
                if debug_metrics:
                    # Store metrics in Flask context
                    g.asha_endpoint_metrics = result.to_dict()

                # Update request data
                processed_data = cast(
                    Dict[str, Any],
                    json.loads(result.output),
                )

            except Exception as e:
                if debug_metrics:
                    g.asha_endpoint_error = str(e)
                # Continue with original data
                processed_data = data

            # Store processed data for the endpoint
            g.asha_endpoint_data = processed_data

            return f(*args, **kwargs)

        return cast(_F, wrapper)

    return decorator
