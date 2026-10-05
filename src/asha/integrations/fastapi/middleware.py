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
FastAPI middleware for ASHA

Automatically processes requests containing prompts through ASHA
security and optimization pipeline.
"""

from typing import Dict, Any, Optional, List, Callable, Awaitable, TYPE_CHECKING, cast

import json

if TYPE_CHECKING:
    from fastapi import Request, Response
    from starlette.middleware.base import BaseHTTPMiddleware

    FASTAPI_AVAILABLE = True
else:
    try:
        from fastapi import Request, Response
        from starlette.middleware.base import BaseHTTPMiddleware

        FASTAPI_AVAILABLE = True
    except ImportError:
        FASTAPI_AVAILABLE = False

        class BaseHTTPMiddleware:
            def __init__(self, app: Any, **kwargs: Any) -> None:
                self.app = app

        class Request:
            pass

        class Response:
            pass


from ...utils.dropin import process


class ASHAMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware that automatically processes prompts through ASHA.

    This middleware intercepts requests containing prompts, processes them
    through ASHA's security and optimization pipeline, and forwards the
    optimized prompts to the actual LLM endpoints.

    Usage:
        from asha.integrations.fastapi import ASHAMiddleware
        app.add_middleware(ASHAMiddleware)

    Configuration:
        app.add_middleware(ASHAMiddleware,
                          privacy=True,
                          token_budget=1200,
                          endpoints=["/chat", "/completions"],
                          prompt_fields=["prompt", "messages", "input"])
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
        Initialize ASHA middleware.

        Args:
            app: FastAPI application instance
            privacy: Enable privacy features
            token_budget: Token budget for optimization
            endpoints: List of endpoints to process
            prompt_fields: List of fields containing prompts
            debug_mode: Enable debug mode
        """
        if not FASTAPI_AVAILABLE:
            raise ImportError(
                "FastAPI is not installed. Install with: pip install fastapi uvicorn\n"
                "Or use the standalone ASHA functions instead."
            )

        super().__init__(app)
        self.privacy = privacy
        self.token_budget = token_budget
        self.endpoints = endpoints or ["/chat", "/completions", "/generate"]
        self.prompt_fields = prompt_fields or ["prompt", "messages", "input", "text"]
        self.debug_mode = debug_mode

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """
        Process request through ASHA pipeline.

        Args:
            request: Incoming request
            call_next: Next middleware in chain

        Returns:
            Processed response
        """
        # Check if this endpoint should be processed
        if not self._should_process_endpoint(request.url.path):
            return await call_next(request)

        # Process the request
        try:
            # Get request body
            body = await self._get_request_body(request)
            if not body:
                return await call_next(request)

            # Extract and process prompts
            processed_body, processing_info = await self._process_prompts(body)

            if self.debug_mode:
                print(
                    f"ASHA Debug: Processed {processing_info['prompts_processed']} prompts"
                )
                print(
                    f"ASHA Debug: Token reduction: {processing_info.get('avg_token_reduction', 0)}%"
                )

            # Create new request with processed body
            processed_request = self._create_processed_request(request, processed_body)

            # Continue with processed request
            response = await call_next(processed_request)

            # Add processing info to response headers (optional)
            if self.debug_mode and processing_info["prompts_processed"] > 0:
                response.headers["X-ASHA-Processed"] = "true"
                response.headers["X-ASHA-Prompts"] = str(
                    processing_info["prompts_processed"]
                )
                response.headers["X-ASHA-Token-Reduction"] = str(
                    processing_info.get("avg_token_reduction", 0)
                )

            return response

        except Exception as e:
            if self.debug_mode:
                print(f"ASHA Error: {str(e)}")

            # Fail gracefully - continue with original request
            return await call_next(request)

    def _should_process_endpoint(self, path: str) -> bool:
        """Check if the endpoint should be processed by ASHA."""
        return any(endpoint in path for endpoint in self.endpoints)

    async def _get_request_body(self, request: Request) -> Optional[Dict[str, Any]]:
        """Extract and parse request body."""
        try:
            if request.method in ["POST", "PUT", "PATCH"]:
                content_type = request.headers.get("content-type", "")

                if "application/json" in content_type:
                    body = await request.body()
                    return cast(Dict[str, Any], json.loads(body.decode()))
                elif "application/x-www-form-urlencoded" in content_type:
                    form_data = await request.form()
                    return dict(form_data)
                elif "multipart/form-data" in content_type:
                    form_data = await request.form()
                    return {key: value for key, value in form_data.items()}

            return None

        except Exception:
            return None

    async def _process_prompts(
        self, body: Dict[str, Any]
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Process all prompts in the request body.

        Args:
            body: Request body dictionary

        Returns:
            Tuple of (processed_body, processing_info)
        """
        processed_body = body.copy()
        processing_info: Dict[str, Any] = {
            "prompts_processed": 0,
            "total_token_reduction": 0,
            "avg_token_reduction": 0,
            "security_threats_detected": 0,
        }

        # Find and process prompts
        prompts_found = self._find_prompts(body)

        for field_path, prompt in prompts_found:
            try:
                # Process prompt through ASHA
                mode = "balanced" if self.privacy else "off"
                proc = process(
                    prompt,
                    mode=mode,
                    token_budget=self.token_budget,
                )
                result = proc.to_dict()

                # Update the field in the processed body
                self._update_field(processed_body, field_path, proc.output)

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

        return processed_body, processing_info

    def _find_prompts(
        self, body: Dict[str, Any], path: str = ""
    ) -> list[tuple[str, str]]:
        """
        Recursively find all prompts in the request body.

        Args:
            body: Body to search
            path: Current field path (for nested objects)

        Returns:
            List of (field_path, prompt) tuples
        """
        prompts = []

        for key, value in body.items():
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
        self, body: Dict[str, Any], field_path: str, new_value: str
    ) -> None:
        """
        Update a field in the body using the field path.

        Args:
            body: Body to update
            field_path: Field path (e.g., "messages[0].content")
            new_value: New value to set
        """
        try:
            # Parse the field path
            parts = field_path.split(".")
            current = body

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

    def _create_processed_request(
        self, original_request: Request, processed_body: Dict[str, Any]
    ) -> Request:
        """
        Create a new request with the processed body.

        Args:
            original_request: Original request
            processed_body: Processed request body

        Returns:
            New request with processed body
        """
        try:
            # Convert processed body back to JSON
            processed_json = json.dumps(processed_body).encode()

            # Create a custom receive function that returns our processed body
            async def receive() -> Dict[str, Any]:
                return {
                    "type": "http.request",
                    "body": processed_json,
                    "more_body": False,
                }

            # Create a new request scope with updated content-length
            scope = dict(original_request.scope)
            scope["headers"] = [
                (k, v)
                for k, v in scope.get("headers", [])
                if k.lower() != b"content-length"
            ] + [(b"content-length", str(len(processed_json)).encode())]

            # Create new request with processed body
            from fastapi import Request

            processed_request = Request(scope, receive)

            return cast(Request, processed_request)

        except Exception as e:
            if self.debug_mode:
                print(f"ASHA Error creating processed request: {str(e)}")
            # Return original request if processing fails
            return original_request


# Convenience function for easy setup
def add_asha_middleware(
    app: Any,
    privacy: bool = True,
    token_budget: int = 1200,
    endpoints: Optional[List[str]] = None,
    debug_mode: bool = False,
) -> None:
    """
    Add ASHA middleware to FastAPI app.

    Args:
        app: FastAPI application
        privacy: Enable privacy features
        token_budget: Token budget for optimization
        endpoints: List of endpoints to process
        debug_mode: Enable debug mode

    Examples:
        from asha.integrations.fastapi import add_asha_middleware
        add_asha_middleware(app, privacy=True, debug_mode=True)
    """
    app.add_middleware(
        ASHAMiddleware,
        privacy=privacy,
        token_budget=token_budget,
        endpoints=endpoints,
        debug_mode=debug_mode,
    )
