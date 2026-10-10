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
Output guard layer for ASHA.

Ensures safe, valid outputs and prevents downstream pipeline breaks.
"""

import json
import re
import xml.etree.ElementTree as ET
from typing import Any, Dict, Union, Optional, List, cast
from dataclasses import dataclass


@dataclass
class ValidationResult:
    """Result of output validation."""

    is_valid: bool
    original_output: Any
    fixed_output: Any
    validation_type: str
    error_message: Optional[str] = None


def safe_repair_json(text: str) -> Union[Dict, List, str]:
    """
    Attempt to repair broken JSON output.

    Args:
        text: Potentially broken JSON text

    Returns:
        Reparsed JSON object or original text if unrecoverable
    """
    try:
        # Try direct parsing first
        return cast(Union[Dict[Any, Any], List[Any], str], json.loads(text))
    except json.JSONDecodeError:
        pass

    # Try to extract JSON from text
    # Look for JSON object pattern
    json_match = re.search(r"\{.*\}", text, re.DOTALL)
    if json_match:
        try:
            return cast(
                Union[Dict[Any, Any], List[Any], str], json.loads(json_match.group())
            )
        except json.JSONDecodeError:
            pass

    # Look for JSON array pattern
    array_match = re.search(r"\[.*\]", text, re.DOTALL)
    if array_match:
        try:
            return cast(
                Union[Dict[Any, Any], List[Any], str], json.loads(array_match.group())
            )
        except json.JSONDecodeError:
            pass

    # Try common JSON fixes
    fixed_text = text.strip()

    # Remove common prefixes/suffixes
    fixed_text = re.sub(r"^[^{]*", "", fixed_text)  # Remove prefix before {
    fixed_text = re.sub(r"[^}]*$", "", fixed_text)  # Remove suffix after }

    # Fix common JSON issues
    fixed_text = re.sub(r",\s*}", "}", fixed_text)  # Remove trailing commas
    # Remove trailing commas in arrays
    fixed_text = re.sub(r",\s*\]", "]", fixed_text)
    # Single quotes to double quotes
    fixed_text = re.sub(r"\'", '"', fixed_text)

    try:
        return cast(Union[Dict[Any, Any], List[Any], str], json.loads(fixed_text))
    except json.JSONDecodeError:
        pass

    # Last resort: return original text
    return text


def safe_repair_xml(text: str) -> str:
    """
    Attempt to repair broken XML output.

    Args:
        text: Potentially broken XML text

    Returns:
        Repaired XML or original text if unrecoverable
    """
    try:
        # Try direct parsing first
        ET.fromstring(text)
        return text
    except ET.ParseError:
        pass

    # Try to extract XML from text
    xml_match = re.search(r"<[^>]+>.*</[^>]+>", text, re.DOTALL)
    if xml_match:
        try:
            ET.fromstring(xml_match.group())
            return xml_match.group()
        except ET.ParseError:
            pass

    # Try common XML fixes
    fixed_text = text.strip()

    # Remove common prefixes/suffixes
    fixed_text = re.sub(r"^[^<]*", "", fixed_text)  # Remove prefix before <
    fixed_text = re.sub(r"[^>]*$", "", fixed_text)  # Remove suffix after >

    try:
        ET.fromstring(fixed_text)
        return fixed_text
    except ET.ParseError:
        pass

    # Return original text if unrecoverable
    return text


def safe_repair_code(
    text: str, language: Optional[str] = None
) -> Union[str, Dict[Any, Any], List[Any]]:
    """
    Attempt to repair broken code output.

    Args:
        text: Potentially broken code text
        language: Programming language hint

    Returns:
        Repaired code or original text if unrecoverable
    """
    # Remove common prefixes/suffixes around code
    fixed_text = text.strip()

    # Remove markdown code fences if present
    fixed_text = re.sub(r"^```[a-zA-Z]*\n", "", fixed_text)
    fixed_text = re.sub(r"\n```$", "", fixed_text)
    fixed_text = re.sub(r"^```[a-zA-Z]*\r?\n", "", fixed_text)
    fixed_text = re.sub(r"\r?\n```$", "", fixed_text)

    # Remove explanatory text before/after code
    if language:
        # Language-specific patterns
        if language.lower() in ["python", "py"]:
            # Remove "Here's the Python code:" etc.
            fixed_text = re.sub(
                r"^.*?python.*?[:\n]\s*", "", fixed_text, flags=re.IGNORECASE
            )
        elif language.lower() in ["javascript", "js"]:
            fixed_text = re.sub(
                r"^.*?javascript.*?[:\n]\s*", "", fixed_text, flags=re.IGNORECASE
            )
        elif language.lower() in ["json"]:
            # JSON should be handled by safe_repair_json
            return safe_repair_json(text)

    return fixed_text


def validate_output(
    output: Any, expected_format: Optional[str] = None
) -> ValidationResult:
    """
    Validate and potentially fix output based on expected format.

    Args:
        output: Output to validate
        expected_format: Expected format (json, xml, code, etc.)

    Returns:
        ValidationResult with validation results
    """
    if expected_format is None:
        # No validation required
        return ValidationResult(
            is_valid=True,
            original_output=output,
            fixed_output=output,
            validation_type="none",
        )

    # Convert to string for processing
    if not isinstance(output, str):
        output_str = str(output)
    else:
        output_str = output

    expected_format = expected_format.lower()

    if expected_format == "json":
        fixed_output = safe_repair_json(output_str)
        is_valid = isinstance(fixed_output, (dict, list))

        return ValidationResult(
            is_valid=is_valid,
            original_output=output,
            fixed_output=fixed_output,
            validation_type="json",
            error_message=None if is_valid else "JSON parsing failed",
        )

    elif expected_format == "xml":
        fixed_output = safe_repair_xml(output_str)
        try:
            ET.fromstring(fixed_output)
            is_valid = True
        except ET.ParseError:
            is_valid = False

        return ValidationResult(
            is_valid=is_valid,
            original_output=output,
            fixed_output=fixed_output,
            validation_type="xml",
            error_message=None if is_valid else "XML parsing failed",
        )

    elif expected_format in [
        "code",
        "python",
        "javascript",
        "js",
        "java",
        "cpp",
        "c++",
    ]:
        fixed_output = safe_repair_code(output_str, expected_format)
        # Basic validation: check if it looks like code
        is_valid = any(
            char in fixed_output
            for char in ["{", "}", "(", ")", ";", "=", "def", "class", "function"]
        )

        return ValidationResult(
            is_valid=is_valid,
            original_output=output,
            fixed_output=fixed_output,
            validation_type="code",
            error_message=None if is_valid else "Code validation inconclusive",
        )

    else:
        # Unknown format, return as-is
        return ValidationResult(
            is_valid=True,
            original_output=output,
            fixed_output=output,
            validation_type="unknown",
        )


def detect_output_format(output: str) -> str:
    """
    Auto-detect output format from content.

    Args:
        output: Output text to analyze

    Returns:
        Detected format (json, xml, code, text)
    """
    output = output.strip()

    # Check for JSON
    if (output.startswith("{") and output.endswith("}")) or (
        output.startswith("[") and output.endswith("]")
    ):
        try:
            json.loads(output)
            return "json"
        except:
            pass

    # Check for XML
    if output.startswith("<") and output.endswith(">"):
        try:
            ET.fromstring(output)
            return "xml"
        except:
            pass

    # Check for code patterns
    code_indicators = [
        "def ",
        "class ",
        "function ",
        "import ",
        "from ",
        "const ",
        "let ",
        "var ",
        "{",
        "}",
        "(",
        ")",
        ";",
        "=",
        "#include",
        "using ",
    ]

    if any(indicator in output for indicator in code_indicators):
        return "code"

    # Default to text
    return "text"


class OutputGuard:
    """
    Comprehensive output guard for ASHA.

    Ensures all outputs are safe and valid before returning to applications.
    """

    def __init__(self, auto_detect: bool = True) -> None:
        """
        Initialize output guard.

        Args:
            auto_detect: Whether to auto-detect output format
        """
        self.auto_detect = auto_detect

    def guard_output(self, output: Any, expected_format: Optional[str] = None) -> Any:
        """
        Guard output with validation and repair.

        Args:
            output: Output to guard
            expected_format: Expected output format

        Returns:
            Guarded output (repaired if needed)
        """
        try:
            if expected_format is None and self.auto_detect:
                # Auto-detect format
                if isinstance(output, str):
                    expected_format = detect_output_format(output)

            # Validate and repair
            result = validate_output(output, expected_format)

            # Return fixed output if validation failed
            return result.fixed_output if not result.is_valid else output

        except Exception:
            # Fail-safe: return original output
            return output

    def guard_streaming_chunk(
        self, chunk: Any, expected_format: Optional[str] = None
    ) -> Any:
        """
        Guard individual streaming chunks.

        Args:
            chunk: Streaming chunk to guard
            expected_format: Expected output format

        Returns:
            Guarded chunk
        """
        # For streaming, we do minimal validation to avoid latency
        try:
            if expected_format == "json" and isinstance(chunk, str):
                # Only try to fix obvious JSON issues in streaming
                if chunk.strip().startswith("{") and not chunk.strip().endswith("}"):
                    # Add missing closing brace if obvious
                    return chunk + "}"
            return chunk
        except Exception:
            return chunk


# Convenience functions
def safe_output(output: Any, expected_format: Optional[str] = None) -> Any:
    """
    Safely validate and repair output.

    Args:
        output: Output to validate
        expected_format: Expected format

    Returns:
        Safe output
    """
    guard = OutputGuard()
    return guard.guard_output(output, expected_format)


def safe_json_output(output: Any) -> Union[Dict, List, str]:
    """
    Safely ensure JSON output.

    Args:
        output: Output to ensure is JSON

    Returns:
        JSON object or repaired string
    """
    if isinstance(output, str):
        return safe_repair_json(output)
    else:
        return cast(Union[Dict[Any, Any], List[Any], str], output)
