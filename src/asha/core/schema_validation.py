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
Schema Validation Mode - Phase 3 Competitive Feature

Competes with structured-output libraries such as Instructor:
- Auto-retry on validation failures
- Error correction and repair
- JSON enforcement
- Pydantic integration

Ensures no broken JSON reaches users while maintaining reliability.
"""

import json
import re
import time
from typing import Dict, List, Any, Optional, Union, Type, Tuple
from dataclasses import dataclass, field
from enum import Enum

# Try to import Pydantic
try:
    from pydantic import BaseModel, ValidationError

    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    print("Warning: Pydantic not available, using basic JSON validation only")


class ValidationStatus(Enum):
    """Validation result status."""

    VALID = "valid"
    INVALID = "invalid"
    REPAIRED = "repaired"
    FAILED = "failed"


@dataclass
class ValidationResult:
    """Result of schema validation."""

    status: ValidationStatus
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    repaired_content: Optional[str]
    original_content: str
    validation_time_ms: float
    retry_count: int
    metadata: Dict[str, Any]


@dataclass
class SchemaConfig:
    """Configuration for schema validation."""

    max_retries: int = 3
    timeout_ms: int = 5000
    enable_auto_repair: bool = True
    strict_mode: bool = False
    enforce_json: bool = True
    repair_strategies: List[str] = field(
        default_factory=lambda: ["basic", "advanced", "fallback"]
    )

    def __post_init__(self) -> None:
        pass


class JSONRepairEngine:
    """Advanced JSON repair and correction engine."""

    def __init__(self) -> None:
        """Initialize JSON repair engine."""
        self.repair_patterns = self._init_repair_patterns()
        self.common_errors = self._init_common_errors()

    def _init_repair_patterns(self) -> List[Dict[str, Any]]:
        """Initialize JSON repair patterns."""
        return [
            # Trailing commas
            {
                "pattern": r",\s*([}\]])",
                "replacement": r"\1",
                "description": "Remove trailing commas",
            },
            # Missing quotes around keys
            {
                "pattern": r"([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:",
                "replacement": r'\1"\2":',
                "description": "Add quotes around unquoted keys",
            },
            # Single quotes instead of double quotes
            {
                "pattern": r"'([^']*)'",
                "replacement": r'"\1"',
                "description": "Convert single quotes to double quotes",
            },
            # Extra commas in arrays
            {
                "pattern": r",\s*]",
                "replacement": r"]",
                "description": "Remove trailing comma in arrays",
            },
            # Missing commas between objects
            {
                "pattern": r"}\s*{",
                "replacement": "},{",
                "description": "Add comma between objects",
            },
        ]

    def _init_common_errors(self) -> Dict[str, str]:
        """Initialize common error patterns and fixes."""
        return {
            "expecting ',' delimiter": "Missing comma in JSON structure",
            "expecting ':' delimiter": "Missing colon after key",
            "expecting property name enclosed in double quotes": "Unquoted property name",
            "Unterminated string starting at": "Unclosed string literal",
            "Expecting value": "Missing value in JSON",
            "Extra data": "Extra content after valid JSON",
        }

    def repair_json(
        self, json_str: str, max_attempts: int = 5
    ) -> Tuple[str, List[str], bool]:
        """
        Attempt to repair invalid JSON.

        Returns:
            Tuple of (repaired_json, repair_log, success)
        """
        original = json_str
        repair_log: List[str] = []

        for attempt in range(max_attempts):
            try:
                # Try parsing as-is
                json.loads(json_str)
                return json_str, repair_log, True
            except json.JSONDecodeError as e:
                error_msg = str(e)
                repair_log.append(f"Attempt {attempt + 1}: {error_msg}")

                # Apply repair strategies
                repaired = self._apply_repairs(json_str, error_msg)
                if repaired != json_str:
                    json_str = repaired
                    repair_log.append(f"Applied repairs to attempt {attempt + 1}")
                else:
                    repair_log.append(
                        f"No repairs applicable for attempt {attempt + 1}"
                    )

                # If this is the last attempt, try aggressive repair
                if attempt == max_attempts - 1:
                    json_str = self._aggressive_repair(json_str)
                    repair_log.append("Applied aggressive repair on final attempt")

        return json_str, repair_log, False

    def _apply_repairs(self, json_str: str, error_msg: str) -> str:
        """Apply targeted repairs based on error message."""
        repaired = json_str

        # Apply pattern-based repairs
        for pattern_info in self.repair_patterns:
            pattern = pattern_info["pattern"]
            replacement = pattern_info["replacement"]

            if re.search(pattern, repaired):
                repaired = re.sub(pattern, replacement, repaired)

        # Apply error-specific repairs
        for error_pattern, fix_desc in self.common_errors.items():
            if error_pattern in error_msg:
                repaired = self._apply_error_specific_repair(repaired, error_pattern)
                break

        return repaired

    def _apply_error_specific_repair(self, json_str: str, error_pattern: str) -> str:
        """Apply repair for specific error pattern."""
        repaired = json_str

        if "expecting ',' delimiter" in error_pattern:
            # Add missing commas
            repaired = re.sub(r"([^,\s])\s*}", r"\1}", repaired)
            repaired = re.sub(r"([^,\s])\s*\]", r"\1]", repaired)

        elif "expecting ':' delimiter" in error_pattern:
            # Add missing colons
            repaired = re.sub(r'"\w+"\s*([^{])', r'":\1', repaired)

        elif "property name enclosed in double quotes" in error_pattern:
            # Quote unquoted property names
            repaired = re.sub(
                r"([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:", r'\1"\2":', repaired
            )

        return repaired

    def _aggressive_repair(self, json_str: str) -> str:
        """Apply aggressive repair strategies."""
        # Remove all non-JSON characters
        cleaned = re.sub(r"[^\x20-\x7E]", "", json_str)

        # Ensure proper JSON structure
        if not cleaned.strip().startswith(("{", "[")):
            cleaned = "{" + cleaned + "}"

        # Balance brackets and braces
        open_braces = cleaned.count("{")
        close_braces = cleaned.count("}")
        open_brackets = cleaned.count("[")
        close_brackets = cleaned.count("]")

        cleaned += "}" * (open_braces - close_braces)
        cleaned += "]" * (open_brackets - close_brackets)

        return cleaned


class SchemaValidator:
    """Main schema validation engine."""

    def __init__(self, config: Optional[SchemaConfig] = None) -> None:
        """Initialize schema validator."""
        self.config = config or SchemaConfig()
        self.repair_engine = JSONRepairEngine()
        self.validation_stats = {
            "total_validations": 0,
            "successful_validations": 0,
            "repairs_attempted": 0,
            "repairs_successful": 0,
            "failures": 0,
        }

    def validate_json(
        self, content: str, schema: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        """
        Validate JSON content against schema.

        Args:
            content: JSON content to validate
            schema: Optional JSON schema for validation

        Returns:
            ValidationResult with validation details
        """
        start_time = time.time()
        self.validation_stats["total_validations"] += 1

        errors: List[str] = []
        warnings: List[str] = []
        repaired_content = None
        retry_count = 0
        status = ValidationStatus.INVALID

        # Try to parse and validate JSON
        try:
            # First, try to parse as JSON
            parsed_json = json.loads(content)

            # Validate against schema if provided
            if schema:
                self._validate_against_schema(parsed_json, schema, errors, warnings)

            if not errors:
                status = ValidationStatus.VALID
                self.validation_stats["successful_validations"] += 1
            else:
                # Try to repair if validation failed
                if self.config.enable_auto_repair:
                    repaired_content, repair_log, success = self._attempt_repair(
                        content, schema
                    )
                    retry_count = len(repair_log)
                    if success:
                        status = ValidationStatus.REPAIRED
                        self.validation_stats["repairs_successful"] += 1
                    else:
                        status = ValidationStatus.FAILED
                        self.validation_stats["failures"] += 1
                    errors.extend(repair_log)
                else:
                    status = ValidationStatus.FAILED
                    self.validation_stats["failures"] += 1

        except json.JSONDecodeError as e:
            errors.append(f"JSON parsing error: {str(e)}")

            # Try to repair JSON
            if self.config.enable_auto_repair:
                repaired_content, repair_log, success = self.repair_engine.repair_json(
                    content, self.config.max_retries
                )
                retry_count = len(repair_log)

                if success:
                    try:
                        # Validate repaired JSON
                        parsed_json = json.loads(repaired_content)
                        if schema:
                            self._validate_against_schema(
                                parsed_json, schema, errors, warnings
                            )

                        if not errors:
                            status = ValidationStatus.REPAIRED
                            self.validation_stats["repairs_successful"] += 1
                        else:
                            status = ValidationStatus.FAILED
                            self.validation_stats["failures"] += 1
                    except json.JSONDecodeError:
                        status = ValidationStatus.FAILED
                        self.validation_stats["failures"] += 1
                else:
                    status = ValidationStatus.FAILED
                    self.validation_stats["failures"] += 1
                    errors.extend(repair_log)
            else:
                status = ValidationStatus.FAILED
                self.validation_stats["failures"] += 1

        validation_time = (time.time() - start_time) * 1000

        return ValidationResult(
            status=status,
            is_valid=status == ValidationStatus.VALID,
            errors=errors,
            warnings=warnings,
            repaired_content=repaired_content,
            original_content=content,
            validation_time_ms=validation_time,
            retry_count=retry_count,
            metadata={
                "schema_provided": schema is not None,
                "auto_repair_enabled": self.config.enable_auto_repair,
                "max_retries": self.config.max_retries,
            },
        )

    def validate_pydantic(self, content: str, model_class: Type) -> ValidationResult:
        """
        Validate content against Pydantic model.

        Args:
            content: JSON content to validate
            model_class: Pydantic model class

        Returns:
            ValidationResult with validation details
        """
        if not PYDANTIC_AVAILABLE:
            return ValidationResult(
                status=ValidationStatus.FAILED,
                is_valid=False,
                errors=["Pydantic not available"],
                warnings=[],
                repaired_content=None,
                original_content=content,
                validation_time_ms=0,
                retry_count=0,
                metadata={"pydantic_available": False},
            )

        start_time = time.time()
        self.validation_stats["total_validations"] += 1

        errors: List[str] = []
        warnings: List[str] = []
        repaired_content = None
        retry_count = 0
        status = ValidationStatus.INVALID

        try:
            # Try to parse JSON first
            parsed_json = json.loads(content)

            # Validate against Pydantic model
            try:
                model_instance = model_class(**parsed_json)
                status = ValidationStatus.VALID
                self.validation_stats["successful_validations"] += 1

                # Convert back to JSON for consistency
                repaired_content = model_instance.json()

            except ValidationError as e:
                errors.extend([f"Validation error: {error}" for error in e.errors()])

                # Try to repair and revalidate
                if self.config.enable_auto_repair:
                    repaired_content, repair_log, success = (
                        self._attempt_pydantic_repair(content, model_class, e.errors())
                    )
                    retry_count = len(repair_log)

                    if success:
                        status = ValidationStatus.REPAIRED
                        self.validation_stats["repairs_successful"] += 1
                    else:
                        status = ValidationStatus.FAILED
                        self.validation_stats["failures"] += 1
                else:
                    status = ValidationStatus.FAILED
                    self.validation_stats["failures"] += 1

        except json.JSONDecodeError as e:
            errors.append(f"JSON parsing error: {str(e)}")

            # Try to repair JSON first, then validate with Pydantic
            if self.config.enable_auto_repair:
                repaired_json, repair_log, json_success = (
                    self.repair_engine.repair_json(content, self.config.max_retries)
                )
                retry_count = len(repair_log)

                if json_success:
                    try:
                        parsed_json = json.loads(repaired_json)
                        model_instance = model_class(**parsed_json)
                        repaired_content = model_instance.json()
                        status = ValidationStatus.REPAIRED
                        self.validation_stats["repairs_successful"] += 1
                    except (ValidationError, json.JSONDecodeError) as e:
                        status = ValidationStatus.FAILED
                        self.validation_stats["failures"] += 1
                        if isinstance(e, ValidationError):
                            errors.extend(
                                [f"Validation error: {error}" for error in e.errors()]
                            )
                else:
                    status = ValidationStatus.FAILED
                    self.validation_stats["failures"] += 1
            else:
                status = ValidationStatus.FAILED
                self.validation_stats["failures"] += 1

        validation_time = (time.time() - start_time) * 1000

        return ValidationResult(
            status=status,
            is_valid=status == ValidationStatus.VALID,
            errors=errors,
            warnings=warnings,
            repaired_content=repaired_content,
            original_content=content,
            validation_time_ms=validation_time,
            retry_count=retry_count,
            metadata={
                "pydantic_model": model_class.__name__,
                "pydantic_available": True,
                "auto_repair_enabled": self.config.enable_auto_repair,
            },
        )

    def _validate_against_schema(
        self, data: Any, schema: Dict[str, Any], errors: List[str], warnings: List[str]
    ) -> None:
        """Validate data against JSON schema."""
        try:
            # Use jsonschema if available
            import jsonschema

            jsonschema.validate(data, schema)
        except ImportError:
            warnings.append("jsonschema not available, skipping schema validation")
        except jsonschema.ValidationError as e:
            errors.append(f"Schema validation error: {e.message}")

    def _attempt_repair(
        self, content: str, schema: Optional[Dict[str, Any]]
    ) -> Tuple[str, List[str], bool]:
        """Attempt to repair content and revalidate."""
        repair_log = []

        # Use JSON repair engine
        repaired_content, log, success = self.repair_engine.repair_json(
            content, self.config.max_retries
        )
        repair_log.extend(log)

        if success and schema:
            try:
                parsed_json = json.loads(repaired_content)
                self._validate_against_schema(parsed_json, schema, repair_log, [])
            except Exception as e:
                repair_log.append(f"Schema validation failed after repair: {e}")
                success = False

        return repaired_content, repair_log, success

    def _attempt_pydantic_repair(
        self, content: str, model_class: Type, validation_errors: List[Any]
    ) -> Tuple[str, List[str], bool]:
        """Attempt to repair content for Pydantic validation."""
        repair_log = []

        # First repair JSON structure
        repaired_content, log, json_success = self.repair_engine.repair_json(
            content, self.config.max_retries
        )
        repair_log.extend(log)

        if not json_success:
            return repaired_content, repair_log, False

        # Try to fix Pydantic-specific issues
        try:
            parsed_json = json.loads(repaired_content)

            # Apply field-level repairs based on validation errors
            for error in validation_errors:
                field_path = ".".join(str(loc) for loc in error["loc"])
                repair_log.append(f"Attempting to fix field: {field_path}")

                # Simple field repairs (can be enhanced)
                if "missing" in str(error.get("type", "")).lower():
                    # Add missing field with default value
                    self._add_missing_field(parsed_json, field_path, repair_log)
                elif "type" in str(error.get("type", "")).lower():
                    # Fix type mismatch
                    self._fix_type_mismatch(parsed_json, field_path, repair_log)

            # Try validation again
            model_instance = model_class(**parsed_json)
            repaired_content = model_instance.json()
            return repaired_content, repair_log, True

        except Exception as e:
            repair_log.append(f"Pydantic repair failed: {e}")
            return repaired_content, repair_log, False

    def _add_missing_field(
        self, data: Dict[str, Any], field_path: str, repair_log: List[str]
    ) -> None:
        """Add missing field with default value."""
        keys = field_path.split(".")
        current = data

        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        # Add missing field with appropriate default
        final_key = keys[-1]
        if final_key not in current:
            current[final_key] = ""  # Default to empty string
            repair_log.append(f"Added missing field: {field_path}")

    def _fix_type_mismatch(
        self, data: Dict[str, Any], field_path: str, repair_log: List[str]
    ) -> None:
        """Fix type mismatch in field."""
        keys = field_path.split(".")
        current = data

        try:
            for key in keys[:-1]:
                current = current[key]

            final_key = keys[-1]
            if final_key in current:
                # Try to convert to appropriate type
                value = current[final_key]
                if isinstance(value, str) and value.isdigit():
                    current[final_key] = int(value)
                    repair_log.append(f"Converted {field_path} to integer")
                elif isinstance(value, str) and value.lower() in ["true", "false"]:
                    current[final_key] = value.lower() == "true"
                    repair_log.append(f"Converted {field_path} to boolean")
        except (KeyError, TypeError):
            pass

    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation statistics."""
        total = self.validation_stats["total_validations"]
        if total == 0:
            return self.validation_stats

        return {
            **self.validation_stats,
            "success_rate": self.validation_stats["successful_validations"] / total,
            "repair_success_rate": self.validation_stats["repairs_successful"]
            / max(1, self.validation_stats["repairs_attempted"]),
            "failure_rate": self.validation_stats["failures"] / total,
        }


class SchemaValidationMode:
    """
    Schema Validation Mode - Phase 3 Competitive Feature

    Integrates with ASHA pipeline to provide:
    - Auto-retry on validation failures
    - Error correction and repair
    - JSON enforcement
    - Pydantic integration
    """

    def __init__(self, config: Optional[SchemaConfig] = None) -> None:
        """Initialize schema validation mode."""
        self.config = config or SchemaConfig()
        self.validator = SchemaValidator(config)
        self.enabled = True

    def enable(self) -> None:
        """Enable schema validation mode."""
        self.enabled = True

    def disable(self) -> None:
        """Disable schema validation mode."""
        self.enabled = False

    def process_with_validation(
        self, content: str, schema: Optional[Union[Dict[str, Any], Type]] = None
    ) -> ValidationResult:
        """
        Process content with schema validation.

        Args:
            content: Content to validate
            schema: JSON schema dict or Pydantic model class

        Returns:
            ValidationResult with validation details
        """
        if not self.enabled:
            return ValidationResult(
                status=ValidationStatus.VALID,
                is_valid=True,
                errors=[],
                warnings=[],
                repaired_content=content,
                original_content=content,
                validation_time_ms=0,
                retry_count=0,
                metadata={"validation_disabled": True},
            )

        # Choose validation method based on schema type
        if (
            isinstance(schema, type)
            and PYDANTIC_AVAILABLE
            and issubclass(schema, BaseModel)
        ):
            return self.validator.validate_pydantic(content, schema)
        elif isinstance(schema, dict) or schema is None:
            return self.validator.validate_json(content, schema)
        else:
            return self.validator.validate_json(content, None)

    def get_mode_info(self) -> Dict[str, Any]:
        """Get schema validation mode information."""
        return {
            "enabled": self.enabled,
            "config": self.config.__dict__,
            "validation_stats": self.validator.get_validation_stats(),
            "pydantic_available": PYDANTIC_AVAILABLE,
            "capabilities": {
                "json_validation": True,
                "pydantic_validation": PYDANTIC_AVAILABLE,
                "auto_repair": self.config.enable_auto_repair,
                "schema_validation": True,
                "error_correction": True,
            },
        }


# Convenience functions for easy usage
def validate_schema(
    content: str,
    schema: Optional[Union[Dict[str, Any], Type]] = None,
    max_retries: int = 3,
) -> ValidationResult:
    """
    Validate content against schema with auto-repair.

    Args:
        content: Content to validate
        schema: JSON schema dict or Pydantic model class
        max_retries: Maximum repair attempts

    Returns:
        ValidationResult with validation details
    """
    config = SchemaConfig(max_retries=max_retries)
    validator = SchemaValidator(config)

    if (
        isinstance(schema, type)
        and PYDANTIC_AVAILABLE
        and issubclass(schema, BaseModel)
    ):
        return validator.validate_pydantic(content, schema)
    elif isinstance(schema, dict) or schema is None:
        return validator.validate_json(content, schema)
    else:
        return validator.validate_json(content, None)


def repair_json(json_str: str, max_attempts: int = 5) -> Tuple[str, bool, List[str]]:
    """
    Repair invalid JSON string.

    Args:
        json_str: Invalid JSON string
        max_attempts: Maximum repair attempts

    Returns:
        Tuple of (repaired_json, success, repair_log)
    """
    repair_engine = JSONRepairEngine()
    repaired, log, success = repair_engine.repair_json(json_str, max_attempts)
    return repaired, success, log


# Quick test function
def test_schema_validation() -> None:
    """Test the schema validation system."""
    print("Testing Schema Validation System:")
    print("=" * 50)

    # Test 1: Invalid JSON repair
    invalid_json = '{"name": "John", "age": 30, "city": "New York",}'
    print(f"\nTest 1 - Invalid JSON:")
    print(f"Original: {invalid_json}")

    result = validate_schema(invalid_json)
    print(f"Status: {result.status.value}")
    print(f"Valid: {result.is_valid}")
    print(f"Repaired: {result.repaired_content}")
    print(f"Errors: {result.errors}")

    # Test 2: Pydantic validation (if available)
    if PYDANTIC_AVAILABLE:
        from pydantic import BaseModel

        class User(BaseModel):
            name: str
            age: int
            email: str

        valid_user = '{"name": "John", "age": 30, "email": "john@example.com"}'
        print(f"\nTest 2 - Pydantic Validation:")
        print(f"Input: {valid_user}")

        result = validate_schema(valid_user, User)
        print(f"Status: {result.status.value}")
        print(f"Valid: {result.is_valid}")
        print(f"Repaired: {result.repaired_content}")

    # Test 3: Complex repair
    complex_invalid = "{'name': 'John', 'age': 30, 'active': true,}"
    print(f"\nTest 3 - Complex Repair:")
    print(f"Original: {complex_invalid}")

    repaired, success, log = repair_json(complex_invalid)
    print(f"Success: {success}")
    print(f"Repaired: {repaired}")
    print(f"Log: {log}")


if __name__ == "__main__":
    test_schema_validation()
