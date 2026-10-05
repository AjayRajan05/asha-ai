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
Format lock system for ASHA.

Prevents optimization from breaking structured prompts (JSON, code, markdown).
"""

import re
import json
from typing import Dict, Any, Tuple, List


class FormatLock:
    """
    Format-aware optimization rules to protect structured prompts.

    This ensures that optimization NEVER breaks:
    - JSON structure and spacing
    - Code blocks and indentation
    - Markdown headers and formatting
    """

    def __init__(self) -> None:
        self.structure_detectors = self._init_structure_detectors()
        self.lock_rules = self._init_lock_rules()

    def detect_structure(self, prompt: str) -> str:
        """
        Detect the structure type of the prompt.

        Returns: "json", "code", "markdown", or "plain"
        """
        # JSON detection (highest priority)
        if self._is_json(prompt):
            return "json"

        # Code block detection
        if self._has_code_blocks(prompt):
            return "code"

        # Markdown detection
        if self._is_markdown(prompt):
            return "markdown"

        return "plain"

    def get_optimization_lock(self, structure_type: str) -> Dict[str, Any]:
        """
        Get optimization lock rules for a structure type.

        Returns dictionary of lock rules that constrain optimization.
        """
        return self.lock_rules.get(structure_type, self.lock_rules["plain"])

    def should_skip_optimization_level(self, structure_type: str, level: int) -> bool:
        """
        Determine if optimization level should be skipped.

        Args:
            structure_type: Type of structure detected
            level: Optimization level (1=L1, 2=L2, 3=L3)

        Returns:
            True if level should be skipped for this structure
        """
        lock_rules = self.get_optimization_lock(structure_type)

        # Skip L2 if prohibited
        if level >= 2 and lock_rules.get("skip_l2", False):
            return True

        # Skip L3 if prohibited
        if level >= 3 and lock_rules.get("skip_l3", False):
            return True

        return False

    def apply_format_preservation(
        self, prompt: str, structure_type: str
    ) -> Dict[str, Any]:
        """
        Apply format-specific preservation rules.

        Returns format preservation metadata.
        """
        preservation_rules = {
            "preserve_whitespace": True,
            "preserve_structure": True,
            "preserve_indentation": True,
            "preserve_quotes": True,
            "preserve_brackets": True,
        }

        if structure_type == "json":
            preservation_rules.update(
                {
                    "preserve_braces": True,
                    "preserve_commas": True,
                    "preserve_colons": True,
                    "string_mode": True,
                }
            )
        elif structure_type == "code":
            preservation_rules.update(
                {
                    "preserve_code_blocks": True,
                    "preserve_comments": True,
                    "preserve_syntax": True,
                }
            )
        elif structure_type == "markdown":
            preservation_rules.update(
                {
                    "preserve_headers": True,
                    "preserve_lists": True,
                    "preserve_links": True,
                    "preserve_emphasis": True,
                }
            )

        return preservation_rules

    def validate_structure_integrity(
        self, original: str, optimized: str, structure_type: str
    ) -> Tuple[bool, List[str]]:
        """
        Validate that optimization preserved structure integrity.

        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []

        if structure_type == "json":
            is_valid, json_issues = self._validate_json_integrity(original, optimized)
            issues.extend(json_issues)

        elif structure_type == "code":
            is_valid, code_issues = self._validate_code_integrity(original, optimized)
            issues.extend(code_issues)

        elif structure_type == "markdown":
            is_valid, md_issues = self._validate_markdown_integrity(original, optimized)
            issues.extend(md_issues)

        return len(issues) == 0, issues

    def _init_structure_detectors(self) -> Dict[str, Any]:
        """Initialize structure detection patterns."""
        return {
            "json_pattern": r"^\s*\{[\s\S]*\}\s*$",
            "code_block_pattern": r"```[\s\S]*?```",
            "inline_code_pattern": r"`[^`]+`",
            "markdown_header_pattern": r"^#{1,6}\s+",
            "markdown_list_pattern": r"^\s*[-*+]\s+",
            "markdown_emphasis_pattern": r"\*[^*]+\*|_[^_]+_|`[^`]+`",
        }

    def _init_lock_rules(self) -> Dict[str, Dict[str, Any]]:
        """Initialize optimization lock rules by structure type."""
        return {
            "json": {
                "skip_l2": True,  # Skip context-aware optimization
                "skip_l3": True,  # Skip advanced optimization
                "preserve_whitespace": True,
                "preserve_structure": True,
                "preserve_braces": True,
                "preserve_commas": True,
                "preserve_colons": True,
                "max_optimization": "L1",  # Only L1 allowed
            },
            "code": {
                "skip_l2": True,  # Skip context-aware optimization
                "skip_l3": True,  # Skip advanced optimization
                "preserve_code_blocks": True,
                "preserve_indentation": True,
                "preserve_comments": True,
                "preserve_syntax": True,
                "max_optimization": "L1",  # Only L1 allowed
            },
            "markdown": {
                "skip_l3": True,  # Skip advanced optimization only
                "skip_l2": False,  # L1/L2 allowed
                "preserve_headers": True,
                "preserve_lists": True,
                "preserve_links": True,
                "preserve_emphasis": True,
                "max_optimization": "L2",  # L1/L2 allowed
            },
            "plain": {
                "skip_l2": False,  # All levels allowed
                "skip_l3": False,
                "max_optimization": "L3",  # All levels allowed
            },
        }

    def _is_json(self, prompt: str) -> bool:
        """Check if prompt is valid JSON."""
        prompt_stripped = prompt.strip()

        # Basic JSON structure check
        if not (prompt_stripped.startswith("{") and prompt_stripped.endswith("}")):
            return False

        try:
            json.loads(prompt_stripped)
            return True
        except (json.JSONDecodeError, ValueError):
            return False

    def _has_code_blocks(self, prompt: str) -> bool:
        """Check if prompt contains code blocks."""
        return bool(re.search(r"```[\s\S]*?```", prompt))

    def _is_markdown(self, prompt: str) -> bool:
        """Check if prompt uses markdown formatting."""
        patterns = self.structure_detectors

        # Check for headers
        if re.search(patterns["markdown_header_pattern"], prompt, re.MULTILINE):
            return True

        # Check for lists
        if re.search(patterns["markdown_list_pattern"], prompt, re.MULTILINE):
            return True

        # Check for emphasis
        if re.search(patterns["markdown_emphasis_pattern"], prompt):
            return True

        return False

    def _validate_json_integrity(
        self, original: str, optimized: str
    ) -> Tuple[bool, List[str]]:
        """Validate JSON structure integrity after optimization."""
        issues = []

        try:
            original_data = json.loads(original.strip())
            optimized_data = json.loads(optimized.strip())

            # Check structure preservation
            if type(original_data) != type(optimized_data):
                issues.append(
                    f"Structure type changed: {type(original_data)} → {type(optimized_data)}"
                )

            # Check key preservation for objects
            if isinstance(original_data, dict) and isinstance(optimized_data, dict):
                original_keys = set(original_data.keys())
                optimized_keys = set(optimized_data.keys())

                missing_keys = original_keys - optimized_keys
                if missing_keys:
                    issues.append(f"Lost keys: {missing_keys}")

                extra_keys = optimized_keys - original_keys
                if extra_keys:
                    issues.append(f"Added keys: {extra_keys}")

            # Check array length preservation
            if isinstance(original_data, list) and isinstance(optimized_data, list):
                if len(original_data) != len(optimized_data):
                    issues.append(
                        f"Array length changed: {len(original_data)} → {len(optimized_data)}"
                    )

        except (json.JSONDecodeError, ValueError) as e:
            issues.append(f"JSON parsing error: {str(e)}")

        return len(issues) == 0, issues

    def _validate_code_integrity(
        self, original: str, optimized: str
    ) -> Tuple[bool, List[str]]:
        """Validate code block integrity after optimization."""
        issues = []

        # Extract code blocks
        original_blocks = re.findall(r"```(\n[\s\S]*?)\n```", original, re.DOTALL)
        optimized_blocks = re.findall(r"```(\n[\s\S]*?)\n```", optimized, re.DOTALL)

        if len(original_blocks) != len(optimized_blocks):
            issues.append(
                f"Code block count changed: {len(original_blocks)} → {len(optimized_blocks)}"
            )

        # Check content preservation
        for i, (orig_block, opt_block) in enumerate(
            zip(original_blocks, optimized_blocks)
        ):
            if orig_block.strip() != opt_block.strip():
                # Check if core logic preserved (simplified check)
                orig_lines = set(orig_block.strip().split("\n"))
                opt_lines = set(opt_block.strip().split("\n"))

                if len(orig_lines - opt_lines) > len(opt_lines - orig_lines) * 2:
                    issues.append(
                        f"Code block {i+1}: Significant content change detected"
                    )

        return len(issues) == 0, issues

    def _validate_markdown_integrity(
        self, original: str, optimized: str
    ) -> Tuple[bool, List[str]]:
        """Validate markdown structure integrity after optimization."""
        issues = []

        # Check header preservation
        original_headers = re.findall(r"^#{1,6}\s+(.+)$", original, re.MULTILINE)
        optimized_headers = re.findall(r"^#{1,6}\s+(.+)$", optimized, re.MULTILINE)

        if len(original_headers) != len(optimized_headers):
            issues.append(
                f"Header count changed: {len(original_headers)} → {len(optimized_headers)}"
            )

        # Check list preservation
        original_lists = re.findall(r"^\s*[-*+]\s+(.+)$", original, re.MULTILINE)
        optimized_lists = re.findall(r"^\s*[-*+]\s+(.+)$", optimized, re.MULTILINE)

        if len(original_lists) != len(optimized_lists):
            issues.append(
                f"List count changed: {len(original_lists)} → {len(optimized_lists)}"
            )

        # Check emphasis preservation
        original_emphasis = re.findall(r"\*[^*]+\*|_[^_]+_|`[^`]+`", original)
        optimized_emphasis = re.findall(r"\*[^*]+\*|_[^_]+_|`[^`]+`", optimized)

        if len(original_emphasis) != len(optimized_emphasis):
            issues.append(
                f"Emphasis count changed: {len(original_emphasis)} → {len(optimized_emphasis)}"
            )

        return len(issues) == 0, issues

    def get_format_report(
        self, original: str, optimized: str, structure_type: str
    ) -> Dict[str, Any]:
        """Generate format preservation report."""
        is_valid, issues = self.validate_structure_integrity(
            original, optimized, structure_type
        )

        return {
            "structure_type": structure_type,
            "is_valid": is_valid,
            "issues": issues,
            "lock_rules_applied": self.get_optimization_lock(structure_type),
            "recommendation": (
                "Structure preserved"
                if is_valid
                else "Structure compromised - review optimization"
            ),
        }
