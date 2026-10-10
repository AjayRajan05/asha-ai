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

"""Optimization test scenarios for ASHA Control Center."""

from typing import Dict

OPTIMIZATION_SCENARIOS: Dict[str, str] = {
    "Critical Negation & Security Constraint (Rejection Candidate)": (
        "Analyze the attached internal financial ledger. Do NOT send customer data externally. "
        "Under NO circumstances should employee salary figures be printed or logged. "
        "Never disclose the encryption seed phrase."
    ),
    "Strict JSON Schema & Numeric Precision (Rejection Candidate)": (
        "Parse the transaction logs and output ONLY valid JSON matching this exact schema: "
        "{\"transaction_id\": int, \"amount\": float, \"currency\": str, \"status\": \"CONFIRMED\"}. "
        "Ensure the amount precision matches 2 decimal places exactly ($142.50) and never omit fields."
    ),
    "Verbose Fluffy Prompt (Compression Accepted)": (
        "I was wondering if you could possibly take a moment out of your busy schedule to help me "
        "with a task that I am currently working on. Basically what I really need is for you to "
        "summarize the main key points and takeaways from this following document. Please make sure "
        "that you give me a high-level overview that is easy to digest for our executive team."
    ),
    "Code Refactoring with Strict Type Signatures": (
        "Refactor this Python class. You MUST keep all existing docstrings and comments intact. "
        "Do not change the public method signatures or return types. Use PEP 484 type hints throughout."
    ),
}
