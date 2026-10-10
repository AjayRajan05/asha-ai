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

"""Single source of truth for PII regex pattern definitions."""

import re
from typing import Dict, List, Pattern

# Domains and tokens that should not trigger PII masking in teaching/docs context
EXAMPLE_EMAIL_DOMAINS = ("example.com", "example.org", "test.com", "localhost")
PLACEHOLDER_EMAIL_LOCALS = frozenset(
    {"test", "user", "sample", "example", "email", "your", "my", "name", "placeholder"}
)


PII_PATTERN_STRINGS: Dict[str, List[str]] = {
    "email": [
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\.[A-Z|a-z]{2,}\b",
        r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
    ],
    "phone": [
        r"\b\d{3}-\d{3}-\d{4}\b",
        # Parentheses are non-word chars; \b before '(' fails after whitespace.
        r"(?<![\w)])\(\d{3}\)\s*\d{3}[-.]?\d{4}\b",
        r"\b\d{3}\.\d{3}\.\d{4}\b",
        r"\b\d{10}\b",
        r"\b\+1[-\s]?\d{3}[-\s]?\d{3}[-\s]?\d{4}\b",
        r"(?<!\w)\+?\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
        # Indian mobiles (+91 / 10-digit starting 6-9)
        r"\b(?:\+91[\s-]?)?[6-9]\d{9}\b",
        r"\b(?:\+91[\s-]?)?\d{5}[\s-]?\d{5}\b",
    ],
    "ssn": [
        r"\b\d{3}-\d{2}-\d{4}\b",
        r"\b\d{3}\s*\d{2}\s*\d{4}\b",
    ],
    "credit_card": [
        r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
        r"\b\d{4}\s\d{4}\s\d{4}\s\d{4}\b",
        r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
        r"\b\d{16}\b",
    ],
    # Indian identity formats
    "aadhaar": [
        r"\b[2-9]\d{3}[\s-]?\d{4}[\s-]?\d{4}\b",
        r"\b[2-9]\d{11}\b",
    ],
    "pan": [
        r"\b[A-Z]{5}[0-9]{4}[A-Z]\b",
    ],
    "gstin": [
        r"\b[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z][0-9A-Z]Z[0-9A-Z]\b",
    ],
    "address": [
        r"\d+\s+[A-Z][a-z]+\s+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd)\b",
        r"\d+\s+[A-Z][a-z]+\s+[A-Z][a-z]+\s+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd)\b",
        r"\b\d+\s+[A-Z][a-z]+\s+(St|Street|Ave|Avenue|Rd|Road|Blvd|Boulevard|Ln|Lane|Dr|Drive|Ct|Court|Way|Place|Pl)\b",
        r"\d+\s+[A-Za-z\s]+,\s*[A-Za-z\s]+,\s*[A-Z]{2}\s*\d{5}",
    ],
    "ip_address": [
        r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",
        r"\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b",
    ],
    "url": [
        # Only URLs containing embedded credentials (user:pass@host) are considered sensitive/PII
        r'[a-zA-Z0-9+_-]+://[^\s:@/]+:[^\s@]+@[^\s<>"{}|\\^`\[\]]+',
    ],
    "zip_code": [
        r"\b\d{5}\b",
        r"\b\d{5}-\d{4}\b",
    ],
    "api_key": [
        r"\bsk-[a-zA-Z0-9]{20,}\b",
        r"\bsk-proj-[a-zA-Z0-9_-]{20,}\b",
        r"\bAKIA[0-9A-Z]{16}\b",
        r"(?i)(?:api[_-]?key|secret[_-]?key)\s*[:=]\s*['\"]?[a-zA-Z0-9_\-]{16,}",
        r"(?i)(?:password|passwd|pwd)\s*[:=]\s*['\"]?[^\s'\"]{6,}",
        r'[a-zA-Z0-9+_-]+://[^\s:@/]+:[^\s@]+@[^\s<>"{}|\\^`\[\]]+',
    ],
    "jwt_token": [
        r"\beyJ[a-zA-Z0-9_-]+\.eyJ[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+\b",
    ],
    "bearer_token": [
        r"(?i)Bearer\s+[a-zA-Z0-9._\-]{20,}",
    ],
}


def compile_pii_patterns(
    flags: int = re.IGNORECASE,
) -> Dict[str, List[Pattern[str]]]:
    """Compile pattern strings into regex objects."""
    compiled: Dict[str, List[Pattern[str]]] = {}
    for pii_type, patterns in PII_PATTERN_STRINGS.items():
        compiled[pii_type] = [re.compile(p, flags) for p in patterns]
    return compiled


def is_example_email(text: str) -> bool:
    """Return True if an email is a known placeholder/teaching address."""
    if "@" not in text:
        return False
    local, domain = text.rsplit("@", 1)
    domain = domain.lower().strip(".")
    if domain not in EXAMPLE_EMAIL_DOMAINS:
        return False
    local_base = local.lower().split("+")[0]
    return local_base in PLACEHOLDER_EMAIL_LOCALS
