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
ASHA Scanner Plugin Architecture.

Provides extensible protocols and registries for registering custom domain-specific
scanners for PII, secrets, and sensitive tokens.
"""

from .base import BaseScanner, PatternScanner
from .registry import (
    ScannerRegistry,
    register_scanner,
    unregister_scanner,
    get_registered_scanners,
    clear_registered_scanners,
)

__all__ = [
    "BaseScanner",
    "PatternScanner",
    "ScannerRegistry",
    "register_scanner",
    "unregister_scanner",
    "get_registered_scanners",
    "clear_registered_scanners",
]
