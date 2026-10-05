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

"""Security package: lazy exports so ``injection_lite`` stays stdlib-cheap."""

from __future__ import annotations

from typing import Any

__all__ = [
    "SecurityLayer",
    "SecurityResult",
    "SecurityLevel",
    "ThreatType",
    "run_security",
    "run_security_only",
    "normalize_security_level",
    "read_security_field",
    "get_sanitized_content",
]


def __getattr__(name: str) -> Any:
    if name in ("SecurityLayer", "SecurityResult", "SecurityLevel", "ThreatType"):
        from . import security_layer as _sl

        return getattr(_sl, name)
    if name in (
        "run_security",
        "run_security_only",
        "normalize_security_level",
        "read_security_field",
        "get_sanitized_content",
    ):
        from . import service as _svc

        return getattr(_svc, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
