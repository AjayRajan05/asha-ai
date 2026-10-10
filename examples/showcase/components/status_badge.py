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

"""Status badge rendering for ASHA Control Center."""

from __future__ import annotations


def get_status_badge_html(status: str) -> str:
    """Return HTML string for status pill badge."""
    st_upper = str(status).upper()
    if st_upper in ("ALLOW", "SAFE", "SUCCESS", "ALLOWED"):
        pill_class = "pill-allow"
    elif st_upper in ("WARN", "REVIEW", "WARNING", "MODERATE"):
        pill_class = "pill-warn"
    elif st_upper in ("BLOCK", "CRITICAL", "BLOCKED", "HIGH", "UNSAFE", "ERROR"):
        pill_class = "pill-block"
    elif st_upper in ("TRANSFORM", "MASKED", "COMPRESSED", "OPTIMIZED"):
        pill_class = "pill-purple"
    else:
        pill_class = "pill-info"

    return f'<span class="pill {pill_class}">{st_upper}</span>'


def render_status_pill(text: str, badge_type: str = "info") -> str:
    """Return HTML status pill supporting explicit badge types ('success', 'warning', 'blocked', 'purple', 'info')."""
    t_lower = badge_type.lower()
    if t_lower in ("success", "allow", "allowed"):
        cls_name = "pill-allow"
    elif t_lower in ("warning", "warn", "review"):
        cls_name = "pill-warn"
    elif t_lower in ("blocked", "block", "error", "critical"):
        cls_name = "pill-block"
    elif t_lower in ("purple", "transform", "optimized", "masked"):
        cls_name = "pill-purple"
    else:
        cls_name = "pill-info"

    return f'<span class="pill {cls_name}">{text}</span>'
