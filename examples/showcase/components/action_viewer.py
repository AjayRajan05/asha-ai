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

"""Action normalization visualizer for ANCHOR governance."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

_COMPONENTS_DIR = Path(__file__).resolve().parent
_SHOWCASE_DIR = _COMPONENTS_DIR.parent
_PROJECT_ROOT = _SHOWCASE_DIR.parent.parent
for _p in (str(_PROJECT_ROOT), str(_SHOWCASE_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import streamlit as st

from examples.showcase.components.status_badge import get_status_badge_html


def render_action_normalization_pipeline(
    raw_action_type: str,
    raw_payload: Dict[str, Any],
    normalized_tool_name: str,
    normalized_arguments: str,
    risk_score: float,
    verdict: str,
    reason: str,
    executed: bool = False,
) -> None:
    """Render the ANCHOR Action Normalization pipeline stages."""
    st.markdown(
        f"""
        <div style="background: #121820; border: 1px solid #273142; border-radius: 8px; padding: 1.2rem; margin: 1rem 0;">
            <div style="font-size: 0.85rem; font-weight: 700; color: #8b949e; text-transform: uppercase; margin-bottom: 0.8rem;">
                ANCHOR Action Governance Pipeline
            </div>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px;">
                <div style="background: #0d1117; padding: 10px; border-radius: 6px; border: 1px solid #1c2430;">
                    <div style="font-size: 0.72rem; color: #8b949e;">1. RAW ACTION</div>
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; color: #f0883e; margin-top: 4px;">{raw_action_type}</div>
                </div>
                <div style="background: #0d1117; padding: 10px; border-radius: 6px; border: 1px solid #1c2430;">
                    <div style="font-size: 0.72rem; color: #8b949e;">2. NORMALIZED TOOL</div>
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; color: #38bdf8; margin-top: 4px;">{normalized_tool_name}</div>
                </div>
                <div style="background: #0d1117; padding: 10px; border-radius: 6px; border: 1px solid #1c2430;">
                    <div style="font-size: 0.72rem; color: #8b949e;">3. RISK SCORE</div>
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; color: #e6edf3; margin-top: 4px;">{risk_score:.2f}</div>
                </div>
                <div style="background: #0d1117; padding: 10px; border-radius: 6px; border: 1px solid #1c2430;">
                    <div style="font-size: 0.72rem; color: #8b949e;">4. VERDICT</div>
                    <div style="margin-top: 4px;">{get_status_badge_html(verdict)}</div>
                </div>
                <div style="background: #0d1117; padding: 10px; border-radius: 6px; border: 1px solid #1c2430;">
                    <div style="font-size: 0.72rem; color: #8b949e;">5. EXECUTION</div>
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; color: {'#3fb950' if executed else '#f85149'}; margin-top: 4px;">
                        {'EXECUTED' if executed else 'BLOCKED (Not Run)'}
                    </div>
                </div>
            </div>
            <div style="font-size: 0.8rem; color: #8b949e; margin-top: 10px;">
                <strong>Decision Basis:</strong> {reason}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
