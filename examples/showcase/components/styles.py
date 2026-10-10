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

"""Developer console styling and theme tokens for ASHA Control Center."""

from __future__ import annotations

import streamlit as st


def inject_styles() -> None:
    """Inject restrained dark developer-console CSS into Streamlit application."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Inter:wght@400;500;600;700&display=swap');

        :root {
            --bg-canvas: #090d13;
            --bg-card: #121820;
            --bg-hover: #1b222d;
            --border-color: #273142;
            --border-focus: #38bdf8;
            --text-primary: #e6edf3;
            --text-secondary: #8b949e;
            --text-tertiary: #6e7681;
            
            --asha-blue: #38bdf8;
            --asha-blue-bg: rgba(56, 189, 248, 0.1);
            --status-allow: #3fb950;
            --status-allow-bg: rgba(63, 185, 80, 0.12);
            --status-warn: #f0883e;
            --status-warn-bg: rgba(240, 136, 62, 0.12);
            --status-block: #f85149;
            --status-block-bg: rgba(248, 81, 73, 0.12);
            --status-transform: #bc8cff;
            --status-transform-bg: rgba(188, 140, 255, 0.12);
        }

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            color: var(--text-primary);
        }

        code, pre, .mono {
            font-family: 'JetBrains Mono', monospace !important;
        }

        header[data-testid="stHeader"] {
            background: transparent !important;
        }

        .block-container {
            padding-top: 4.5rem !important;
            padding-bottom: 2.5rem !important;
            padding-left: 2rem !important;
            padding-right: 2rem !important;
            max-width: 100% !important;
            box-sizing: border-box !important;
        }

        #MainMenu, footer { visibility: hidden; }

        /* Hero / Header Container */
        .console-header {
            background: linear-gradient(180deg, #131b26 0%, #0d121a 100%);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 1.2rem 1.6rem;
            margin-top: 0.25rem;
            margin-bottom: 1.4rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 16px;
            flex-wrap: wrap;
            box-sizing: border-box;
            width: 100%;
        }

        .console-header-left {
            min-width: 0;
            flex: 1 1 auto;
        }

        .console-title {
            display: flex;
            align-items: center;
            gap: 8px;
            margin: 0;
            flex-wrap: wrap;
        }

        .console-logo {
            color: #38bdf8;
            font-size: 1.25rem;
            font-weight: 800;
            letter-spacing: -0.01em;
            white-space: nowrap;
        }

        .console-heading {
            color: #ffffff;
            font-size: 1.2rem;
            font-weight: 700;
            letter-spacing: -0.01em;
            white-space: nowrap;
        }

        .console-subtitle {
            font-size: 0.82rem;
            color: #8b949e;
            margin-top: 4px;
            line-height: 1.4;
            word-break: break-word;
        }

        .console-header-right {
            display: flex;
            align-items: center;
            gap: 12px;
            flex-shrink: 0;
        }

        .console-mode-label {
            font-size: 0.78rem;
            color: #8b949e;
            white-space: nowrap;
        }

        .console-badge {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.72rem;
            padding: 2px 8px;
            border-radius: 4px;
            font-weight: 600;
            text-transform: uppercase;
            white-space: nowrap;
        }

        .badge-preview {
            background: var(--asha-blue-bg);
            color: var(--asha-blue);
            border: 1px solid rgba(56, 189, 248, 0.3);
        }

        /* Metric cards */
        .metric-card {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 6px;
            padding: 1rem 1.2rem;
            text-align: left;
        }

        .metric-label {
            font-size: 0.78rem;
            font-weight: 500;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.04em;
            margin-bottom: 0.3rem;
        }

        .metric-value {
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.6rem;
            font-weight: 700;
            color: #ffffff;
            letter-spacing: -0.02em;
        }

        .metric-sub {
            font-size: 0.75rem;
            color: var(--text-tertiary);
            margin-top: 0.2rem;
        }

        /* Status Pills */
        .pill {
            font-family: 'JetBrains Mono', monospace;
            display: inline-flex;
            align-items: center;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.78rem;
            font-weight: 600;
        }

        .pill-allow {
            background: var(--status-allow-bg);
            color: var(--status-allow);
            border: 1px solid rgba(63, 185, 80, 0.3);
        }

        .pill-warn {
            background: var(--status-warn-bg);
            color: var(--status-warn);
            border: 1px solid rgba(240, 136, 62, 0.3);
        }

        .pill-block {
            background: var(--status-block-bg);
            color: var(--status-block);
            border: 1px solid rgba(248, 81, 73, 0.3);
        }

        .pill-info {
            background: var(--asha-blue-bg);
            color: var(--asha-blue);
            border: 1px solid rgba(56, 189, 248, 0.3);
        }

        .pill-purple {
            background: var(--status-transform-bg);
            color: var(--status-transform);
            border: 1px solid rgba(188, 140, 255, 0.3);
        }

        /* Pipeline Visualizer */
        .pipeline-container {
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 0.8rem 1rem;
            margin: 1rem 0;
            overflow-x: auto;
        }

        .pipeline-node {
            display: flex;
            flex-direction: column;
            align-items: center;
            text-align: center;
            padding: 0.4rem 0.8rem;
            border-radius: 6px;
            min-width: 90px;
        }

        .node-active {
            background: rgba(56, 189, 248, 0.08);
            border: 1px solid var(--asha-blue);
        }

        .node-done {
            background: rgba(63, 185, 80, 0.08);
            border: 1px solid rgba(63, 185, 80, 0.4);
        }

        .node-blocked {
            background: rgba(248, 81, 73, 0.08);
            border: 1px solid rgba(248, 81, 73, 0.4);
        }

        .node-pending {
            background: transparent;
            border: 1px dashed var(--border-color);
            opacity: 0.6;
        }

        .pipeline-arrow {
            color: var(--text-tertiary);
            font-size: 1.1rem;
            margin: 0 4px;
        }

        /* Diff Containers */
        .diff-box {
            background: #0d1117;
            border: 1px solid var(--border-color);
            border-radius: 6px;
            padding: 0.9rem;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
            white-space: pre-wrap;
            word-break: break-all;
        }

        .diff-del {
            background: rgba(248, 81, 73, 0.15);
            color: #ff7b72;
            text-decoration: line-through;
            padding: 1px 3px;
            border-radius: 3px;
        }

        .diff-add {
            background: rgba(63, 185, 80, 0.15);
            color: #7ee787;
            padding: 1px 3px;
            border-radius: 3px;
        }

        /* Event timeline */
        .timeline-event {
            display: flex;
            align-items: flex-start;
            gap: 12px;
            padding: 8px 12px;
            border-left: 2px solid var(--border-color);
            margin-left: 8px;
            font-size: 0.85rem;
        }

        .timeline-time {
            font-family: 'JetBrains Mono', monospace;
            color: var(--text-tertiary);
            font-size: 0.76rem;
            min-width: 85px;
        }

        .timeline-name {
            font-family: 'JetBrains Mono', monospace;
            font-weight: 600;
        }

        /* Callout / Warning box */
        .console-alert {
            background: rgba(240, 136, 62, 0.08);
            border: 1px solid rgba(240, 136, 62, 0.3);
            border-radius: 6px;
            padding: 0.8rem 1rem;
            font-size: 0.85rem;
            margin: 0.8rem 0;
            color: #d29922;
        }

        .console-alert-danger {
            background: rgba(248, 81, 73, 0.08);
            border: 1px solid rgba(248, 81, 73, 0.3);
            border-radius: 6px;
            padding: 0.8rem 1rem;
            font-size: 0.85rem;
            margin: 0.8rem 0;
            color: #ff7b72;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
