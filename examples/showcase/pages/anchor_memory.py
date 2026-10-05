"""ASHA Control Center - Memory Security Lab.

Interactive laboratory demonstrating ANCHOR MemoryGuard:
- Cognitive poisoning detection
- Memory write scope validation
- Instruction injection into agent memory
- Real verdict and risk scoring
"""

import sys
import time
from pathlib import Path

_PAGES_DIR = Path(__file__).resolve().parent
_SHOWCASE_DIR = _PAGES_DIR.parent
_PROJECT_ROOT = _SHOWCASE_DIR.parent.parent
for _p in (str(_PROJECT_ROOT), str(_SHOWCASE_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import streamlit as st

from examples.showcase.scenarios.memory import MEMORY_SCENARIOS
from examples.showcase.runtime.trace import Trace, EventType
from examples.showcase.runtime.session import record_trace
from examples.showcase.components.header import render_header
from examples.showcase.components.status_badge import render_status_pill
from examples.showcase.components.object_inspector import render_object_inspector
from examples.showcase.components.code_viewer import render_python_code


def render() -> None:
    render_header(title="MEMORY SECURITY FIREWALL", subtitle="Protect agent memory stores from cognitive poisoning and scope writes")


    col_sel, col_src, col_scope = st.columns([2, 1, 1])
    with col_sel:
        preset_keys = list(MEMORY_SCENARIOS.keys())
        selected_key = st.selectbox("Select Scenario Preset", preset_keys, index=0)
        preset = MEMORY_SCENARIOS[selected_key]
    with col_src:
        source = st.selectbox(
            "Source Origin",
            ["User", "Tool Execution", "Retrieved Document", "Agent Reflection"],
            index=0 if preset.source == "user" else 1,
        )
    with col_scope:
        scope = st.selectbox(
            "Target Scope",
            ["session", "user_preferences", "long_term", "system_credentials"],
            index=0,
        )

    mission_goal = st.text_input(
        "Active Mission Goal",
        value="Help the user manage personal documents and customer feedback notes safely.",
        help="The active mission contract against which the memory operation is evaluated.",
    )

    memory_content = st.text_area(
        "Memory Content Payload",
        value=preset.content,
        height=140,
        help="The raw text payload being written or updated in memory.",
    )

    run_btn = st.button("Inspect Memory Write", type="primary", use_container_width=True)

    if run_btn:
        trace = Trace(provider="ANCHOR_MEMORY_GUARD", model="memory_guard_v1")
        t0 = time.perf_counter()

        trace.add_event(
            EventType.INPUT_RECEIVED,
            status="INFO",
            reason=f"Memory write submitted from {source} to scope '{scope}'",
            metadata={"source": source, "scope": scope, "chars": len(memory_content)},
        )

        try:
            from asha.runtime.anchor.mission import MissionCompiler
            from asha.runtime.anchor.memory_guard import MemoryGuard
            from asha.runtime.anchor.verdicts import Verdict

            trace.add_event(
                EventType.COMPILATION_STARTED,
                status="INFO",
                reason="Compiling active MissionContract...",
            )
            compiler = MissionCompiler()
            contract = compiler.compile(mission_goal)
            trace.add_event(
                EventType.COMPILATION_COMPLETED,
                status="SUCCESS",
                reason=f"Mission compiled: {contract.goal[:50]}...",
                metadata={"allowed_scopes": contract.allowed_memory_scopes},
            )

            trace.add_event(
                EventType.POLICY_CHECK,
                status="INFO",
                reason="MemoryGuard scanning payload for poisoning & scope integrity...",
            )
            guard = MemoryGuard()
            event = guard.normalize_memory_event(
                operation="write",
                content=memory_content,
                scope=scope,
            )
            verdict = guard.evaluate_memory(event, contract, full_content=memory_content)

            is_blocked = verdict.verdict is Verdict.BLOCK or str(verdict.verdict) == "Verdict.BLOCK"
            dt_ms = (time.perf_counter() - t0) * 1000.0

            if is_blocked:
                trace.add_event(
                    EventType.ACTION_BLOCKED,
                    status="BLOCKED",
                    duration_ms=dt_ms,
                    reason=f"Memory write BLOCKED: {verdict.reason}",
                    metadata={"risk_score": verdict.risk_score, "scope": scope},
                )
            else:
                trace.add_event(
                    EventType.ACTION_ALLOWED,
                    status="SUCCESS",
                    duration_ms=dt_ms,
                    reason=f"Memory write ALLOWED: {verdict.reason}",
                    metadata={"risk_score": verdict.risk_score, "scope": scope},
                )

            trace.add_event(
                EventType.REQUEST_COMPLETE,
                status="SUCCESS" if not is_blocked else "BLOCKED",
                duration_ms=dt_ms,
                reason="Memory evaluation completed.",
            )

            record_trace(trace)

            # Display Results Pipeline
            st.markdown("### Execution Pipeline & Decision")
            
            # Step Visualization
            s1_col, s2_col, s3_col, s4_col, s5_col = st.columns(5)
            with s1_col:
                st.markdown("**1. Source**")
                st.caption(f"{source}")
                st.markdown(render_status_pill("VALID", "success"), unsafe_allow_html=True)
            with s2_col:
                st.markdown("**2. Scope Check**")
                scope_ok = scope in contract.allowed_memory_scopes or scope == "session"
                st.caption(f"'{scope}'")
                st.markdown(
                    render_status_pill("PERMITTED" if scope_ok else "DENIED", "success" if scope_ok else "blocked"),
                    unsafe_allow_html=True,
                )
            with s3_col:
                st.markdown("**3. Poisoning Scan**")
                st.caption(f"Risk: {verdict.risk_score:.2f}")
                st.markdown(
                    render_status_pill("ALERT" if verdict.risk_score > 0.5 else "CLEAN", "blocked" if verdict.risk_score > 0.5 else "success"),
                    unsafe_allow_html=True,
                )
            with s4_col:
                st.markdown("**4. Contract Match**")
                st.caption("Contextual")
                st.markdown(render_status_pill("ALIGNED" if not is_blocked else "CONFLICT", "success" if not is_blocked else "warning"), unsafe_allow_html=True)
            with s5_col:
                st.markdown("**5. Decision**")
                st.caption(f"{dt_ms:.1f} ms")
                decision_label = "BLOCKED" if is_blocked else ("REVIEW" if str(verdict.verdict).endswith("REVIEW") else "ALLOWED")
                decision_badge = "blocked" if is_blocked else ("warning" if decision_label == "REVIEW" else "success")
                st.markdown(render_status_pill(decision_label, decision_badge), unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Details card
            card_border = "#ef4444" if is_blocked else "#10b981"
            bg_color = "rgba(239, 68, 68, 0.05)" if is_blocked else "rgba(16, 185, 129, 0.05)"
            st.markdown(
                f"""
                <div style="background: {bg_color}; border: 1px solid {card_border}; border-radius: 8px; padding: 16px; margin-bottom: 20px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <span style="font-weight: 700; font-size: 1.1rem; color: {'#f87171' if is_blocked else '#34d399'};">
                            {'🛡️ MEMORY WRITE BLOCKED' if is_blocked else '✅ MEMORY WRITE ALLOWED'}
                        </span>
                        <span style="font-family: monospace; font-size: 0.85rem; color: #94a3b8;">
                            Risk Score: {verdict.risk_score:.2f} / 1.00
                        </span>
                    </div>
                    <p style="margin: 0; color: #cbd5e1; font-size: 0.95rem;">
                        <strong>Enforcement Reason:</strong> {verdict.reason}
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Object Inspector
            render_object_inspector(
                raw_obj=verdict,
                title="MemoryVerdict Object Inspector",
                human_data={
                    "Decision": str(verdict.verdict),
                    "Risk Score": f"{verdict.risk_score:.3f}",
                    "Reason": verdict.reason,
                    "Payload Characters": len(memory_content),
                    "Target Scope": scope,
                    "Source Origin": source,
                    "Evaluation Latency": f"{dt_ms:.2f} ms",
                },
            )

            # Copy Python
            st.markdown("<br>", unsafe_allow_html=True)
            python_code = f"""from asha.runtime.anchor.mission import MissionCompiler
from asha.runtime.anchor.memory_guard import MemoryGuard

# 1. Compile active mission contract
compiler = MissionCompiler()
contract = compiler.compile({repr(mission_goal)})

# 2. Initialize MemoryGuard firewall
guard = MemoryGuard()

# 3. Normalize memory operation
event = guard.normalize_memory_event(
    operation="write",
    content={repr(memory_content)},
    scope={repr(scope)},
)

# 4. Evaluate against mission integrity & poisoning models
verdict = guard.evaluate_memory(event, contract, full_content={repr(memory_content)})

print("Verdict:", verdict.verdict)
print("Risk:", verdict.risk_score)
print("Reason:", verdict.reason)
"""
            render_python_code(python_code, title="ANCHOR Memory Evaluation Code")

        except Exception as ex:
            st.error(f"Memory evaluation error: {ex}")
            trace.add_event(EventType.ERROR, status="ERROR", reason=str(ex))
            record_trace(trace)


if __name__ == "__main__":
    render()

