"""ASHA Control Center - Intent Drift Lab.

Demonstrates ANCHOR runtime intent drift detection across multi-step agent trajectories:
- Trajectory execution through AnchorRuntime
- Progressive alignment and drift score tracking
- Cumulative risk calculation (RiskSummary)
- Safe simulated tool execution
"""

import sys
import time
from pathlib import Path
from typing import Any, Dict, List

_PAGES_DIR = Path(__file__).resolve().parent
_SHOWCASE_DIR = _PAGES_DIR.parent
_PROJECT_ROOT = _SHOWCASE_DIR.parent.parent
for _p in (str(_PROJECT_ROOT), str(_SHOWCASE_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import streamlit as st

from examples.showcase.scenarios.intent_drift import DRIFT_TRAJECTORIES
from examples.showcase.runtime.trace import Trace, EventType
from examples.showcase.runtime.session import record_trace
from examples.showcase.components.header import render_header
from examples.showcase.components.status_badge import render_status_pill
from examples.showcase.components.object_inspector import render_object_inspector
from examples.showcase.components.code_viewer import render_python_code


def render() -> None:
    render_header(title="INTENT DRIFT LAB", subtitle="Monitor agent trajectories, detect mission departure, and bound off-task execution")


    trajectory_names = list(DRIFT_TRAJECTORIES.keys())
    selected_traj_name = st.selectbox("Select Agent Trajectory Scenario", trajectory_names, index=0)
    traj_data = DRIFT_TRAJECTORIES[selected_traj_name]

    mission_prompt = st.text_input(
        "Initial Agent Mission Prompt",
        value=traj_data["mission"],
        help="The baseline mission contract established before agent execution begins.",
    )

    st.markdown("### Configured Trajectory Steps")
    steps: List[Dict[str, Any]] = traj_data["steps"]

    # Preview steps in an expander
    with st.expander("Inspect Configured Steps", expanded=True):
        for s in steps:
            col_num, col_desc, col_act, col_exp = st.columns([1, 4, 3, 2])
            with col_num:
                st.markdown(f"**Step {s['step']}**")
            with col_desc:
                st.markdown(f"**{s['name']}**")
                st.caption(s["description"])
            with col_act:
                st.code(f"{s['action_type']}: {s['payload'].get('tool_name', '')}", language="json")
            with col_exp:
                exp_v = s.get("expected_verdict", "ALLOW")
                badge_type = "success" if exp_v == "ALLOW" else ("warning" if exp_v == "WARN" else "blocked")
                st.markdown(f"Expected: {render_status_pill(exp_v, badge_type)}", unsafe_allow_html=True)

    run_btn = st.button("Execute Trajectory Through ANCHOR", type="primary", use_container_width=True)

    if run_btn:
        trace = Trace(provider="ANCHOR_RUNTIME", model="anchor_trajectory_governor")
        t0 = time.perf_counter()

        trace.add_event(
            EventType.INPUT_RECEIVED,
            status="INFO",
            reason=f"Initializing ANCHOR runtime with baseline mission: '{mission_prompt}'",
            metadata={"mission": mission_prompt, "step_count": len(steps)},
        )

        try:
            from asha.runtime.anchor import AnchorRuntime
            from asha.runtime.anchor.verdicts import Verdict

            allowed_tools = ["search_flights", "compare_fares", "fetch_booking_terms", "read_customer_data"]
            runtime = AnchorRuntime(
                mission=mission_prompt,
                allowed_tools=allowed_tools,
                risk_tolerance="LOW",
            )

            results: List[Dict[str, Any]] = []

            for s in steps:
                step_num = s["step"]
                act_type = s["action_type"]
                payload = s["payload"]

                # Trace event for step attempt
                trace.add_event(
                    EventType.ACTION_NORMALIZED,
                    status="INFO",
                    reason=f"Step {step_num}: Normalizing action '{s['name']}' ({act_type})",
                    metadata={"step": step_num, "payload": payload},
                )

                # Real ANCHOR evaluation
                t_step_start = time.perf_counter()
                allowed = runtime.evaluate_action_request(
                    action_type=act_type,
                    payload=payload,
                    record=True,
                    raise_on_block=False,
                )
                t_step_ms = (time.perf_counter() - t_step_start) * 1000.0

                latest_action = runtime.state.actions_history[-1] if runtime.state.actions_history else None
                latest_approval = runtime.state.approval_history[-1] if runtime.state.approval_history else None
                latest_risk = runtime.state.risk_history[-1] if runtime.state.risk_history else None

                verdict_val = latest_approval.verdict.verdict.value if latest_approval else ("ALLOW" if allowed else "BLOCK")
                reason_val = latest_approval.verdict.reason if latest_approval else ("Action permitted" if allowed else "Blocked by policy")

                # Simulated execution if allowed
                sim_result = "Action executed safely in simulation sandbox." if allowed else "Action suppressed by ANCHOR firewall."

                if allowed:
                    trace.add_event(
                        EventType.ACTION_ALLOWED,
                        status="SUCCESS",
                        duration_ms=t_step_ms,
                        reason=f"Step {step_num} ALLOWED: {reason_val}",
                        metadata={"step": step_num, "tool": payload.get("tool_name", "")},
                    )
                else:
                    trace.add_event(
                        EventType.ACTION_BLOCKED,
                        status="BLOCKED",
                        duration_ms=t_step_ms,
                        reason=f"Step {step_num} BLOCKED: {reason_val}",
                        metadata={"step": step_num, "tool": payload.get("tool_name", "")},
                    )

                results.append({
                    "step": step_num,
                    "name": s["name"],
                    "action_type": act_type,
                    "payload": payload,
                    "allowed": allowed,
                    "verdict": verdict_val,
                    "reason": reason_val,
                    "risk_summary": latest_risk,
                    "latency_ms": t_step_ms,
                    "sim_result": sim_result,
                })

            total_ms = (time.perf_counter() - t0) * 1000.0
            trace.add_event(
                EventType.REQUEST_COMPLETE,
                status="SUCCESS",
                duration_ms=total_ms,
                reason="Trajectory evaluation completed.",
            )
            record_trace(trace)

            # Metrics row
            st.markdown("### Trajectory Security Assessment")
            final_risk = runtime.state.risk_history[-1] if runtime.state.risk_history else None

            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            with m_col1:
                st.metric("Total Steps", len(steps))
            with m_col2:
                drift_score = final_risk.drift_score if final_risk else 0.0
                st.metric("Final Drift Score", f"{drift_score:.2f}")
            with m_col3:
                align_score = final_risk.alignment_score if final_risk else 1.0
                st.metric("Mean Alignment", f"{align_score:.2f}")
            with m_col4:
                sev = final_risk.severity.value if final_risk else "LOW"
                badge = "success" if sev == "LOW" else ("warning" if sev == "MEDIUM" else "blocked")
                st.markdown(f"**Cumulative Severity:**<br>{render_status_pill(sev, badge)}", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Step-by-step display
            st.markdown("### Step-by-Step Enforcement Timeline")
            for res in results:
                is_ok = res["allowed"]
                border = "#10b981" if is_ok else "#ef4444"
                bg = "rgba(16, 185, 129, 0.04)" if is_ok else "rgba(239, 68, 68, 0.05)"
                icon = "✓" if is_ok else "🚫"
                badge_type = "success" if is_ok else "blocked"

                st.markdown(
                    f"""
                    <div style="background: {bg}; border: 1px solid {border}; border-radius: 8px; padding: 14px; margin-bottom: 12px;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                            <span style="font-weight: 700; font-size: 1rem; color: {'#34d399' if is_ok else '#f87171'};">
                                {icon} Step {res['step']}: {res['name']}
                            </span>
                            <div>
                                {render_status_pill(res['verdict'], badge_type)}
                                <span style="font-family: monospace; font-size: 0.8rem; color: #94a3b8; margin-left: 8px;">
                                    {res['latency_ms']:.1f} ms
                                </span>
                            </div>
                        </div>
                        <p style="margin: 0 0 6px 0; color: #cbd5e1; font-size: 0.88rem;">
                            <strong>Enforcement:</strong> {res['reason']}
                        </p>
                        <p style="margin: 0; color: #64748b; font-size: 0.82rem; font-family: monospace;">
                            Simulation: {res['sim_result']}
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Object Inspector for state & final risk
            st.markdown("<br>", unsafe_allow_html=True)
            render_object_inspector(
                raw_obj=final_risk,
                title="ANCHOR Final RiskSummary Object",
                human_data={
                    "Drift Score": f"{final_risk.drift_score:.3f}" if final_risk else "N/A",
                    "Alignment Score": f"{final_risk.alignment_score:.3f}" if final_risk else "N/A",
                    "Total Risk Score": f"{final_risk.total_risk_score:.1f}" if final_risk else "N/A",
                    "Severity": str(final_risk.severity.value) if final_risk else "N/A",
                    "Total Steps Evaluated": len(results),
                    "Actions Permitted": sum(1 for r in results if r["allowed"]),
                    "Actions Blocked": sum(1 for r in results if not r["allowed"]),
                },
            )

            # Copy Python
            st.markdown("<br>", unsafe_allow_html=True)
            python_code = f"""from asha.runtime.anchor import AnchorRuntime

# 1. Initialize ANCHOR runtime with baseline mission
runtime = AnchorRuntime(
    mission={repr(mission_prompt)},
    allowed_tools={allowed_tools},
    risk_tolerance="LOW",
)

# 2. Evaluate steps sequentially
steps = {steps}

for s in steps:
    allowed = runtime.evaluate_action_request(
        action_type=s["action_type"],
        payload=s["payload"],
        record=True,
    )
    print(f"Step {{s['step']}} ({{s['name']}}): {{'ALLOWED' if allowed else 'BLOCKED'}}")

final_risk = runtime.state.risk_history[-1]
print("Final Drift Score:", final_risk.drift_score)
print("Severity:", final_risk.severity)
"""
            render_python_code(python_code, title="Trajectory Intent Drift Code")

        except Exception as ex:
            st.error(f"Trajectory execution error: {ex}")
            trace.add_event(EventType.ERROR, status="ERROR", reason=str(ex))
            record_trace(trace)


if __name__ == "__main__":
    render()

