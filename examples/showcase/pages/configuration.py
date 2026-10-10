"""ASHA Control Center - PolicyConfig & Presets.

Interactive configuration builder for PolicyConfig and Profile:
- Standard enterprise profiles: STRICT, BALANCED, DEVELOPER, PERMISSIVE
- Granular knobs: strictness, optimization aggressiveness, token budgets, threat blocking
- Clear documentation of supported vs unavailable capabilities
- Live effective configuration JSON preview & copyable Python code
"""

import sys
from pathlib import Path

_PAGES_DIR = Path(__file__).resolve().parent
_SHOWCASE_DIR = _PAGES_DIR.parent
_PROJECT_ROOT = _SHOWCASE_DIR.parent.parent
for _p in (str(_PROJECT_ROOT), str(_SHOWCASE_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import streamlit as st

from examples.showcase.components.header import render_header
from examples.showcase.components.status_badge import render_status_pill
from examples.showcase.components.object_inspector import render_object_inspector
from examples.showcase.components.code_viewer import render_python_code


def render() -> None:
    render_header(title="POLICY CONFIGURATION", subtitle="Deterministic policy profiles, SI alignment guardrails, and custom thresholds")


    from asha.core.policy_config import PolicyConfig, PolicyMode, Profile

    tab_preset, tab_custom, tab_matrix = st.tabs([
        "Profile Presets",
        "Custom PolicyConfig Builder",
        "Feature Availability Matrix",
    ])

    # -------------------------------------------------------------
    # TAB 1: Profile Presets
    # -------------------------------------------------------------
    with tab_preset:
        st.markdown("### Zero-Config Enterprise Security Profiles")
        st.markdown(
            "<p style='color: #94a3b8; font-size: 0.85rem;'>"
            "Pre-tuned combinations of detection thresholds, token budgets, and fail-closed policies."
            "</p>",
            unsafe_allow_html=True,
        )

        preset_choice = st.selectbox(
            "Select Security Profile",
            ["BALANCED (Recommended)", "STRICT (Regulated / High-Risk)", "SI_ALIGNMENT (Superintelligence / Agent Alignment)", "DEVELOPER (Local Debugging)", "PERMISSIVE (Audit Only)"],
            index=0,
        )

        if "STRICT" in preset_choice:
            profile = Profile.STRICT()
            badge = "blocked"
            desc = "Maximum-security posture. Aggressive 0.65 injection threshold, 1000-token cap, 5-step agent ceiling, deterministic mode, fail-closed."
        elif "SI_ALIGNMENT" in preset_choice:
            profile = Profile.SI_ALIGNMENT()
            badge = "info"
            desc = "Autonomous Superintelligence and agent alignment. Deterministic fail-closed, 0.60 injection threshold, 4096-token agent context, 10-step recursive execution cap."
        elif "DEVELOPER" in preset_choice:
            profile = Profile.DEVELOPER()
            badge = "purple"
            desc = "Optimized for local developer iteration. High visibility, verbose debug info, generous 4096-token budget."
        elif "PERMISSIVE" in preset_choice:
            profile = Profile.PERMISSIVE()
            badge = "warning"
            desc = "Audit and observation mode. Never blocks requests; logs detected anomalies without halting user pipelines."
        else:
            profile = Profile.BALANCED()
            badge = "success"
            desc = "Default enterprise production profile. Balances strong PII and injection protection with low false positives."

        p_col1, p_col2 = st.columns([3, 1])
        with p_col1:
            st.markdown(f"**Profile Mode:** `{profile.mode}` | **Status:** {render_status_pill(profile._name, badge)}", unsafe_allow_html=True)
            st.caption(desc)
        with p_col2:
            st.metric("Token Budget", f"{profile.token_budget} tok")

        p_m1, p_m2, p_m3, p_m4 = st.columns(4)
        with p_m1:
            st.metric("Injection Cutoff", f"{profile.injection_threshold:.2f}")
        with p_m2:
            st.metric("Max Agent Steps", profile.max_agent_steps)
        with p_m3:
            st.metric("Fail Closed", "True" if profile.fail_closed else "False")
        with p_m4:
            st.metric("Rehydrate PII", "True" if profile.rehydrate_pii else "False")

        st.markdown("<br>", unsafe_allow_html=True)
        render_python_code(
            f"""from asha import Profile, SecureLLM, wrap_llm

# Initialize zero-config profile
profile = Profile.{profile._name}()

# Apply to any client instance
# client = wrap_llm(raw_client, profile=profile)
""",
            title=f"Profile.{profile._name}() Python Usage",
        )

    # -------------------------------------------------------------
    # TAB 2: Custom PolicyConfig Builder
    # -------------------------------------------------------------
    with tab_custom:
        st.markdown("### Fine-Grained PolicyConfig Knobs")

        c1, c2, c3 = st.columns(3)
        with c1:
            mode_str = st.selectbox("Policy Mode", ["balanced", "strict", "lite", "off"], index=0)
        with c2:
            pii_mode = st.selectbox("PII Engine", ["hybrid", "rule", "ml_only"], index=0)
        with c3:
            security_level = st.selectbox("Security Level", ["medium", "high", "low"], index=0)

        c4, c5 = st.columns(2)
        with c4:
            pii_strictness = st.slider("PII Strictness Threshold", 0.0, 1.0, 0.8, 0.05)
        with c5:
            opt_level = st.slider("Optimization Aggressiveness", 0.0, 1.0, 0.5, 0.05)

        t1, t2, t3, t4 = st.columns(4)
        with t1:
            threat_blocking = st.checkbox("Threat Blocking", value=True)
        with t2:
            reversible_pii = st.checkbox("Reversible Masking", value=True)
        with t3:
            deterministic_mode = st.checkbox("Deterministic Mode", value=False)
        with t4:
            preserve_intent = st.checkbox("Preserve Intent Check", value=True)

        token_reduction_limit = st.slider("Max Token Reduction Ceiling", 0.1, 0.9, 0.8, 0.05)

        # Build real PolicyConfig
        policy = PolicyConfig(
            mode=PolicyMode(mode_str),
            pii_strictness=pii_strictness,
            optimization_level=opt_level,
            threat_blocking=threat_blocking,
            deterministic=deterministic_mode,
            max_token_reduction=token_reduction_limit,
            pii_mode=pii_mode,
            reversible=reversible_pii,
            preserve_intent=preserve_intent,
            security_level=security_level,
        )

        st.markdown("### Effective Policy Configuration")
        render_object_inspector(
            raw_obj=policy,
            title="PolicyConfig Dataclass Inspection",
            human_data={
                "Mode": policy.mode.value,
                "PII Strictness": f"{policy.pii_strictness:.2f}",
                "Optimization Level": f"{policy.optimization_level:.2f}",
                "Threat Blocking": str(policy.threat_blocking),
                "PII Engine": policy.pii_mode,
                "Reversible Masking": str(policy.reversible),
                "Deterministic": str(policy.deterministic),
                "Max Reduction": f"{int(policy.max_token_reduction * 100)}%",
            },
        )

        st.markdown("<br>", unsafe_allow_html=True)
        py_snippet = f"""from asha import PolicyConfig, process

policy = PolicyConfig(
    mode={repr(policy.mode.value)},
    pii_strictness={policy.pii_strictness},
    optimization_level={policy.optimization_level},
    threat_blocking={policy.threat_blocking},
    deterministic={policy.deterministic},
    pii_mode={repr(policy.pii_mode)},
    reversible={policy.reversible},
)

# Pass directly to process() or runtime agents
result = process("User prompt...", policy=policy)
"""
        render_python_code(py_snippet, title="Custom PolicyConfig Code Snippet")

    # -------------------------------------------------------------
    # TAB 3: Feature Availability Matrix
    # -------------------------------------------------------------
    with tab_matrix:
        st.markdown("### Feature Availability & Limitations")
        st.markdown(
            "<p style='color: #94a3b8; font-size: 0.85rem;'>"
            "Transparency regarding what ASHA currently provides vs what is intentionally excluded or planned."
            "</p>",
            unsafe_allow_html=True,
        )

        feature_matrix = [
            {"Capability": "Dual-Stage PII Masking (Rule + ML)", "Status": "Supported", "Notes": "Reversible token vault with session scoping."},
            {"Capability": "Prompt Injection Detection", "Status": "Supported", "Notes": "Fast-path heuristics + calibrated classifier."},
            {"Capability": "Token Optimization & Constraint Validation", "Status": "Supported", "Notes": "Preserves negation, numbers, format; rolls back on violation."},
            {"Capability": "ANCHOR Mission Contract Governance", "Status": "Supported", "Notes": "Immutable baseline contract with tool and domain firewalling."},
            {"Capability": "Action Type Normalization", "Status": "Supported", "Notes": "Maps command_execution/tool_call to canonical action events."},
            {"Capability": "Memory Write Firewall", "Status": "Supported", "Notes": "Detects cognitive poisoning and scope escalation."},
            {"Capability": "Intent Drift Tracking", "Status": "Supported", "Notes": "Calculates trajectory drift and alignment decay."},
            {"Capability": "RAG Context Indirect Injection Scanning", "Status": "Supported", "Notes": "Scans retrieved chunks and delimiter breakouts."},
            {"Capability": "Hardware Enclave Attestation (SGX/SEV)", "Status": "Unavailable", "Notes": "Out of scope for software-only Python library."},
            {"Capability": "Multi-Cluster Automatic Failover", "Status": "Unavailable", "Notes": "Application-level gateway; infrastructure handled upstream."},
        ]

        st.dataframe(feature_matrix, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    render()

