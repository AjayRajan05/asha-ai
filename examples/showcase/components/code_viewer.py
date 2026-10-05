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

"""Copy Python code generator component for ASHA Control Center."""

from __future__ import annotations

import streamlit as st


def render_python_code(code: str, title: str = "Python Integration Code") -> None:
    """Render a clean Python code block with heading and copy support."""
    st.markdown(f"#### {title}")
    st.code(code.strip(), language="python")


def render_copy_python(
    operation: str = "process",
    prompt: str = "Hello user@domain.com",
    mode: str = "balanced",
    token_budget: int = 1200,
    pii_mode: str = "rule",
    reversible: bool = False,
) -> None:
    """Render a clean 'Copy Python' code block matching the active configuration."""
    if operation == "process":
        code = f"""from asha import process, PolicyConfig

# Configure ASHA security policy
policy = PolicyConfig(
    mode="{mode}",
    pii_mode="{pii_mode}",
    reversible={reversible},
)

# Process prompt with semantic preservation and honest token optimization
result = process(
    {repr(prompt)},
    policy=policy,
    token_budget={token_budget},
)

print("Protected output:", result.output)
print("Security verdict:", result.security.safe)
print("Tokens saved:", result.metrics.tokens_saved)
"""
    elif operation == "sanitize":
        code = f"""from asha import sanitize, PolicyConfig

policy = PolicyConfig(mode="{mode}", pii_mode="{pii_mode}")
result = sanitize({repr(prompt)}, policy=policy)

print("Sanitized prompt:", result.output)
print("Detected threats:", result.security_result.threats)
"""
    elif operation == "optimize":
        code = f"""from asha import optimize

# Run token optimization with constraint preservation guarantees
result = optimize(
    {repr(prompt)},
    token_budget={token_budget},
)

print("Optimized text:", result.optimized)
print("Candidate accepted:", result.metadata.accepted)
print("Tokens saved:", result.metadata.tokens_saved)
"""
    elif operation == "wrap_llm":
        code = f"""import openai
from asha import wrap_llm, PolicyConfig

# Transparent non-mutating client wrapper
raw_client = openai.OpenAI()
client = wrap_llm(raw_client, mode="{mode}")

# Use client exactly as usual
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{{"role": "user", "content": {repr(prompt)}}}]
)
print(response.choices[0].message.content)
"""
    elif operation == "anchor":
        code = f"""from asha.runtime.anchor import AnchorRuntime, MissionContract

mission = MissionContract(
    goal="Help user safely inspect customer tickets",
    allowed_tools=["read_ticket", "search_knowledgebase"],
    forbidden_tools=["bash", "execute_command", "network_egress"],
)

runtime = AnchorRuntime(mission=mission)
# Intercept incoming agent action
decision = runtime.evaluate_action(
    action_type="command_execution",
    payload={{"command": "cat /etc/passwd"}},
)
print("Governance decision:", decision.decision)
"""
    else:
        code = f"""import asha

result = asha.process({repr(prompt)})
print(result)
"""

    render_python_code(code, title="Python Integration Code")
