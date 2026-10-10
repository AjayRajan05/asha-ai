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

"""Real-world use-case tests: Multi-turn conversational PII vaulting and session isolation."""

from __future__ import annotations

import pytest
from asha import PIIVault


def test_multiturn_conversation_preserves_consistent_tokens() -> None:
    """In a multi-turn chat, repeating the same PII must map to the same placeholder token."""
    vault = PIIVault()

    # Turn 1: User introduces themselves and their phone number
    turn1_user = "Hi, I am reaching out from user.primary@enterprise.com, phone 555-123-4567."
    turn1_masked = vault.mask(turn1_user)

    assert "user.primary@enterprise.com" not in turn1_masked
    assert "555-123-4567" not in turn1_masked
    assert vault.has_pii is True

    # Identify assigned email token
    import re
    email_tokens = re.findall(r"<EMAIL_\d+>", turn1_masked)
    assert len(email_tokens) == 1
    assigned_email_token = email_tokens[0]

    # Turn 2: User repeats their email later in the conversation
    turn2_user = "Please verify: did you send the confirmation to user.primary@enterprise.com?"
    turn2_masked = vault.mask(turn2_user)

    # Must reuse the same placeholder token so LLM retains referential entity consistency
    assert assigned_email_token in turn2_masked
    assert "user.primary@enterprise.com" not in turn2_masked

    # Turn 3: LLM replies referencing the placeholder token
    llm_raw_reply = f"Yes, the statement has been delivered to {assigned_email_token} successfully."
    user_visible_reply = vault.unmask(llm_raw_reply)

    assert "user.primary@enterprise.com" in user_visible_reply
    assert assigned_email_token not in user_visible_reply


def test_session_isolation_between_distinct_vaults() -> None:
    """Separate client sessions have independent vaults that do not leak or cross-contaminate."""
    vault_alice = PIIVault(session_id="session_alice")
    vault_bob = PIIVault(session_id="session_bob")

    masked_alice = vault_alice.mask("Customer alice@bank.com balance check")
    masked_bob = vault_bob.mask("Customer bob@bank.com balance check")

    assert "alice@bank.com" not in masked_alice
    assert "bob@bank.com" not in masked_bob

    # Alice's vault cannot unmask Bob's tokens and vice-versa
    unmasked_by_alice = vault_alice.unmask(masked_bob)
    assert "bob@bank.com" not in unmasked_by_alice


def test_pii_vault_clear_resets_session_state() -> None:
    """Clearing a vault safely removes all tracked entities and mappings."""
    vault = PIIVault()
    masked = vault.mask("Token sk-secret12345678901234567890 for auth")
    assert vault.has_pii is True

    vault.clear()
    assert vault.has_pii is False
    assert len(vault) == 0

    # Unmasking after clear cannot restore cleared secrets
    result = vault.unmask(masked)
    assert "sk-secret12345678901234567890" not in result
