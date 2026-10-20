from __future__ import annotations
"""Contract: experimental APIs warn; core APIs do not."""


import warnings

import pytest

from asha import process
from asha.exceptions import ASHAExperimentalWarning
from asha.integrations import auto_patch, disable_auto_patch
from asha.runtime.adapters.factory import SmartRoutingAdapter


def test_process_does_not_emit_experimental_warning() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ASHAExperimentalWarning)
        process("hello world", mode="off")
    assert not [w for w in caught if issubclass(w.category, ASHAExperimentalWarning)]


def test_auto_patch_emits_experimental_warning() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ASHAExperimentalWarning)
        try:
            auto_patch(enable=False, verbose=False)
        finally:
            disable_auto_patch()
    assert any(issubclass(w.category, ASHAExperimentalWarning) for w in caught)


def test_smart_routing_adapter_emits_experimental_warning() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ASHAExperimentalWarning)
        SmartRoutingAdapter({"chat": "mock"})
    assert any(issubclass(w.category, ASHAExperimentalWarning) for w in caught)


def test_experimental_warning_message_points_at_docs() -> None:
    with pytest.warns(ASHAExperimentalWarning, match="experimental-features"):
        SmartRoutingAdapter({"chat": "mock"})
