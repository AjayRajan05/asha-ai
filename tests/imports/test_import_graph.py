from __future__ import annotations
# Copyright 2026 Ajay Rajan
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Import graph smoke tests for canonical module paths."""


import importlib


def test_core_engines_importable() -> None:
    mod = importlib.import_module("asha.core.engines")
    assert callable(mod.sanitize_text)
    assert callable(mod.compile_prompt)
    assert callable(mod.optimize_tokens)
    sanitized = mod.sanitize_text("Contact alice@acme.com for help")
    assert "alice@acme.com" not in sanitized.output


def test_runtime_prompt_processor_importable() -> None:
    mod = importlib.import_module("asha.runtime")
    assert hasattr(mod, "PromptProcessor")
    assert hasattr(mod, "ExecutionProfile")
    assert not hasattr(mod, "Processor")


def test_integrations_wrap_llm_importable() -> None:
    mod = importlib.import_module("asha.integrations")
    assert callable(mod.wrap_llm)
    assert callable(mod.auto_patch)

    class _Bad:
        pass

    import pytest

    with pytest.raises(ValueError) as exc_info:
        mod.wrap_llm(_Bad())
    message = str(exc_info.value).lower()
    assert (
        "unsupported" in message
        or "client" in message
        or "adapter" in message
        or "wrap" in message
    )


def test_compat_legacy_results_importable() -> None:
    mod = importlib.import_module("asha.compat.legacy_results")
    assert callable(mod.to_legacy_pipeline_dict)
    from asha import process

    result = process("Contact alice@acme-corp.com", mode="balanced")
    legacy = mod.to_legacy_pipeline_dict(result)
    assert isinstance(legacy, dict)
    assert legacy.get("success") is True or "prompts" in legacy
    assert "alice@acme-corp.com" not in result.output
    prompts = legacy.get("prompts") or {}
    # Original may retain raw input; sanitized/compiled/optimized must not.
    assert "alice@acme-corp.com" not in str(prompts.get("sanitized", ""))
    assert "alice@acme-corp.com" not in str(prompts.get("compiled", ""))
    assert "alice@acme-corp.com" not in str(prompts.get("optimized", ""))


def test_types_results_importable() -> None:
    mod = importlib.import_module("asha.types")
    assert hasattr(mod, "ProcessResult")
    from asha import process

    result = process("Contact alice@acme-corp.com")
    assert type(result).__name__ == "ProcessResult"
    assert "alice@acme-corp.com" not in result.output
    assert len(result.output) > 0


def test_core_ml_importable_without_heavy_deps() -> None:
    import sys

    heavy = ("sentence_transformers", "sklearn", "lightgbm", "kenlm")
    before = {name for name in heavy if name in sys.modules}

    mod = importlib.import_module("asha.core.ml")
    # Touch public names (triggers lazy submodule loads, not heavy deps).
    assert callable(mod.get_encoder)
    assert callable(mod.load_thresholds)
    assert callable(mod.scan_secrets)

    after = {name for name in heavy if name in sys.modules}
    assert after == before
