from __future__ import annotations
"""Tests verifying all public APIs are importable directly from the root asha package."""


import asyncio


def test_root_api_imports():
    from asha import (
        process,
        sanitize,
        optimize,
        wrap_llm,
        Agent,
        PolicyConfig,
        ANCHOR,
        anchor,
        SecureLLM,
        Profile,
        PIIVault,
        scan_rag_context,
        AgentCostGuard,
    )
    assert callable(process)
    assert callable(sanitize)
    assert callable(optimize)
    assert callable(wrap_llm)
    assert Agent is not None
    assert PolicyConfig is not None
    assert ANCHOR is not None
    assert anchor is not None
    assert SecureLLM is not None
    assert Profile is not None
    assert PIIVault is not None
    assert callable(scan_rag_context)
    assert AgentCostGuard is not None


def test_import_exceptions():
    from asha import (
        ASHAError,
        ASHAProcessingError,
        ASHAAnchorBlocked,
        AgentBudgetExceeded,
    )
    for exc in (ASHAError, ASHAProcessingError, ASHAAnchorBlocked, AgentBudgetExceeded):
        assert issubclass(exc, Exception)


def test_hasattr_checks():
    import asha
    assert hasattr(asha, "process")
    assert hasattr(asha, "sanitize")
    assert hasattr(asha, "optimize")
    assert hasattr(asha, "wrap_llm")
    assert hasattr(asha, "PolicyConfig")
    assert hasattr(asha, "ANCHOR")
    assert hasattr(asha, "anchor")
    assert hasattr(asha, "Agent")
    assert hasattr(asha, "SecureLLM")
    assert hasattr(asha, "Profile")
    assert hasattr(asha, "PIIVault")
    assert hasattr(asha, "scan_rag_context")
    assert hasattr(asha, "AgentCostGuard")


def test_all_symbols_are_importable():
    """Every symbol in __all__ must be resolvable via getattr."""
    import asha

    failures = []
    for name in asha.__all__:
        try:
            val = getattr(asha, name)
            assert val is not None or name == "__version__"
        except Exception as exc:
            failures.append(f"{name}: {exc}")
    assert not failures, "Symbols in __all__ not importable: " + str(failures)


def test_import_async_variants():
    from asha import process_async, sanitize_async, optimize_async

    assert asyncio.iscoroutinefunction(process_async)
    assert asyncio.iscoroutinefunction(sanitize_async)
    assert asyncio.iscoroutinefunction(optimize_async)
