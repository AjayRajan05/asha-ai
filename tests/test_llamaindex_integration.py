"""LlamaIndex integration tests - Gap 18.

Tests the LlamaIndex plugin module.  wrap_query_engine and related utilities
are tested via mock objects; the actual llama_index package is optional.
"""

import pytest

from asha.utils.dropin import process


# ---------------------------------------------------------------------------
# Module structure tests (no LlamaIndex required)
# ---------------------------------------------------------------------------


def test_llamaindex_plugin_importable():
    from asha.integrations.llamaindex.plugin import wrap_query_engine

    class _MockQueryEngine:
        def query(self, query_bundle):
            return f"importable:{query_bundle.query_str}"

    wrapped = wrap_query_engine(_MockQueryEngine(), privacy=False)
    assert wrapped is not None
    assert type(wrapped).__name__ == "ASHAQueryEngine"


def test_llamaindex_plugin_exports_wrap_query_engine():
    from asha.integrations.llamaindex.plugin import wrap_query_engine

    assert callable(wrap_query_engine)

    class _MockQE:
        def query(self, bundle):
            return f"exports:{getattr(bundle, 'query_str', bundle)}"

    wrapped = wrap_query_engine(_MockQE(), privacy=False)
    assert type(wrapped).__name__ == "ASHAQueryEngine"


def test_llamaindex_plugin_exports_asha_llm():
    from asha.integrations.llamaindex.plugin import ASHALLM

    assert ASHALLM is not None
    assert isinstance(ASHALLM, type) or callable(ASHALLM)


# ---------------------------------------------------------------------------
# wrap_query_engine: mock query engine (no LlamaIndex install needed)
# ---------------------------------------------------------------------------


def test_wrap_query_engine_returns_wrapped():
    pytest.importorskip("llama_index.core")
    from asha.integrations.llamaindex.plugin import wrap_query_engine
    from llama_index.core.schema import QueryBundle

    received = []

    class _MockQueryEngine:
        def query(self, query_bundle):
            received.append(query_bundle.query_str)
            return f"result:{query_bundle.query_str}"

    qe = _MockQueryEngine()
    wrapped = wrap_query_engine(qe, privacy=False)
    result = wrapped.query(QueryBundle(query_str="What is 2+2?"))
    assert received
    assert "result:" in str(result)
    assert "What is 2+2?" in received[0] or "2+2" in received[0]


def _make_query_bundle(query_str: str):
    """Create a minimal QueryBundle-compatible object (works with or without llama_index)."""

    class _QueryBundle:
        def __init__(self, qs):
            self.query_str = qs
            self.custom_embedding_strs = None
            self.embedding_strs = None

    return _QueryBundle(query_str)


def test_wrap_query_engine_processes_query():
    """Requires llama_index for the QueryBundle reconstruction inside the plugin."""
    pytest.importorskip("llama_index.core")
    from asha.integrations.llamaindex.plugin import wrap_query_engine
    from llama_index.core.schema import QueryBundle

    received = []

    class _MockQueryEngine:
        def query(self, bundle):
            received.append(bundle.query_str)
            return f"response to: {bundle.query_str}"

    qe = _MockQueryEngine()
    wrapped = wrap_query_engine(qe, privacy=False)
    result = wrapped.query(QueryBundle(query_str="What is 2+2?"))
    assert "response to" in str(result)


def test_wrap_query_engine_masks_pii():
    """PII in queries should be masked - requires llama_index for QueryBundle."""
    pytest.importorskip("llama_index.core")
    from asha.integrations.llamaindex.plugin import wrap_query_engine
    from llama_index.core.schema import QueryBundle

    received = []

    class _MockQueryEngine:
        def query(self, bundle):
            received.append(bundle.query_str)
            return f"result:{bundle.query_str}"

    qe = _MockQueryEngine()
    # Use a non-placeholder address (user@example.com is intentionally left
    # unmasked as a known teaching/placeholder local+domain pair).
    pii_email = "alice.chen@acme-corp.com"
    wrapped = wrap_query_engine(qe, privacy=True)
    wrapped.query(QueryBundle(query_str=f"Find documents about {pii_email}"))
    assert received
    assert pii_email not in received[0]


# ---------------------------------------------------------------------------
# Smoke test with real LlamaIndex (skipped if not installed)
# ---------------------------------------------------------------------------


def test_llamaindex_real_import_smoke():
    pytest.importorskip("llama_index")
    from asha.integrations.llamaindex.plugin import wrap_query_engine
    from llama_index.core.schema import QueryBundle

    class _MockQueryEngine:
        def query(self, query_bundle):
            return f"smoke:{query_bundle.query_str}"

    wrapped = wrap_query_engine(_MockQueryEngine(), privacy=False)
    result = wrapped.query(QueryBundle(query_str="ping"))
    assert str(result) == "smoke:ping"
