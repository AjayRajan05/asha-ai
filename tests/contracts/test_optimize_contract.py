"""Contract tests - optimize() must never invoke security."""

import ast
import inspect

import pytest

from asha import optimize
from asha.core import engines
from asha.types.results import OptimizeResult


def test_optimize_preserves_pii_email():
    email = "john@example.com"
    result = optimize(email)
    assert isinstance(result, OptimizeResult)
    assert email in result.output


def test_optimize_tokens_does_not_import_security_module():
    source = inspect.getsource(engines.optimize_tokens)
    tree = ast.parse(source)
    imports = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    for node in imports:
        if isinstance(node, ast.ImportFrom) and node.module:
            assert (
                "security" not in node.module
            ), f"optimize_tokens must not import {node.module}"
        for alias in getattr(node, "names", []):
            name = alias.name
            assert "security" not in name.lower()


def test_sanitize_does_not_call_optimize_engine(monkeypatch):
    calls = {"optimize": 0}

    def fake_optimize(*args, **kwargs):
        calls["optimize"] += 1
        raise AssertionError("sanitize must not call optimize_tokens")

    monkeypatch.setattr(engines, "optimize_tokens", fake_optimize)
    from asha import sanitize

    result = sanitize("hello@example.com")
    assert calls["optimize"] == 0
    assert "hello@example.com" not in result.output
