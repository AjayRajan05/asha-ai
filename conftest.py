"""Pytest configuration - ensure local src/ is used before site-packages."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Union

import pytest

from asha.types.results import OptimizeResult, ProcessResult, SanitizeResult

# Do NOT set ASHA_DISABLE_ML here (doing so previously skipped rule-based safety).
# HF toxic-bert is opt-in via ASHA_ENABLE_HF_SAFETY=1.

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

ResultType = Union[ProcessResult, SanitizeResult, OptimizeResult, str, dict]


def output_of(result: ResultType) -> str:
    """Extract prompt text from ProcessResult or legacy shapes."""
    if isinstance(result, (ProcessResult, SanitizeResult, OptimizeResult)):
        return result.output
    if isinstance(result, dict):
        return str(
            result.get("optimized")
            or result.get("sanitized")
            or result.get("output")
            or ""
        )
    return str(result)


def metrics_of(result: ResultType) -> dict:
    if isinstance(result, ProcessResult) and result.metrics:
        return result.metrics.to_dict()
    if isinstance(result, OptimizeResult) and result.metrics:
        return result.metrics.to_dict()
    if isinstance(result, dict):
        return dict(result.get("metrics") or result)
    return {}


@pytest.fixture(scope="session", autouse=True)
def _preload_minilm_once():
    """Load MiniLM once before any test can pollute torch/numpy.

    Some CLI/transformers paths reload numpy and break subsequent torch
    imports on Windows. Sharing one loaded SentenceTransformer keeps real
    embedding gates available for the whole suite.
    """
    try:
        from asha.core.ml.embeddings import get_encoder

        get_encoder().ensure_loaded()
    except Exception:
        pass


@pytest.fixture(autouse=True)
def _reset_auto_patch_state():
    """Prevent auto_patch tests from leaking global SDK patch state."""
    yield
    try:
        from asha.utils.auto_patch import auto_patch, enable_auto_patch

        enable_auto_patch()
        auto_patch(enable=False)
    except Exception:
        pass
