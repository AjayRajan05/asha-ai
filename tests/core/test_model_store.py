from __future__ import annotations
"""Tests for automatic detector model store (Phase 2)."""


import zipfile
from pathlib import Path

import pytest

from asha.core.ml import model_store as ms


def _make_tiny_bundle(zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr(
            "injection/lite/hashed_clf.json",
            '{"weights_b64":"","weights_dtype":"f32","bias":0.0,"n_bits":8,"ngram_range":[3,5]}',
        )
        # Empty weights_b64 may fail loaders; marker + lite file is enough for store tests.
        zf.writestr(".asha-models-0.4.2.ok", "asha-models 0.4.2\n")
        zf.writestr(
            "injection/lite/char_ngram_lm.json",
            '{"n":3,"unseen_log_prob":-10,"alphabet_hint":32,"log_probs":{},'
            '"benign_baseline_ppl":8,"ppl_scale":4}',
        )


def test_candidate_paths_include_cache_and_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ASHA_MODELS_DIR", raising=False)
    monkeypatch.setenv("ASHA_MODELS_CACHE", str(tmp_path / "cache"))
    paths = ms.candidate_model_paths("injection", "fusion_meta.joblib")
    assert any("cache" in str(p) for p in paths)
    assert any(p.name == "fusion_meta.joblib" for p in paths)


def test_ensure_models_uses_local_zip_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "models-cache"
    local_zip = tmp_path / "asha-models-0.4.2.zip"
    _make_tiny_bundle(local_zip)

    monkeypatch.setenv("ASHA_MODELS_CACHE", str(cache))
    monkeypatch.setenv("ASHA_MODELS_ZIP", str(local_zip))
    monkeypatch.delenv("ASHA_MODELS_DIR", raising=False)
    monkeypatch.delenv("ASHA_DISABLE_MODEL_DOWNLOAD", raising=False)
    monkeypatch.setenv("ASHA_MODELS_URL", "https://example.invalid/asha-models.zip")
    monkeypatch.setattr(ms, "_ensured", False)
    monkeypatch.setattr(ms, "_download_attempted", False)

    root = ms.ensure_models(cache)
    assert (root / "injection" / "lite" / "hashed_clf.json").is_file()
    assert (root / ".asha-models-0.4.2.ok").is_file()


def test_download_disabled_skips_fetch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "empty-cache"
    monkeypatch.setenv("ASHA_MODELS_CACHE", str(cache))
    monkeypatch.setenv("ASHA_DISABLE_MODEL_DOWNLOAD", "1")
    monkeypatch.delenv("ASHA_MODELS_DIR", raising=False)
    monkeypatch.setattr(ms, "_ensured", False)
    monkeypatch.setattr(ms, "_download_attempted", False)

    root = ms.ensure_models(cache)
    assert root == cache.resolve()
    assert not (root / ".asha-models-0.4.2.ok").is_file()
