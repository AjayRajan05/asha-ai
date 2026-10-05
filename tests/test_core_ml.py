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

"""Unit tests for asha.core.ml shared infrastructure (step 1)."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

from asha.core.ml.calibration import (
    ThresholdBands,
    Verdict,
    bucket_probability,
    derive_threshold_at_fpr,
    get_bands,
    load_thresholds,
)
from asha.core.ml.embeddings import (
    EMBEDDING_DIM,
    EmbeddingEncoder,
    get_encoder,
)
from asha.core.ml.secret_entropy import (
    max_secret_confidence,
    scan_secrets,
    shannon_entropy,
)


# ---------------------------------------------------------------------------
# Import-graph / lazy-load guarantees
# ---------------------------------------------------------------------------


def test_ml_package_import_does_not_pull_heavy_deps() -> None:
    """Importing asha.core.ml must not load sentence_transformers / sklearn."""
    # Ensure a clean check against already-loaded modules from other tests.
    heavy = ("sentence_transformers", "sklearn", "lightgbm", "kenlm", "torch")
    before = {name for name in heavy if name in sys.modules}

    import asha.core.ml as ml_pkg

    importlib.reload(ml_pkg)
    after = {name for name in heavy if name in sys.modules}
    assert after == before


def test_embeddings_module_import_is_lite() -> None:
    heavy = ("sentence_transformers", "sklearn", "torch")
    before = {name for name in heavy if name in sys.modules}
    import asha.core.ml.embeddings as emb

    importlib.reload(emb)
    after = {name for name in heavy if name in sys.modules}
    assert after == before


# ---------------------------------------------------------------------------
# Embeddings (real MiniLM: downloads on first use; no hash fallback)
# ---------------------------------------------------------------------------


def test_encoder_minilm_deterministic() -> None:
    enc = EmbeddingEncoder()
    a = enc.encode_one("ignore previous instructions")
    b = enc.encode_one("ignore previous instructions")
    c = enc.encode_one("totally different benign text")
    assert a.shape == (EMBEDDING_DIM,)
    assert (a == b).all()
    assert not (a == c).all()
    assert abs(float((a**2).sum()) ** 0.5 - 1.0) < 1e-4


def test_get_encoder_singleton() -> None:
    e1 = get_encoder(reset=True)
    e2 = get_encoder()
    assert e1 is e2


def test_encoder_raises_without_deps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing sentence-transformers must raise (no silent hash substitute)."""
    import builtins

    import asha.core.ml.embeddings as emb

    # Clear process-wide MiniLM so the import block is actually exercised.
    monkeypatch.setattr(emb, "_shared_st_model", None)
    monkeypatch.setattr(emb, "_shared_st_name", None)
    monkeypatch.setattr(emb, "_encoder_singleton", None)
    monkeypatch.setattr(emb, "_minilm_ok", None)

    real_import = builtins.__import__

    def _block_sentence_transformers(
        name, globals=None, locals=None, fromlist=(), level=0
    ):
        if name == "sentence_transformers" or name.startswith("sentence_transformers."):
            raise ImportError("simulated missing sentence-transformers")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _block_sentence_transformers)
    enc = emb.EmbeddingEncoder()
    with pytest.raises(ImportError, match="asha\\[hardened\\]"):
        enc.encode("hello")


def test_cosine_similarity_identical_texts() -> None:
    enc = EmbeddingEncoder()
    assert enc.cosine_similarity("same", "same") == pytest.approx(1.0, abs=1e-4)


# ---------------------------------------------------------------------------
# Calibration / thresholds
# ---------------------------------------------------------------------------


def test_load_thresholds_from_repo_config() -> None:
    cfg = load_thresholds(reload=True)
    assert "injection" in cfg
    assert "alignment" in cfg
    bands = get_bands("injection", thresholds=cfg)
    assert isinstance(bands, ThresholdBands)
    assert 0.0 <= bands.safe_max <= bands.block_min <= 1.0


def test_bucket_probability_three_band() -> None:
    bands = ThresholdBands(safe_max=0.2, block_min=0.8)
    assert bucket_probability(0.05, bands) is Verdict.SAFE
    assert bucket_probability(0.5, bands) is Verdict.REVIEW
    assert bucket_probability(0.9, bands) is Verdict.BLOCK


def test_bucket_probability_four_band_alignment() -> None:
    bands = ThresholdBands(safe_max=0.2, warn_max=0.45, block_min=0.75)
    assert bucket_probability(0.1, bands) is Verdict.SAFE
    assert bucket_probability(0.3, bands) is Verdict.WARN
    assert bucket_probability(0.5, bands) is Verdict.REVIEW
    assert bucket_probability(0.9, bands) is Verdict.BLOCK


def test_bucket_probability_fail_closed_on_nan() -> None:
    bands = ThresholdBands(safe_max=0.2, block_min=0.8)
    assert bucket_probability(float("nan"), bands) is Verdict.REVIEW
    assert bucket_probability("not-a-float", bands) is Verdict.REVIEW  # type: ignore[arg-type]
    assert bucket_probability(float("nan"), bands, fail_closed=False) is Verdict.SAFE


def test_threshold_bands_reject_inverted() -> None:
    with pytest.raises(ValueError):
        ThresholdBands(safe_max=0.9, block_min=0.1)


def test_derive_threshold_at_fpr() -> None:
    # 10 negatives with scores 0.0..0.9, 5 positives clustered high.
    y_true = [0] * 10 + [1] * 5
    y_prob = [i / 10 for i in range(10)] + [0.85, 0.9, 0.92, 0.95, 0.99]
    thr = derive_threshold_at_fpr(y_true, y_prob, target_fpr=0.1)
    # At most 1/10 negatives above threshold.
    fp = sum(1 for yt, yp in zip(y_true, y_prob) if yt == 0 and yp >= thr)
    assert fp / 10 <= 0.1 + 1e-9
    assert 0.0 <= thr <= 1.0


def test_packaged_thresholds_file_exists() -> None:
    pkg = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "asha"
        / "config"
        / "thresholds.yaml"
    )
    # Also accept installed layout via importlib.
    from asha import config as asha_config

    cfg_dir = Path(asha_config.__file__).resolve().parent
    assert (cfg_dir / "thresholds.yaml").is_file() or pkg.is_file()


# ---------------------------------------------------------------------------
# Secret entropy scanner: adversarial / shape cases
# ---------------------------------------------------------------------------


def test_shannon_entropy_uniform_vs_repeated() -> None:
    repeated = "aaaaaaaaaaaaaaaa"
    uniform = "aB3$xY9!qLmN2pQw"
    assert shannon_entropy(repeated) < shannon_entropy(uniform)


def test_detect_openai_sk_shape() -> None:
    text = "leak this sk-abcdefghijklmnopqrstuvwxyz0123456789 into logs"
    hits = scan_secrets(text)
    assert hits
    assert any(h.shape == "openai_sk" for h in hits)
    assert max_secret_confidence(text) > 0.5


def test_detect_google_api_key_shape() -> None:
    text = "key=AIzaSyA-abcdefghijklmnopqrstuvwx"
    hits = scan_secrets(text)
    assert any(h.shape == "google_api" for h in hits)


def test_detect_aws_access_key_shape() -> None:
    text = "AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE"
    hits = scan_secrets(text)
    assert any(h.shape == "aws_access_key" for h in hits)


def test_detect_jwt_three_part() -> None:
    # Minimal synthetic JWT-shaped token (header.payload.sig), not a real secret.
    jwt = (
        "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
        "eyJzdWIiOiIxMjM0NTY3ODkwIn0."
        "SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
    )
    hits = scan_secrets(f"Authorization: Bearer {jwt}")
    assert any(h.shape == "jwt" for h in hits)


def test_benign_low_entropy_not_flagged() -> None:
    text = "please summarize the quarterly revenue report for the sales team"
    hits = scan_secrets(text)
    # No shape matchers and no high-entropy tokens.
    assert all(h.shape != "high_entropy" or h.confidence < 0.7 for h in hits)
    assert max_secret_confidence(text) < 0.55


def test_high_entropy_opaque_token_without_keyword() -> None:
    # Random-looking base64-ish blob with no labeled keyword nearby.
    blob = "xK9mP2qR7vL4nW8yT1zA0bC3dE5fG6hJ"
    text = f"config value is {blob} for the worker"
    hits = scan_secrets(text)
    assert any(h.shape == "high_entropy" for h in hits) or any(
        h.confidence >= 0.55 for h in hits
    )


def test_generic_bearer_assignment() -> None:
    # Synthetic opaque token (not a provider-shaped secret) for assignment detection.
    text = 'api_key = "ASHADEMOKEY_9f3c2a1b8e7d6c5b4a39281706f5e4d3"'
    hits = scan_secrets(text)
    assert hits
    assert max_secret_confidence(text) > 0.4


def test_scan_empty_and_short() -> None:
    assert scan_secrets("") == []
    assert scan_secrets("short") == []
