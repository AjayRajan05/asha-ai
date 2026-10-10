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

"""Lazy-loaded MiniLM sentence embedding encoder (encode-only, never generative).

Used by injection detection, memory guard, mission classifier, and optimizer
safety checks. Callers must share a single instance via :func:`get_encoder`.

No hash/pseudo-embedding fallback: if MiniLM cannot load, raise with install
guidance so callers download ``asha[hardened]`` / fix torch rather than
silently scoring with non-semantic vectors.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, List, Optional, Sequence, Union

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

TextInput = Union[str, Sequence[str]]

# Process-wide shared SentenceTransformer: reloading MiniLM mid-process can
# native-crash on Windows (ACCESS_VIOLATION); never unload once loaded.
_shared_st_model: Optional[Any] = None
_shared_st_name: Optional[str] = None
_shared_st_lock = threading.Lock()


def _get_shared_sentence_transformer(model_name: str) -> Any:
    global _shared_st_model, _shared_st_name
    with _shared_st_lock:
        if _shared_st_model is not None and _shared_st_name == model_name:
            return _shared_st_model
        from sentence_transformers import SentenceTransformer

        # Downloads from HuggingFace Hub on first use when not cached.
        _shared_st_model = SentenceTransformer(model_name)
        _shared_st_name = model_name
        return _shared_st_model


class EmbeddingEncoder:
    """Encode text to fixed-size vectors. Heavy deps load on first encode()."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        *,
        allow_hash_fallback: bool = False,
    ) -> None:
        if allow_hash_fallback:
            logger.warning(
                "allow_hash_fallback is ignored: MiniLM is required. "
                "Install with: pip install asha[hardened]"
            )
        self.model_name = model_name
        self.allow_hash_fallback = False
        self._model: Optional[Any] = None
        self._lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def using_fallback(self) -> bool:
        """Always False (hash fallback removed)."""
        return False

    def _load_model(self) -> None:
        try:
            import numpy as np  # noqa: F401 (preload to prevent native thread crashes on Windows)
            self._model = _get_shared_sentence_transformer(self.model_name)
        except Exception as exc:
            raise ImportError(
                "sentence-transformers/torch MiniLM is required for embedding "
                f"gates (failed with {type(exc).__name__}: {exc}). "
                "Install with: pip install asha[hardened] "
                f"and ensure model {self.model_name!r} can download."
            ) from exc

    def ensure_loaded(self) -> None:
        if self.is_loaded:
            return
        with self._lock:
            self._load_model()

    def encode(
        self,
        texts: TextInput,
        *,
        normalize: bool = True,
    ) -> "object":
        """Return an ``(n, EMBEDDING_DIM)`` numpy array of embeddings."""
        import numpy as np

        self.ensure_loaded()
        if isinstance(texts, str):
            batch: List[str] = [texts]
        else:
            batch = list(texts)
        if not batch:
            return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)

        model = self._model
        assert model is not None
        raw = model.encode(
            batch,
            normalize_embeddings=normalize,
            show_progress_bar=False,
        )
        vectors = np.asarray(raw, dtype=np.float32)
        if normalize:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-12)
            vectors = vectors / norms
        return vectors

    def encode_one(self, text: str, *, normalize: bool = True) -> "object":
        import numpy as np

        matrix = self.encode(text, normalize=normalize)
        return np.asarray(matrix)[0]

    def cosine_similarity(self, a: str, b: str) -> float:
        import numpy as np

        va = np.asarray(self.encode_one(a, normalize=True), dtype=np.float32)
        vb = np.asarray(self.encode_one(b, normalize=True), dtype=np.float32)
        return float(np.dot(va, vb))


_encoder_singleton: Optional[EmbeddingEncoder] = None
_singleton_lock = threading.Lock()
_minilm_ok: Optional[bool] = None


def get_encoder(
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    allow_hash_fallback: bool = False,
    reset: bool = False,
) -> EmbeddingEncoder:
    """Return the process-wide shared MiniLM encoder instance."""
    global _encoder_singleton
    if reset:
        with _singleton_lock:
            # Keep the underlying SentenceTransformer; only refresh the wrapper.
            _encoder_singleton = EmbeddingEncoder(model_name=model_name)
            if _shared_st_model is not None and _shared_st_name == model_name:
                _encoder_singleton._model = _shared_st_model
            return _encoder_singleton
    if _encoder_singleton is None:
        with _singleton_lock:
            if _encoder_singleton is None:
                _encoder_singleton = EmbeddingEncoder(model_name=model_name)
    return _encoder_singleton


def minilm_loadable(*, force: bool = False) -> bool:
    """True when MiniLM can be loaded in-process (downloads if needed)."""
    global _minilm_ok
    if _shared_st_model is not None:
        _minilm_ok = True
        return True
    if not force and _minilm_ok is not None:
        return _minilm_ok
    try:
        get_encoder().ensure_loaded()
        _minilm_ok = True
        return True
    except Exception:
        _minilm_ok = False
        return False
