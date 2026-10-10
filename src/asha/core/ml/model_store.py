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

"""Hardened / lite detector model store with automatic download.

Distribution (Phase 2 option A):
  - Base ``pip install asha`` needs no models (lite heuristics still run).
  - When hardened artifacts are required, ASHA downloads the official
    ``asha-models`` bundle into a user cache **without prompting**.
  - Override with ``ASHA_MODELS_DIR`` or ``ASHA_MODELS_URL``.
  - Opt out (CI / air-gap): ``ASHA_DISABLE_MODEL_DOWNLOAD=1``.

Public docs must not publish thresholds or training recipes; this module only
fetches the released artifact bundle.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import threading
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# Official release asset for asha==0.4.2 detector weights (auto-fetched).
DEFAULT_MODELS_URL = (
    "https://github.com/AjayRajan05/asha-si/releases/download/"
    "asha-models-0.4.2/asha-models-0.4.2.zip"
)
MODELS_BUNDLE_VERSION = "0.4.2"
_MARKER_NAME = f".asha-models-{MODELS_BUNDLE_VERSION}.ok"

# SHA-256 of the official asha-models-0.4.2.zip release asset.
# Update this constant whenever a new model bundle is released.
# Compute with: sha256sum asha-models-0.4.2.zip
MODELS_BUNDLE_SHA256: str = "e9b61729eb27146bea194572165464af1059720941571e2a38f48d65a290efb5"

# Only these URL prefixes are trusted for model downloads.
# file: URIs are permitted for local/CI testing.
_ALLOWED_URL_PREFIXES = (
    "https://github.com/AjayRajan05/",
    "https://raw.githubusercontent.com/AjayRajan05/",
    "file:",
)

_lock = threading.Lock()
_ensured = False
_download_attempted = False


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def download_disabled() -> bool:
    return _env_truthy("ASHA_DISABLE_MODEL_DOWNLOAD")


def models_url() -> str:
    return os.environ.get("ASHA_MODELS_URL", DEFAULT_MODELS_URL).strip()


def default_cache_models_dir() -> Path:
    """User cache location for auto-downloaded models (no permission prompt)."""
    override = os.environ.get("ASHA_MODELS_CACHE")
    if override:
        return Path(override).expanduser().resolve()
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return Path(xdg).expanduser().resolve() / "asha" / "models"
    return Path.home().resolve() / ".cache" / "asha" / "models"


def get_models_dir(*, ensure: bool = False) -> Path:
    """Return the active models directory.

    Priority:
      1. ``ASHA_MODELS_DIR`` (explicit local / private tree)
      2. Auto-download cache (``~/.cache/asha/models`` or ``ASHA_MODELS_CACHE``)
      3. ``./models`` under cwd (dev checkout)
    """
    env = os.environ.get("ASHA_MODELS_DIR")
    if env:
        path = Path(env).expanduser().resolve()
        if ensure:
            ensure_models(path)
        return path

    cache = default_cache_models_dir()
    if ensure:
        ensure_models(cache)
        return cache

    cwd_models = (Path.cwd() / "models").resolve()
    if cwd_models.is_dir() and any(cwd_models.rglob("*")):
        return cwd_models
    if (cache / _MARKER_NAME).is_file() or any(cache.rglob("*.joblib")):
        return cache
    return cache


def candidate_model_paths(*relative_parts: str) -> List[Path]:
    """Ordered candidate paths for a model file under the models tree."""
    rel = Path(*relative_parts)
    paths: List[Path] = []
    env = os.environ.get("ASHA_MODELS_DIR")
    if env:
        paths.append(Path(env).expanduser() / rel)
    paths.append(default_cache_models_dir() / rel)
    paths.append(Path.cwd() / "models" / rel)
    # Editable / source layouts relative to this file.
    here = Path(__file__).resolve()
    for parent in list(here.parents)[:6]:
        paths.append(parent / "models" / rel)
    # Deduplicate while preserving order.
    seen = set()
    out: List[Path] = []
    for p in paths:
        key = str(p)
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def resolve_model_file(*relative_parts: str, ensure: bool = True) -> Optional[Path]:
    """Find a model file, optionally triggering auto-download first."""
    if ensure and not download_disabled():
        ensure_models()
    for path in candidate_model_paths(*relative_parts):
        if path.is_file():
            return path
    return None


def _marker_path(root: Path) -> Path:
    return root / _MARKER_NAME


def _bundle_looks_populated(root: Path) -> bool:
    if _marker_path(root).is_file():
        return True
    # Partial / legacy trees still usable.
    probes = (
        root / "injection" / "fusion_meta.joblib",
        root / "injection" / "lite" / "hashed_clf.json",
        root / "chain_guard" / "transition_matrix.json",
    )
    return any(p.is_file() for p in probes)


def _validate_models_url(url: str) -> None:
    """Raise ValueError if *url* is not from a trusted origin."""
    for prefix in _ALLOWED_URL_PREFIXES:
        if url.startswith(prefix):
            return
    raise ValueError(
        f"ASHA: untrusted ASHA_MODELS_URL '{url}'. "
        f"Allowed prefixes: {_ALLOWED_URL_PREFIXES}. "
        "Set ASHA_DISABLE_MODEL_DOWNLOAD=1 to skip the download entirely."
    )


def ensure_models(root: Optional[Path] = None) -> Path:
    """Ensure detector artifacts exist under ``root`` (download if needed).

    Emits a WARNING log before any network egress so users are never
    silently surprised. Set ``ASHA_DISABLE_MODEL_DOWNLOAD=1`` to opt out.
    Safe to call from hot paths (thread-locked, downloads at most once).
    """
    global _ensured, _download_attempted
    target = (root or default_cache_models_dir()).expanduser().resolve()
    if _bundle_looks_populated(target):
        _ensured = True
        return target

    with _lock:
        if _bundle_looks_populated(target):
            _ensured = True
            return target
        if download_disabled():
            logger.info(
                "ASHA model download disabled (ASHA_DISABLE_MODEL_DOWNLOAD); "
                "using lite/heuristic fallbacks when artifacts are missing."
            )
            return target
        if _download_attempted and not _bundle_looks_populated(target):
            return target
        _download_attempted = True
        url = models_url()
        url_trusted = True
        try:
            _validate_models_url(url)
        except ValueError as exc:
            logger.error(str(exc))
            url_trusted = False
        if url_trusted:
            logger.warning(
                "ASHA: downloading ML detector models (~4 MB) from %s "
                "to %s. Set ASHA_DISABLE_MODEL_DOWNLOAD=1 to skip.",
                url,
                target,
            )
        try:
            if url_trusted:
                _download_and_extract(url, target)
            else:
                raise ValueError("URL not trusted; skipping network fetch")
        except Exception as exc:
            # Dev convenience: use a locally built release zip if present.
            local_zip = _local_bundle_zip()
            if local_zip is not None:
                try:
                    logger.info("Using local models bundle %s", local_zip)
                    _extract_zip(local_zip, target)
                except Exception as local_exc:
                    logger.warning(
                        "ASHA could not install detector models from %s (%s) "
                        "or local bundle %s (%s). Continuing with lite/heuristic "
                        "fallbacks (not a silent 'secure' guarantee for the "
                        "hardened path).",
                        url,
                        exc,
                        local_zip,
                        local_exc,
                    )
                    return target
            else:
                logger.warning(
                    "ASHA could not download detector models from %s (%s). "
                    "Continuing with lite/heuristic fallbacks (not a silent "
                    "'secure' guarantee for the hardened path).",
                    url,
                    exc,
                )
                return target
        if _bundle_looks_populated(target):
            _marker_path(target).write_text(
                f"asha-models {MODELS_BUNDLE_VERSION}\n", encoding="utf-8"
            )
            _ensured = True
            logger.info("ASHA detector models ready at %s", target)
        return target


def _local_bundle_zip() -> Optional[Path]:
    env = os.environ.get("ASHA_MODELS_ZIP")
    candidates: List[Path] = []
    if env:
        candidates.append(Path(env).expanduser())
    here = Path(__file__).resolve()
    for parent in list(here.parents)[:6]:
        candidates.append(parent / "dist" / f"asha-models-{MODELS_BUNDLE_VERSION}.zip")
    candidates.append(Path.cwd() / "dist" / f"asha-models-{MODELS_BUNDLE_VERSION}.zip")
    for path in candidates:
        if path.is_file():
            return path
    return None


def _verify_bundle_checksum(archive: Path) -> None:
    """Verify SHA-256 of the downloaded archive when MODELS_BUNDLE_SHA256 is set.

    Raises ``ValueError`` if the digest does not match.
    Skips silently when MODELS_BUNDLE_SHA256 is empty (pre-release or CI override).
    """
    import hashlib

    expected = MODELS_BUNDLE_SHA256.strip()
    if not expected:
        logger.debug(
            "ASHA: MODELS_BUNDLE_SHA256 is not set; skipping checksum verification."
        )
        return
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != expected:
        raise ValueError(
            f"ASHA: model bundle checksum mismatch. "
            f"Expected {expected!r}, got {digest!r}. "
            "The downloaded file may be corrupt or tampered with."
        )
    logger.info("ASHA: model bundle checksum OK (%s)", digest[:16])


def _download_and_extract(url: str, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="asha-models-") as tmp:
        tmp_path = Path(tmp)
        archive = tmp_path / "asha-models.zip"
        logger.info("Downloading ASHA detector models from %s", url)
        if url.startswith("file:"):
            from urllib.parse import urlparse, unquote

            parsed = urlparse(url)
            src = Path(unquote(parsed.path))
            if os.name == "nt" and str(src).startswith("/"):
                # file:///C:/path → /C:/path on Windows
                src = Path(str(src)[1:])
            shutil.copy2(src, archive)
        else:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": f"asha/{MODELS_BUNDLE_VERSION} model-store"},
            )
            with (
                urllib.request.urlopen(req, timeout=120) as resp,
                archive.open("wb") as out,
            ):
                shutil.copyfileobj(resp, out)
        _verify_bundle_checksum(archive)
        _extract_zip(archive, dest)


def _extract_zip(archive: Path, dest: Path) -> None:
    """Extract *archive* into *dest* with Zip Slip prevention.

    Each member path is resolved and validated to be inside *extract_root*
    before extraction. An attacker-controlled zip with ``../../`` traversal
    entries will raise ``ValueError`` and abort the extraction.
    """
    dest.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="asha-models-x-") as tmp:
        extract_root = (Path(tmp) / "extracted").resolve()
        extract_root.mkdir()
        with zipfile.ZipFile(archive, "r") as zf:
            for member in zf.infolist():
                member_dest = (extract_root / member.filename).resolve()
                # Guard against Zip Slip path traversal.
                if not str(member_dest).startswith(str(extract_root) + os.sep) and \
                        member_dest != extract_root:
                    raise ValueError(
                        f"ASHA: Zip Slip detected: refusing to extract "
                        f"'{member.filename}' outside of '{extract_root}'."
                    )
                zf.extract(member, extract_root)
        payload = _find_models_root(extract_root)
        _merge_tree(payload, dest)


def _find_models_root(extracted: Path) -> Path:
    """Accept zip layouts with or without a top-level models/ folder."""
    if (extracted / "injection").is_dir() or (extracted / "chain_guard").is_dir():
        return extracted
    nested = extracted / "models"
    if nested.is_dir():
        return nested
    for child in extracted.iterdir():
        if child.is_dir() and (
            (child / "injection").is_dir() or (child / "models" / "injection").is_dir()
        ):
            if (child / "models").is_dir():
                return child / "models"
            return child
    return extracted


def _merge_tree(src: Path, dest: Path) -> None:
    for path in src.rglob("*"):
        if path.is_dir():
            continue
        rel = path.relative_to(src)
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def required_hardened_files() -> Tuple[str, ...]:
    return (
        "injection/fusion_meta.joblib",
        "injection/embedding_rf.joblib",
        "injection/perplexity_gb.joblib",
        "memory_guard/poisoning_rf.joblib",
        "chain_guard/transition_matrix.json",
        "mission/domain_clf.joblib",
        "intent/intent_clf.joblib",
        "alignment/isotonic_breakpoints.json",
    )


def missing_hardened_files(root: Optional[Path] = None) -> List[str]:
    base = root or get_models_dir(ensure=False)
    return [rel for rel in required_hardened_files() if not (base / rel).is_file()]
