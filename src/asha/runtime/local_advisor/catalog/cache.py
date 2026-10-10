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

"""JSON cache for local model catalog."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Optional

from ..constants import CACHE_TTL_SECONDS


def cache_dir() -> Path:
    base = os.environ.get("ASHA_CACHE_DIR")
    if base:
        return Path(base) / "local_models"
    return Path.home() / ".cache" / "asha" / "local_models"


def cache_path(name: str = "models.json") -> Path:
    path = cache_dir()
    path.mkdir(parents=True, exist_ok=True)
    return path / name


def load_cache(
    name: str = "models.json", ttl: int = CACHE_TTL_SECONDS
) -> Optional[list[dict[str, Any]]]:
    path = cache_path(name)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if time.time() - payload.get("cached_at", 0) > ttl:
            return None
        data = payload.get("models")
        return data if isinstance(data, list) else None
    except Exception:
        return None


def save_cache(models: list[dict[str, Any]], name: str = "models.json") -> None:
    path = cache_path(name)
    payload = {"cached_at": time.time(), "models": models}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
