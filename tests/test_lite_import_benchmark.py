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

"""Cold-import benchmark for lite-tier ASHA surface (no heavy ML deps)."""

from __future__ import annotations

import ast
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

HEAVY_MODULES = (
    "numpy",
    "sklearn",
    "torch",
    "transformers",
    "sentence_transformers",
)

# Hard budget after lazy package __init__ fix. Measured cold import of
# asha.core.text + injection_lite + LiteInjectionDetector() is typically
# tens of ms; 200ms (350ms on Windows) leaves headroom for antivirus / disk jitter.
_COLD_IMPORT_BUDGET_MS = 350.0 if sys.platform == "win32" else 200.0


@pytest.fixture(autouse=True)
def _isolate_imports():
    """Best-effort: record heavy modules present before each test."""
    before = {name for name in HEAVY_MODULES if name in sys.modules}
    yield
    after = {name for name in HEAVY_MODULES if name in sys.modules}
    newly_loaded = after - before
    assert not newly_loaded, f"lite path pulled in heavy modules: {newly_loaded}"


def test_injection_lite_detect_no_heavy_deps():
    from asha.core.security.injection_lite import LiteInjectionDetector

    detector = LiteInjectionDetector()
    result = detector.detect("summarize quarterly revenue for the board")
    assert result.mode == "lite"
    assert result.probability >= 0.0


def test_cold_import_text_and_injection_lite():
    """True cold import in a fresh interpreter (avoids polluted parent sys.modules)."""
    repo_src = str(Path(__file__).resolve().parents[1] / "src")
    src = textwrap.dedent(
        f"""
        import sys, time
        sys.path.insert(0, {repo_src!r})
        t0 = time.perf_counter()
        import asha.core.text  # noqa: F401
        import asha.core.security.injection_lite as lite_mod
        _ = lite_mod.LiteInjectionDetector()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        heavy = [m for m in {list(HEAVY_MODULES)!r} if m in sys.modules]
        print(f"ELAPSED_MS={{elapsed_ms:.1f}}")
        print(f"HEAVY={{heavy!r}}")
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", src],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    elapsed_ms = None
    heavy: list[str] = []
    for line in proc.stdout.splitlines():
        if line.startswith("ELAPSED_MS="):
            elapsed_ms = float(line.split("=", 1)[1])
        if line.startswith("HEAVY="):
            heavy = ast.literal_eval(line.split("=", 1)[1])
    assert elapsed_ms is not None, proc.stdout
    assert heavy == [], f"lite cold import loaded heavy modules: {heavy}"
    assert (
        elapsed_ms < _COLD_IMPORT_BUDGET_MS
    ), f"cold import {elapsed_ms:.0f}ms exceeds {_COLD_IMPORT_BUDGET_MS:.0f}ms budget"
