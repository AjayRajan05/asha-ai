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

"""Tests ensuring all Showcase pages and components import cleanly."""

from __future__ import annotations

import importlib
from pathlib import Path
import pytest


def test_import_all_showcase_pages() -> None:
    """Every page module in examples/showcase/pages must import cleanly."""
    pages_dir = Path(__file__).resolve().parent.parent.parent / "examples" / "showcase" / "pages"
    assert pages_dir.exists(), f"Pages dir not found at {pages_dir}"

    page_files = [f for f in pages_dir.glob("*.py") if f.name != "__init__.py"]
    assert len(page_files) >= 15, f"Expected at least 15 page files, found {len(page_files)}"

    for page_file in page_files:
        module_name = f"examples.showcase.pages.{page_file.stem}"
        mod = importlib.import_module(module_name)
        assert hasattr(mod, "render"), f"Module {module_name} must export a render callable"
        assert callable(getattr(mod, "render")), f"{module_name}.render must be callable"


def test_import_all_showcase_components() -> None:
    """Every component in examples/showcase/components must import cleanly."""
    comp_dir = Path(__file__).resolve().parent.parent.parent / "examples" / "showcase" / "components"
    assert comp_dir.exists(), f"Components dir not found at {comp_dir}"

    comp_files = [f for f in comp_dir.glob("*.py") if f.name != "__init__.py"]
    for comp_file in comp_files:
        module_name = f"examples.showcase.components.{comp_file.stem}"
        importlib.import_module(module_name)


def test_import_app_entrypoint() -> None:
    """App entrypoint must import cleanly."""
    app_mod = importlib.import_module("examples.showcase.app")
    assert hasattr(app_mod, "main")
