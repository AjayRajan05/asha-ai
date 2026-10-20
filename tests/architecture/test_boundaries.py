from __future__ import annotations
# Copyright 2026 Ajay Rajan
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Architecture boundary enforcement tests."""


import ast
import re
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src" / "asha"

FORBIDDEN_IMPORTS = {
    "core": {"runtime", "integrations", "compat"},
    "runtime": {"integrations", "compat"},
    "types": {"compat", "runtime", "integrations"},
    "utils": {"compat"},
}

_RELATIVE_FORBIDDEN = re.compile(
    r"from\s+\.+(runtime|integrations|compat)(?:\.|\s+import)"
)


def _read_absolute_imports(module_path: Path) -> set[str]:
    source = module_path.read_text(encoding="utf-8-sig")
    tree = ast.parse(source)
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            top = node.module.split(".")[0]
            if top in {"runtime", "integrations", "compat"}:
                found.add(top)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top in {"runtime", "integrations", "compat"}:
                    found.add(top)
    return found


def _read_relative_forbidden(module_path: Path, layer: str) -> set[str]:
    source = module_path.read_text(encoding="utf-8-sig")
    forbidden = FORBIDDEN_IMPORTS[layer]
    found: set[str] = set()
    for match in _RELATIVE_FORBIDDEN.finditer(source):
        target = match.group(1)
        if target in forbidden:
            found.add(target)
    return found


def _py_files_under(prefix: str) -> list[Path]:
    base = SRC / prefix
    if not base.exists():
        return []
    return [p for p in base.rglob("*.py") if p.name != "__init__.py"]


@pytest.mark.parametrize("layer", list(FORBIDDEN_IMPORTS))
def test_layer_does_not_import_forbidden_modules(layer: str) -> None:
    forbidden = FORBIDDEN_IMPORTS[layer]
    violations: list[str] = []
    for path in _py_files_under(layer):
        bad = (
            _read_absolute_imports(path) | _read_relative_forbidden(path, layer)
        ) & forbidden
        if bad:
            rel = path.relative_to(SRC)
            violations.append(f"{rel} imports forbidden {sorted(bad)}")
    assert not violations, "\n".join(violations)


def test_no_legacy_in_runtime() -> None:
    runtime = SRC / "runtime"
    legacy_names = {"legacy_dict", "legacy_pipeline", "legacy_results"}
    for path in runtime.rglob("*.py"):
        if path.stem in legacy_names or "legacy" in path.name:
            pytest.fail(f"Legacy module remains in runtime: {path}")


def test_no_legacy_in_core() -> None:
    core = SRC / "core"
    for path in core.rglob("*legacy*"):
        if path.suffix == ".py":
            pytest.fail(f"Legacy module remains in core: {path}")


def test_no_pipeline_facade() -> None:
    assert not (SRC / "compat" / "legacy_pipeline.py").exists()
    assert not (SRC / "compat" / "async_pipeline.py").exists()
    assert not (SRC / "pipeline").exists()
    for path in SRC.rglob("*.py"):
        source = path.read_text(encoding="utf-8-sig")
        if re.search(r"^class Pipeline\b", source, re.MULTILINE):
            pytest.fail(f"Pipeline class found in {path.relative_to(SRC)}")


def test_compat_legacy_results_only() -> None:
    compat_py = [p for p in (SRC / "compat").glob("*.py") if p.name != "__init__.py"]
    names = {p.name for p in compat_py}
    assert names <= {"legacy_results.py", "warnings.py"}
