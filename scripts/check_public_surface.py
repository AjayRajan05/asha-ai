#!/usr/bin/env python3
"""Fail if the public tree contains secrets or private-artifact paths.

Usage:
    python scripts/check_public_surface.py

Local ``training/``, ``data/``, ``models/``, etc. may exist on disk when
gitignored; this check only fails if they are *tracked* or not ignored
(would ship in a clean clone / sdist).
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FORBIDDEN_DIRS = (
    "training",
    "asha_training_data_generators",
    "data",
    "models",
    "config",  # root config/ only: packaged thresholds live under src/asha/config/
    ".private-artifacts",
    "asha-private-artifacts",
    "jailbreak_llms",
)

FORBIDDEN_FILES = (
    ".env",
    "docs_detail.md",
    "find_empty_tests.py",
    "fake_tests_report.json",
    "anchor_telemetry.jsonl",
    "llm_augment.py",
    "build_asha_dataset.py",
    "build_best_dataset.py",
    "improve_data_quality.py",
    "Data_README.md.md",
    "To-Do.md",
    "asha_training_data_generators.zip",
    "injection_malicious.jsonl",
    "injection_benign.jsonl",
)

# Real Google API key shape; ignore obvious test placeholders.
GOOGLE_KEY = re.compile(r"AIza[0-9A-Za-z_-]{30,}")
OPENAI_LIVE = re.compile(r"(?<![a-zA-Z0-9])sk-(?!1234567890)[a-zA-Z0-9]{20,}")

SKIP_DIR_NAMES = {
    ".git",
    ".venv",
    "venv",
    ".venv_asha_clean_test",
    ".venv-mypy-ci",
    "site",
    "build",
    "dist",
    "__pycache__",
    ".pytest_cache",
    "node_modules",
    "graphify-out",
    ".agents",
}


def _git_ignored(rel: str) -> bool:
    """True if ``rel`` is ignored by git (safe to keep locally)."""
    try:
        proc = subprocess.run(
            ["git", "check-ignore", "-q", rel],
            cwd=ROOT,
            capture_output=True,
            check=False,
        )
        return proc.returncode == 0
    except OSError:
        return False


def _git_tracked(rel: str) -> bool:
    try:
        proc = subprocess.run(
            ["git", "ls-files", "--error-unmatch", rel],
            cwd=ROOT,
            capture_output=True,
            check=False,
        )
        return proc.returncode == 0
    except OSError:
        return False


def main() -> int:
    errors: list[str] = []

    for name in FORBIDDEN_DIRS:
        p = ROOT / name
        if not p.exists():
            continue
        if _git_ignored(name) and not _git_tracked(name):
            continue
        errors.append(f"forbidden directory present (not gitignored): {name}/")

    for name in FORBIDDEN_FILES:
        p = ROOT / name
        if not p.exists():
            continue
        if _git_ignored(name) and not _git_tracked(name):
            continue
        errors.append(f"forbidden file present (not gitignored): {name}")

    # Scratch suite logs at repo root
    for p in ROOT.glob("_*.txt"):
        if not _git_ignored(p.name):
            errors.append(f"scratch log present: {p.name}")
    for p in ROOT.glob("_*.xml"):
        if not _git_ignored(p.name):
            errors.append(f"scratch junit present: {p.name}")

    # Retrieve all public tracked and untracked-unignored files via git ls-files
    try:
        proc = subprocess.run(
            ["git", "ls-files", "-c", "-o", "--exclude-standard"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        public_files = proc.stdout.splitlines()
    except Exception as exc:
        print(f"Error querying git ls-files: {exc}")
        return 1

    for rel_s in public_files:
        rel = Path(rel_s)
        if any(part in SKIP_DIR_NAMES or part.startswith(".venv") for part in rel.parts):
            continue

        p = ROOT / rel
        if not p.is_file():
            continue

        # Model binaries must not be in public tree
        if p.suffix.lower() in {".joblib", ".pkl"}:
            errors.append(f"model weight in public tree: {rel_s}")
            continue

        if p.suffix.lower() not in {".py", ".md", ".toml", ".yml", ".yaml", ".json", ".txt", ".example"}:
            continue

        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        # Test / benchmark fixtures intentionally contain fake sk-… strings.
        if rel.parts[0] in {"tests", "benchmarks", "examples"}:
            if GOOGLE_KEY.search(text):
                errors.append(f"possible Google API key in {rel_s}")
            continue

        if GOOGLE_KEY.search(text):
            errors.append(f"possible Google API key in {rel_s}")
        if OPENAI_LIVE.search(text):
            errors.append(f"possible live OpenAI-style key in {rel_s}")

    # Version pin consistency for the 0.4.2 release build
    init_v = re.search(r'__version__\s*=\s*"([^"]+)"', (ROOT / "src/asha/__init__.py").read_text(encoding="utf-8"))
    py_v = re.search(r'^version\s*=\s*"([^"]+)"', (ROOT / "pyproject.toml").read_text(encoding="utf-8"), re.M)
    if not init_v or not py_v or init_v.group(1) != py_v.group(1):
        errors.append("version mismatch between pyproject.toml and src/asha/__init__.py")
    elif py_v.group(1) != "0.4.2":
        errors.append(f"expected release pin 0.4.2, found {py_v.group(1)}")

    if errors:
        print("PUBLIC SURFACE CHECK FAILED:")
        for e in errors:
            print(f"  - {e}")
        return 1

    print("PUBLIC SURFACE CHECK OK (0.4.2, no private artifacts/secrets in public tree)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
