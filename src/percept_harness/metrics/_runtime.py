"""Shared preflight, subprocess and output helpers for the metric wrappers.

Standard library only, so it also works in the metric environments.
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

def verify_revision(source: Path, expected: str | None, *, runner=None) -> dict[str, str | None]:
    if not (source / ".git").exists():
        raise ValueError(f"Source is not a git checkout: {source}")
    if expected is None:
        return {"revision": None}
    result = (runner or subprocess.run)(["git", "-C", str(source), "rev-parse", "HEAD"],
                                        check=True, capture_output=True, text=True)
    actual = result.stdout.strip()
    if actual != expected:
        raise ValueError(f"{source} revision mismatch: expected {expected}, got {actual}")
    return {"revision": actual}


def verify_source(root: Path, files, record: dict, revision: str) -> dict:
    for relative in files:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Missing required source file: {path}")
    return {**record, **verify_revision(root, revision)}


def run_logged(command, *, log: Path, label: str, mode: str = "x", **kwargs) -> None:
    with log.open(mode, encoding="utf-8") as stream:
        completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, **kwargs)
    if completed.returncode:
        raise RuntimeError(f"{label} exited with {completed.returncode}; see {log}")


def write_json_exclusive(path: Path, value, *, newline: bool = True, parents: bool = False) -> str:
    """Create a new, strictly valid JSON file and return its text (without the newline)."""
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    if parents:
        path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(text + "\n" if newline else text)
    return text


def run_cli(label: str, function, *, allow_nan: bool = False) -> int:
    try:
        print(json.dumps(function(), ensure_ascii=False, indent=2, allow_nan=allow_nan))
        return 0
    except Exception as error:
        print(f"{label} failed: {error}", file=sys.stderr)
        return 1
