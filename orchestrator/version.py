"""Single source of the Vox / Apollo version: <repo>/VERSION."""

from __future__ import annotations

from pathlib import Path

VERSION_FILE = Path(__file__).resolve().parent.parent / "VERSION"


def version() -> str:
    try:
        text = VERSION_FILE.read_text(encoding="utf-8").strip()
    except OSError:
        return "unknown"
    return text or "unknown"
