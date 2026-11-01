"""Where Apollo lives. Everything path-like in config resolves through here.

APOLLO_HOME overrides the install root; otherwise it is the repository root (the parent of orchestrator/).
"""
from __future__ import annotations

import os
from pathlib import Path

APOLLO_HOME = Path(os.environ.get("APOLLO_HOME") or Path(__file__).resolve().parent.parent)


def apollo_path(value: str | os.PathLike[str]) -> str:
    """~ expands; a relative path is relative to APOLLO_HOME; an absolute path is kept."""
    p = Path(value).expanduser()
    return str(p if p.is_absolute() else APOLLO_HOME / p)
