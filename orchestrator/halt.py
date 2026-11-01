"""The global halt flag written by `fortress-halt` and lifted by `fortress-resume`.

While it exists Apollo neither starts nor keeps running.
"""

from __future__ import annotations

import os
from pathlib import Path

HALT_MESSAGE = "Halted by fortress-halt. Run fortress-resume to lift it."


def halt_flag() -> Path:
    base = os.environ.get("FORTRESS_HALT_STATE") or str(Path.home() / ".local/state/fortress")
    return Path(base) / "HALT"


def is_halted() -> bool:
    return halt_flag().exists()
