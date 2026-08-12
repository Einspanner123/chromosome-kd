"""Rebuild every version-2 paper figure from its editable source."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCES = ROOT / "sources"

FIGURE_SCRIPTS = [
    SOURCES / "fig00_speed_accuracy.py",
    SOURCES / "fig01_system_overview.py",
    SOURCES / "fig02_lqcr_principle.py",
    SOURCES / "fig07_qualitative.py",
]


def main() -> None:
    for script in FIGURE_SCRIPTS:
        print(f"[figures] building {script.name}")
        subprocess.run([sys.executable, str(script)], cwd=SOURCES, check=True)


if __name__ == "__main__":
    main()
