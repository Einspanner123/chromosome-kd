"""Rebuild every version-2 paper figure from its editable source."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCES = ROOT / "sources"

FIGURE_SCRIPTS = [
    SOURCES / "fig01_system_overview.py",
    SOURCES / "fig02_lqcr_principle.py",
    SOURCES / "fig04_precision_bottleneck.py",
    SOURCES / "fig06_efficiency.py",
    SOURCES / "fig07_qualitative.py",
]

PENDING_SCRIPTS = [
    SOURCES / "fig03_lqcr_clean_replication.py",
    SOURCES / "fig05_lqcr_stratified_effect.py",
]


def main() -> None:
    for script in FIGURE_SCRIPTS:
        print(f"[figures] building {script.name}")
        subprocess.run([sys.executable, str(script)], cwd=SOURCES, check=True)
    for script in PENDING_SCRIPTS:
        print(f"[figures] pending evidence: {script.name}")


if __name__ == "__main__":
    main()
