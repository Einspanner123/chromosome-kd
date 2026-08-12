#!/usr/bin/env python3
"""Relocate non-submission evidence records out of the manuscript asset tree."""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "tools/experiment_db/experiments.db"
MOVES = {
    "lqcr-d2-final-seed42": (
        "docs/paper/latex/figures/v2/data/source_lqcr_final_only_seed42.json",
        "tools/experiment_db/evidence_sources/legacy_paper_assets/source_lqcr_final_only_seed42.json",
    ),
    "tradeoff-a6000-20260811": (
        "docs/paper/latex/figures/v2/data/source_fps_a6000_karyoflow.json",
        "tools/experiment_db/evidence_sources/legacy_paper_assets/source_fps_a6000_karyoflow.json",
    ),
    "tradeoff-dino-a6000-20260811": (
        "docs/paper/latex/figures/v2/data/source_fps_a6000_dino.json",
        "tools/experiment_db/evidence_sources/legacy_paper_assets/source_fps_a6000_dino.json",
    ),
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    con = sqlite3.connect(DB)
    for artifact_id, (_, new_relative) in MOVES.items():
        new_path = ROOT / new_relative
        if not new_path.is_file():
            raise FileNotFoundError(new_path)
        row = con.execute(
            "SELECT sha256 FROM evidence_artifact WHERE artifact_id=?", (artifact_id,)
        ).fetchone()
        if row is None:
            raise RuntimeError(f"missing artifact: {artifact_id}")
        actual = digest(new_path)
        if actual != row[0]:
            raise RuntimeError(f"SHA mismatch for {artifact_id}: {actual} != {row[0]}")
        con.execute(
            "UPDATE evidence_artifact SET path=?, notes=? WHERE artifact_id=?",
            (new_relative, "Internal evidence retained outside the manuscript submission tree.", artifact_id),
        )
    con.commit()
    print(f"updated {len(MOVES)} evidence paths")


if __name__ == "__main__":
    main()
