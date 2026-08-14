#!/usr/bin/env python3
"""Build a deterministic repository-cleanup baseline manifest.

Tracked files use Git blob identities. Large runtime trees receive metadata-only
summaries here; paper-critical payload hashes remain owned by evidence records,
and P5 creates a relocation manifest before moving any artifact.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ARTIFACT_ROOTS = ("work_dirs", "results", "analysis", "experiments/analysis")


def run_git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), *args], text=True
    ).strip()


def tracked_files(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    output = run_git(root, "ls-files", "-s")
    for line in output.splitlines():
        metadata, path = line.split("\t", 1)
        mode, blob, stage = metadata.split()
        file_path = root / path
        records.append(
            {
                "path": path,
                "mode": mode,
                "blob": blob,
                "stage": int(stage),
                "size_bytes": file_path.lstat().st_size,
            }
        )
    return records


def summarize_tree(path: Path) -> dict[str, Any]:
    if not os.path.lexists(path):
        return {"exists": False, "files": 0, "bytes": 0, "top_level": []}

    total_files = 0
    total_bytes = 0
    top_level: list[dict[str, Any]] = []
    for child in sorted(path.iterdir(), key=lambda item: item.name):
        child_files = 0
        child_bytes = 0
        if child.is_symlink():
            child_files = 1
            child_bytes = child.lstat().st_size
        elif child.is_file():
            child_files = 1
            child_bytes = child.stat().st_size
        else:
            for dirpath, _, filenames in os.walk(child):
                base = Path(dirpath)
                for filename in filenames:
                    candidate = base / filename
                    try:
                        child_bytes += candidate.lstat().st_size
                        child_files += 1
                    except FileNotFoundError:
                        continue
        total_files += child_files
        total_bytes += child_bytes
        top_level.append(
            {"name": child.name, "files": child_files, "bytes": child_bytes}
        )
    return {
        "exists": True,
        "files": total_files,
        "bytes": total_bytes,
        "top_level": top_level,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    tracked = tracked_files(root)
    top_counts = Counter(record["path"].split("/", 1)[0] for record in tracked)
    manifest = {
        "schema_version": 1,
        "plan_id": "repo-cleanup-20260815",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository_root_name": root.name,
        "git": {
            "commit": run_git(root, "rev-parse", "HEAD"),
            "branch": run_git(root, "branch", "--show-current"),
            "status_porcelain": run_git(root, "status", "--porcelain"),
        },
        "tracked": {
            "count": len(tracked),
            "bytes": sum(record["size_bytes"] for record in tracked),
            "top_level_counts": dict(sorted(top_counts.items())),
            "files": tracked,
        },
        "artifact_summaries": {
            relative: summarize_tree(root / relative) for relative in ARTIFACT_ROOTS
        },
    }

    output = args.output
    if not output.is_absolute():
        output = root / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
