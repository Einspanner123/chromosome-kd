#!/usr/bin/env python3
"""Aggregate already verified paper-facing test summaries without inference."""

from __future__ import annotations

import json
from pathlib import Path

import d2_test_unified as evaluator
import paper_test_multiseed as runner
import test_inference_ablations as ablations


ROOT = Path(__file__).resolve().parents[2]
GROUPS = ["sota", "solver", "topk", "lqcr", "distill", "renewal_d1"]


def main() -> None:
    records = []
    for paper_id, (dataset, _) in runner.selected_specs(GROUPS).items():
        evaluator.configure_dataset(dataset)
        root = ROOT / "results/paper_test_multiseed" / paper_id / paper_id
        by_seed = {}
        for path in root.glob("*/summary.json"):
            summary = json.loads(path.read_text())
            seed = int(summary["protocol"]["seed"])
            if seed not in runner.SEEDS:
                continue
            existing = by_seed.get(seed)
            if existing is not None and existing[0]["protocol_hash"] != summary["protocol_hash"]:
                raise RuntimeError(f"ambiguous {paper_id} seed {seed}")
            by_seed[seed] = (summary, path)
        missing = set(runner.SEEDS) - set(by_seed)
        if missing:
            raise RuntimeError(f"missing {paper_id}: {sorted(missing)}")
        for seed in runner.SEEDS:
            summary, path = by_seed[seed]
            extra_metrics = {}
            if "lqcr_beta_" in paper_id:
                extra_metrics = ablations.high_iou_ap(
                    path.parent / "predictions.bbox.json", evaluator.ANN_PATH)
            records.append({
                "paper_id": paper_id,
                "dataset": dataset.upper(),
                "seed": seed,
                "summary": summary,
                "extra_metrics": extra_metrics,
                "source_summary": str(path.relative_to(ROOT)),
                "source_summary_sha256": runner.sha256(path),
            })
    output = runner.finalize(records)
    print(output.relative_to(ROOT))


if __name__ == "__main__":
    main()
