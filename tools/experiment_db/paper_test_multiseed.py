#!/usr/bin/env python3
"""Run paper-facing test evaluations with three deterministic inference seeds.

This is an inference-stability audit of a fixed checkpoint, not a substitute
for independently trained checkpoints. Results retain that distinction in the
database protocol and manuscript.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import sqlite3

import d2_test_unified as evaluator
import test_inference_ablations as ablations


ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "tools/experiment_db/experiments.db"
SEEDS = (42, 123, 789)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def custom_specs() -> dict[str, tuple[str, dict]]:
    return {
        "rf_heun": ("d2", ablations.SUITES["solver"]["heun_4"]),
        "euler_1": ("d2", ablations.SUITES["solver"]["euler_1"]),
        "euler_4": ("d2", ablations.SUITES["solver"]["euler_4"]),
        "dpmpp_1": ("d2", ablations.SUITES["solver"]["dpm_solver_pp_1"]),
        "dpmpp_4": ("d2", ablations.SUITES["solver"]["dpm_solver_pp_4"]),
        "topk_k500_on": ("d2", ablations.SUITES["topk_renewal"]["k500_renewal_on"]),
        "topk_k500_off": ("d2", ablations.SUITES["topk_renewal"]["k500_renewal_off"]),
        "topk_k200_on": ("d2", ablations.SUITES["topk_renewal"]["k200_renewal_on"]),
        "topk_k100_on": ("d2", ablations.SUITES["topk_renewal"]["k100_renewal_on"]),
        "topk_k100_off": ("d2", ablations.SUITES["topk_renewal"]["k100_renewal_off"]),
        **{
            f"lqcr_beta_{key.split('_', 1)[1]}": ("d2", spec)
            for key, spec in ablations.SUITES["lqcr_beta"].items()
        },
        "head_student_h3": ("d2", {
            "label": "Head-distilled H3 student",
            "config": "experiments/configs/ldmdet/directions/mainline_ablation_24obj/s1_h3_s4_24obj.py",
            "checkpoint": "work_dirs/h3_distill_plan_a_24obj/best_coco_bbox_mAP_epoch_10.pth",
        }),
    }


def renewal_specs_d1() -> dict[str, tuple[str, dict]]:
    base = evaluator.MODEL_SETS["d1"]["karyoflow"]
    output = {}
    for state in (True, False):
        spec = dict(base)
        spec["label"] = f"Dataset 1 renewal {'on' if state else 'off'}"
        spec["overrides"] = {"model": {"bbox_head": {"box_renewal": state}}}
        output[f"d1_renewal_{'on' if state else 'off'}"] = ("d1", spec)
    return output


def selected_specs(groups: list[str]) -> dict[str, tuple[str, dict]]:
    output = {}
    if "sota" in groups:
        for dataset in ("d1", "d2"):
            for name, spec in evaluator.MODEL_SETS[dataset].items():
                output[f"{dataset}_sota_{name}"] = (dataset, spec)
    custom = custom_specs()
    mapping = {
        "solver": ("rf_heun", "euler_1", "euler_4", "dpmpp_1", "dpmpp_4"),
        "topk": ("topk_k500_on", "topk_k500_off", "topk_k200_on",
                 "topk_k100_on", "topk_k100_off"),
        "lqcr": tuple(key for key in custom if key.startswith("lqcr_beta_")),
        "distill": ("head_student_h3",),
    }
    for group in groups:
        for key in mapping.get(group, ()):
            output[f"d2_{key}"] = custom[key]
    if "renewal_d1" in groups:
        output.update(renewal_specs_d1())
    return output


def run(groups: list[str], gpu: int, plan: bool) -> list[dict]:
    records = []
    for paper_id, (dataset, spec) in selected_specs(groups).items():
        evaluator.configure_dataset(dataset)
        evaluator.RESULT_ROOT = ROOT / "results/paper_test_multiseed" / paper_id
        evaluator.FAMILY = "paper_inference_seed_test"
        evaluator.ARTIFACT_PREFIX = f"paper-test-{paper_id}"
        evaluator.PROTOCOL_VERSION = "paper-test-three-inference-seed-v1"
        profile = evaluator.load_annotation_profile()
        code_sha = evaluator.code_fingerprint()
        for seed in SEEDS:
            evaluator.SEED = seed
            summary = evaluator.run_model(paper_id, spec, profile, code_sha, gpu, plan)
            if not plan:
                summary_path = (evaluator.RESULT_ROOT / paper_id /
                                summary["protocol_hash"] / "summary.json")
                extra_metrics = {}
                if "lqcr_beta_" in paper_id:
                    prediction_path = summary_path.parent / "predictions.bbox.json"
                    extra_metrics = ablations.high_iou_ap(
                        prediction_path, evaluator.ANN_PATH)
                records.append({
                    "paper_id": paper_id,
                    "dataset": dataset.upper(),
                    "seed": seed,
                    "summary": summary,
                    "extra_metrics": extra_metrics,
                    "source_summary": str(summary_path.relative_to(ROOT)),
                    "source_summary_sha256": sha256(summary_path),
                })
    return records


def finalize(records: list[dict]) -> Path:
    grouped = {}
    for row in records:
        grouped.setdefault(row["paper_id"], []).append(row)
    aggregates = []
    for paper_id, rows in sorted(grouped.items()):
        rows.sort(key=lambda row: row["seed"])
        metrics = {}
        metric_names = list(rows[0]["summary"]["metrics_exact"])
        metric_names.extend(rows[0].get("extra_metrics", {}))
        for metric in metric_names:
            values = [row["summary"]["metrics_exact"].get(
                metric, row.get("extra_metrics", {}).get(metric)) for row in rows]
            mean = sum(values) / len(values)
            sample_std = (sum((x - mean) ** 2 for x in values) /
                          (len(values) - 1)) ** 0.5
            metrics[metric] = {"values": values, "mean": mean,
                               "sample_std": sample_std}
        aggregates.append({
            "paper_id": paper_id,
            "dataset": rows[0]["dataset"],
            "checkpoint": rows[0]["summary"]["protocol"]["checkpoint_path"],
            "checkpoint_sha256": rows[0]["summary"]["protocol"]["checkpoint_sha256"],
            "seeds": list(SEEDS),
            "replication_unit": "inference_seed_fixed_checkpoint",
            "metrics": metrics,
            "runs": [{"seed": row["seed"],
                      "source_summary": row["source_summary"],
                      "source_summary_sha256": row["source_summary_sha256"]}
                     for row in rows],
        })
    evidence = {
        "schema_version": 1,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "description": "Paper-facing held-out test inference stability with a fixed checkpoint and seeds 42/123/789.",
        "warning": "Inference-seed replication does not estimate training-seed uncertainty.",
        "results": aggregates,
    }
    payload = (json.dumps(evidence, indent=2, sort_keys=True) + "\n").encode()
    digest = hashlib.sha256(payload).hexdigest()
    relative = Path("tools/experiment_db/evidence_sources") / f"paper_test_multiseed_{digest[:12]}.json"
    path = ROOT / relative
    path.write_bytes(payload)
    artifact_id = f"paper-test-multiseed-{digest[:12]}"
    conn = sqlite3.connect(DB)
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("""INSERT OR REPLACE INTO evidence_artifact
        (artifact_id,server,path,sha256,kind,generated_at,status,notes)
        VALUES (?,'ross',?,?,?,?,'verified',?)""",
        (artifact_id, str(relative), digest, "fixed_checkpoint_three_inference_seed_test",
         evidence["generated_at"], evidence["warning"]))
    for row in aggregates:
        for metric, summary in row["metrics"].items():
            result_id = f"paper-3iseed-{row['paper_id']}-{metric.lower()}"
            protocol = {
                "checkpoint": row["checkpoint"],
                "checkpoint_sha256": row["checkpoint_sha256"],
                "seeds": row["seeds"],
                "values": summary["values"],
                "sample_std": summary["sample_std"],
                "replication_unit": row["replication_unit"],
                "split": "test",
            }
            conn.execute("""INSERT OR REPLACE INTO controlled_result
                (result_id,family,variant,dataset,split,seed,metric,value,unit,
                 baseline_result_id,delta,protocol_json,artifact_id,evidence_level,
                 paper_eligible,notes) VALUES (?,?,?,?,'test',?,?,?,'absolute',
                 NULL,NULL,?,?,'controlled',1,?)""",
                (result_id, "paper_three_inference_seed_test", row["paper_id"],
                 row["dataset"], "inference_mean(42,123,789)", metric,
                 summary["mean"], json.dumps(protocol, sort_keys=True), artifact_id,
                 "Fixed-checkpoint inference-seed mean; not training-seed uncertainty."))
    conn.commit()
    conn.close()
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", nargs="+",
                        choices=("sota", "solver", "topk", "lqcr", "distill",
                                 "renewal_d1"),
                        default=("sota", "solver", "topk", "lqcr", "distill"))
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    records = run(args.groups, args.gpu_id, args.plan)
    if not args.plan:
        print(f"aggregate: {finalize(records).relative_to(ROOT)}")


if __name__ == "__main__":
    main()
