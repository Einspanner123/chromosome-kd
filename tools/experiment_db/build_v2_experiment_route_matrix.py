#!/usr/bin/env python3
"""Build the authoritative v2 paper experiment route matrix.

The legacy ledger supplies only the stable experiment scope and descriptive
fields.  All executable configuration identities, evidence links, and live
statuses are replaced here with v2-reviewed values.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
from collections import Counter
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
LEGACY = ROOT / "tools/experiment_db/manifests/master_experiment_ledger_v1.json"
OUTPUT = ROOT / "experiments/configs/v2/manifests/paper_experiment_route_matrix.yaml"
CSV_OUTPUT = ROOT / "tools/experiment_db/exports/paper_experiment_route_matrix.csv"
DOC_OUTPUT = ROOT / "docs/experiments/PAPER_EXPERIMENT_ROUTE_MATRIX.md"
DB = ROOT / "tools/experiment_db/experiments.db"

D1_MATRIX = "experiments/configs/v2/matrices/d1_inhouse1700.yaml"
D2_MATRIX = "experiments/configs/v2/matrices/d2_taichung.yaml"
D2_HISTORY_MATRIX = "experiments/configs/v2/matrices/d2_taichung_history.yaml"
D1_GENERATION_MATRIX = "experiments/configs/v2/matrices/d1_inhouse1700_generation_ablation.yaml"
D2_GENERATION_MATRIX = "experiments/configs/v2/matrices/d2_taichung_generation_ablation.yaml"
D2_SOTA_COMPLETION_MATRIX = "experiments/configs/v2/matrices/d2_taichung_sota_completion.yaml"

METHODS = {
    "diffusiondet": "experiments/configs/v2/methods/diffusiondet_ddpm.py",
    "rf_heun": "experiments/configs/v2/methods/rf_heun.py",
    "karyoflow": "experiments/configs/v2/methods/karyoflow.py",
    "karyoflow_lqcr": "experiments/configs/v2/methods/karyoflow_lqcr.py",
    "dino_r50": "experiments/configs/v2/methods/dino_r50.py",
    "rtmdet_l": "experiments/configs/v2/methods/rtmdet_l.py",
    "cascade_rcnn_r50": "experiments/configs/v2/methods/cascade_rcnn_r50.py",
    "yolox_s": "experiments/configs/v2/methods/yolox_s.py",
    "strict_g0": "experiments/configs/v2/methods/strict_g0_ddpm_linear_scaleshift.py",
    "strict_g1": "experiments/configs/v2/methods/strict_g1_rf_linear_scaleshift.py",
    "strict_g2": "experiments/configs/v2/methods/strict_g2_rf_shifted_scaleshift.py",
}

D2_LEGACY = {
    "diffusiondet": (
        "experiments/configs/v2/methods/diffusiondet_ddpm_legacy.py",
        "experiments/configs/v2/matrices/d2_diffusiondet_history.yaml",
        "tools/experiment_db/compatibility_reports/d2_sota_diffusiondet.json",
        "STRUCTURAL_ONLY",
    ),
    "dino_r50": (
        "experiments/configs/v2/methods/dino_r50_legacy.py",
        "experiments/configs/v2/matrices/d2_dino_rtmdet_history.yaml",
        "tools/experiment_db/compatibility_reports/d2_sota_dino_r50.json",
        "EXACT",
    ),
    "rtmdet_l": (
        "experiments/configs/v2/methods/rtmdet_l_legacy.py",
        "experiments/configs/v2/matrices/d2_dino_rtmdet_history.yaml",
        "tools/experiment_db/compatibility_reports/d2_sota_rtmdet_l.json",
        "EXACT",
    ),
    "cascade_rcnn_r50": (
        "experiments/configs/v2/methods/cascade_rcnn_r50_legacy.py",
        "experiments/configs/v2/matrices/d2_cascade_history.yaml",
        "tools/experiment_db/compatibility_reports/d2_sota_cascade_rcnn.json",
        "EXACT",
    ),
    "yolox_s": (
        "experiments/configs/v2/methods/yolox_s_legacy.py",
        "experiments/configs/v2/matrices/d2_yolox_history.yaml",
        "tools/experiment_db/compatibility_reports/d2_sota_yolox_s.json",
        "EXACT",
    ),
}

STATUS_OVERRIDE = {
    "D1I.SOTA.karyoflow": "PLANNED",
    "D1I.SOTA.diffusiondet": "PLANNED",
    "D1I.SOTA.dino_r50": "PLANNED",
    "D1I.SOTA.rtmdet_l": "PLANNED",
    "D1I.SOTA.cascade_rcnn_r50": "PLANNED",
    "D1I.SOTA.yolox_s": "PLANNED",
    "D1I.SOTA.karyoflow_lqcr": "BLOCKED_PARENT",
    "D1I.ABL.G0": "PLANNED",
    "D1I.ABL.G1": "PLANNED",
    "D1I.ABL.G2": "PLANNED",
    "D1I.ABL.G3": "PLANNED",
    "D1I.INF.solver_steps": "BLOCKED_PARENT",
    "D1I.INF.topk_renewal": "BLOCKED_PARENT",
    "D1I.DEC.beta_val": "BLOCKED_PARENT",
    "D1I.DEC.strict_subsets": "BLOCKED_PREDICTIONS",
    "D1I.DEP.distill_h3": "BLOCKED_IMPLEMENTATION",
    "D1I.DEP.GACS": "BLOCKED_PARENT",
    "D1I.DEP.speed": "BLOCKED_CHECKPOINTS",
    "D2.SOTA.diffusiondet.missing_train2": "PLANNED",
    "D2.SOTA.dino_r50.missing_train2": "PLANNED",
    "D2.SOTA.rtmdet_l.missing_train2": "PLANNED",
    "D2.SOTA.cascade_rcnn_r50.missing_train2": "PLANNED",
    "D2.SOTA.yolox_s.missing_train2": "PLANNED",
    "D2.ABL.strict.G0": "PLANNED",
    "D2.ABL.strict.G1": "PLANNED",
    "D2.ABL.strict.G2": "PLANNED",
    "D2.INF.solver_steps.train3": "PLANNED",
    "D2.INF.topk_renewal.train3": "PLANNED",
    "D2.DEP.distill_h3.existing": "COMPLETED_EVIDENCE_ONLY",
    "D2.DEP.distill_h3.train3": "BLOCKED_IMPLEMENTATION",
}

PROTOCOLS = {
    "D1I.INF.solver_steps": "experiments/configs/v2/ablations/d1_solver_test.yaml",
    "D1I.INF.topk_renewal": "experiments/configs/v2/ablations/d1_topk_renewal_test.yaml",
    "D1I.DEC.beta_val": "experiments/configs/v2/ablations/d1_lqcr_beta_val.yaml",
    "D1I.DEC.strict_subsets": "tools/experiment_db/protocols/d2_difficulty_strata.json",
    "D1I.DEP.speed": "experiments/configs/v2/deployment/a6000_speed_protocol.yaml",
    "D1I.DEP.GACS": "experiments/configs/v2/deployment/gacs_dynamic_policy.yaml",
    "D2.INF.solver_steps.fixed1": "experiments/configs/v2/ablations/d2_solver_test.yaml",
    "D2.INF.solver_steps.train3": "experiments/configs/v2/ablations/d2_solver_train3_test.yaml",
    "D2.INF.topk_renewal.fixed1": "experiments/configs/v2/ablations/d2_topk_renewal_test.yaml",
    "D2.INF.topk_renewal.train3": "experiments/configs/v2/ablations/d2_topk_renewal_train3_test.yaml",
    "D2.DEC.beta_test.fixed1": "experiments/configs/v2/ablations/d2_lqcr_beta_test.yaml",
    "D2.DEC.strict_subsets": "tools/experiment_db/protocols/d2_difficulty_strata.json",
    "D2.DEP.speed": "experiments/configs/v2/deployment/a6000_speed_protocol.yaml",
    "D2.DEP.GACS": "experiments/configs/v2/deployment/gacs_dynamic_policy.yaml",
}

CONFIG_STATE_OVERRIDE = {
    "D1I.SOTA.dino_r50": "READY",
    "D1I.SOTA.rtmdet_l": "READY",
    "D1I.SOTA.cascade_rcnn_r50": "READY",
    "D1I.SOTA.yolox_s": "READY",
    "D1I.ABL.G0": "READY",
    "D1I.ABL.G1": "READY",
    "D1I.ABL.G2": "READY",
    "D1I.DEP.distill_h3": "IMPLEMENTATION_MISSING",
    "D1I.DEP.GACS": "PROTOCOL_READY_PARENT_PENDING",
    "D1I.INF.solver_steps": "PROTOCOL_READY_PARENT_PENDING",
    "D1I.INF.topk_renewal": "PROTOCOL_READY_PARENT_PENDING",
    "D1I.DEC.beta_val": "PROTOCOL_READY_PARENT_PENDING",
    "D1I.DEC.strict_subsets": "DATASET_ADAPTATION_REQUIRED",
    "D2.ABL.strict.G0": "READY",
    "D2.ABL.strict.G1": "READY",
    "D2.ABL.strict.G2": "READY",
    "D2.DEP.distill_h3.train3": "IMPLEMENTATION_MISSING",
    "D2.DEP.GACS": "PROTOCOL_READY",
    "D2.INF.solver_steps.train3": "PROTOCOL_READY",
    "D2.INF.topk_renewal.train3": "PROTOCOL_READY",
}

NOTES_OVERRIDE = {
    "D1I.SOTA.karyoflow": "No active v2 run is registered; previous self1700 runs are invalid or superseded.",
    "D1I.ABL.G3": "Reuses the planned D1 KaryoFlow training triplet after strict config audit.",
    "D2.ABL.historical_chain": "Historical foundation comparison only; never interpret adjacent rows as isolated cumulative effects.",
    "D2.DEP.distill_h3.existing": "Inference identity is EXACT; the archived historical distillation training implementation is not executable in cleaned ldmdet.",
    "D2.DEP.speed": "Historical latency is valid under its recorded protocol but PARTIAL against the strict rerun protocol.",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def config_for(row: dict) -> tuple[str, str, str, str]:
    """Return method, matrix, protocol, configuration state."""
    rid, dataset, variant = row["ledger_id"], row["dataset_id"], row["variant"]
    method = ""
    matrix = D1_MATRIX if dataset == "D1_INHOUSE1700_V2" else D2_MATRIX if dataset == "D2" else ""
    protocol = PROTOCOLS.get(rid, "")
    state = CONFIG_STATE_OVERRIDE.get(rid, "READY")

    if rid in {"D1I.SOTA.karyoflow", "D1I.ABL.G3"}:
        method = METHODS["karyoflow"]
    elif rid == "D1I.SOTA.diffusiondet":
        method = METHODS["diffusiondet"]
    elif rid == "D1I.SOTA.karyoflow_lqcr":
        method = METHODS["karyoflow_lqcr"]
    elif rid.startswith("D1I.SOTA.") and variant in METHODS:
        method = METHODS[variant]
    elif rid in {"D1I.ABL.G0", "D1I.ABL.G1", "D1I.ABL.G2"}:
        stage = rid.rsplit('.', 1)[-1].lower()
        method = METHODS[f"strict_{stage}"]
        matrix = D1_GENERATION_MATRIX
    elif rid.startswith("D1I.INF.") or rid == "D1I.DEC.beta_val":
        method = METHODS["karyoflow_lqcr" if rid == "D1I.DEC.beta_val" else "karyoflow"]
    elif rid == "D1I.DEP.GACS":
        method = METHODS["karyoflow"]
    elif rid.startswith("D2.SOTA.karyoflow_lqcr"):
        method = "experiments/configs/v2/methods/karyoflow_ot_lqcr_legacy.py"
        matrix = D2_HISTORY_MATRIX
        state = "EXACT"
    elif rid in {"D2.SOTA.karyoflow_train3", "D2.ABL.strict.G3"}:
        method = "experiments/configs/v2/methods/karyoflow_ot_legacy.py"
        matrix = D2_HISTORY_MATRIX
        state = "EXACT"
    elif rid.startswith("D2.SOTA.") and variant in D2_LEGACY:
        if rid.endswith(".existing"):
            method, history_matrix, _, compatibility = D2_LEGACY[variant]
            matrix = history_matrix
            state = compatibility
        else:
            method = METHODS[variant]
            matrix = D2_SOTA_COMPLETION_MATRIX
            state = "READY"
    elif rid in {"D2.ABL.strict.G0", "D2.ABL.strict.G1", "D2.ABL.strict.G2"}:
        stage = rid.rsplit('.', 1)[-1].lower()
        method = METHODS[f"strict_{stage}"]
        matrix = D2_GENERATION_MATRIX
        state = "READY"
    elif rid == "D2.ABL.historical_chain":
        protocol = "experiments/configs/v2/manifests/d2_paper_experiment_inventory.yaml"
        state = "MIGRATED_MIXED_HISTORY"
    elif rid == "D2.DEP.distill_h3.existing":
        method = "experiments/configs/v2/methods/karyoflow_ot_h3_student_legacy.py"
        matrix = "experiments/configs/v2/matrices/d2_h3_student_history.yaml"
        protocol = "experiments/configs/v2/deployment/h3_distillation_training_archive.yaml"
        state = "EXACT_INFERENCE_ARCHIVED_TRAINING"
    elif rid == "D2.DEP.distill_h3.train3":
        method = ""
        protocol = "experiments/configs/v2/deployment/h3_distillation_training_archive.yaml"
    elif rid == "D2.DEP.GACS":
        method = "experiments/configs/v2/methods/karyoflow_ot_legacy.py"
        matrix = D2_HISTORY_MATRIX
        state = "PROTOCOL_READY"
    elif rid.startswith("D2.INF.") or rid.startswith("D2.DEC.") or rid == "D2.DEP.speed":
        method = "experiments/configs/v2/methods/karyoflow_ot_legacy.py"
        matrix = D2_HISTORY_MATRIX
        state = "PROTOCOL_READY" if rid.endswith("fixed1") or rid in {
            "D2.DEC.beta_test.fixed1", "D2.DEC.strict_subsets", "D2.DEP.speed"
        } else state
    elif dataset == "D1_COMPOSITE2200_LEGACY":
        protocol = "tools/experiment_db/experiments.db"
        state = "ARCHIVED"

    return method, matrix, protocol, state


def build_rows() -> list[dict]:
    legacy_rows = json.loads(LEGACY.read_text())["experiments"]
    rows = []
    for old in legacy_rows:
        row = dict(old)
        rid = row["ledger_id"]
        if row["dataset_id"] == "D1_INHOUSE1700_V1":
            row["dataset_id"] = "D1_INHOUSE1700_V2"
        method, matrix, protocol, config_state = config_for(row)
        row["legacy_config_path"] = row.pop("config_path")
        row["legacy_config_status"] = row.pop("config_status")
        row["method_config"] = method
        row["matrix_config"] = matrix
        row["protocol_config"] = protocol
        row["config_state"] = config_state
        row["status"] = STATUS_OVERRIDE.get(rid, row["status"])
        row["parameters"] = row.get("parameters", "")
        row["server_plan"] = row.pop("executor_plan")
        row["output_template"] = row.pop("work_dir_template")
        if row["dataset_id"] == "D1_INHOUSE1700_V2":
            row["output_template"] = row["output_template"].replace(
                "work_dirs/self1700/", "work_dirs/v2/d1_inhouse1700/"
            ).replace("results/self1700/", "results/v2/d1_inhouse1700/")
        elif rid.endswith(".missing_train2"):
            row["output_template"] = f"work_dirs/v2/d2_taichung/sota_completion/{row['variant']}/trainseed_{{training_seed}}"
        elif rid.startswith("D2.ABL.strict.G") and rid != "D2.ABL.strict.G3":
            row["output_template"] = f"work_dirs/v2/d2_taichung/strict_generation/{rid.rsplit('.', 1)[-1].lower()}/trainseed_{{training_seed}}"
        elif rid == "D2.INF.solver_steps.train3":
            row["output_template"] = "results/v2/d2_taichung/inference/solver/{parent}/{solver}_{steps}"
        elif rid == "D2.INF.topk_renewal.train3":
            row["output_template"] = "results/v2/d2_taichung/inference/topk_renewal/{parent}/{variant}"
        elif rid == "D2.DEP.distill_h3.train3":
            row["output_template"] = "work_dirs/v2/d2_taichung/deployment/h3/{parent}"
        elif rid == "D2.DEP.GACS":
            row["output_template"] = "results/v2/d2_taichung/deployment/gacs/{parent}"
        row["database_ids"] = {
            "train_run_ids": [] if row["dataset_id"] == "D1_INHOUSE1700_V2" else [
                x for x in (row.pop("train_run_ids", "") or "").split(";") if x
            ],
            "result_family": row.pop("result_family", ""),
            "evidence_artifact_ids": [
                x for x in (row.pop("evidence_artifact_ids", "") or "").split(";") if x
            ],
        }
        if rid == "LEGACY.D1.composite":
            row["database_ids"]["result_family"] = "all D1 families except D1_INHOUSE1700_V2"
        if rid.startswith("D2.SOTA.") and row["variant"] in D2_LEGACY:
            row["compatibility_report"] = D2_LEGACY[row["variant"]][2]
        elif rid in {"D2.SOTA.karyoflow_train3", "D2.SOTA.karyoflow_lqcr_train3", "D2.ABL.strict.G3"}:
            row["compatibility_report"] = "tools/experiment_db/compatibility_reports/d2_mainline_history_v1_index.json"
        elif rid == "D2.DEP.distill_h3.existing":
            row["compatibility_report"] = "tools/experiment_db/compatibility_reports/d2_head_student_h3.json"
        else:
            row["compatibility_report"] = ""
        if row["dataset_id"] == "D1_INHOUSE1700_V2":
            row["swanlab_project"] = "KaryoFlow-Self1700-V2" if row["execution_kind"] in {"full_train", "short_train"} else ""
            row["notes"] = NOTES_OVERRIDE.get(rid, "No active v2 run is currently registered.")
        else:
            row["notes"] = NOTES_OVERRIDE.get(rid, row.get("notes", ""))
            if row["dataset_id"] == "D2" and row["execution_kind"] in {"full_train", "short_train"} and row["status"].startswith(("PLANNED", "BLOCKED")):
                row["swanlab_project"] = "KaryoFlow-Dataset2-V2"
                row["swanlab_run_template"] = "{variant}_seed{training_seed}"
        rows.append(row)
    return rows


def flatten(row: dict) -> dict:
    return {
        "route_id": row["ledger_id"],
        "dataset_id": row["dataset_id"],
        "layer": row["layer"],
        "family": row["family"],
        "variant": row["variant"],
        "execution_kind": row["execution_kind"],
        "run_count": row["run_count"],
        "training_seeds": row["training_seeds"],
        "inference_seeds": row["inference_seeds"],
        "status": row["status"],
        "priority": row["priority"],
        "replication_unit": row["replication_unit"],
        "config_state": row["config_state"],
        "method_config": row["method_config"],
        "matrix_config": row["matrix_config"],
        "protocol_config": row["protocol_config"],
        "parameters": row["parameters"],
        "server_plan": row["server_plan"],
        "output_template": row["output_template"],
        "train_run_ids": ";".join(row["database_ids"]["train_run_ids"]),
        "result_family": row["database_ids"]["result_family"],
        "evidence_artifact_ids": ";".join(row["database_ids"]["evidence_artifact_ids"]),
        "compatibility_report": row["compatibility_report"],
        "swanlab_project": row["swanlab_project"],
        "swanlab_run_template": row["swanlab_run_template"],
        "parent_route_id": row["parent_ledger_id"],
        "paper_role": row["paper_role"],
        "acceptance_gate": row["acceptance_gate"],
        "notes": row["notes"],
    }


def write_doc(payload: dict, manifest_sha: str, artifact_id: str) -> None:
    rows = payload["experiments"]
    statuses = Counter(r["status"] for r in rows)
    lines = [
        "# KaryoFlow v2 完整实验路线矩阵",
        "",
        "> 这是双数据集实验调度、配置解析、数据库登记和论文数字校对的唯一权威入口。",
        "> YAML 是源文件；本 Markdown、CSV 与 SQLite 表均由生成器派生，禁止手工修改派生文件。",
        "",
        "## 权威身份",
        "",
        f"- YAML：`{OUTPUT.relative_to(ROOT)}`",
        f"- YAML SHA-256：`{manifest_sha}`",
        f"- 数据库 artifact：`{artifact_id}`",
        f"- 实验组：{len(rows)}；展开运行：{sum(int(r['run_count']) for r in rows)}。",
        f"- 状态分布：{', '.join(f'{k}={v}' for k, v in sorted(statuses.items()))}。",
        "",
        "## 数据与统计口径",
        "",
        "- `D1_INHOUSE1700_V2`：1190/170/340，纯自建、D2 类别 ID 对齐、group-disjoint 70/10/20 划分。",
        "- `D2`：3500/500/1000，台中公开数据集 70/10/20 划分。",
        "- 训练复现以不同训练 checkpoint 为统计单位；固定 checkpoint 的推理 seed 不得冒充训练 seed。",
        "- 主精度只允许 held-out test；validation 只用于 checkpoint/超参数选择。",
        "- 效率只允许按 A6000 严格协议比较；服务器名称只存在内部路线矩阵，不进入论文。",
        "",
        "## 路线矩阵",
    ]
    for dataset in ("D1_INHOUSE1700_V2", "D2", "D1_COMPOSITE2200_LEGACY"):
        lines += ["", f"### {dataset}", ""]
        for layer in ("Detector comparison", "Generation", "Decision", "Deployment", "Archive"):
            selected = [r for r in rows if r["dataset_id"] == dataset and r["layer"] == layer]
            if not selected:
                continue
            lines += [f"#### {layer}", "", "| ID | Variant | Kind / runs | Seeds | Status | Config state | v2 method / protocol | Server | Output | DB/evidence |", "|---|---|---:|---|---|---|---|---|---|---|"]
            for row in selected:
                ref = row["method_config"] or row["protocol_config"] or "—"
                evidence = row["database_ids"]["result_family"] or ";".join(row["database_ids"]["evidence_artifact_ids"])
                db = f"route:{row['ledger_id']}" + (f"; {evidence}" if evidence else "")
                seeds = f"train={row['training_seeds']}; infer={row['inference_seeds']}"
                lines.append(
                    f"| `{row['ledger_id']}` | {row['variant']} | {row['execution_kind']} / {row['run_count']} | {seeds} | **{row['status']}** | {row['config_state']} | `{ref}` | {row['server_plan']} | `{row['output_template']}` | `{db}` |"
                )
            lines.append("")
            for row in selected:
                lines.append(
                    f"- `{row['ledger_id']}`：参数={row['parameters']}；验收={row['acceptance_gate']}；论文用途={row['paper_role']}；备注={row['notes']}"
                )
    lines += [
        "",
        "## 当前执行结论",
        "",
        "- D1_INHOUSE1700_V2 当前没有有效 active v2 训练；旧 V1 SwanLab/registry 运行均不得继续显示为 RUNNING。",
        "- D2 已有论文证据已迁移；传统基线仍是固定 checkpoint 推理复现，而非三训练种子。",
        "- 严格 G0→G1→G2→G3 三训练种子消融尚未完成，当前历史链只能作描述性比较。",
        "- H3 推理身份可精确复现，但历史蒸馏训练实现仍需恢复；GACS 保持可选部署扩展。",
        "",
    ]
    DOC_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    DOC_OUTPUT.write_text("\n".join(lines), encoding="utf-8")


def register_db(rows: list[dict], manifest_sha: str, artifact_id: str) -> None:
    con = sqlite3.connect(DB)
    with con:
        con.execute(
            """CREATE TABLE IF NOT EXISTS experiment_route_matrix_v2 (
               route_id TEXT PRIMARY KEY, dataset_id TEXT NOT NULL, layer TEXT NOT NULL,
               family TEXT NOT NULL, variant TEXT NOT NULL, execution_kind TEXT NOT NULL,
               run_count INTEGER NOT NULL, training_seeds TEXT, inference_seeds TEXT,
               status TEXT NOT NULL, priority TEXT NOT NULL, replication_unit TEXT NOT NULL,
               config_state TEXT NOT NULL, method_config TEXT, matrix_config TEXT,
               protocol_config TEXT, server_plan TEXT NOT NULL, output_template TEXT NOT NULL,
               database_refs_json TEXT NOT NULL, swanlab_project TEXT,
               parent_route_id TEXT, paper_role TEXT NOT NULL, acceptance_gate TEXT NOT NULL,
               source_manifest_sha256 TEXT NOT NULL, record_json TEXT NOT NULL)
            """
        )
        con.execute("DELETE FROM experiment_route_matrix_v2")
        con.executemany(
            """INSERT INTO experiment_route_matrix_v2 VALUES
               (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            [
                (
                    r["ledger_id"], r["dataset_id"], r["layer"], r["family"], r["variant"],
                    r["execution_kind"], int(r["run_count"]), r["training_seeds"], r["inference_seeds"],
                    r["status"], r["priority"], r["replication_unit"], r["config_state"],
                    r["method_config"], r["matrix_config"], r["protocol_config"], r["server_plan"],
                    r["output_template"], json.dumps(r["database_ids"], sort_keys=True),
                    r["swanlab_project"], r["parent_ledger_id"], r["paper_role"],
                    r["acceptance_gate"], manifest_sha, json.dumps(r, sort_keys=True),
                )
                for r in rows
            ],
        )
        con.execute(
            "DELETE FROM evidence_artifact WHERE artifact_id LIKE 'paper-route-matrix-v2-%'"
        )
        con.execute(
            """INSERT OR REPLACE INTO evidence_artifact
               (artifact_id,server,path,sha256,kind,generated_at,status,notes)
               VALUES(?,?,?,?,?,NULL,'verified',?)""",
            (
                artifact_id, "ross", str(OUTPUT.relative_to(ROOT)), manifest_sha,
                "paper_experiment_route_matrix_v2",
                f"{len(rows)} groups; {sum(int(r['run_count']) for r in rows)} expanded runs",
            ),
        )


def main() -> None:
    rows = build_rows()
    payload = {
        "schema_version": 2,
        "matrix_id": "karyoflow.paper_experiment_route.v2",
        "authoritative": True,
        "supersedes": [
            "tools/experiment_db/manifests/master_experiment_ledger_v1.json",
            "tools/experiment_db/exports/master_experiment_ledger.csv",
            "docs/experiments/MASTER_EXPERIMENT_LEDGER.md",
        ],
        "dataset_scope": {
            "D1_INHOUSE1700_V2": {
                "split_images": [1190, 170, 340], "role": "primary in-house cohort",
                "manifest_artifact_id": "dataset-self1700-48d90fed63ec",
                "manifest_sha256": "48d90fed63ecc107b374a316effc1e5ab0d63b7c4bd9110d33ddd16e1f43146c",
                "annotation_sha256": {
                    "train": "e62704aa3ce9f7b58170946cca78458ff5331fc4316ba946eace6a1dbc6c8ed1",
                    "val": "9065c7b5fb4df2a9f9c794dfe83c655ddf6695388d2bde562171b208c6d3a5b9",
                    "test": "883696b8e60cc901cfe92b3f009d8c60e7b8cefbb5ba9ce3c742c3720343f08f",
                },
            },
            "D2": {
                "split_images": [3500, 500, 1000], "role": "public Taichung cohort",
                "annotation_sha256": {
                    "train": "218ae0c96b41f342975ac52f14dfba19f74763f4583129e17c178c4a0fbff026",
                    "val": "bcf0f99908e46f838b2924b801458594f2ed923b1dc1166e851395313ec53a92",
                    "test": "110fd2804f435b04a1eee969cb28666a0818b2886040a57cbae2dd05f2767495",
                },
            },
            "D1_COMPOSITE2200_LEGACY": {"role": "archive only; mixed-source and non-comparable"},
        },
        "status_policy": {
            "COMPLETED_*": "evidence exists; scope is qualified by the suffix",
            "PLANNED": "v2 configuration is ready but no active valid run is registered",
            "BLOCKED_*": "a named prerequisite is missing",
            "PARTIAL_LEGACY": "legacy evidence exists but does not meet the current protocol",
            "ARCHIVED_NONCOMPARABLE": "retained only for provenance",
        },
        "experiments": rows,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False, width=120), encoding="utf-8")
    manifest_sha = sha256(OUTPUT)
    artifact_id = f"paper-route-matrix-v2-{manifest_sha[:12]}"

    flat = [flatten(r) for r in rows]
    CSV_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with CSV_OUTPUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(flat)
    write_doc(payload, manifest_sha, artifact_id)
    register_db(rows, manifest_sha, artifact_id)
    print(json.dumps({
        "status": "PASS", "groups": len(rows),
        "expanded_runs": sum(int(r["run_count"]) for r in rows),
        "manifest_sha256": manifest_sha, "artifact_id": artifact_id,
    }, indent=2))


if __name__ == "__main__":
    main()
