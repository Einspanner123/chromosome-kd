"""Register paper evidence that predates the curated manifest.

The script is idempotent and intentionally source-backed: every inserted row
references a Git-sized JSON artifact with a verified SHA-256 identity.
"""

import hashlib
import json
import os
import sqlite3


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DB = os.path.join(HERE, "experiments.db")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def artifact(conn, artifact_id, path, kind, notes):
    absolute = os.path.join(ROOT, path)
    if not os.path.isfile(absolute):
        raise FileNotFoundError(path)
    conn.execute(
        """INSERT OR REPLACE INTO evidence_artifact
        (artifact_id,server,path,sha256,kind,generated_at,status,notes)
        VALUES (?,?,?,?,?,'2026-08-12','verified',?)""",
        (artifact_id, "ross", path, sha256(absolute), kind, notes),
    )


def result(conn, *, result_id, family, variant, dataset, split, seed, metric,
           value, artifact_id, protocol, unit="absolute", baseline=None,
           delta=None, level="controlled", eligible=True, notes=None):
    conn.execute(
        """INSERT OR REPLACE INTO controlled_result
        (result_id,family,variant,dataset,split,seed,metric,value,unit,
         baseline_result_id,delta,protocol_json,artifact_id,evidence_level,
         paper_eligible,notes) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (result_id, family, variant, dataset, split, seed, metric, value, unit,
         baseline, delta, json.dumps(protocol, sort_keys=True), artifact_id,
         level, int(eligible), notes),
    )


def main():
    conn = sqlite3.connect(DB)
    conn.execute("PRAGMA foreign_keys=ON")

    artifact(
        conn, "dataset-splits-20260812",
        "tools/experiment_db/evidence_sources/dataset_splits_20260812.json",
        "dataset_split_audit",
        "Exact train/val/test image, annotation and class counts with annotation hashes.",
    )
    split_source = json.load(open(os.path.join(
        ROOT, "tools/experiment_db/evidence_sources/dataset_splits_20260812.json")))
    for row in split_source["datasets"]:
        common = dict(
            family="dataset_inventory", variant="annotation_file",
            dataset=row["dataset"], split=row["split"], seed="none",
            artifact_id="dataset-splits-20260812",
            protocol={"annotation": row["annotation"],
                      "annotation_sha256": row["annotation_sha256"]},
            unit="count",
        )
        prefix = f"dataset-{row['dataset'].lower()}-{row['split']}"
        result(conn, result_id=f"{prefix}-images", metric="images",
               value=row["images"], **common)
        result(conn, result_id=f"{prefix}-annotations", metric="annotations",
               value=row["annotations"], **common)
        result(conn, result_id=f"{prefix}-classes", metric="classes",
               value=row["classes"], **common)

    artifact(
        conn, "d1-rf-ddpm-test-20260730",
        "tools/experiment_db/evidence_sources/d1_rf_ddpm_test_20260730.json",
        "multi_seed_test_evaluation",
        "D1 held-out test evaluation of RF+Heun and DDPM, three checkpoint seeds.",
    )
    d1 = json.load(open(os.path.join(
        ROOT, "tools/experiment_db/evidence_sources/d1_rf_ddpm_test_20260730.json")))
    wanted = {
        "D1 KaryoFlow RF+Heun seed42": ("RF_Heun", "42"),
        "D1 KaryoFlow RF+Heun seed123": ("RF_Heun", "123"),
        "D1 KaryoFlow RF+Heun seed789": ("RF_Heun", "789"),
        "D1 DiffusionDet DDPM seed42": ("DDPM", "42"),
        "D1 DiffusionDet DDPM seed123": ("DDPM", "123"),
        "D1 DiffusionDet DDPM seed789": ("DDPM", "789"),
    }
    values = {"RF_Heun": [], "DDPM": []}
    for row in d1["results"]:
        if row["label"] not in wanted:
            continue
        variant, seed = wanted[row["label"]]
        values[variant].append(row["mAP"])
        result(
            conn, result_id=f"d1-rf-ddpm-{variant.lower()}-seed{seed}-map",
            family="rf_ddpm_d1_test", variant=variant, dataset="D1",
            split="test", seed=seed, metric="mAP", value=row["mAP"],
            artifact_id="d1-rf-ddpm-test-20260730",
            protocol={"images": row["n_images"], "config": row["config"],
                      "checkpoint": row["checkpoint"]},
        )
    for variant, vals in values.items():
        mean = sum(vals) / len(vals)
        std = (sum((x - mean) ** 2 for x in vals) / (len(vals) - 1)) ** 0.5
        result(
            conn, result_id=f"d1-rf-ddpm-{variant.lower()}-mean-map",
            family="rf_ddpm_d1_test", variant=variant, dataset="D1",
            split="test", seed="mean(42,123,789)", metric="mAP", value=mean,
            artifact_id="d1-rf-ddpm-test-20260730",
            protocol={"images": 220, "values": vals, "sample_std": std},
        )

    artifact(
        conn, "annotation-perturbation-d2-test",
        "tools/experiment_db/evidence_sources/annotation_perturbation_d2_test.json",
        "annotation_perturbation_test",
        "D2 test annotations perturbed with bbox-center Gaussian noise and label flips; predictions fixed.",
    )
    perturb = json.load(open(os.path.join(
        ROOT, "tools/experiment_db/evidence_sources/annotation_perturbation_d2_test.json")))
    table = perturb["experiment_1_noise_robustness"]["table"]
    for key, metrics in table.items():
        sigma = int(key.split("sigma=")[1].split("_")[0])
        flip = float(key.split("flip=")[1])
        variant = f"sigma{sigma}px_flip{flip:g}"
        for metric, value in metrics.items():
            result(
                conn,
                result_id=f"annotation-d2-{variant}-{metric.lower()}",
                family="annotation_perturbation", variant=variant,
                dataset="D2", split="test", seed="42", metric=metric,
                value=value, artifact_id="annotation-perturbation-d2-test",
                protocol={"images": 1000, "bbox_center_noise_sigma_px": sigma,
                          "label_flip_fraction": flip,
                          "images_and_predictions_fixed": True,
                          "checkpoint": perturb["experiment_1_noise_robustness"]["checkpoint"]},
                level="diagnostic", eligible=True,
                notes="Annotation sensitivity diagnostic, not detector robustness training.",
            )

    # Preserve the superseded +0.082 comparison without allowing it into paper claims.
    result(
        conn, result_id="legacy-d2-rf-vs-a0-map-delta",
        family="legacy_protocol", variant="RF_Heun_vs_A0_SGD12",
        dataset="D2", split="val", seed="42", metric="mAP_delta",
        value=0.082, artifact_id="d1-rf-ddpm-test-20260730",
        protocol={"baseline": "A0 SGD/12-epoch, mAP 0.774",
                  "comparison": "RF+Heun, mAP 0.856",
                  "superseded_by": "unified AdamW/150-epoch held-out test comparison"},
        unit="absolute_delta", level="diagnostic", eligible=False,
        notes="Historical mixed-training-protocol comparison; prohibited in current paper claims.",
    )
    conn.commit()
    print("registered dataset inventory, D1 RF/DDPM test, and annotation perturbation evidence")


if __name__ == "__main__":
    main()
