#!/usr/bin/env python3
"""Small dependency-free regression tests for the evidence gate."""

from copy import deepcopy
from validate_run_evidence import validate
from pathlib import Path

SHA = "a" * 64


def record():
    artifact = {"path": "placeholder", "sha256": SHA}
    return {
        "schema_version": "1.0", "record_id": "d2__model__seed-42",
        "record_type": "evaluation", "dataset_id": "D2", "split": "test",
        "annotation": artifact, "num_images": 1000, "training_seed": 42,
        "inference_seed": 42, "replication_unit": "independent_training_seed",
        "config": artifact, "checkpoint": artifact, "source_log": artifact,
        "code": {"git_commit": "abcdef1", "dirty": False},
        "evaluation_protocol": {
            "metric_definition": "COCO bbox mAP@[.50:.95]", "max_dets": [1, 10, 100],
            "eval_code_version": "abcdef1", "candidate_count": 500,
            "solver": "DPM-Solver++", "steps": 4, "nfe": 4,
        },
        "selection_source": "validation:best_mAP", "status": "verified",
        "metrics": {name: 0.5 for name in ("mAP", "AP50", "AP75", "AP_S", "AP_M", "AP_L")},
    }


def main():
    base = record()
    assert not validate(base, Path.cwd(), False)
    bad = deepcopy(base); bad["split"] = "val"
    assert any("split=test" in error for error in validate(bad, Path.cwd(), False))
    bad = deepcopy(base); del bad["metrics"]["AP_L"]
    assert any("AP_L" in error for error in validate(bad, Path.cwd(), False))
    bad = deepcopy(base); bad["replication_unit"] = "paired_final_stage_intervention"
    assert any("parent_train_run_id" in error for error in validate(bad, Path.cwd(), False))
    bad = deepcopy(base); bad["test_tuned"] = True
    assert any("test-tuned" in error for error in validate(bad, Path.cwd(), False))
    print("PASS evidence gate regression tests")


if __name__ == "__main__":
    main()
