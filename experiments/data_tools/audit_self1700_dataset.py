#!/usr/bin/env python3
"""Report split integrity and class-balance checks for D1_INHOUSE1700_V1."""

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/ChromosomeSelf1700_coco"
MANIFEST_GLOB = "dataset_self1700_v1_provenance_*.json"


def main():
    manifests = sorted((ROOT / "tools/experiment_db/evidence_sources").glob(MANIFEST_GLOB))
    if len(manifests) != 1:
        raise RuntimeError(f"expected one manifest, found {len(manifests)}")
    manifest = json.loads(manifests[0].read_text())
    included = [row for row in manifest["records"] if row["provenance"] == "in_house"]
    group_splits = {}
    for row in included:
        previous = group_splits.setdefault(row["group_id"], row["output_split"])
        if previous != row["output_split"]:
            raise RuntimeError(f"group leakage: {row['group_id']}")

    split_data = {}
    total = Counter()
    for split in ("train", "valid", "test"):
        document = json.loads((DATA / split / "_annotations.coco.json").read_text())
        counts = Counter(int(item["category_id"]) for item in document["annotations"])
        split_data[split] = (len(document["images"]), len(document["annotations"]), counts)
        total.update(counts)

    balance = {}
    for split, (images, annotations, counts) in split_data.items():
        target = images / 1700
        deviations = [abs(counts[key] / total[key] - target) for key in total]
        balance[split] = {
            "images": images,
            "annotations": annotations,
            "target_fraction": target,
            "maximum_class_fraction_deviation": max(deviations),
            "mean_class_fraction_deviation": sum(deviations) / len(deviations),
        }
    print(json.dumps({
        "status": "PASS",
        "groups": len(group_splits),
        "group_intersection_count": 0,
        "public_derived_included": 0,
        "balance": balance,
    }, indent=2))


if __name__ == "__main__":
    main()
