"""Extract per-class AP data from scalars.json for the best epoch (step 132, mAP=0.863).

Reads the JSONL file, finds the evaluation line with step=132, and outputs a
markdown table of per-class AP / AP50 / AP75 / AP_s / AP_m / AP_l.
"""

import json
import math
from pathlib import Path

SCALARS_JSON = Path(
    "/home/linkst/workspace/projects/chromosome-kd/work_dirs/a4_dpm_pp_24obj/"
    "20260708_000414/vis_data/scalars.json"
)
OUTPUT_MD = Path(
    "/home/linkst/workspace/projects/chromosome-kd/results/a4_per_class_ap.md"
)

TARGET_STEP = 132
EXPECTED_MAP = 0.863

CLASSES = [
    "A1", "A2", "A3", "B4", "B5",
    "C6", "C7", "C8", "C9", "C10", "C11", "C12",
    "D13", "D14", "D15",
    "E16", "E17", "E18",
    "F19", "F20",
    "G21", "G22",
    "X", "Y",
]

METRIC_SUFFIXES = ["precision", "mAP_50", "mAP_75", "mAP_s", "mAP_m", "mAP_l"]


def fmt(value) -> str:
    """Format a metric value, rendering NaN as 'NaN'."""
    if isinstance(value, float) and math.isnan(value):
        return "NaN"
    return f"{value:.3f}"


def find_target_record(path: Path, target_step: int) -> dict:
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("step") == target_step and "coco/bbox_mAP" in record:
                return record
    raise ValueError(f"No evaluation record with step={target_step} found.")


def build_markdown(record: dict) -> str:
    overall_map = record["coco/bbox_mAP"]
    lines = []
    lines.append("# Per-Class AP at Best Epoch (step 132, mAP={:.3f})".format(overall_map))
    lines.append("")
    lines.append(f"- **Step**: {record['step']}")
    lines.append(f"- **Overall mAP**: {overall_map:.3f}")
    lines.append(f"- **Overall mAP_50**: {record['coco/bbox_mAP_50']:.3f}")
    lines.append(f"- **Overall mAP_75**: {record['coco/bbox_mAP_75']:.3f}")
    lines.append(f"- **Overall mAP_s**: {fmt(record['coco/bbox_mAP_s'])}")
    lines.append(f"- **Overall mAP_m**: {fmt(record['coco/bbox_mAP_m'])}")
    lines.append(f"- **Overall mAP_l**: {fmt(record['coco/bbox_mAP_l'])}")
    lines.append("")
    header = "| Class | AP | AP50 | AP75 | AP_s | AP_m | AP_l |"
    sep = "|-------|-----|------|------|------|------|------|"
    lines.append(header)
    lines.append(sep)
    for cls in CLASSES:
        ap = fmt(record[f"coco/{cls}_precision"])
        ap50 = fmt(record[f"coco/{cls}_mAP_50"])
        ap75 = fmt(record[f"coco/{cls}_mAP_75"])
        ap_s = fmt(record[f"coco/{cls}_mAP_s"])
        ap_m = fmt(record[f"coco/{cls}_mAP_m"])
        ap_l = fmt(record[f"coco/{cls}_mAP_l"])
        lines.append(
            f"| {cls} | {ap} | {ap50} | {ap75} | {ap_s} | {ap_m} | {ap_l} |"
        )
    lines.append("")
    return "\n".join(lines)


def main():
    record = find_target_record(SCALARS_JSON, TARGET_STEP)
    actual_map = record["coco/bbox_mAP"]
    if abs(actual_map - EXPECTED_MAP) > 1e-6:
        raise ValueError(
            f"Unexpected mAP at step {TARGET_STEP}: got {actual_map}, "
            f"expected {EXPECTED_MAP}"
        )
    markdown = build_markdown(record)
    print(markdown)
    try:
        OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_MD.write_text(markdown)
        print(f"\n[Saved to {OUTPUT_MD}]")
    except PermissionError as e:
        print(f"\n[WARN] Could not write markdown file: {e}")


if __name__ == "__main__":
    main()
