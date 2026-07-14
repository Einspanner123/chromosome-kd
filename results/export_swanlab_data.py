"""SwanLab 数据导出脚本 (v4)

用途: 从 SwanLab 导出关键实验的 mAP 数据
运行: conda run -n chromo python results/export_swanlab_data.py
"""

import json
import sys

try:
    import swanlab
except ImportError:
    print("ERROR: swanlab not installed.")
    sys.exit(1)

API_KEY = "Huzvq1fnDeqOwgQo2AMAI"
USERNAME = "einspanner"

# 关键 metric keys
METRIC_KEYS = [
    "coco/bbox_mAP",
    "coco/bbox_mAP_50",
    "coco/bbox_mAP_75",
    "coco/bbox_mAP_s",
    "coco/bbox_mAP_m",
    "coco/bbox_mAP_l",
]

# 需要导出的实验
TARGET_EXPERIMENTS = {
    "ldmdet-mainline-ablation-24obj": {
        "a0_baseline": "A0 baseline",
        "a1_rf_heun": "A1 RF+Heun",
        "a2_rf_heun_adaln": "A2 AdaLN",
        "a3_full_sota": "A3 StochOT",
        "a4_dpm_pp": "A4 DPM-Solver++",
    },
    "ldmdet-ablation": {
        "chromo_24obj_random_seed42": "24obj Random seed 42",
        "chromo_24obj_random_seed123": "24obj Random seed 123",
        "chromo_24obj_random_seed789": "24obj Random seed 789",
    },
    "chromosome-kd-benchmark-24obj": {
        "benchmark_diffusiondet": "DiffusionDet",
        "dino": "DINO",
        "cascade": "Cascade R-CNN",
        "yolox": "YOLOX-S",
    },
}


def main():
    print("Logging in to SwanLab...")
    swanlab.login(api_key=API_KEY)
    api = swanlab.Api()

    results = {}

    for project_name, target_exps in TARGET_EXPERIMENTS.items():
        path = f"{USERNAME}/{project_name}"
        print(f"\n{'='*60}")
        print(f"Project: {path}")
        print(f"{'='*60}")

        try:
            runs = list(api.runs(path))
            print(f"  Total runs: {len(runs)}")

            for target_key, desc in target_exps.items():
                matched = [r for r in runs if target_key.lower() in r.name.lower()]

                if not matched:
                    print(f"\n  [NOT FOUND] {target_key} ({desc})")
                    continue

                for run in matched:
                    print(f"\n  [FOUND] {run.name} ({desc})")
                    exp_data = {"name": run.name, "project": project_name, "description": desc}

                    try:
                        # 用 keys 参数获取 mAP 数据
                        df = run.metrics(keys=METRIC_KEYS)
                        print(f"    DataFrame shape: {df.shape}")
                        print(f"    Columns: {df.columns.tolist()}")

                        if df.shape[0] > 0:
                            # 获取每个 metric 的最大值
                            for col in df.columns:
                                if col == 'index' or col == '_step':
                                    continue
                                vals = df[col].dropna()
                                if len(vals) > 0:
                                    max_val = vals.max()
                                    exp_data[col] = {"max": float(max_val), "count": len(vals)}
                                    print(f"    {col}: max={max_val:.4f} ({len(vals)} points)")
                        else:
                            print(f"    DataFrame is empty")
                            # 尝试 run.json() 获取 summary
                            rj = run.json()
                            if isinstance(rj, dict):
                                summary = rj.get('summary', {})
                                for k, v in summary.items():
                                    if 'mAP' in k:
                                        exp_data[k] = v
                                        print(f"    summary {k}: {v}")

                    except Exception as e:
                        print(f"    ERROR: {e}")

                    results[f"{project_name}/{run.name}"] = exp_data

        except Exception as e:
            print(f"  ERROR: {e}")

    # 保存
    output_path = "results/swanlab_export.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n{'='*60}")
    print(f"Saved to: {output_path}")
    print(f"Total: {len(results)}")


if __name__ == "__main__":
    main()
