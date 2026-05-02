
import argparse
import subprocess
import sys


def run_eval(config_path, checkpoint_path, sampling_steps, work_dir_suffix):
    cmd = [
        sys.executable, "tools/test.py",
        config_path,
        checkpoint_path,
        "--work-dir", f"work_dirs/eval_multistep/{work_dir_suffix}",
        "--cfg-options",
        f"model.bbox_head.sampling_timesteps={sampling_steps}",
    ]
    print(f"\n{'='*60}")
    print(f"Running: sampling_steps={sampling_steps}, work_dir={work_dir_suffix}")
    print(f"{'='*60}")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )
    return result.stdout + result.stderr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["reflow_v5", "trd_full", "both"], default="both")
    args = parser.parse_args()

    configs = {
        "reflow_v5": {
            "config": "projects/LDMDet/configs/ldmdet_flowdet_adaln_reflow.py",
            "checkpoint": "work_dirs/ldmdet_flowdet_adaln_reflow_v5/best_coco_bbox_mAP_epoch_1.pth",
            "steps": [1, 2, 4, 8],
        },
        "trd_full": {
            "config": "projects/LDMDet/configs/ldmdet_flowdet_adaln_trd_full.py",
            "checkpoint": "work_dirs/ldmdet_flowdet_adaln_trd_full/best_coco_bbox_mAP_epoch_63.pth",
            "steps": [1, 2, 4, 8],
        },
    }

    if args.model != "both":
        configs = {k: v for k, v in configs.items() if k == args.model}

    for model_name, cfg in configs.items():
        print(f"\n\n{'#'*60}")
        print(f"# Evaluating: {model_name}")
        print(f"# Config: {cfg['config']}")
        print(f"# Checkpoint: {cfg['checkpoint']}")
        print(f"# Steps to evaluate: {cfg['steps']}")
        print(f"{'#'*60}")

        for steps in cfg["steps"]:
            suffix = f"{model_name}_steps{steps}"
            output = run_eval(
                cfg["config"],
                cfg["checkpoint"],
                steps,
                suffix,
            )
            for line in output.split("\n"):
                if "bbox_mAP" in line:
                    print(f"  [steps={steps}] {line.strip()}")


if __name__ == "__main__":
    main()
