import argparse
import subprocess
import sys


PYTHON = "/home/linkst/miniconda3/envs/chromo/bin/python"
CWD = "/home/linkst/workplace/chromo/chromosome-kd"


def run_eval(config_path, checkpoint_path, sampling_steps, work_dir_suffix):
    cmd = [
        PYTHON, "tools/test.py",
        config_path,
        checkpoint_path,
        "--work-dir", f"work_dirs/eval_multistep/{work_dir_suffix}",
        "--cfg-options",
        f"model.bbox_head.sampling_timesteps={sampling_steps}",
    ]
    print(f"\n{'='*60}")
    print(f"Running: {work_dir_suffix}, sampling_steps={sampling_steps}")
    print(f"{'='*60}")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=CWD,
    )
    return result.stdout + result.stderr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["adaln", "baseline", "both"], default="both")
    args = parser.parse_args()

    configs = {
        "adaln": {
            "config": "projects/LDMDet/configs/ldmdet_flowdet_adaln.py",
            "checkpoint": "work_dirs/ldmdet_flowdet_adaln/best_coco_bbox_mAP_epoch_56.pth",
            "steps": [1, 2, 4, 8],
        },
        "baseline": {
            "config": "projects/LDMDet/configs/ldmdet_baseline.py",
            "checkpoint": "work_dirs/ldmdet_baseline/best_coco_bbox_mAP_epoch_68.pth",
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
