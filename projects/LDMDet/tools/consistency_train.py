"""
Consistency Distillation 训练脚本 — LDMDet

将多步 FlowDet teacher 蒸馏为单步 student。

用法:
  python projects/LDMDet/tools/consistency_train.py \
    --teacher-config projects/LDMDet/configs/ldmdet_flowdet_adaln_trd_full.py \
    --teacher-checkpoint work_dirs/ldmdet_flowdet_adaln_trd_full/best_coco_bbox_mAP_epoch_63.pth \
    --student-config projects/LDMDet/configs/ldmdet_flowdet_adaln_consistency.py \
    --work-dir work_dirs/ldmdet_flowdet_adaln_consistency
"""

import argparse
import os
import sys
from copy import deepcopy

import torch
from mmengine.config import Config
from mmengine.runner import Runner

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))


def parse_args():
    parser = argparse.ArgumentParser(description="LDMDet Consistency Distillation")
    parser.add_argument("--teacher-config", required=True)
    parser.add_argument("--teacher-checkpoint", required=True)
    parser.add_argument("--student-config", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--ema-rate", type=float, default=0.999)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    student_cfg = Config.fromfile(args.student_config)
    student_cfg.work_dir = args.work_dir

    student_cfg.model.bbox_head.use_consistency = True
    student_cfg.model.bbox_head.consistency_ema_rate = args.ema_rate

    if args.resume:
        student_cfg.resume = True
    else:
        student_cfg.load_from = args.teacher_checkpoint

    runner = Runner.from_cfg(student_cfg)
    runner.train()


if __name__ == "__main__":
    main()
