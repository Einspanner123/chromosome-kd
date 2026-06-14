"""下游任务评估: LDMDet mAP对比

评估生成数据对LDMDet检测性能的提升效果。
对比: 原始数据训练 vs 原始+生成数据训练 的mAP差异。
"""

import os
import subprocess
import sys
from typing import Dict


def evaluate_ldmdet(
    config_path: str,
    work_dir: str,
    checkpoint_path: str,
    data_root: str,
    ann_file: str,
    gpu_id: int = 0,
) -> Dict[str, float]:
    """使用LDMDet在指定数据集上评估mAP

    Args:
        config_path: LDMDet配置文件路径
        work_dir: 工作目录
        checkpoint_path: LDMDet checkpoint路径
        data_root: 评估数据集根目录
        ann_file: 评估标注文件路径
        gpu_id: GPU ID
    Returns:
        dict with mAP metrics
    """
    cmd = [
        sys.executable,
        'projects/LDMDet/tools/test.py',
        config_path,
        checkpoint_path,
        '--work-dir',
        work_dir,
    ]

    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            timeout=3600,
        )

        # 解析mAP结果
        metrics = _parse_mAP_from_output(result.stdout)
        return metrics

    except subprocess.TimeoutExpired:
        return {'error': 'Evaluation timed out'}
    except Exception as e:
        return {'error': str(e)}


def compare_augmentation_effect(
    baseline_config: str,
    baseline_ckpt: str,
    augmented_config: str,
    augmented_ckpt: str,
    val_data_root: str,
    val_ann_file: str,
    gpu_id: int = 0,
) -> Dict[str, float]:
    """对比数据增强效果

    分别在原始和增强数据上训练的模型，在相同验证集上评估mAP。

    Args:
        baseline_config: 基线配置 (原始数据训练)
        baseline_ckpt: 基线checkpoint
        augmented_config: 增强配置 (原始+生成数据训练)
        augmented_ckpt: 增强checkpoint
        val_data_root: 验证集根目录
        val_ann_file: 验证标注文件
        gpu_id: GPU ID
    Returns:
        dict with baseline_mAP, augmented_mAP, improvement
    """
    # 评估基线
    baseline_metrics = evaluate_ldmdet(
        config_path=baseline_config,
        work_dir='work_dirs/eval_baseline',
        checkpoint_path=baseline_ckpt,
        data_root=val_data_root,
        ann_file=val_ann_file,
        gpu_id=gpu_id,
    )

    # 评估增强
    augmented_metrics = evaluate_ldmdet(
        config_path=augmented_config,
        work_dir='work_dirs/eval_augmented',
        checkpoint_path=augmented_ckpt,
        data_root=val_data_root,
        ann_file=val_ann_file,
        gpu_id=gpu_id,
    )

    baseline_map = baseline_metrics.get('coco/bbox_mAP', 0.0)
    augmented_map = augmented_metrics.get('coco/bbox_mAP', 0.0)

    return {
        'baseline_mAP': baseline_map,
        'augmented_mAP': augmented_map,
        'improvement': augmented_map - baseline_map,
        'improvement_pct': (
            (augmented_map - baseline_map) / (baseline_map + 1e-8)
        )
        * 100,
        'baseline_detail': baseline_metrics,
        'augmented_detail': augmented_metrics,
    }


def _parse_mAP_from_output(output: str) -> Dict[str, float]:
    """从MMDetection测试输出中解析mAP指标"""
    metrics = {}

    for line in output.split('\n'):
        line = line.strip()
        # 匹配 "coco/bbox_mAP: 0.4560" 格式
        if 'bbox_mAP' in line:
            parts = line.split(':')
            if len(parts) == 2:
                key = parts[0].strip()
                try:
                    value = float(parts[1].strip())
                    metrics[key] = value
                except ValueError:
                    pass

    return metrics
