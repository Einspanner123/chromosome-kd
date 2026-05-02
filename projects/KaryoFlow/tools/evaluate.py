"""
KaryoFlow 评估工具

计算排列准确率、配对准确率、分组准确率等指标。
支持与 baseline (random, greedy, hungarian) 对比。
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import torch
from scipy.stats import kendalltau

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from karyoflow.constants import (
    CLASS_TO_SLOTS,
    DENVER_GROUPS,
    HOMOLOG_PAIRS,
    NUM_SLOTS,
    SLOT_ORDER,
    SLOT_TO_GROUP,
)


def position_accuracy(pred_perm, gt_perm):
    """位置准确率: 每个位置独立判断"""
    if isinstance(pred_perm, list):
        pred_perm = torch.tensor(pred_perm)
        gt_perm = torch.tensor(gt_perm)
    return (pred_perm == gt_perm).float().mean().item()


def permutation_accuracy(pred_perm, gt_perm):
    """排列准确率: 整个排列完全正确"""
    if isinstance(pred_perm, list):
        pred_perm = torch.tensor(pred_perm)
        gt_perm = torch.tensor(gt_perm)
    return (pred_perm == gt_perm).all().float().item()


def pair_accuracy(pred_perm, gt_perm):
    """配对准确率: 同源对是否正确配对 (不考虑对内顺序)

    对每对同源染色体 (slot_a, slot_b):
    如果 {pred[slot_a], pred[slot_b]} == {gt[slot_a], gt[slot_b]}，则正确
    """
    if isinstance(pred_perm, list):
        pred_perm = torch.tensor(pred_perm)
        gt_perm = torch.tensor(gt_perm)

    correct = 0
    for slot_a, slot_b in HOMOLOG_PAIRS:
        pred_set = {pred_perm[slot_a].item(), pred_perm[slot_b].item()}
        gt_set = {gt_perm[slot_a].item(), gt_perm[slot_b].item()}
        if pred_set == gt_set:
            correct += 1

    return correct / max(len(HOMOLOG_PAIRS), 1)


def group_accuracy(pred_perm, gt_perm):
    """Denver 分组准确率: 每个组内的染色体是否全部正确

    Returns: dict of group_name → accuracy
    """
    if isinstance(pred_perm, list):
        pred_perm = torch.tensor(pred_perm)
        gt_perm = torch.tensor(gt_perm)

    group_results = {}
    for group_name, classes in DENVER_GROUPS.items():
        slots = []
        for cls_name in classes:
            slots.extend(CLASS_TO_SLOTS[cls_name])

        # 组内所有 slot 是否正确
        pred_group = set(pred_perm[s].item() for s in slots)
        gt_group = set(gt_perm[s].item() for s in slots)
        group_results[group_name] = 1.0 if pred_group == gt_group else 0.0

    return group_results


def kendall_tau_distance(pred_perm, gt_perm):
    """Kendall Tau 距离 (归一化到 [0, 1])"""
    if isinstance(pred_perm, torch.Tensor):
        pred_perm = pred_perm.tolist()
        gt_perm = gt_perm.tolist()
    tau, _ = kendalltau(pred_perm, gt_perm)
    return (1 - tau) / 2  # 归一化到 [0, 1]


def evaluate_all(predictions, ground_truths):
    """批量评估所有指标

    Args:
        predictions: list of (N,) permutations
        ground_truths: list of (N,) permutations

    Returns:
        dict of metrics
    """
    pos_accs = []
    perm_accs = []
    pair_accs = []
    group_accs = defaultdict(list)
    kt_dists = []

    for pred, gt in zip(predictions, ground_truths):
        pos_accs.append(position_accuracy(pred, gt))
        perm_accs.append(permutation_accuracy(pred, gt))
        pair_accs.append(pair_accuracy(pred, gt))

        ga = group_accuracy(pred, gt)
        for g, a in ga.items():
            group_accs[g].append(a)

        kt_dists.append(kendall_tau_distance(pred, gt))

    metrics = {
        "position_accuracy": sum(pos_accs) / len(pos_accs) * 100,
        "permutation_accuracy": sum(perm_accs) / len(perm_accs) * 100,
        "pair_accuracy": sum(pair_accs) / len(pair_accs) * 100,
        "kendall_tau_distance": sum(kt_dists) / len(kt_dists),
        "num_samples": len(predictions),
    }

    for g, accs in group_accs.items():
        metrics[f"group_{g}_accuracy"] = sum(accs) / len(accs) * 100

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-json", type=str, required=True)
    parser.add_argument("--gt-json", type=str, required=True)
    args = parser.parse_args()

    with open(args.pred_json) as f:
        predictions = json.load(f)
    with open(args.gt_json) as f:
        ground_truths = json.load(f)

    metrics = evaluate_all(predictions, ground_truths)
    print(json.dumps(metrics, indent=2))
