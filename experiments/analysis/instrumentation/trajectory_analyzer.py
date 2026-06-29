"""扩散采样轨迹分析 — 挖掘扩散过程内在问题

通过采集每个采样步 (t=T→0) 的中间状态, 分析:
- box 演化: 哪一步导致定位误差? box 是否随步数收敛?
- cls 收敛: 分类置信度是否随 t 收敛? 还是震荡?
- x0 质量: 每步的 x0 预测质量如何? 早期 x0 是否可靠?
- renewal 修正: box_renewal 实际修正了多少? 是否有效?

使用方法 (推理时):
    # 在 DiffusionDetHead.predict 中采集
    collector = TrajectoryCollector()
    results, trajectory = head.predict(features, img_metas, return_trajectory=True)
    for step_idx, (cls_logits, pred_bboxes) in enumerate(trajectory):
        collector.record(step_idx, t_curr, t_next, cls_logits, pred_bboxes, x0_raw, ...)
    analyzer = TrajectoryAnalyzer(collector.trajectory)
    report = analyzer.full_report()
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


class TrajectoryCollector:
    """采集扩散采样每步的中间状态.

    在 DiffusionDetHead.predict 的采样循环中调用 record() 采集每步状态.
    """

    def __init__(self):
        self.trajectory: list[dict] = []

    def record(
        self,
        step_idx: int,
        t_curr: float,
        t_next: float,
        cls_logits: torch.Tensor,        # (bs, n_proposals, n_classes)
        pred_bboxes: torch.Tensor,       # (bs, n_proposals, 4) xyxy
        x0_raw: torch.Tensor,            # (bs, n_proposals, 4) x0 预测
        x_raw_before_step: torch.Tensor,  # (bs, n_proposals, 4) 采样步前
        x_raw_after_step: torch.Tensor,   # (bs, n_proposals, 4) 采样步后
        x_raw_after_renewal: Optional[torch.Tensor] = None,  # box_renewal 后 (可选)
    ):
        """记录一个采样步的中间状态 (detach 避免占用计算图)"""
        entry = {
            'step_idx': step_idx,
            't_curr': float(t_curr),
            't_next': float(t_next),
            'cls_logits': cls_logits.detach().cpu(),
            'pred_bboxes': pred_bboxes.detach().cpu(),
            'x0_raw': x0_raw.detach().cpu(),
            'x_raw_before_step': x_raw_before_step.detach().cpu(),
            'x_raw_after_step': x_raw_after_step.detach().cpu(),
        }
        if x_raw_after_renewal is not None:
            entry['x_raw_after_renewal'] = x_raw_after_renewal.detach().cpu()
        self.trajectory.append(entry)


class TrajectoryAnalyzer:
    """分析扩散采样轨迹.

    Args:
        trajectory: TrajectoryCollector.trajectory 列表
    """

    def __init__(self, trajectory: list[dict]):
        self.trajectory = trajectory
        self.n_steps = len(trajectory)

    def _box_iou(self, boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """计算两组框的 IoU (逐对应).

        Args:
            boxes1, boxes2: (..., 4) xyxy

        Returns:
            (...) IoU
        """
        # 通用广播
        area1 = (boxes1[..., 2] - boxes1[..., 0]).clamp(min=0) * \
                (boxes1[..., 3] - boxes1[..., 1]).clamp(min=0)
        area2 = (boxes2[..., 2] - boxes2[..., 0]).clamp(min=0) * \
                (boxes2[..., 3] - boxes2[..., 1]).clamp(min=0)
        lt = torch.max(boxes1[..., :2], boxes2[..., :2])
        rb = torch.min(boxes1[..., 2:], boxes2[..., 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[..., 0] * wh[..., 1]
        union = area1 + area2 - inter
        return inter / union.clamp(min=1e-8)

    def analyze_box_evolution(self) -> dict:
        """分析 box 随采样步的演化.

        Returns:
            per_step_iou_to_final: 每步 pred_bboxes 与最终 pred_bboxes 的平均 IoU
            per_step_l1_to_final: 每步 pred_bboxes 与最终的 L1 距离
            convergence_step: 收敛步 (IoU 不再显著变化的最早步)
            box_variance_per_step: 每步 box 的方差 (反映稳定性)
        """
        if self.n_steps < 2:
            return {'per_step_iou_to_final': [], 'per_step_l1_to_final': [],
                    'convergence_step': 0, 'box_variance_per_step': []}

        final_bboxes = self.trajectory[-1]['pred_bboxes']
        ious, l1s, variances = [], [], []
        for entry in self.trajectory:
            cur = entry['pred_bboxes']
            iou = self._box_iou(cur, final_bboxes).mean().item()
            l1 = (cur - final_bboxes).abs().mean().item()
            var = cur.var(dim=-2).mean().item()  # proposals 间方差
            ious.append(iou)
            l1s.append(l1)
            variances.append(var)

        # 收敛步: IoU 变化 < 0.01 的最早步
        convergence_step = self.n_steps - 1
        for i in range(1, self.n_steps):
            if abs(ious[i] - ious[i - 1]) < 0.01:
                convergence_step = i
                break

        return {
            'per_step_iou_to_final': ious,
            'per_step_l1_to_final': l1s,
            'convergence_step': convergence_step,
            'box_variance_per_step': variances,
        }

    def analyze_cls_convergence(self) -> dict:
        """分析 cls_logits 随采样步的收敛性.

        Returns:
            per_step_entropy: 每步平均熵 (越低越确定)
            per_step_top1_top2_gap: 每步 top1-top2 logit 间距 (越大越确定)
            is_converged: cls 是否随步数收敛 (后期熵稳定)
            convergence_step: 收敛步
        """
        entropies, gaps = [], []
        for entry in self.trajectory:
            logits = entry['cls_logits']  # (bs, n, C)
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * (probs.clamp(min=1e-8)).log()).sum(dim=-1)  # (bs, n)
            entropies.append(entropy.mean().item())
            # top1-top2 gap
            top2, _ = probs.topk(2, dim=-1)
            gap = (top2[..., 0] - top2[..., 1]).mean().item()
            gaps.append(gap)

        # 收敛: 后 1/3 步的熵变化 < 0.05
        is_converged = False
        convergence_step = self.n_steps - 1
        if self.n_steps >= 3:
            late_start = max(1, self.n_steps // 2)
            late_changes = [abs(entropies[i] - entropies[i - 1])
                            for i in range(late_start, self.n_steps)]
            is_converged = all(c < 0.05 for c in late_changes) if late_changes else True
            if is_converged:
                convergence_step = late_start

        return {
            'per_step_entropy': entropies,
            'per_step_top1_top2_gap': gaps,
            'is_converged': is_converged,
            'convergence_step': convergence_step,
        }

    def analyze_x0_quality(self) -> dict:
        """分析 x0 预测质量.

        Returns:
            per_step_x0_to_final_iou: 每步 x0_raw 与最终 pred_bboxes 的 IoU
            x0_stability: x0 步间稳定性 (相邻步 x0 差异的均值)
            early_x0_quality: 早期 (前 1/3) x0 平均 IoU
        """
        if self.n_steps < 2:
            return {'per_step_x0_to_final_iou': [], 'x0_stability': 0.0,
                    'early_x0_quality': 0.0}

        final_bboxes = self.trajectory[-1]['pred_bboxes']
        ious = []
        for entry in self.trajectory:
            iou = self._box_iou(entry['x0_raw'], final_bboxes).mean().item()
            ious.append(iou)

        # 稳定性: 相邻步 x0 差异
        stabilities = []
        for i in range(1, self.n_steps):
            diff = (self.trajectory[i]['x0_raw'] -
                    self.trajectory[i - 1]['x0_raw']).abs().mean().item()
            stabilities.append(diff)
        x0_stability = sum(stabilities) / len(stabilities) if stabilities else 0.0

        # 早期 x0 质量
        early_end = max(1, self.n_steps // 3)
        early_x0_quality = sum(ious[:early_end]) / early_end

        return {
            'per_step_x0_to_final_iou': ious,
            'x0_stability': x0_stability,
            'early_x0_quality': early_x0_quality,
        }

    def analyze_renewal(self) -> dict:
        """分析 box_renewal 的修正幅度.

        Returns:
            per_step_renewal_delta: 每步 renewal 修正的 L1 幅度
            renewal_effective: renewal 是否有效 (平均修正量 > 0.001)
            renewal_direction: 修正方向 (正值=扩大, 负值=缩小)
        """
        deltas = []
        for entry in self.trajectory:
            if 'x_raw_after_renewal' not in entry:
                deltas.append(0.0)
                continue
            before = entry['x_raw_after_step']
            after = entry['x_raw_after_renewal']
            delta = (after - before).abs().mean().item()
            deltas.append(delta)

        avg_delta = sum(deltas) / len(deltas) if deltas else 0.0
        renewal_effective = avg_delta > 0.001

        # 修正方向 (wh 维度的平均变化)
        directions = []
        for entry in self.trajectory:
            if 'x_raw_after_renewal' not in entry:
                continue
            before_wh = entry['x_raw_after_step'][..., 2:] - entry['x_raw_after_step'][..., :2]
            after_wh = entry['x_raw_after_renewal'][..., 2:] - entry['x_raw_after_renewal'][..., :2]
            directions.append((after_wh.mean() - before_wh.mean()).item())
        renewal_direction = sum(directions) / len(directions) if directions else 0.0

        return {
            'per_step_renewal_delta': deltas,
            'renewal_effective': renewal_effective,
            'renewal_direction': renewal_direction,
        }

    def full_report(self) -> dict:
        """生成完整轨迹分析报告."""
        box_evo = self.analyze_box_evolution()
        cls_conv = self.analyze_cls_convergence()
        x0_qual = self.analyze_x0_quality()
        renewal = self.analyze_renewal()

        # 文字分析结论
        analyses = []
        if self.n_steps >= 2:
            # box 演化
            if box_evo['convergence_step'] < self.n_steps - 1:
                analyses.append(
                    f"box 在第 {box_evo['convergence_step']} 步已收敛 "
                    f"(IoU 变化 <0.01), 后续步可能浪费"
                )
            else:
                analyses.append(
                    f"box 在 {self.n_steps} 步内未完全收敛, "
                    f"最终 IoU 变化={abs(box_evo['per_step_iou_to_final'][-1] - box_evo['per_step_iou_to_final'][-2]):.4f}"
                )
            # cls 收敛
            if not cls_conv['is_converged']:
                analyses.append(
                    f"cls_logits 未收敛 (后期熵仍变化), "
                    f"可能存在分类不确定性"
                )
            # x0 质量
            if x0_qual['early_x0_quality'] < 0.5:
                analyses.append(
                    f"早期 x0 质量差 (IoU={x0_qual['early_x0_quality']:.3f}), "
                    f"扩散起点预测不可靠"
                )
            # renewal
            if renewal['renewal_effective']:
                analyses.append(
                    f"box_renewal 有效 (平均修正={sum(renewal['per_step_renewal_delta'])/len(renewal['per_step_renewal_delta']):.4f})"
                )
            else:
                analyses.append("box_renewal 修正量过小, 可能未起作用")

        return {
            'n_steps': self.n_steps,
            'box_evolution': box_evo,
            'cls_convergence': cls_conv,
            'x0_quality': x0_qual,
            'renewal': renewal,
            'analysis': ' | '.join(analyses) if analyses else '步数不足, 无法分析',
        }
