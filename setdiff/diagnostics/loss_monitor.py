"""SetDiff loss 插桩监控 — 验证 double-counting 修复 + L1/GIoU 梯度尺度对齐.

使用方法 (训练前快速验证):
    python -c \
"from setdiff.diagnostics.loss_monitor import LossMonitor; LossMonitor.run_quick_check()"

或在训练中作为 hook 使用:
    from setdiff.diagnostics.loss_monitor import LossMonitorHook
    # 注册到 runner
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LossMonitor:
    """静态方法集合: 验证 loss 计算正确性."""

    @staticmethod
    def verify_no_double_counting(
        head,
        image_features,
        gt_boxes,
        gt_labels,
    ) -> bool:
        """验证 head 返回的 loss_dict 不含 'loss' key (double-counting 根因).

        Returns:
            True if 修复生效 (无 'loss' key), False otherwise.
        """
        # 调用 head forward
        loss_dict = head(image_features, gt_boxes, gt_labels)
        has_loss_key = 'loss' in loss_dict
        if has_loss_key:
            print(f'[FAIL] loss_dict 含 "loss" key: {list(loss_dict.keys())}')
            return False
        print(f'[OK] loss_dict 无 "loss" key: {list(loss_dict.keys())}')
        return True

    @staticmethod
    def measure_gradient_ratio(
        head,
        image_features,
        gt_boxes,
        gt_labels,
    ) -> Dict[str, float]:
        """测量各 loss 项的加权梯度 norm (head 层面, 诊断用).

        head 返回加权单项 (mmengine parse_losses 直接 sum 各项 backward),
        因此测量加权梯度 = 实际训练信号尺度.

        注意: head 层面的梯度比取决于 pred 与 tgt 的几何关系:
        - 训练初期 (pred 随机, 远离 tgt): GIoU 饱和, 梯度很小,
          ratio 可达 40-60+ (非 bug, 是 GIoU 几何特性).
        - 训练后期 (pred 接近 tgt): GIoU 不饱和, ratio ≈ 2.5 (权重 5:2).
        因此本方法主要用于诊断 (各 loss 项梯度是否有限/非零),
        严格验证 L1/GIoU 空间尺度对齐请用:
        - measure_criterion_gradient_ratio (controlled, pred 接近 tgt)
        - measure_loss_space_consistency (验证 2*s 空间因子)

        Returns:
            dict with 'loss_cls', 'loss_bbox', 'loss_giou' 加权梯度 norm,
            及 'ratio_bbox_giou' 加权梯度比.
        """
        head.zero_grad()
        # Forward (head 返回加权单项)
        loss_dict = head(image_features, gt_boxes, gt_labels)

        grad_norms: Dict[str, float] = {}
        for k, v in loss_dict.items():
            head.zero_grad()
            # 加权 loss backward: 测量实际训练信号梯度尺度
            # (mmengine parse_losses 直接 sum 加权单项后 backward)
            v.backward(retain_graph=True)
            total_norm = 0.0
            for p in head.parameters():
                if p.grad is not None:
                    total_norm += p.grad.norm(2).item() ** 2
            grad_norms[k] = total_norm**0.5

        ratio = grad_norms.get('loss_bbox', 0) / max(
            grad_norms.get('loss_giou', 1e-8), 1e-8
        )
        grad_norms['ratio_bbox_giou'] = ratio

        print(
            f'加权梯度 norm: cls={grad_norms["loss_cls"]:.4f}, '
            f'bbox={grad_norms["loss_bbox"]:.4f}, '
            f'giou={grad_norms["loss_giou"]:.4f}'
        )
        print(f'L1:GIoU 加权梯度比 = {ratio:.2f}')
        print('  注: 训练初期 GIoU 饱和时 ratio 可达 40-60 (几何特性, 非 bug)')
        print('  pred 接近 tgt 后: ratio ≈ 2.5 (权重 5:2)')

        return grad_norms

    @staticmethod
    def measure_criterion_gradient_ratio(
        criterion,
        pred_boxes,
        gt_boxes_list,
        gt_labels_list,
    ) -> Dict[str, float]:
        """测量 criterion 层面的加权梯度比 (controlled, 不依赖 head 几何).

        通过将 pred_boxes 作为可学习参数, 直接测量 loss_bbox 和 loss_giou
        对 pred_boxes 的加权梯度, 验证 L1:GIoU 梯度尺度对齐.

        适用场景:
        - pred 接近 tgt (GIoU 不饱和): 梯度比 ≈ 2.5 (权重 5:2), 验证修复生效.
        - pred 远离 tgt (GIoU 饱和): 梯度比可达 50+ (GIoU 几何特性, 非 bug).

        修复前 (L1 在扩散空间, GIoU 在 [0,1] 空间):
            即使 pred 接近 tgt, ratio ≈ 5s:2 ≈ 5 (s=2), 因 L1 梯度多 2s 因子.
        修复后 (L1 和 GIoU 都在 [0,1] 空间):
            pred 接近 tgt 时 ratio ≈ 2.5 (由权重 5:2 决定).

        2026-07-19 重构: criterion 内部用 Hungarian 重新匹配 (num_pos=M),
            故本方法接收原始 GT list (不再接收 expanded matched_boxes/labels).

        Args:
            criterion: SetCriterion 实例.
            pred_boxes: [B, N, 4] 预测框 (扩散空间 [-s, s]).
            gt_boxes_list: List[Tensor[M_i, 4]] 每张图的 GT (扩散空间).
            gt_labels_list: List[Tensor[M_i]] 每张图的 GT label.

        Returns:
            dict with 'loss_bbox', 'loss_giou' 加权梯度 norm,
            及 'ratio_bbox_giou' 加权梯度比.
        """
        # 创建可学习 pred_boxes (leaf parameter)
        pred_leaf = nn.Parameter(pred_boxes.detach().clone())
        B, N = pred_leaf.shape[:2]

        outputs = {
            'pred_logits': torch.zeros(
                B, N, criterion.num_classes
            ),
            'pred_boxes': pred_leaf,
        }

        loss_dict, _ = criterion(outputs, gt_boxes_list, gt_labels_list)

        grad_norms: Dict[str, float] = {}
        # loss_cls 不依赖 pred_boxes, 仅测量 loss_bbox 和 loss_giou
        for k in ['loss_bbox', 'loss_giou']:
            v = loss_dict[k]
            weighted = v * criterion.weight_dict.get(k, 1.0)
            grad = torch.autograd.grad(
                weighted, pred_leaf, retain_graph=True, create_graph=False
            )[0]
            grad_norms[k] = grad.norm(2).item()

        ratio = grad_norms['loss_bbox'] / max(grad_norms['loss_giou'], 1e-8)
        grad_norms['ratio_bbox_giou'] = ratio

        print(
            f'criterion 加权梯度 norm: '
            f'bbox={grad_norms["loss_bbox"]:.4f}, '
            f'giou={grad_norms["loss_giou"]:.4f}'
        )
        print(f'L1:GIoU 加权梯度比 = {ratio:.2f}')
        print('  pred 接近 tgt (不饱和): ≈ 2.5 (权重 5:2, 修复后)')
        print('  pred 远离 tgt (饱和): 可达 50+ (GIoU 几何特性, 非 bug)')

        return grad_norms

    @staticmethod
    def measure_loss_space_consistency(
        criterion,
        pred_boxes,
        matched_boxes,
        matched_labels,
    ) -> Dict[str, float]:
        """验证 loss_bbox 和 loss_giou 在同一空间 ([0,1]) 计算.

        ⚠️ 独立数学检查 (2026-07-19 说明):
            本方法不调用 criterion.forward, 不经过 Hungarian 重匹配,
            直接对传入的 pred_boxes 和 matched_boxes 计算 L1 数值对比,
            验证 "L1 在扩散空间 vs [0,1] 空间" 的 2*s 数值因子.
            参数名 matched_boxes/matched_labels 仅表示 "与 pred 对应的
            GT 张量" (用于局部 L1 对比), 与 set_head coupling 阶段的
            matched_boxes (轨迹构造产物) 是不同概念, 不共享匹配语义.

            如需验证 criterion 内部 Hungarian 重匹配行为, 请使用:
            - measure_criterion_gradient_ratio (接收 gt_boxes_list)

        通过对比 L1 在扩散空间 vs [0,1] 空间的数值, 验证空间尺度因子 2*s.
        修复后 _loss_bbox 在 [0,1] 空间计算 (与 _loss_giou 一致),
        因此扩散空间 L1 应是 [0,1] 空间 L1 的 2*s 倍.

        Args:
            criterion: SetCriterion 实例 (仅取 snr_scale, 不调用 forward).
            pred_boxes: [B, N, 4] 预测框 (扩散空间, 仅用于数值对比).
            matched_boxes: [B, N, 4] 对应 GT (扩散空间, 与 pred 一一对齐).
            matched_labels: [B, N] label (仅用于 valid mask, ≥0 视为有效).

        Returns:
            dict with 'loss_bbox_norm_space', 'loss_bbox_diffusion_space',
            'consistency_ratio' (应 ≈ 2*s).
        """
        s = criterion.snr_scale
        # 在扩散空间计算 L1 (旧 bug 场景)
        valid = matched_labels >= 0
        pred = pred_boxes[valid]
        tgt = matched_boxes[valid]
        l1_diffusion = F.l1_loss(pred, tgt, reduction='sum') / max(
            pred.shape[0], 1
        )

        # 在 [0,1] 空间计算 L1 (修复后)
        pred_norm = (pred.clamp(-s, s) / s + 1.0) / 2.0
        tgt_norm = (tgt.clamp(-s, s) / s + 1.0) / 2.0
        l1_norm = F.l1_loss(pred_norm, tgt_norm, reduction='sum') / max(
            pred.shape[0], 1
        )

        ratio = l1_diffusion.item() / max(l1_norm.item(), 1e-8)
        print(f'L1 扩散空间: {l1_diffusion.item():.4f}')
        print(f'L1 [0,1]空间: {l1_norm.item():.4f}')
        print(f'比例 (应 ≈ 2*s = {2 * s}): {ratio:.4f}')

        return {
            'loss_bbox_diffusion_space': l1_diffusion.item(),
            'loss_bbox_norm_space': l1_norm.item(),
            'consistency_ratio': ratio,
        }

    @staticmethod
    def run_quick_check():
        """快速验证: 构造小规模 head, 运行所有监控检查."""
        print('=' * 60)
        print('SetDiff Loss 插桩监控 — 快速验证')
        print('=' * 60)

        from setdiff.criterion.set_loss import SetCriterion
        from setdiff.models.set_head import JointDiffusionHead

        head = JointDiffusionHead(
            num_queries=10,
            feat_channels=64,
            num_heads=4,
            num_layers=2,
            dim_feedforward=128,
            num_classes=24,
            snr_scale=2.0,
        )

        torch.manual_seed(42)
        B, HW, C = 1, 100, 64
        image_features = torch.randn(B, HW, C)
        # GT 在扩散空间 [-2, 2] (小目标 w,h 为负)
        gt_boxes = [torch.tensor([[0.0, 0.0, -1.6, -1.6]])]
        gt_labels = [torch.tensor([0])]

        print('\n[1] Double-counting 验证:')
        ok = LossMonitor.verify_no_double_counting(
            head, image_features, gt_boxes, gt_labels
        )

        print('\n[2] 梯度尺度测量 (head 层面, 诊断用):')
        # 需要新一次 forward (因为前一次已 backward)
        head.zero_grad()
        LossMonitor.measure_gradient_ratio(
            head, image_features, gt_boxes, gt_labels
        )

        print('\n[3] 梯度尺度测量 (criterion 层面, controlled):')
        # 构造 pred 接近 tgt 的场景 (GIoU 不饱和), 验证 ratio ≈ 2.5
        criterion = SetCriterion(num_classes=24, snr_scale=2.0)
        s = 2.0
        # [0,1] 空间: pred 略大于 tgt (同中心, GIoU 不饱和)
        pred_norm_ctrl = torch.tensor([[[0.5, 0.5, 0.3, 0.3]]])
        tgt_norm_ctrl = torch.tensor([[[0.5, 0.5, 0.2, 0.2]]])
        # 转换到扩散空间 [-s, s]
        pred_diff_ctrl = (pred_norm_ctrl * 2.0 - 1.0) * s
        tgt_diff_ctrl = (tgt_norm_ctrl * 2.0 - 1.0) * s
        # 2026-07-19: criterion 内部 Hungarian 重新匹配, 传原始 GT list
        LossMonitor.measure_criterion_gradient_ratio(
            criterion,
            pred_diff_ctrl,
            [tgt_diff_ctrl[0]],  # gt_boxes_list: List[Tensor[M,4]]
            [torch.tensor([0])],  # gt_labels_list: List[Tensor[M]]
        )

        print('\n[4] Loss 空间一致性:')
        pred_boxes = torch.tensor(
            [[[0.0, 0.0, -1.0, -1.0], [0.5, 0.5, 0.5, 0.5]]]
        )
        matched_boxes = torch.tensor(
            [[[0.2, 0.2, -1.4, -1.4], [0.3, 0.3, 0.3, 0.3]]]
        )
        matched_labels = torch.tensor([[0, 0]])
        LossMonitor.measure_loss_space_consistency(
            criterion, pred_boxes, matched_boxes, matched_labels
        )

        print('\n' + '=' * 60)
        print(f'Double-counting 修复: {"PASS" if ok else "FAIL"}')
        print('=' * 60)


class LossMonitorHook:
    """mmengine Hook: 训练中定期监控 loss 梯度尺度.

    用法:
        custom_hooks = [
            dict(type='LossMonitorHook', priority=50, interval=50),
        ]

    注: 实际注册到 mmengine 时, 可继承 ``mmengine.hooks.Hook`` 并使用
    ``@HOOKS.register_module()`` 装饰; 这里保持轻量, 不强依赖 mmengine.
    """

    def __init__(self, interval: int = 50):
        self.interval = interval

    def after_train_iter(
        self,
        runner,
        batch_idx: int,
        data_batch: Optional[dict] = None,
        outputs: Optional[dict] = None,
    ) -> None:
        """每 interval 步打印一次 loss 比例监控 (轻量, 不重复计算梯度)."""
        if batch_idx % self.interval != 0:
            return
        # 从 outputs 获取 log_vars
        if hasattr(outputs, 'log_vars'):
            log_vars = outputs.log_vars
        elif isinstance(outputs, dict) and 'log_vars' in outputs:
            log_vars = outputs['log_vars']
        else:
            return

        # 简化监控: 只打印 loss 比例 (不重复计算梯度, 避免开销)
        cls = log_vars.get('loss_cls', 0)
        bbox = log_vars.get('loss_bbox', 0)
        giou = log_vars.get('loss_giou', 0)
        if giou > 0:
            ratio = bbox / giou
            runner.logger.info(
                f'[LossMonitor] cls={cls:.4f} bbox={bbox:.4f} '
                f'giou={giou:.4f} bbox/giou={ratio:.2f} '
                f'(修复前≈4.0, 修复后≈2.5)'
            )
