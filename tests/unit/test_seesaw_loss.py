"""方向I-1: SeesawLoss 单元测试

测试 Seesaw Loss for Long-Tailed Instance Segmentation (CVPR 2021) 的实现。
核心验证:
1. 损失形状和基本前向
2. Seesaw 惩罚: 高频负类获得更小惩罚
3. 累计样本计数 buffer 更新
4. 误分类自校准惩罚
5. 梯度回传
6. 边界条件 (全背景、单类)
"""

import pytest
import torch
import torch.nn.functional as F

from ldmdet.criterion.losses import SeesawLoss


class TestSeesawLossShape:
    """测试损失输出形状和基本性质"""

    def test_scalar_output(self):
        """损失应为标量"""
        loss_fn = SeesawLoss(num_classes=24, p=0.8)
        pred = torch.randn(100, 24)
        target = torch.randint(0, 25, (100,))  # 0-23 + 24=background
        loss = loss_fn(pred, target)
        assert loss.dim() == 0

    def test_positive_loss(self):
        """非完美预测时损失应 > 0"""
        loss_fn = SeesawLoss(num_classes=24, p=0.8, reduction='sum')
        pred = torch.randn(50, 24)
        target = torch.randint(0, 24, (50,))
        loss = loss_fn(pred, target)
        assert loss > 0

    def test_perfect_prediction_low_loss(self):
        """完美预测时损失应接近 0"""
        loss_fn = SeesawLoss(num_classes=24, p=0.8, gamma=0, alpha=-1)
        # 正样本 logit 大正数, 负样本 logit 大负数
        pred = torch.full((10, 24), -100.0)
        target = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        pred[range(10), target] = 100.0
        loss = loss_fn(pred, target)
        assert loss < 0.01

    def test_loss_weight_scaling(self):
        """loss_weight 应正确缩放损失"""
        pred = torch.randn(50, 24)
        target = torch.randint(0, 25, (50,))
        loss_fn1 = SeesawLoss(num_classes=24, loss_weight=1.0)
        loss_fn2 = SeesawLoss(num_classes=24, loss_weight=2.0)
        l1 = loss_fn1(pred, target)
        l2 = loss_fn2(pred, target)
        assert abs((l2 / l1).item() - 2.0) < 0.01


class TestSeesawPenalty:
    """测试 Seesaw 惩罚机制 (mmdet 官方方向: ratio = N_j / N_y, clamp(max=1.0))

    语义: 头类样本(N_y大)对尾类负类(N_j小)时 S<1 衰减, 保护尾类不被头类压制.
    """

    def test_head_sample_tail_negative_decays(self):
        """头类样本对尾类负类应被衰减 (S < 1 when N_j << N_y)

        场景: 样本是头类0(N_0=1000), 负类1是尾类(N_1=10)
        期望: S_{0,1} = (N_1/N_0)^p = (10/1000)^0.8 ≈ 0.025 (衰减)
        """
        loss_fn = SeesawLoss(num_classes=3, p=0.8, gamma=0, alpha=-1,
                             reduction='sum', loss_weight=1.0)
        loss_fn.cum_samples = torch.tensor([1000.0, 10.0, 10.0])

        # 头类0正样本, 尾类1和2的负logit较高(产生负损失)
        pred = torch.tensor([[5.0, 5.0, 5.0]])
        target = torch.tensor([0])  # 头类0正样本
        loss = loss_fn(pred, target)

        # 手动计算
        p_pos = torch.sigmoid(torch.tensor(5.0))  # 正类 sigmoid
        base_pos = -torch.log(p_pos)  # 正类损失
        base_neg = -torch.log(1 - p_pos)  # 负类损失
        # S_{0,1} = (N_1/N_0)^p = (10/1000)^0.8 ≈ 0.025 (尾类负类, 衰减)
        # S_{0,2} = (N_2/N_0)^p = (10/1000)^0.8 ≈ 0.025 (尾类负类, 衰减)
        s_0_1 = (10.0 / 1000.0) ** 0.8
        s_0_2 = (10.0 / 1000.0) ** 0.8
        expected_loss = base_pos + base_neg * s_0_1 + base_neg * s_0_2
        assert abs(loss.item() - expected_loss.item()) < 0.01

    def test_tail_sample_head_negative_no_decay(self):
        """尾类样本对头类负类不应衰减 (S = 1 when N_j >> N_y)

        场景: 样本是尾类1(N_1=10), 负类0是头类(N_0=1000)
        期望: S_{1,0} = min(N_0/N_1, 1)^p = min(100, 1)^0.8 = 1.0 (不衰减)
        """
        loss_fn = SeesawLoss(num_classes=3, p=0.8, gamma=0, alpha=-1,
                             reduction='sum', loss_weight=1.0)
        loss_fn.cum_samples = torch.tensor([1000.0, 10.0, 10.0])

        # 尾类1正样本, 头类0和尾类2的负logit较高
        pred = torch.tensor([[5.0, 5.0, 5.0]])
        target = torch.tensor([1])  # 尾类1正样本
        loss = loss_fn(pred, target)

        # 手动计算
        p_pos = torch.sigmoid(torch.tensor(5.0))
        base_pos = -torch.log(p_pos)
        base_neg = -torch.log(1 - p_pos)
        # S_{1,0} = min(1000/10, 1)^0.8 = 1.0 (头类负类, 不衰减)
        # S_{1,2} = min(10/10, 1)^0.8 = 1.0 (等频, 不衰减)
        s_1_0 = 1.0
        s_1_2 = 1.0
        expected_loss = base_pos + base_neg * s_1_0 + base_neg * s_1_2
        assert abs(loss.item() - expected_loss.item()) < 0.01

    def test_no_amplification_head_to_tail(self):
        """验证 Bug 修复: 头类样本对尾类负类不应放大 (S <= 1)"""
        loss_fn = SeesawLoss(num_classes=3, p=0.8, gamma=0, alpha=-1,
                             reduction='sum', loss_weight=1.0)
        loss_fn.cum_samples = torch.tensor([1000.0, 10.0, 10.0])

        # 头类0正样本
        pred = torch.tensor([[5.0, 5.0, 5.0]])
        target = torch.tensor([0])
        loss_head_sample = loss_fn(pred, target)

        # 对比: 如果 S 被错误放大, 损失会远大于标准 BCE
        # 标准 BCE (无 Seesaw): pos + neg + neg
        p_pos = torch.sigmoid(torch.tensor(5.0))
        standard_bce = -torch.log(p_pos) + 2 * (-torch.log(1 - p_pos))

        # Seesaw 应衰减尾类负类, 所以损失应 < 标准 BCE
        assert loss_head_sample.item() < standard_bce.item(), \
            '头类样本对尾类负类应衰减, 损失应小于标准 BCE'

    def test_equal_frequency_no_scaling(self):
        """各类频率相等时, Seesaw 因子 = 1, 退化为标准 BCE"""
        loss_fn = SeesawLoss(num_classes=3, p=0.8, gamma=0, alpha=-1,
                             reduction='sum', loss_weight=1.0)
        loss_fn.cum_samples = torch.tensor([100.0, 100.0, 100.0])

        focal_pred = torch.randn(20, 3)
        target = torch.randint(0, 3, (20,))

        seesaw_loss = loss_fn(focal_pred, target)

        # 用标准 BCE 对比
        one_hot = torch.zeros(20, 3)
        one_hot[range(20), target] = 1.0
        bce_loss = F.binary_cross_entropy_with_logits(focal_pred, one_hot, reduction='sum')

        assert abs(seesaw_loss.item() - bce_loss.item()) < 0.1


class TestCumulativeSamples:
    """测试累计样本计数 buffer"""

    def test_buffer_updates(self):
        """forward 后 cum_samples 应更新"""
        loss_fn = SeesawLoss(num_classes=5, p=0.8)
        initial = loss_fn.cum_samples.clone()
        target = torch.tensor([0, 0, 1, 2, 2, 2, 3, 4, 4, 4])
        pred = torch.randn(10, 5)
        loss_fn(pred, target)
        # 每类初始=1, 加上本批次: 类0=+2, 类1=+1, 类2=+3, 类3=+1, 类4=+3
        expected = torch.tensor([3.0, 2.0, 4.0, 2.0, 4.0])
        assert torch.allclose(loss_fn.cum_samples, expected)

    def test_background_not_counted(self):
        """背景样本 (target == num_classes) 不应更新 cum_samples"""
        loss_fn = SeesawLoss(num_classes=3, p=0.8)
        target = torch.tensor([0, 1, 3, 3, 2])  # 3=background
        pred = torch.randn(5, 3)
        loss_fn(pred, target)
        # 类0=+1, 类1=+1, 类2=+1, 背景不计
        expected = torch.tensor([2.0, 2.0, 2.0])
        assert torch.allclose(loss_fn.cum_samples, expected)

    def test_buffer_persistent(self):
        """cum_samples 应是 buffer (随 state_dict 保存)"""
        loss_fn = SeesawLoss(num_classes=3, p=0.8)
        assert 'cum_samples' in dict(loss_fn.named_buffers())


class TestMisclassificationPenalty:
    """测试误分类自校准惩罚"""

    def test_misclassified_restores_penalty(self):
        """误分类时, Seesaw 衰减被恢复 (S=1)"""
        loss_fn = SeesawLoss(num_classes=3, p=0.8, gamma=0, alpha=-1, reduction='sum')
        # 类0高频, 类1低频
        loss_fn.cum_samples = torch.tensor([1000.0, 10.0, 10.0])

        # 类1正样本, 但类0的logit远高于类1 (误分类为类0)
        pred_misclassified = torch.tensor([[10.0, -10.0, -10.0]])
        target = torch.tensor([1])

        # 正确分类 (类1 logit高)
        pred_correct = torch.tensor([[-10.0, 10.0, -10.0]])

        loss_mis = loss_fn(pred_misclassified, target)
        # reset buffer (forward 会更新)
        loss_fn.cum_samples = torch.tensor([1000.0, 10.0, 10.0])
        loss_correct = loss_fn(pred_correct, target)

        # 误分类的损失应远大于正确分类
        assert loss_mis > loss_correct

    def test_correct_classification_decays_tail(self):
        """正确分类时, 头类样本对尾类负类的梯度被 Seesaw 衰减"""
        loss_fn = SeesawLoss(num_classes=3, p=0.8, gamma=0, alpha=-1,
                             reduction='sum', loss_weight=1.0)
        loss_fn.cum_samples = torch.tensor([1000.0, 10.0, 10.0])

        # 头类0正样本(logit=10, 正确分类), 尾类1和2的负logit也高(logit=5)
        pred = torch.tensor([[10.0, 5.0, 5.0]])
        target = torch.tensor([0])  # 头类0正样本
        loss = loss_fn(pred, target)

        # 手动验证: 尾类1和2的负损失被 S_{0,j} = (N_j/N_0)^p ≈ 0.025 衰减
        p0 = torch.sigmoid(torch.tensor(10.0))  # 正类
        p1 = torch.sigmoid(torch.tensor(5.0))   # 负类1
        p2 = torch.sigmoid(torch.tensor(5.0))   # 负类2
        # S_{0,1} = (N_1/N_0)^p = (10/1000)^0.8 ≈ 0.025 (尾类负类, 衰减)
        # S_{0,2} = (N_2/N_0)^p = (10/1000)^0.8 ≈ 0.025 (尾类负类, 衰减)
        s_0_1 = (10.0 / 1000.0) ** 0.8
        s_0_2 = (10.0 / 1000.0) ** 0.8
        pos_loss = -torch.log(p0)
        neg_loss_1 = -torch.log(1 - p1) * s_0_1
        neg_loss_2 = -torch.log(1 - p2) * s_0_2
        expected = pos_loss + neg_loss_1 + neg_loss_2
        assert abs(loss.item() - expected.item()) < 0.01


class TestSeesawBackward:
    """测试梯度回传"""

    def test_gradient_flows(self):
        """梯度应正确回传到 pred"""
        loss_fn = SeesawLoss(num_classes=5, p=0.8)
        pred = torch.randn(20, 5, requires_grad=True)
        target = torch.randint(0, 6, (20,))
        loss = loss_fn(pred, target)
        loss.backward()
        assert pred.grad is not None
        assert not torch.isnan(pred.grad).any()

    def test_gradient_shape(self):
        """梯度形状应与 pred 一致"""
        loss_fn = SeesawLoss(num_classes=5, p=0.8)
        pred = torch.randn(20, 5, requires_grad=True)
        target = torch.randint(0, 5, (20,))
        loss = loss_fn(pred, target)
        loss.backward()
        assert pred.grad.shape == pred.shape


class TestSeesawEdgeCases:
    """测试边界条件"""

    def test_all_background(self):
        """全部背景样本时应正常返回"""
        loss_fn = SeesawLoss(num_classes=5, p=0.8)
        pred = torch.randn(10, 5)
        target = torch.full((10,), 5)  # 全背景
        loss = loss_fn(pred, target)
        assert loss.dim() == 0
        assert not torch.isnan(loss)

    def test_single_class(self):
        """只有单一正类时应正常工作"""
        loss_fn = SeesawLoss(num_classes=3, p=0.8)
        pred = torch.randn(10, 3)
        target = torch.tensor([1, 1, 1, 1, 1, 3, 3, 3, 3, 3])
        loss = loss_fn(pred, target)
        assert loss.dim() == 0
        assert not torch.isnan(loss)

    def test_empty_batch(self):
        """空 batch 应返回 0 损失"""
        loss_fn = SeesawLoss(num_classes=3, p=0.8)
        pred = torch.randn(0, 3)
        target = torch.tensor([], dtype=torch.long)
        loss = loss_fn(pred, target)
        assert loss.item() == 0.0

    def test_two_dim_target(self):
        """target 为 2D 时应正确处理 (与 FocalLoss 接口一致)"""
        loss_fn = SeesawLoss(num_classes=5, p=0.8)
        pred = torch.randn(4, 10, 5)  # [bs, N, C]
        target = torch.randint(0, 6, (4, 10))  # [bs, N]
        loss = loss_fn(pred, target)
        assert loss.dim() == 0

    def test_reduction_mean(self):
        """reduction='mean' 应返回均值"""
        loss_fn = SeesawLoss(num_classes=5, p=0.8, reduction='mean', loss_weight=1.0)
        pred = torch.randn(20, 5)
        target = torch.randint(0, 6, (20,))
        loss = loss_fn(pred, target)
        assert loss.dim() == 0
        # mean 应小于 sum
        loss_fn_sum = SeesawLoss(num_classes=5, p=0.8, reduction='sum', loss_weight=1.0)
        loss_sum = loss_fn_sum(pred, target)
        assert loss < loss_sum

    def test_flat_input_no_reshape(self):
        """target 为 one-hot 格式时应正确转换 (与 FocalLoss else 分支兼容)"""
        loss_fn = SeesawLoss(num_classes=5, p=0.8, loss_weight=1.0)
        # pred: [N, C], target: [N, C] one-hot -> 走 else 分支, argmax 转换
        pred = torch.randn(10, 5)
        # 构造 one-hot target
        target_idx = torch.randint(0, 5, (10,))
        target_onehot = torch.zeros(10, 5)
        target_onehot[range(10), target_idx] = 1.0
        # one-hot 输入应与类别索引输入产生相同结果
        loss_onehot = loss_fn(pred, target_onehot)
        loss_fn2 = SeesawLoss(num_classes=5, p=0.8, loss_weight=1.0)
        loss_idx = loss_fn2(pred, target_idx)
        assert torch.allclose(loss_onehot, loss_idx, atol=1e-5)
