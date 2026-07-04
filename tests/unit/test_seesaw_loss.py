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
    """测试 Seesaw 惩罚机制"""

    def test_high_freq_negative_less_penalty(self):
        """高频负类应获得更小的惩罚 (S_ij < 1 when N_j >> N_i)"""
        loss_fn = SeesawLoss(num_classes=3, p=0.8, gamma=0, alpha=-1,
                             reduction='sum', loss_weight=1.0)
        # 设置累计样本: 类0=1000(高频), 类1=10(低频), 类2=10(低频)
        loss_fn.cum_samples = torch.tensor([1000.0, 10.0, 10.0])

        # 一个类1的正样本, 预测为类0和类2的负logit相同
        pred = torch.tensor([[0.0, 10.0, 0.0]])  # 正样本是类1, 但所有logit=0/10
        # 实际: 让正样本logit低, 负样本logit高, 产生负损失
        pred = torch.tensor([[-5.0, -5.0, -5.0]])  # 全部低, 正样本类1也低
        target = torch.tensor([1])  # 类1正样本

        # 对比: 类0(高频, N=1000) vs 类2(低频, N=10) 的负损失
        # S_{1,0} = (1000/10)^0.8 = 100^0.8 ≈ 40, 但这是 > 1, 意味着类0的惩罚更大?
        # 不对! S_ij = (N_j/N_i)^p, 当 N_j > N_i 时 S > 1, 惩罚更大
        # 但 Seesaw 的目的是: 尾类(低频)的正样本, 对头类(高频)的负梯度应该被衰减
        # 所以应该是: 对于类i的正样本, 类j的负梯度乘以 S_ij = (N_j/N_i)^p
        # 当 N_j >> N_i (头类j对尾类i): S_ij >> 1... 这不对
        #
        # 重新看论文: S_ij = (N_j / N_i)^p, 但衰减的是 logit, 不是放大!
        # 论文公式: p_j = softmax(z_j), L = -log(p_y / (p_y + sum S_{y,j} * p_j))
        # 当 S < 1 时, p_j 的贡献减小 → 尾类正样本对头类的负梯度减小
        # 所以 S_{y,j} = (N_j / N_y)^p, 当 N_j >> N_y: S >> 1, 头类贡献更大? 不对
        #
        # 再看: S_{ij} = min(1, N_i/N_j)^p? 不, 论文是 S_{ij} = (N_j/N_i)^{-p} = (N_i/N_j)^p
        # 即: 对于类i正样本, 类j的衰减因子 = (N_i / N_j)^p
        # 当 N_j >> N_i (头类j, 尾类i): (N_i/N_j)^p << 1 → 头类负梯度被大幅衰减 ✓
        # 当 N_j ≈ N_i: (N_i/N_j)^p ≈ 1 → 正常惩罚 ✓
        loss = loss_fn(pred, target)

        # 验证: 高频类(类0)的负损失 < 低频类(类2)的负损失
        # 手动计算
        # 正样本类1, logit全-5, sigmoid(-5)≈0.0067
        # 负损失类0: (1-0) * p^0 * log(1-p) * S_{1,0}, S_{1,0} = (10/1000)^0.8
        # 负损失类2: (1-0) * p^0 * log(1-p) * S_{1,2}, S_{1,2} = (10/10)^0.8 = 1
        # S_{1,0} = (10/1000)^0.8 = 0.01^0.8 ≈ 0.025
        # 所以类0的负损失 ≈ 0.025 * base, 类2的负损失 ≈ 1 * base
        # 类0惩罚 < 类2惩罚 ✓
        p_neg = torch.sigmoid(torch.tensor(-5.0))  # 负类 sigmoid
        base_pos = -torch.log(torch.sigmoid(torch.tensor(-5.0)))  # 正类损失
        base_neg = -torch.log(1 - p_neg)  # 负类损失
        s_1_0 = (10.0 / 1000.0) ** 0.8
        s_1_2 = (10.0 / 10.0) ** 0.8
        expected_loss = base_pos + base_neg * s_1_0 + base_neg * s_1_2
        assert abs(loss.item() - expected_loss.item()) < 0.01

    def test_equal_frequency_no_scaling(self):
        """各类频率相等时, Seesaw 因子 ≈ 1, 退化为标准 focal loss"""
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

    def test_correct_classification_decays_head(self):
        """正确分类时, 头类负梯度被 Seesaw 衰减"""
        loss_fn = SeesawLoss(num_classes=3, p=0.8, gamma=0, alpha=-1,
                             reduction='sum', loss_weight=1.0)
        loss_fn.cum_samples = torch.tensor([1000.0, 10.0, 10.0])

        # 类1正样本, 类0也是高logit (但类1更高, 正确分类)
        pred = torch.tensor([[5.0, 10.0, 5.0]])
        target = torch.tensor([1])
        loss = loss_fn(pred, target)

        # 手动验证: 类0的负损失被 S_{1,0} = (10/1000)^0.8 ≈ 0.025 衰减
        p0 = torch.sigmoid(torch.tensor(5.0))
        p1 = torch.sigmoid(torch.tensor(10.0))
        p2 = torch.sigmoid(torch.tensor(5.0))
        s_1_0 = (10.0 / 1000.0) ** 0.8
        s_1_2 = (10.0 / 10.0) ** 0.8
        # 正损失: -log(sigmoid(10))
        pos_loss = -torch.log(p1)
        # 负损失: 类0 衰减, 类2 不衰减
        neg_loss_0 = -torch.log(1 - p0) * s_1_0
        neg_loss_2 = -torch.log(1 - p2) * s_1_2
        expected = pos_loss + neg_loss_0 + neg_loss_2
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
