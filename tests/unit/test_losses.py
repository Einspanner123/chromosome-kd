"""测试损失函数"""

import torch

from ldmdet.criterion.losses import FocalLoss, GIoULoss, L1Loss


class TestFocalLoss:
    def test_shape(self):
        loss_fn = FocalLoss()
        pred = torch.randn(10, 24)
        target = torch.randint(0, 24, (10,))
        loss = loss_fn(pred, target)
        assert loss.dim() == 0  # scalar
        assert loss > 0

    def test_perfect_prediction(self):
        loss_fn = FocalLoss(gamma=0, alpha=-1)  # gamma=0 → standard BCE
        pred = torch.tensor([[100.0, -100.0], [-100.0, 100.0]])
        target = torch.tensor([0, 1])
        loss = loss_fn(pred, target)
        # 完美预测: loss ≈ 0
        assert loss < 0.01


class TestL1Loss:
    def test_shape(self):
        loss_fn = L1Loss()
        pred = torch.randn(10, 4)
        target = torch.randn(10, 4)
        loss = loss_fn(pred, target)
        assert loss.dim() == 0

    def test_zero_diff(self):
        loss_fn = L1Loss(reduction='mean')
        pred = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        target = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        loss = loss_fn(pred, target)
        assert loss == 0.0


class TestGIoULoss:
    def test_shape(self):
        loss_fn = GIoULoss()
        pred = torch.tensor([[0.1, 0.1, 0.9, 0.9], [0.2, 0.2, 0.8, 0.8]])
        target = torch.tensor([[0.0, 0.0, 1.0, 1.0], [0.3, 0.3, 0.7, 0.7]])
        loss = loss_fn(pred, target)
        assert loss.dim() == 0

    def test_perfect_overlap(self):
        loss_fn = GIoULoss(reduction='mean')
        bbox = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
        loss = loss_fn(bbox, bbox)
        assert abs(loss.item()) < 0.001
