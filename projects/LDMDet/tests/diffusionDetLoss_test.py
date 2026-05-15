import sys
from pathlib import Path

import torch

# 将项目目录添加到 python 路径
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from mods.loss import (
    DiffusionDetCriterion,
    DiffusionDetMatcher,
    FocalLoss,
    GIoULoss,
    L1Loss,
)


def test_losses():
    print('Testing basic losses...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. FocalLoss
    focal_loss = FocalLoss(
        use_sigmoid=True, alpha=0.25, gamma=2.0, loss_weight=2.0
    ).to(device)
    pred = torch.randn(2, 10, 80).to(device)
    target = torch.randint(0, 80, (2, 10)).to(device)
    loss_f = focal_loss(pred, target)
    assert loss_f.shape == ()
    assert loss_f >= 0

    # 2. L1Loss
    l1_loss = L1Loss(loss_weight=5.0).to(device)
    pred_box = torch.randn(2, 10, 4).to(device)
    target_box = torch.randn(2, 10, 4).to(device)
    loss_l1 = l1_loss(pred_box, target_box)
    assert loss_l1.shape == ()
    assert loss_l1 >= 0

    # 3. GIoULoss
    giou_loss = GIoULoss(loss_weight=2.0).to(device)
    pred_box = torch.tensor([[10, 10, 50, 50]], dtype=torch.float32).to(device)
    target_box = torch.tensor([[15, 15, 55, 55]], dtype=torch.float32).to(
        device
    )
    loss_giou = giou_loss(pred_box, target_box)
    assert loss_giou.shape == ()
    assert loss_giou >= 0


def test_criterion_forward():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = 80

    matcher = DiffusionDetMatcher(cost_class=2.0, cost_bbox=5.0, cost_giou=2.0)
    loss_cls = FocalLoss(
        use_sigmoid=True, alpha=0.25, gamma=2.0, loss_weight=2.0
    )
    loss_bbox = L1Loss(loss_weight=5.0)
    loss_giou = GIoULoss(loss_weight=2.0)

    criterion = DiffusionDetCriterion(
        num_classes=num_classes,
        matcher=matcher,
        loss_cls=loss_cls,
        loss_bbox=loss_bbox,
        loss_giou=loss_giou,
        deep_supervision=True,
    ).to(device)

    # 模拟网络输出
    bs = 2
    num_proposals = 100
    outputs = {
        'pred_logits': torch.randn(bs, num_proposals, num_classes).to(device),
        'pred_boxes': torch.rand(bs, num_proposals, 4).to(
            device
        ),  # 归一化 xyxy
        'aux_outputs': [
            {
                'pred_logits': torch.randn(bs, num_proposals, num_classes).to(
                    device
                ),
                'pred_boxes': torch.rand(bs, num_proposals, 4).to(device),
            }
            for _ in range(2)
        ],
    }

    # 模拟真值
    targets = [
        {
            'labels': torch.tensor([1, 5, 10], dtype=torch.long).to(device),
            'bboxes': torch.rand(3, 4).to(device),  # 归一化 xyxy
            'img_shape': (800, 800),
        },
        {
            'labels': torch.tensor([2, 8], dtype=torch.long).to(device),
            'bboxes': torch.rand(2, 4).to(device),
            'img_shape': (800, 1000),
        },
    ]

    losses = criterion(outputs, targets)

    # 检查损失字典
    assert 'loss_cls' in losses
    assert 'loss_bbox' in losses
    assert 'loss_giou' in losses
    assert 'aux_0_loss_cls' in losses
    assert 'aux_1_loss_bbox' in losses

    for k, v in losses.items():
        assert v.shape == ()
        assert not torch.isnan(v)
    print('Criterion forward test passed!')


if __name__ == '__main__':
    test_losses()
    test_criterion_forward()
    print('All tests passed!')
