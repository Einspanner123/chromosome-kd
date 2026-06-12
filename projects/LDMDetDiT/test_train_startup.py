"""快速验证: train_noise_source='grid' 能否正常启动训练"""
import sys

import torch

sys.path.insert(0, 'projects/LDMDetDiT')


def test_noise_gen():
    from mods.dit_head import DiTDiffusionDetHead
    from mods.rectified_flow import RectifiedFlow

    N, bs = 100, 2
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    head = DiTDiffusionDetHead(
        num_classes=24, feat_channels=384, num_proposals=N,
        train_noise_source='grid', box_init_mode='spatial_prior',
        use_adaln_zero=False,
    ).to(device)

    # 1. _make_train_noise 形状
    noise = head._make_train_noise(device)
    assert noise.shape == (N, 4), f"shape: {noise.shape}"
    print(f"1. _make_train_noise shape: {noise.shape} ✓")

    # 2. 每次调用结果不同
    n0 = head._make_train_noise(device)
    n1 = head._make_train_noise(device)
    assert not torch.allclose(n0, n1)
    print("2. noise varies between calls ✓")

    # 3. grid noise vs randn: grid 确定性高 (std 更小)
    n_grid = head._make_train_noise(device)
    n_rand = torch.randn(N, 4, device=device)
    print(f"3. grid std={n_grid.std():.3f} vs randn std={n_rand.std():.3f} (grid should be ~1.8-2.0)")

    # 4. _build_training_targets 端到端
    rf = RectifiedFlow()
    gt_bboxes = [
        torch.tensor([[50, 50, 120, 120], [200, 80, 280, 200]], dtype=torch.float32, device=device),
        torch.tensor([[30, 60, 150, 180]], dtype=torch.float32, device=device),
    ]
    gt_labels = [
        torch.tensor([0, 5], dtype=torch.long, device=device),
        torch.tensor([3], dtype=torch.long, device=device),
    ]
    img_metas = [{'img_shape': (512, 512)}, {'img_shape': (512, 512)}]
    t = torch.tensor([0.3, 0.9], device=device)

    targets = head._normalize_targets(gt_bboxes, gt_labels, img_metas, bs)
    x_boxes, x_starts, x_noises, matched, probs = head._build_training_targets(
        bs, device, t, targets, gt_bboxes, img_metas)

    for i in range(bs):
        xb, xs, xn = x_boxes[i], x_starts[i], x_noises[i]
        # x_noisy = (1-t)*x_start + t*noise
        expected = (1-t[i]) * xs + t[i] * xn
        diff = (xb - expected).abs().max().item()
        assert diff < 1e-5, f"batch {i}: diff={diff:.6f}"
    print("4. _build_training_targets: x_noisy = (1-t)*x_start + t*noise ✓")

    # 5. 验证 t=1 时 x_noisy ≈ noise (归一化后是网格分布，不是中心聚集)
    t1 = torch.tensor([1.0, 1.0], device=device)
    xb1, xs1, xn1, _, _ = head._build_training_targets(
        bs, device, t1, targets, gt_bboxes, img_metas)
    for i in range(bs):
        assert torch.allclose(xb1[i], xn1[i], atol=1e-4), "t=1: x_noisy ≠ noise"
    # grid 噪声不应聚集在 0 附近 (中心聚集的典型 randn 问题)
    print(f"5. t=1: x_noisy = noise (grid), center_bias={xn1[0][:, :2].norm(dim=-1).mean():.3f} "
          f"(grid should be ~2-3, not ~0) ✓")

    # 6. forward 前向传播
    features = [torch.randn(bs, 384, 128, 128, device=device)]
    curr_bboxes_batch = torch.cat([
        head._raw_to_xyxy(x_boxes[i].unsqueeze(0), img_metas[i:i+1])
        for i in range(bs)])
    t_input = t * head.timesteps
    outputs = head(features, curr_bboxes_batch, t_input, img_metas=img_metas)
    all_cls, all_pred_bboxes, _, all_x0_raw, _, _ = outputs
    assert all_cls[-1].shape == (bs, N, 24)
    assert all_pred_bboxes[-1].shape == (bs, N, 4)
    print(f"6. forward: cls={list(all_cls[-1].shape)} pred_bboxes={list(all_pred_bboxes[-1].shape)} ✓")

    print(f"\n{'='*60}")
    print("ALL CHECKS PASSED — 训练可正常启动")
    print(f"{'='*60}")


if __name__ == '__main__':
    test_noise_gen()
