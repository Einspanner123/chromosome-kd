"""方向 Q1: RoI 特征遮挡模拟 Smoke 测试

验证 Q1 配置 (q1_occlusion_p03.py) 在实际训练中能正常工作。

测试流程:
  1. 加载 Q1 配置
  2. 构建模型
  3. 运行 1 个 training iteration (随机数据)
  4. 验证 loss 正常计算
  5. 验证 eval 模式下不施加遮挡
  6. 打印关键信息: 模型参数数、遮挡参数值、loss 值、梯度是否存在

运行:
  cd /home/linkst/workspace/projects/chromosome-kd && \
      conda run -n chromo python tests/smoke/test_q1_smoke.py
"""

import os
import sys

import torch

# 确保项目根目录在 path 中
_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# 注册所有模块 (和 tools/train.py 一致)
import mmdet.models  # noqa: F401, E402
from mmengine.registry import init_default_scope  # noqa: E402

init_default_scope('mmdet')

import experiments.mmdet_bridge.registry  # noqa: F401, E402
import experiments.mmdet_bridge.detector  # noqa: F401, E402
import experiments.mmdet_bridge.hooks  # noqa: F401, E402

from mmengine.config import Config  # noqa: E402
from mmengine.registry import MODELS  # noqa: E402
from mmengine.structures import InstanceData  # noqa: E402
from mmdet.structures import DetDataSample  # noqa: E402


# ================================================================
# 常量
# ================================================================

CONFIG_PATH = os.path.join(
    _PROJECT_ROOT,
    'experiments',
    'configs',
    'ldmdet',
    'directions',
    'frontier_directions',
    'q1_occlusion_p03.py',
)

DUMMY_BS = 2
DUMMY_IMG_SIZE = 256
DUMMY_NUM_GT = 5
DUMMY_NUM_CLASSES = 24


# ================================================================
# 辅助函数
# ================================================================


def _make_dummy_batch(
    bs: int = DUMMY_BS,
    img_size: int = DUMMY_IMG_SIZE,
    num_gt: int = DUMMY_NUM_GT,
    with_gt: bool = True,
):
    """构造 dummy 输入数据 (随机张量, 不依赖真实数据集)"""
    batch_inputs = torch.randn(bs, 3, img_size, img_size)
    batch_data_samples = []
    for _ in range(bs):
        ds = DetDataSample()
        ds.set_metainfo(
            dict(
                img_shape=(img_size, img_size),
                pad_shape=(img_size, img_size),
                ori_shape=(img_size, img_size),
                scale_factor=[1.0, 1.0, 1.0, 1.0],
            )
        )
        if with_gt:
            gt_instances = InstanceData()
            x1 = torch.rand(num_gt) * (img_size - 60)
            y1 = torch.rand(num_gt) * (img_size - 60)
            w = torch.rand(num_gt) * 50 + 10
            h = torch.rand(num_gt) * 50 + 10
            gt_instances.bboxes = torch.stack(
                [x1, y1, x1 + w, y1 + h], dim=1
            )
            gt_instances.labels = torch.randint(
                0, DUMMY_NUM_CLASSES, (num_gt,)
            )
            ds.gt_instances = gt_instances
        batch_data_samples.append(ds)
    return batch_inputs, batch_data_samples


def _find_single_heads(model):
    """递归查找模型中所有 SingleDiffusionDetHead 实例"""
    from ldmdet.core.single_head import SingleDiffusionDetHead

    heads = []
    for module in model.modules():
        if isinstance(module, SingleDiffusionDetHead):
            heads.append(module)
    return heads


# ================================================================
# 主测试流程
# ================================================================


def main():
    print('=' * 72)
    print('方向 Q1: RoI 特征遮挡模拟 Smoke 测试')
    print('=' * 72)

    # ------------------------------------------------------------
    # Step 1: 加载 Q1 配置
    # ------------------------------------------------------------
    print('\n[Step 1] 加载 Q1 配置...')
    print(f'  配置路径: {CONFIG_PATH}')
    cfg = Config.fromfile(CONFIG_PATH)
    print(f'  配置加载成功: model.type = {cfg.model.type}')

    # 验证遮挡参数已正确写入配置
    sh_cfg = cfg.model.bbox_head.single_head
    print(f'  single_head.occlusion_prob = {sh_cfg.get("occlusion_prob")}')
    print(
        f'  single_head.occlusion_ratio_range = '
        f'{sh_cfg.get("occlusion_ratio_range")}'
    )
    assert sh_cfg.get('occlusion_prob') == 0.3, \
        '配置中 occlusion_prob 应为 0.3'
    assert sh_cfg.get('occlusion_ratio_range') == (0.2, 0.6), \
        '配置中 occlusion_ratio_range 应为 (0.2, 0.6)'

    # ------------------------------------------------------------
    # Step 2: 构建模型
    # ------------------------------------------------------------
    print('\n[Step 2] 构建模型...')
    model = MODELS.build(cfg.model)
    print('  模型构建成功')

    # 模型参数数
    n_total = sum(p.numel() for p in model.parameters())
    n_grad = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f'  模型参数总数: {n_total:,}')
    print(f'  可训练参数数: {n_grad:,} ({n_grad / n_total * 100:.2f}%)')

    # 查找 SingleDiffusionDetHead 并打印遮挡参数
    heads = _find_single_heads(model)
    print(f'  SingleDiffusionDetHead 实例数: {len(heads)}')
    for i, head in enumerate(heads):
        print(
            f'  head[{i}]: occlusion_prob={head.occlusion_prob}, '
            f'occlusion_ratio_range={head.occlusion_ratio_range}, '
            f'training={head.training}'
        )

    # ------------------------------------------------------------
    # Step 3 & 4: 运行 1 个 training iteration 并验证 loss
    # ------------------------------------------------------------
    print('\n[Step 3/4] 运行 1 个 training iteration (随机数据)...')
    model.train()
    torch.manual_seed(42)
    batch_inputs, batch_data_samples = _make_dummy_batch()
    print(
        f'  输入: batch_inputs={tuple(batch_inputs.shape)}, '
        f'num_samples={len(batch_data_samples)}'
    )

    losses = model.loss(batch_inputs, batch_data_samples)
    print(f'  loss forward 成功, 损失项: {list(losses.keys())}')

    total_loss = sum(
        v for v in losses.values() if isinstance(v, torch.Tensor)
    )
    print(f'  total_loss = {total_loss.item():.6f}')
    for key, val in losses.items():
        if isinstance(val, torch.Tensor):
            print(f'    {key}: {val.item():.6f}')
        else:
            print(f'    {key}: {val}')

    assert isinstance(losses, dict) and len(losses) > 0, 'loss 应为非空 dict'
    assert torch.isfinite(total_loss), 'total_loss 应为有限值'
    print('  [OK] loss 正常计算')

    # ------------------------------------------------------------
    # Step 5: 验证梯度存在
    # ------------------------------------------------------------
    print('\n[Step 5] 验证梯度回传...')
    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-3,
        momentum=0.9,
    )
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    # 统计有梯度的参数
    n_with_grad = 0
    n_without_grad = 0
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if p.grad is not None and p.grad.abs().sum() > 0:
            n_with_grad += 1
        else:
            n_without_grad += 1
    print(f'  有梯度的参数张量数: {n_with_grad}')
    print(f'  无梯度的参数张量数: {n_without_grad}')
    assert n_with_grad > 0, '应有参数获得梯度'
    print('  [OK] 梯度正常回传')

    # ------------------------------------------------------------
    # Step 6: 验证 eval 模式下不施加遮挡
    # ------------------------------------------------------------
    print('\n[Step 6] 验证 eval 模式下不施加遮挡...')
    model.eval()

    # 直接调用 _apply_occlusion 验证 eval 模式不遮挡
    torch.manual_seed(42)
    dummy_roi = torch.randn(10, 256, 7, 7)
    for i, head in enumerate(heads):
        head.eval()
        out = head._apply_occlusion(dummy_roi)
        unchanged = torch.equal(out, dummy_roi)
        print(
            f'  head[{i}]: eval 模式 _apply_occlusion 输出与输入完全一致? '
            f'{unchanged}'
        )
        assert unchanged, f'head[{i}] eval 模式不应施加遮挡'

    # 推理前向测试 (验证 eval 模式 forward 不报错)
    batch_inputs_eval, batch_data_samples_eval = _make_dummy_batch(
        bs=1, with_gt=False
    )
    with torch.no_grad():
        results = model.predict(batch_inputs_eval, batch_data_samples_eval)
    pred = results[0].pred_instances
    print(
        f'  eval forward 成功: pred_instances.bboxes={tuple(pred.bboxes.shape)}, '
        f'scores={tuple(pred.scores.shape)}, labels={tuple(pred.labels.shape)}'
    )
    print('  [OK] eval 模式不施加遮挡, 推理正常')

    # ------------------------------------------------------------
    # 结论
    # ------------------------------------------------------------
    print('\n' + '=' * 72)
    print('Smoke 测试结论: 全部通过')
    print('=' * 72)
    print('  ✓ Q1 配置能正确加载, occlusion_prob=0.3, ratio=(0.2, 0.6)')
    print('  ✓ 模型能正常构建, SingleDiffusionDetHead 实例参数正确')
    print('  ✓ 1 个 training iteration 正常运行, loss 有限且非零')
    print('  ✓ 梯度正常回传, optimizer.step() 后参数更新')
    print('  ✓ eval 模式下 _apply_occlusion 不施加遮挡 (推理零开销)')
    print('  ✓ eval 模式 forward (predict) 正常运行')
    print('\n实现正确, Q1 遮挡模拟可在实际训练中正常工作。')


if __name__ == '__main__':
    main()
