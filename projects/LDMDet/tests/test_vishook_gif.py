"""
LDMDet PredictionVisHook 增量 GIF 生成测试

红绿重构 TDD:
  Red:   测试增量 GIF 追加行为 (当前全量重建，测试应失败)
  Green: 实现增量追加逻辑
  Refactor: 清理代码

运行方式:
  PYTHONPATH=. python projects/LDMDet/tests/test_vishook_gif.py
"""

import sys
import tempfile
import os
from pathlib import Path

sys.path.insert(0, 'projects/LDMDet')

from PIL import Image, ImageDraw


def _make_epoch_img(vis_dir, epoch, img_idx, stem, color_rgb):
    """创建模拟 epoch 可视化图片。"""
    img = Image.new('RGB', (400, 300), color=(30, 30, 50))
    draw = ImageDraw.Draw(img)
    draw.rectangle([100, 80, 250, 200], outline=color_rgb, width=2)
    draw.text((10, 10), f'Epoch {epoch}', fill='white')
    fname = f'epoch_{epoch:03d}_img{img_idx}_{stem}.jpg'
    img.save(vis_dir / fname)
    return fname


def _count_gif_frames(gif_path):
    """统计 GIF 帧数。"""
    if not gif_path.exists():
        return 0
    img = Image.open(gif_path)
    return img.n_frames


def _get_gif_frame_durations(gif_path):
    """获取 GIF 每帧的 duration (ms)。"""
    if not gif_path.exists():
        return []
    img = Image.open(gif_path)
    durations = []
    try:
        while True:
            durations.append(img.info.get('duration', 100))
            img.seek(img.tell() + 1)
    except EOFError:
        pass
    return durations


# ============================================================
# Red: 增量追加测试 (当前实现为全量重建，以下测试应失败)
# ============================================================

def test_incremental_gif_append():
    """测试: 已有 GIF 时，新 epoch 只追加一帧而非全量重建。

    验证点:
    1. 第 1 个 epoch: 不生成 GIF (需要 >=2 帧)
    2. 第 2 个 epoch: 生成 2 帧 GIF
    3. 第 3 个 epoch: 追加第 3 帧，GIF 变为 3 帧
    4. 第 4 个 epoch: 追加第 4 帧，GIF 变为 4 帧
    5. 末帧 duration 应为 2000ms，其余为 500ms
    """
    from hooks import PredictionVisHook

    hook = PredictionVisHook(num_images=2)

    with tempfile.TemporaryDirectory() as tmpdir:
        vis_dir = Path(tmpdir)
        gif_dir = vis_dir / 'gifs'

        # Epoch 1: 只有 1 帧，不应生成 GIF
        _make_epoch_img(vis_dir, 1, 0, 'img_A', (255, 0, 0))
        hook._generate_gifs(vis_dir, 1)
        gif_path = gif_dir / 'img_A.gif'
        assert not gif_path.exists(), \
            f'Epoch 1: 不应生成 GIF (只有 1 帧), 但文件存在: {gif_path}'

        # Epoch 3: 现在有 2 帧，应生成 2 帧 GIF
        _make_epoch_img(vis_dir, 3, 0, 'img_A', (0, 255, 0))
        hook._generate_gifs(vis_dir, 3)
        assert gif_path.exists(), \
            f'Epoch 3: 应生成 GIF, 但文件不存在'
        n_frames = _count_gif_frames(gif_path)
        assert n_frames == 2, \
            f'Epoch 3: GIF 应有 2 帧, 实际 {n_frames}'

        # Epoch 5: 增量追加第 3 帧
        _make_epoch_img(vis_dir, 5, 0, 'img_A', (0, 0, 255))
        hook._generate_gifs(vis_dir, 5)
        n_frames = _count_gif_frames(gif_path)
        assert n_frames == 3, \
            f'Epoch 5: GIF 应有 3 帧 (增量追加), 实际 {n_frames}'

        # Epoch 7: 增量追加第 4 帧
        _make_epoch_img(vis_dir, 7, 0, 'img_A', (255, 255, 0))
        hook._generate_gifs(vis_dir, 7)
        n_frames = _count_gif_frames(gif_path)
        assert n_frames == 4, \
            f'Epoch 7: GIF 应有 4 帧 (增量追加), 实际 {n_frames}'

        # 验证 duration: 前 3 帧 500ms，末帧 2000ms
        durations = _get_gif_frame_durations(gif_path)
        assert len(durations) == 4, \
            f'duration 数量应为 4, 实际 {len(durations)}'
        for i in range(3):
            assert durations[i] == 500, \
                f'第 {i} 帧 duration 应为 500ms, 实际 {durations[i]}'
        assert durations[3] == 2000, \
            f'末帧 duration 应为 2000ms, 实际 {durations[3]}'

    print('test_incremental_gif_append PASSED')


def test_incremental_gif_multiple_images():
    """测试: 多张图各自独立增量追加。"""
    from hooks import PredictionVisHook

    hook = PredictionVisHook(num_images=4)

    with tempfile.TemporaryDirectory() as tmpdir:
        vis_dir = Path(tmpdir)
        gif_dir = vis_dir / 'gifs'

        # Epoch 1: 2 张图各 1 帧
        _make_epoch_img(vis_dir, 1, 0, 'img_A', (255, 0, 0))
        _make_epoch_img(vis_dir, 1, 1, 'img_B', (0, 255, 0))
        hook._generate_gifs(vis_dir, 1)
        assert not (gif_dir / 'img_A.gif').exists()
        assert not (gif_dir / 'img_B.gif').exists()

        # Epoch 3: 各追加 1 帧
        _make_epoch_img(vis_dir, 3, 0, 'img_A', (0, 0, 255))
        _make_epoch_img(vis_dir, 3, 1, 'img_B', (255, 255, 0))
        hook._generate_gifs(vis_dir, 3)
        assert _count_gif_frames(gif_dir / 'img_A.gif') == 2
        assert _count_gif_frames(gif_dir / 'img_B.gif') == 2

        # Epoch 5: 只 img_A 有新数据
        _make_epoch_img(vis_dir, 5, 0, 'img_A', (128, 128, 0))
        hook._generate_gifs(vis_dir, 5)
        assert _count_gif_frames(gif_dir / 'img_A.gif') == 3
        assert _count_gif_frames(gif_dir / 'img_B.gif') == 2  # 无新帧

    print('test_incremental_gif_multiple_images PASSED')


def test_incremental_gif_no_duplicate_epochs():
    """测试: 重复调用 _generate_gifs 不会重复追加已有帧。"""
    from hooks import PredictionVisHook

    hook = PredictionVisHook(num_images=2)

    with tempfile.TemporaryDirectory() as tmpdir:
        vis_dir = Path(tmpdir)
        gif_dir = vis_dir / 'gifs'

        # Epoch 1 + 3
        _make_epoch_img(vis_dir, 1, 0, 'img_A', (255, 0, 0))
        _make_epoch_img(vis_dir, 3, 0, 'img_A', (0, 255, 0))
        hook._generate_gifs(vis_dir, 3)
        assert _count_gif_frames(gif_dir / 'img_A.gif') == 2

        # 重复调用 (模拟意外重复)
        hook._generate_gifs(vis_dir, 3)
        n_frames = _count_gif_frames(gif_dir / 'img_A.gif')
        assert n_frames == 2, \
            f'重复调用不应增加帧数, 实际 {n_frames}'

    print('test_incremental_gif_no_duplicate_epochs PASSED')


def test_incremental_gif_duration_update():
    """测试: 追加新帧后，原末帧 duration 从 2000ms 变为 500ms，新末帧为 2000ms。"""
    from hooks import PredictionVisHook

    hook = PredictionVisHook(num_images=2)

    with tempfile.TemporaryDirectory() as tmpdir:
        vis_dir = Path(tmpdir)
        gif_dir = vis_dir / 'gifs'

        _make_epoch_img(vis_dir, 1, 0, 'img_A', (255, 0, 0))
        _make_epoch_img(vis_dir, 3, 0, 'img_A', (0, 255, 0))
        hook._generate_gifs(vis_dir, 3)

        # 2 帧: 第 1 帧 500ms, 第 2 帧 2000ms
        durations = _get_gif_frame_durations(gif_dir / 'img_A.gif')
        assert durations[0] == 500, f'第 1 帧应为 500ms, 实际 {durations[0]}'
        assert durations[1] == 2000, f'第 2 帧 (末帧) 应为 2000ms, 实际 {durations[1]}'

        # 追加第 3 帧
        _make_epoch_img(vis_dir, 5, 0, 'img_A', (0, 0, 255))
        hook._generate_gifs(vis_dir, 5)

        # 3 帧: 前 2 帧 500ms, 第 3 帧 2000ms
        durations = _get_gif_frame_durations(gif_dir / 'img_A.gif')
        assert durations[0] == 500, f'第 1 帧应为 500ms, 实际 {durations[0]}'
        assert durations[1] == 500, f'第 2 帧应为 500ms (不再是末帧), 实际 {durations[1]}'
        assert durations[2] == 2000, f'第 3 帧 (新末帧) 应为 2000ms, 实际 {durations[2]}'

    print('test_incremental_gif_duration_update PASSED')


def test_incremental_gif_preserves_deleted_source():
    """测试: 旧 epoch 图片被删除后，增量 GIF 仍保留已有帧。

    这是增量 vs 全量重建的核心行为差异:
    - 全量重建: 重新扫描 vis_dir，找不到已删除的 epoch 图片 → 丢帧
    - 增量追加: 只追加新帧，旧帧已在 GIF 中 → 不丢帧
    """
    from hooks import PredictionVisHook

    hook = PredictionVisHook(num_images=2)

    with tempfile.TemporaryDirectory() as tmpdir:
        vis_dir = Path(tmpdir)
        gif_dir = vis_dir / 'gifs'

        # Epoch 1 + 3: 生成 2 帧 GIF
        f1 = _make_epoch_img(vis_dir, 1, 0, 'img_A', (255, 0, 0))
        f3 = _make_epoch_img(vis_dir, 3, 0, 'img_A', (0, 255, 0))
        hook._generate_gifs(vis_dir, 3)
        assert _count_gif_frames(gif_dir / 'img_A.gif') == 2

        # 删除 epoch 1 的源图片 (模拟磁盘清理)
        (vis_dir / f1).unlink()

        # Epoch 5: 追加新帧
        _make_epoch_img(vis_dir, 5, 0, 'img_A', (0, 0, 255))
        hook._generate_gifs(vis_dir, 5)

        # 增量追加: GIF 应有 3 帧 (epoch 1 的帧仍在 GIF 中)
        n_frames = _count_gif_frames(gif_dir / 'img_A.gif')
        assert n_frames == 3, \
            f'删除源图后 GIF 应保留 3 帧, 实际 {n_frames} (全量重建会丢帧)'

    print('test_incremental_gif_preserves_deleted_source PASSED')


def test_incremental_gif_metadata_tracking():
    """测试: 通过元数据文件追踪已处理的 epoch，避免重复追加。"""
    from hooks import PredictionVisHook

    hook = PredictionVisHook(num_images=2)

    with tempfile.TemporaryDirectory() as tmpdir:
        vis_dir = Path(tmpdir)
        gif_dir = vis_dir / 'gifs'

        _make_epoch_img(vis_dir, 1, 0, 'img_A', (255, 0, 0))
        _make_epoch_img(vis_dir, 3, 0, 'img_A', (0, 255, 0))
        hook._generate_gifs(vis_dir, 3)

        # 元数据文件应记录已处理的 epoch
        meta_path = gif_dir / 'img_A.json'
        assert meta_path.exists(), \
            f'元数据文件应存在: {meta_path}'

        import json
        with open(meta_path) as f:
            meta = json.load(f)
        assert 'processed_epochs' in meta, \
            f'元数据应包含 processed_epochs 字段'
        assert set(meta['processed_epochs']) == {1, 3}, \
            f'已处理 epoch 应为 [1, 3], 实际 {meta["processed_epochs"]}'

        # 追加 epoch 5
        _make_epoch_img(vis_dir, 5, 0, 'img_A', (0, 0, 255))
        hook._generate_gifs(vis_dir, 5)

        with open(meta_path) as f:
            meta = json.load(f)
        assert set(meta['processed_epochs']) == {1, 3, 5}, \
            f'已处理 epoch 应为 [1, 3, 5], 实际 {meta["processed_epochs"]}'

    print('test_incremental_gif_metadata_tracking PASSED')


# ============================================================
# main
# ============================================================
if __name__ == '__main__':
    print('=' * 60)
    print('PredictionVisHook Incremental GIF Tests (Red Phase)')
    print('=' * 60)

    test_incremental_gif_append()
    test_incremental_gif_multiple_images()
    test_incremental_gif_no_duplicate_epochs()
    test_incremental_gif_duration_update()
    test_incremental_gif_preserves_deleted_source()
    test_incremental_gif_metadata_tracking()

    print('\n' + '=' * 60)
    print('ALL PASSED')
    print('=' * 60)
