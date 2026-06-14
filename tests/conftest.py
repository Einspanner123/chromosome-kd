"""测试共享 fixtures"""

import pytest
import torch


@pytest.fixture
def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def dummy_img_meta():
    from ldmdet.data.structures import ImageMeta
    return ImageMeta(img_shape=(800, 1216))


@pytest.fixture
def dummy_noise():
    return torch.randn(10, 4)


@pytest.fixture
def dummy_gt():
    return torch.randn(3, 4)


@pytest.fixture
def dummy_labels():
    return torch.tensor([0, 1, 2], dtype=torch.long)


@pytest.fixture
def cost_matrix():
    """5x3 cost matrix for coupling tests"""
    return torch.tensor([
        [1.0, 5.0, 3.0],
        [4.0, 2.0, 6.0],
        [3.0, 4.0, 1.0],
        [5.0, 1.0, 4.0],
        [2.0, 3.0, 5.0],
    ])
