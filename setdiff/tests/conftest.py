"""共享测试 fixtures 和工具函数"""

import torch
import pytest


@pytest.fixture(autouse=True)
def deterministic_seed():
    """每个测试前设置确定性种子"""
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    yield
