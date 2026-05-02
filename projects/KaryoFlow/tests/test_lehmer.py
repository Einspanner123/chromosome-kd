"""Lehmer code 单元测试"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from karyoflow.lehmer import (
    batch_lehmer_to_permutation,
    batch_permutation_to_lehmer,
    lehmer_to_permutation,
    permutation_to_lehmer,
    validate_lehmer,
    validate_permutation,
)


def test_roundtrip_small():
    """小排列的往返测试"""
    perm = [2, 0, 3, 1]
    lehmer = permutation_to_lehmer(perm)
    assert lehmer == [2, 0, 1, 0], f"Expected [2,0,1,0], got {lehmer}"
    recovered = lehmer_to_permutation(lehmer)
    assert recovered == perm, f"Expected {perm}, got {recovered}"
    print("  test_roundtrip_small PASSED")


def test_identity():
    """恒等排列"""
    perm = list(range(10))
    lehmer = permutation_to_lehmer(perm)
    assert lehmer == [0] * 10
    assert lehmer_to_permutation(lehmer) == perm
    print("  test_identity PASSED")


def test_reverse():
    """逆序排列"""
    perm = [3, 2, 1, 0]
    lehmer = permutation_to_lehmer(perm)
    assert lehmer == [3, 2, 1, 0]
    assert lehmer_to_permutation(lehmer) == perm
    print("  test_reverse PASSED")


def test_roundtrip_46():
    """46 元素排列往返"""
    import random
    random.seed(42)
    perm = list(range(46))
    random.shuffle(perm)

    lehmer = permutation_to_lehmer(perm)
    assert validate_lehmer(lehmer)
    recovered = lehmer_to_permutation(lehmer)
    assert recovered == perm
    print("  test_roundtrip_46 PASSED")


def test_batch_roundtrip():
    """批量 Tensor 往返"""
    import random
    random.seed(42)

    B, N = 4, 46
    perms = []
    for _ in range(B):
        p = list(range(N))
        random.shuffle(p)
        perms.append(p)

    perm_tensor = torch.tensor(perms)
    lehmer_tensor = batch_permutation_to_lehmer(perm_tensor)
    recovered = batch_lehmer_to_permutation(lehmer_tensor)

    assert (recovered == perm_tensor).all(), "Batch roundtrip failed"
    print("  test_batch_roundtrip PASSED")


def test_validate():
    assert validate_permutation([0, 1, 2, 3])
    assert not validate_permutation([0, 1, 1, 3])
    assert validate_lehmer([2, 0, 1, 0])
    assert not validate_lehmer([2, 0, 4, 0])  # l_2 max = 1
    print("  test_validate PASSED")


if __name__ == "__main__":
    print("Running Lehmer tests...")
    test_roundtrip_small()
    test_identity()
    test_reverse()
    test_roundtrip_46()
    test_batch_roundtrip()
    test_validate()
    print("All Lehmer tests PASSED!")
