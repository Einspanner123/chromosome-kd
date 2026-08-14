"""Regression tests for training and validation RNG control."""

import random

import numpy as np
import pytest
import torch
from mmengine.config import Config

from experiments.runners.reproducibility import (
    FixedValidationSeedHook,
    apply_training_seed,
)


def _draw_rng_values():
    return random.random(), np.random.rand(), torch.rand(4)


def _seed_cpu_rngs(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def test_cli_training_seed_overrides_config_and_preserves_policy():
    cfg = Config(
        dict(randomness=dict(seed=5, deterministic=True, diff_rank_seed=True))
    )

    apply_training_seed(cfg, 123)

    assert cfg.randomness == dict(
        seed=123, deterministic=True, diff_rank_seed=True
    )


def test_cli_training_seed_populates_missing_randomness():
    cfg = Config(dict())

    apply_training_seed(cfg, 789)

    assert cfg.randomness == dict(
        seed=789, deterministic=False, diff_rank_seed=False
    )


@pytest.mark.parametrize('seed', [-1, 2**32])
def test_invalid_training_seed_is_rejected(seed):
    with pytest.raises(ValueError, match='--seed'):
        apply_training_seed(Config(dict()), seed)


def test_validation_seed_replays_identical_inference_rng():
    hook = FixedValidationSeedHook(42)

    _seed_cpu_rngs(1)
    hook.before_val(None)
    first = _draw_rng_values()
    hook.after_val(None)

    _seed_cpu_rngs(999)
    hook.before_val(None)
    second = _draw_rng_values()
    hook.after_val(None)

    assert first[0] == second[0]
    assert first[1] == second[1]
    torch.testing.assert_close(first[2], second[2], rtol=0, atol=0)


def test_validation_rng_scope_restores_training_streams():
    hook = FixedValidationSeedHook(42)
    original_deterministic = torch.backends.cudnn.deterministic
    original_benchmark = torch.backends.cudnn.benchmark

    try:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        _seed_cpu_rngs(7)
        expected = _draw_rng_values()

        _seed_cpu_rngs(7)
        hook.before_val(None)
        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False
        _draw_rng_values()
        hook.after_val(None)
        actual = _draw_rng_values()

        assert actual[0] == expected[0]
        assert actual[1] == expected[1]
        torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)
        assert torch.backends.cudnn.deterministic is False
        assert torch.backends.cudnn.benchmark is True
    finally:
        torch.backends.cudnn.deterministic = original_deterministic
        torch.backends.cudnn.benchmark = original_benchmark


def test_validation_rng_scope_rejects_unbalanced_calls():
    hook = FixedValidationSeedHook(42)

    with pytest.raises(RuntimeError, match='not initialized'):
        hook.after_val(None)

    hook.before_val(None)
    with pytest.raises(RuntimeError, match='Nested validation'):
        hook.before_val(None)
    hook.after_val(None)
