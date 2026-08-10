"""Reproducibility controls shared by LDMDet training runners."""

import random

import numpy as np
import torch
from mmengine.hooks import Hook


def _validate_seed(seed: int, name: str) -> None:
    """Validate seeds accepted by Python, NumPy, PyTorch and MMEngine."""
    if not 0 <= seed < 2**32:
        raise ValueError(f'{name} must be in [0, 2**32), got {seed}')


def apply_training_seed(cfg, seed):
    """Make ``--seed`` authoritative for MMEngine runner randomness.

    Preserve config-specific determinism/rank settings while overriding only
    the seed. Setting an environment variable is insufficient because
    MMEngine reads ``cfg.randomness`` when constructing ``Runner``.
    """
    if seed is None:
        return
    _validate_seed(seed, '--seed')
    randomness = dict(cfg.get('randomness', {}))
    randomness['seed'] = seed
    randomness.setdefault('deterministic', False)
    randomness.setdefault('diff_rank_seed', False)
    cfg.randomness = randomness


class FixedValidationSeedHook(Hook):
    """Use fixed validation RNG without perturbing stochastic training.

    LDMDet samples initial proposals and box-renewal replacements with
    ``torch.randn`` at inference time. Saving and restoring all RNG streams
    makes validation comparable across epochs while keeping the stochastic
    training trajectory exactly as if validation had consumed no randomness.
    """

    priority = 'VERY_HIGH'

    def __init__(self, seed: int = 42):
        _validate_seed(seed, '--val-seed')
        self.seed = seed
        self._saved_state = None

    @staticmethod
    def _seed_all(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def before_val(self, runner) -> None:
        if self._saved_state is not None:
            raise RuntimeError('Nested validation RNG scopes are not supported')
        self._saved_state = {
            'python': random.getstate(),
            'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': (
                torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else None
            ),
        }
        self._seed_all(self.seed)

    def after_val(self, runner) -> None:
        if self._saved_state is None:
            raise RuntimeError('Validation RNG scope was not initialized')
        random.setstate(self._saved_state['python'])
        np.random.set_state(self._saved_state['numpy'])
        torch.set_rng_state(self._saved_state['torch'])
        if self._saved_state['cuda'] is not None:
            torch.cuda.set_rng_state_all(self._saved_state['cuda'])
        self._saved_state = None
