#!/usr/bin/env python3
"""Bind one validation-selected checkpoint to a preregistered training run."""

from __future__ import annotations
import argparse
from pathlib import Path

from tools.experiments.registry import register_selected_checkpoint


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-run-id', required=True)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--selection-source', required=True)
    parser.add_argument('--selection-metric', default='coco/bbox_mAP')
    parser.add_argument('--selection-value', type=float)
    args = parser.parse_args()
    digest = register_selected_checkpoint(
        args.train_run_id,
        args.checkpoint,
        selection_source=args.selection_source,
        selection_metric=args.selection_metric,
        selection_value=args.selection_value,
    )
    print(f'train_run_id={args.train_run_id} checkpoint_sha256={digest}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
