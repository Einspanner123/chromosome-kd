#!/usr/bin/env python3
"""Build the authoritative v2 paper experiment route matrix.

The legacy ledger supplies only the stable experiment scope and descriptive
fields.  All executable configuration identities, evidence links, and live
statuses are replaced here with v2-reviewed values.
"""

from __future__ import annotations
import csv
import hashlib
import json
import os
import re
import sqlite3
from collections import Counter
from copy import deepcopy
from pathlib import Path

import yaml

from tools.experiments.matrix import (
    default_work_dir,
    resolve_config,
    scientific_hash,
)

ROOT = Path(__file__).resolve().parents[2]
LEGACY = (
    ROOT / 'tools/experiment_db/manifests/master_experiment_ledger_v1.json'
)
OUTPUT = ROOT / 'experiments/manifests/paper_experiment_route_matrix.yaml'
CSV_OUTPUT = (
    ROOT / 'tools/experiment_db/exports/paper_experiment_route_matrix.csv'
)
DOC_OUTPUT = ROOT / 'docs/experiments/PAPER_EXPERIMENT_ROUTE_MATRIX.md'
DB = ROOT / 'tools/experiment_db/experiments.db'
CLAIM_MANIFEST = ROOT / 'experiments/manifests/paper_claim_manifest.yaml'
SCHEDULER_STATE = ROOT / 'work_dirs/v2/scheduler/state.json'

D1_MATRIX = 'experiments/configs/matrices/d1_inhouse1700.yaml'
D2_MATRIX = 'experiments/configs/matrices/d2_taichung.yaml'
D2_HISTORY_MATRIX = 'experiments/configs/matrices/d2_taichung_history.yaml'
D1_GENERATION_MATRIX = (
    'experiments/configs/matrices/d1_inhouse1700_generation_ablation.yaml'
)
D2_GENERATION_MATRIX = (
    'experiments/configs/matrices/d2_taichung_generation_ablation.yaml'
)
D2_SOTA_COMPLETION_MATRIX = (
    'experiments/configs/matrices/d2_taichung_sota_completion.yaml'
)
D1_DISTILL_MATRIX = (
    'experiments/configs/matrices/d1_inhouse1700_head_distill.yaml'
)
D2_DISTILL_MATRIX = (
    'experiments/configs/matrices/d2_taichung_head_distill_canonical.yaml'
)
D2_CANONICAL_DATASET = (
    'D2_TAICHUNG5000_V2'  # legacy builder token; normalized before export
)

METHODS = {
    'diffusiondet': 'experiments/configs/methods/diffusiondet_ddpm.py',
    'rf_heun': 'experiments/configs/methods/rf_heun.py',
    'karyoflow': 'experiments/configs/methods/karyoflow.py',
    'karyoflow_lqcr': 'experiments/configs/methods/karyoflow_lqcr.py',
    'dino_r50': 'experiments/configs/methods/dino_r50.py',
    'rtmdet_l': 'experiments/configs/methods/rtmdet_l.py',
    'cascade_rcnn_r50': 'experiments/configs/methods/cascade_rcnn_r50.py',
    'yolox_s': 'experiments/configs/methods/yolox_s.py',
    'strict_g0': 'experiments/configs/methods/strict_g0_ddpm_linear_scaleshift.py',
    'strict_g1': 'experiments/configs/methods/strict_g1_rf_linear_scaleshift.py',
    'strict_g2': 'experiments/configs/methods/strict_g2_rf_shifted_scaleshift.py',
    'strict_g3': 'experiments/configs/methods/strict_g3_rf_shifted_adaln_zero.py',
    'h3_distill': 'experiments/configs/methods/karyoflow_h3_distill.py',
    'ot_h3_distill': 'experiments/configs/methods/karyoflow_ot_h3_distill.py',
}

D2_LEGACY = {
    'diffusiondet': (
        'experiments/configs/methods/diffusiondet_ddpm_legacy.py',
        'experiments/configs/matrices/d2_diffusiondet_history.yaml',
        'tools/experiment_db/compatibility_reports/d2_sota_diffusiondet.json',
        'STRUCTURAL_ONLY',
    ),
    'dino_r50': (
        'experiments/configs/methods/dino_r50_legacy.py',
        'experiments/configs/matrices/d2_dino_rtmdet_history.yaml',
        'tools/experiment_db/compatibility_reports/d2_sota_dino_r50.json',
        'EXACT',
    ),
    'rtmdet_l': (
        'experiments/configs/methods/rtmdet_l_legacy.py',
        'experiments/configs/matrices/d2_dino_rtmdet_history.yaml',
        'tools/experiment_db/compatibility_reports/d2_sota_rtmdet_l.json',
        'EXACT',
    ),
    'cascade_rcnn_r50': (
        'experiments/configs/methods/cascade_rcnn_r50_legacy.py',
        'experiments/configs/matrices/d2_cascade_history.yaml',
        'tools/experiment_db/compatibility_reports/d2_sota_cascade_rcnn.json',
        'EXACT',
    ),
    'yolox_s': (
        'experiments/configs/methods/yolox_s_legacy.py',
        'experiments/configs/matrices/d2_yolox_history.yaml',
        'tools/experiment_db/compatibility_reports/d2_sota_yolox_s.json',
        'EXACT',
    ),
}

STATUS_OVERRIDE = {
    'D1I.SOTA.karyoflow': 'COMPLETED_CENTRAL_TEST_VERIFIED',
    'D1I.SOTA.diffusiondet': 'TEST_QUEUED_1_OF_3_COMPLETE',
    'D1I.SOTA.dino_r50': 'PLANNED',
    'D1I.SOTA.rtmdet_l': 'PLANNED',
    'D1I.SOTA.cascade_rcnn_r50': 'PLANNED',
    'D1I.SOTA.yolox_s': 'PLANNED',
    'D1I.SOTA.karyoflow_lqcr': 'COMPLETED_CENTRAL_TEST_VERIFIED',
    'D1I.ABL.G0': 'TEST_QUEUED_1_OF_3_COMPLETE',
    'D1I.ABL.G1': 'TRAIN3_COMPLETE_TEST_QUEUED',
    'D1I.ABL.G2': 'TRAIN3_COMPLETE_TEST_QUEUED',
    'D1I.ABL.G3': 'TRAIN3_COMPLETE_SYNC_PENDING_TEST_QUEUED',
    'D1I.INF.solver_steps': 'QUEUED_SERIAL_A4000',
    'D1I.INF.topk_renewal': 'QUEUED_SERIAL_A4000',
    'D1I.DEC.beta_val': 'BLOCKED_PARENT',
    'D1I.DEC.strict_subsets': 'BLOCKED_PREDICTIONS',
    'D1I.DEP.distill_h3': 'BLOCKED_PARENT',
    'D1I.DEP.GACS': 'BLOCKED_PARENT',
    'D1I.DEP.speed': 'BLOCKED_CHECKPOINTS',
    'D2.SOTA.diffusiondet.missing_train2': 'PLANNED',
    'D2.SOTA.dino_r50.missing_train2': 'PLANNED',
    'D2.SOTA.rtmdet_l.missing_train2': 'PLANNED',
    'D2.SOTA.cascade_rcnn_r50.missing_train2': 'PLANNED',
    'D2.SOTA.yolox_s.missing_train2': 'PLANNED',
    'D2.ABL.strict.G0': 'PLANNED',
    'D2.ABL.strict.G1': 'PLANNED',
    'D2.ABL.strict.G2': 'PLANNED',
    'D2.ABL.strict.G3': 'PLANNED',
    'D2.INF.solver_steps.train3': 'PLANNED',
    'D2.INF.topk_renewal.train3': 'PLANNED',
    'D2.DEP.distill_h3.existing': 'COMPLETED_EVIDENCE_ONLY',
    'D2.DEP.distill_h3.train3': 'PLANNED',
}

PROTOCOLS = {
    'D1I.INF.solver_steps': 'experiments/configs/ablations/d1_solver_test.yaml',
    'D1I.INF.topk_renewal': 'experiments/configs/ablations/d1_topk_renewal_test.yaml',
    'D1I.DEC.beta_val': 'experiments/configs/ablations/d1_lqcr_beta_val.yaml',
    'D1I.DEC.strict_subsets': 'tools/experiment_db/protocols/d2_difficulty_strata.json',
    'D1I.DEP.speed': 'experiments/configs/deployment/a6000_speed_protocol.yaml',
    'D1I.DEP.GACS': 'experiments/configs/deployment/gacs_dynamic_policy.yaml',
    'D1I.DEP.distill_h3': 'experiments/configs/deployment/h3_distillation_protocol.yaml',
    'D2.INF.solver_steps.fixed1': 'experiments/configs/ablations/d2_solver_test.yaml',
    'D2.INF.solver_steps.train3': 'experiments/configs/ablations/d2_solver_train3_test.yaml',
    'D2.INF.topk_renewal.fixed1': 'experiments/configs/ablations/d2_topk_renewal_test.yaml',
    'D2.INF.topk_renewal.train3': 'experiments/configs/ablations/d2_topk_renewal_train3_test.yaml',
    'D2.DEC.beta_test.fixed1': 'experiments/configs/ablations/d2_lqcr_beta_test.yaml',
    'D2.DEC.strict_subsets': 'tools/experiment_db/protocols/d2_difficulty_strata.json',
    'D2.DEP.speed': 'experiments/configs/deployment/a6000_speed_protocol.yaml',
    'D2.DEP.GACS': 'experiments/configs/deployment/gacs_dynamic_policy.yaml',
    'D2.DEP.distill_h3.train3': 'experiments/configs/deployment/h3_distillation_protocol.yaml',
}

SERVER_PLAN_OVERRIDE = {
    'D1I.ABL.G0': '42=ross:A6000:0;123=workstation:A5000:0;789=workstation:A4000:1',
    'D1I.ABL.G1': '42=ross:A6000:0;123=workstation:A5000:0;789=workstation:A5000:0',
    'D1I.ABL.G2': '42=ross:A6000:0;123=workstation:A5000:0;789=workstation:A5000:0',
    'D1I.ABL.G3': '42=ross:A6000:0;123=ross:A6000:0;789=workstation:A5000:0',
}


CONFIG_STATE_OVERRIDE = {
    'D1I.SOTA.dino_r50': 'READY',
    'D1I.SOTA.rtmdet_l': 'READY',
    'D1I.SOTA.cascade_rcnn_r50': 'READY',
    'D1I.SOTA.yolox_s': 'READY',
    'D1I.ABL.G0': 'READY',
    'D1I.ABL.G1': 'READY',
    'D1I.ABL.G2': 'READY',
    'D1I.DEP.distill_h3': 'READY_PARENT_PENDING',
    'D1I.DEP.GACS': 'PROTOCOL_READY_PARENT_PENDING',
    'D1I.INF.solver_steps': 'PROTOCOL_READY',
    'D1I.INF.topk_renewal': 'PROTOCOL_READY',
    'D1I.DEC.beta_val': 'PROTOCOL_READY_PARENT_PENDING',
    'D1I.DEC.strict_subsets': 'DATASET_ADAPTATION_REQUIRED',
    'D2.ABL.strict.G0': 'READY',
    'D2.ABL.strict.G1': 'READY',
    'D2.ABL.strict.G2': 'READY',
    'D2.DEP.distill_h3.train3': 'READY',
    'D2.DEP.GACS': 'PROTOCOL_READY',
    'D2.INF.solver_steps.train3': 'PROTOCOL_READY',
    'D2.INF.topk_renewal.train3': 'PROTOCOL_READY',
}

NOTES_OVERRIDE = {
    'D1I.SOTA.karyoflow': (
        'Canonical v2 seeds 42, 123, and 789 completed independent training '
        'and central held-out test evaluation (340 images; inference seed 42); '
        'all three immutable config digests were preserved during evaluation.'
    ),
    'D1I.SOTA.karyoflow_lqcr': (
        'All three parent-matched quality-head runs completed training, passed '
        'the final-only tensor audit (590 shared tensors unchanged; five '
        'quality-head tensors added), and completed central held-out test '
        'evaluation with immutable prediction and metric evidence.'
    ),
    'D1I.ABL.G0': (
        'All three independent G0 runs completed with distinct validation-best '
        'checkpoint SHA values. Seed 789 has verified test evidence; seeds 42 '
        'and 123 are queued for the same held-out-test protocol.'
    ),
    'D1I.ABL.G1': (
        'All three independent G1 runs completed training and are queued for '
        'serial held-out-test evaluation on the accuracy worker.'
    ),
    'D1I.ABL.G2': (
        'All three G2 training runs, validation-selected checkpoints, and '
        'completion artifacts are centrally registered and verified. Their '
        'three serial held-out-test evaluations remain queued.'
    ),
    'D1I.ABL.G3': (
        "All three G3 training processes completed successfully. Seed 789's "
        'validation-best checkpoint is transferring to the central evidence '
        'store over an intermittent connection; its held-out test remains '
        'dependency-gated until import. A brief epoch-1 Ross fallback is excluded '
        'from evidence. DPM++ remains a separate inference comparison.'
    ),
    'D1I.INF.solver_steps': (
        'All 36 fixed-parent test evaluations are queued serially: three '
        'independent KaryoFlow parents by Euler/Heun/DPM++ and one to four steps.'
    ),
    'D1I.INF.topk_renewal': (
        'All 30 fixed-parent test evaluations are queued serially: three '
        'independent KaryoFlow parents by five candidate budgets and renewal on/off.'
    ),
    'D2.ABL.historical_chain': 'Historical foundation comparison only; never interpret adjacent rows as isolated cumulative effects.',
    'D2.DEP.distill_h3.existing': 'Inference identity is EXACT; the archived historical distillation training implementation is not executable in cleaned ldmdet.',
    'D2.DEP.speed': 'Historical latency is valid under its recorded protocol but PARTIAL against the strict rerun protocol.',
}


def _scheduler_test_progress() -> dict[str, dict[str, int]]:
    """Summarize only postprocess-verified D1 test-completion tasks.

    A test contributes to ``completed`` only after evidence validation, database
    import, and the scheduler's global database audit have all succeeded.  This
    keeps the route matrix behind (never ahead of) the evidence database.
    """
    groups = {
        'G0': {'completed': 1, 'running': 0, 'queued': 0, 'total': 3},
        'G1': {'completed': 0, 'running': 0, 'queued': 0, 'total': 3},
        'G2': {'completed': 0, 'running': 0, 'queued': 0, 'total': 3},
        'G3': {'completed': 0, 'running': 0, 'queued': 0, 'total': 3},
        'solver': {'completed': 0, 'running': 0, 'queued': 0, 'total': 36},
        'topk': {'completed': 0, 'running': 0, 'queued': 0, 'total': 30},
    }
    if not SCHEDULER_STATE.exists():
        return groups
    try:
        state = json.loads(SCHEDULER_STATE.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return groups

    def group_for(task_id: str) -> str | None:
        if not task_id.endswith('-ross') or not task_id.startswith('d1-test-'):
            return None
        if task_id.startswith('d1-test-solver-'):
            return 'solver'
        if task_id.startswith('d1-test-topk-'):
            return 'topk'
        match = re.match(r'd1-test-g([0-3])-seed\d+-ross$', task_id)
        return f'G{match.group(1)}' if match else None

    for queue_name, counter_name in (
        ('pending', 'queued'),
        ('running', 'running'),
    ):
        for task in state.get(queue_name, []):
            group = group_for(task.get('task_id', ''))
            if group:
                groups[group][counter_name] += 1
    for task in state.get('completed', []):
        group = group_for(task.get('task_id', ''))
        if (
            group
            and task.get('process_status') == 'succeeded'
            and task.get('postprocess', {}).get('status') == 'succeeded'
        ):
            groups[group]['completed'] += 1
    completing = os.environ.get('KARYOFLOW_COMPLETING_TASK_ID', '')
    completing_group = group_for(completing)
    already_counted = any(
        task.get('task_id') == completing
        and task.get('process_status') == 'succeeded'
        and task.get('postprocess', {}).get('status') == 'succeeded'
        for task in state.get('completed', [])
    )
    if completing_group and not already_counted:
        groups[completing_group]['completed'] += 1
    return groups


def _scheduler_d2_sota_progress() -> dict[str, dict[str, int]]:
    """Summarize the two new training runs completing each D2 train3 row."""
    task_slugs = {
        'diffusiondet': 'diffusiondet',
        'dino_r50': 'dino',
        'rtmdet_l': 'rtmdet',
        'cascade_rcnn_r50': 'cascade',
        'yolox_s': 'yolox',
    }
    groups = {
        variant: {'completed': 0, 'running': 0, 'queued': 0, 'total': 2}
        for variant in task_slugs
    }
    if not SCHEDULER_STATE.exists():
        return groups
    try:
        state = json.loads(SCHEDULER_STATE.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return groups

    def variant_for(task_id: str) -> str | None:
        for variant, slug in task_slugs.items():
            if task_id.startswith(f'd2-sota-{slug}-seed'):
                return variant
        return None

    for queue_name, counter_name in (
        ('pending', 'queued'),
        ('running', 'running'),
    ):
        for task in state.get(queue_name, []):
            variant = variant_for(task.get('task_id', ''))
            if variant:
                groups[variant][counter_name] += 1
    for task in state.get('completed', []):
        variant = variant_for(task.get('task_id', ''))
        if (
            variant
            and task.get('process_status') == 'succeeded'
            and task.get('postprocess', {}).get('status') == 'succeeded'
        ):
            groups[variant]['completed'] += 1
    return groups


def _scheduler_d1_sota_progress() -> dict[str, dict[str, int]]:
    """Summarize the three scheduled D1 training runs for each SOTA baseline."""
    task_slugs = {
        'dino_r50': 'dino',
        'rtmdet_l': 'rtmdet',
        'cascade_rcnn_r50': 'cascade',
        'yolox_s': 'yolox',
    }
    groups = {
        variant: {'completed': 0, 'running': 0, 'queued': 0, 'total': 3}
        for variant in task_slugs
    }
    if not SCHEDULER_STATE.exists():
        return groups
    try:
        state = json.loads(SCHEDULER_STATE.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return groups

    def variant_for(task_id: str) -> str | None:
        for variant, slug in task_slugs.items():
            if task_id.startswith(f'd1-sota-{slug}-seed'):
                return variant
        return None

    for queue_name, counter_name in (
        ('pending', 'queued'),
        ('running', 'running'),
    ):
        for task in state.get(queue_name, []):
            variant = variant_for(task.get('task_id', ''))
            if variant:
                groups[variant][counter_name] += 1
    for task in state.get('completed', []):
        variant = variant_for(task.get('task_id', ''))
        if (
            variant
            and task.get('process_status') == 'succeeded'
            and task.get('postprocess', {}).get('status') == 'succeeded'
        ):
            groups[variant]['completed'] += 1
    return groups


def _d1_sota_status(progress: dict[str, int]) -> str:
    done, total = progress['completed'], progress['total']
    if done >= total:
        return 'TRAIN3_COMPLETE_TEST_PENDING'
    if progress['running']:
        return f'TRAINING_ROSS_{done}_OF_{total}_COMPLETE'
    if progress['queued']:
        return f'QUEUED_ROSS_{done}_OF_{total}_COMPLETE'
    return 'PLANNED'


def _d2_sota_status(progress: dict[str, int]) -> str:
    done, total = progress['completed'], progress['total']
    if done >= total:
        return 'TRAIN3_COMPLETE_TEST_PENDING'
    if progress['running']:
        return f'TRAINING_WORKSTATION_{done}_OF_{total}_COMPLETE'
    if progress['queued']:
        return f'QUEUED_WORKSTATION_{done}_OF_{total}_COMPLETE'
    return 'DEFERRED_UNTIL_D1_COMPLETE'


def _progress_status(progress: dict[str, int], *, train3: bool = False) -> str:
    done, total = progress['completed'], progress['total']
    if done >= total:
        return (
            'COMPLETED_TEST_VERIFIED'
            if total > 3
            else 'COMPLETED_TEST3_VERIFIED'
        )
    if progress['running']:
        prefix = 'TRAIN3_COMPLETE_TEST_RUNNING' if train3 else 'TEST_RUNNING'
    else:
        prefix = 'TRAIN3_COMPLETE_TEST_QUEUED' if train3 else 'TEST_QUEUED'
    return f'{prefix}_{done}_OF_{total}_COMPLETE'


def _apply_d1_test_progress(rows: list[dict]) -> None:
    """Overlay live test evidence progress on otherwise static route metadata."""
    progress = _scheduler_test_progress()
    by_id = {row['ledger_id']: row for row in rows}
    for stage in ('G0', 'G1', 'G2', 'G3'):
        row = by_id[f'D1I.ABL.{stage}']
        item = progress[stage]
        row['status'] = _progress_status(item, train3=stage != 'G0')
        row['notes'] = (
            f'All three independent {stage} training runs completed. '
            f'Held-out test evidence: {item["completed"]}/{item["total"]} passed '
            'evaluation, immutable evidence validation, database import, and '
            'global audit; remaining evaluations run serially on Ross A6000.'
        )
    # DiffusionDet is the G0 generation configuration and shares its evidence.
    by_id['D1I.SOTA.diffusiondet']['status'] = _progress_status(progress['G0'])
    by_id['D1I.SOTA.diffusiondet']['notes'] = by_id['D1I.ABL.G0']['notes']

    sota_progress = _scheduler_d1_sota_progress()
    for variant, item in sota_progress.items():
        row = by_id[f'D1I.SOTA.{variant}']
        row['status'] = _d1_sota_status(item)
        row['server_plan'] = (
            'three independent training runs serialized on Ross A6000 while '
            'workstation is unavailable; held-out tests follow each completed run'
        )
        row['notes'] = (
            f'Ross training completion: {item["completed"]}/{item["total"]}; '
            f'running={item["running"]}, queued={item["queued"]}. '
            'A run is counted complete only after training evidence finalization.'
        )

    for key, route_id, label in (
        ('solver', 'D1I.INF.solver_steps', 'solver/step'),
        ('topk', 'D1I.INF.topk_renewal', 'Top-K/renewal'),
    ):
        item = progress[key]
        if item['completed'] >= item['total']:
            status = 'COMPLETED_TEST_VERIFIED'
        elif item['running']:
            status = f'RUNNING_SERIAL_ROSS_{item["completed"]}_OF_{item["total"]}_COMPLETE'
        else:
            status = f'QUEUED_SERIAL_ROSS_{item["completed"]}_OF_{item["total"]}_COMPLETE'
        by_id[route_id]['status'] = status
        by_id[route_id]['notes'] = (
            f'{item["completed"]}/{item["total"]} fixed-parent {label} held-out '
            'test evaluations passed the evidence/database audit; remaining '
            'tasks are serialized on Ross A6000.'
        )


def _method_name(path: str) -> str:
    return Path(path).stem


def _resolved_training_metadata(row: dict) -> dict | None:
    """Resolve canonical training facts instead of copying prose summaries."""
    if row['execution_kind'] not in {'full_train', 'short_train'}:
        return None
    if not row.get('method_config') or not row.get('matrix_config'):
        return None
    if row['config_state'] not in {'READY', 'READY_PARENT_PENDING'}:
        return None
    seed_text = str(row.get('training_seeds', ''))
    seeds = [int(x) for x in seed_text.split(',') if x.strip().isdigit()]
    if not seeds:
        return None
    method_name = _method_name(row['method_config'])
    cfg, _ = resolve_config(row['matrix_config'], method_name, seeds[0])
    optimizer = cfg.optim_wrapper.optimizer
    work_dir = str(default_work_dir(cfg, seeds[0]).relative_to(ROOT))
    work_template = work_dir.replace(
        f'trainseed_{seeds[0]}', 'trainseed_{training_seed}'
    )
    return {
        'config_id': cfg.experiment.config_id,
        'method_id': cfg.experiment.method_id,
        'batch_size': int(cfg.train_dataloader.batch_size),
        'max_epochs': int(cfg.train_cfg.max_epochs),
        'optimizer': str(optimizer['type']),
        'base_lr': float(optimizer['lr']),
        'scientific_config_sha256': scientific_hash(cfg),
        'output_template': work_template,
    }


def _scientific_parameters(text: str, has_resolved_training: bool) -> str:
    """Drop stale operational claims when the resolver is authoritative."""
    if not has_resolved_training:
        return text
    text = re.sub(r';?\s*batch\s*=\s*\d+(?:\s*\([^)]*\))?', '', text)
    text = re.sub(r';?\s*\d+\s*epochs?', '', text)
    return re.sub(r';\s*;', ';', text).strip(' ;')


def _active_registered_runs(row: dict, resolved: dict) -> list[str]:
    """Return only active registrations with the exact scientific identity."""
    connection = sqlite3.connect(DB)
    try:
        records = connection.execute(
            'SELECT train_run_id FROM train_run_registry '
            'WHERE dataset_id=? AND method=? AND scientific_config_sha256=? '
            "AND status IN ('planned','running') ORDER BY training_seed",
            (
                row['dataset_id'],
                resolved['method_id'],
                resolved['scientific_config_sha256'],
            ),
        ).fetchall()
    finally:
        connection.close()
    return [record[0] for record in records]


def _upgrade_scientific_routes(rows: list[dict]) -> list[dict]:
    """Separate canonical paper routes from verified historical identities."""
    by_id = {row['ledger_id']: row for row in rows}

    legacy_parent = by_id['D2.SOTA.karyoflow_train3']
    legacy_parent['ledger_id'] = 'D2.HIST.karyoflow_ot_train3'
    legacy_parent['layer'] = 'Archive'
    legacy_parent['family'] = 'Verified historical detector'
    legacy_parent['paper_role'] = 'Historical OT-coupling reference only'
    legacy_parent['priority'] = 'ARCHIVE'

    legacy_lqcr = by_id['D2.SOTA.karyoflow_lqcr_train3']
    legacy_lqcr['ledger_id'] = 'D2.HIST.karyoflow_ot_lqcr_train3'
    legacy_lqcr['layer'] = 'Archive'
    legacy_lqcr['family'] = 'Verified historical decision model'
    legacy_lqcr['paper_role'] = 'Historical OT-parent LQCR reference only'
    legacy_lqcr['priority'] = 'ARCHIVE'
    legacy_lqcr['parent_ledger_id'] = legacy_parent['ledger_id']

    for variant in D2_LEGACY:
        historical = by_id[f'D2.SOTA.{variant}.existing']
        historical['layer'] = 'Archive'
        historical['family'] = 'Historical detector point estimate'
        historical['priority'] = 'ARCHIVE'
        historical['paper_role'] = 'Publisher-split historical context only'
        canonical = by_id[f'D2.SOTA.{variant}.missing_train2']
        canonical.update(
            ledger_id=f'D2.SOTA.{variant}.canonical_train3',
            dataset_id=D2_CANONICAL_DATASET,
            run_count=3,
            training_seeds='42,123,789',
            family='SOTA',
            status='PLANNED',
            priority='P0',
            paper_role='Canonical leakage-repaired D2 baseline',
            acceptance_gate='three independently trained models; six held-out-test metrics',
            parent_ledger_id='',
            notes='Historical publisher-split checkpoints are not reused.',
        )

    canonical_parent = deepcopy(legacy_parent)
    canonical_parent.update(
        ledger_id='D2.SOTA.karyoflow_canonical_train3',
        dataset_id=D2_CANONICAL_DATASET,
        layer='Detector comparison',
        family='SOTA',
        variant='karyoflow',
        execution_kind='full_train',
        status='PLANNED',
        priority='P0',
        training_seeds='42,123,789',
        config_state='READY',
        method_config=METHODS['karyoflow'],
        matrix_config=D2_MATRIX,
        protocol_config='',
        parameters='RF; shifted t; AdaLN-Zero; DPM++ 4-step; K=500; 6 heads; random coupling',
        database_ids={
            'train_run_ids': [],
            'result_family': 'sota_d2_canonical_test',
            'evidence_artifact_ids': [],
        },
        compatibility_report='',
        swanlab_project='KaryoFlow-Dataset2-V2',
        swanlab_run_template='{variant}_seed{training_seed}',
        parent_ledger_id='',
        paper_role='Canonical D2 main detector table',
        acceptance_gate='three canonical random-coupling checkpoints; six held-out-test metrics',
        notes=(
            'Deferred at completed epoch 3 after confirming that all paper '
            'D2 evidence follows the publisher-provided original split; retain '
            'only as an optional split-sensitivity experiment.'
        ),
        server_plan=(
            '42=ross:A6000:0;123=workstation:A5000:0;789=workstation:A4000:1'
        ),
    )
    canonical_lqcr = deepcopy(legacy_lqcr)
    canonical_lqcr.update(
        ledger_id='D2.SOTA.karyoflow_canonical_lqcr_train3',
        dataset_id=D2_CANONICAL_DATASET,
        layer='Decision',
        family='SOTA',
        variant='karyoflow_lqcr',
        execution_kind='short_train',
        status='BLOCKED_PARENT',
        priority='P0',
        training_seeds='42,123,789',
        config_state='READY_PARENT_PENDING',
        method_config=METHODS['karyoflow_lqcr'],
        matrix_config=D2_MATRIX,
        protocol_config='',
        parameters='parent-matched final-only quality head; beta selected on validation',
        database_ids={
            'train_run_ids': [],
            'result_family': 'paired_lqcr_d2_canonical_train3',
            'evidence_artifact_ids': [],
        },
        compatibility_report='',
        swanlab_project='KaryoFlow-Dataset2-V2',
        swanlab_run_template='{variant}_seed{training_seed}',
        parent_ledger_id=canonical_parent['ledger_id'],
        paper_role='Canonical D2 LQCR paired effect',
        acceptance_gate='one frozen-parent child per seed; final-only tensor audit; six test metrics',
        notes='Required before the manuscript is refreshed.',
    )
    rows.extend([canonical_parent, canonical_lqcr])

    d1_g3 = by_id['D1I.ABL.G3']
    d1_g3.update(
        parameters='RF; shifted t; AdaLN-Zero; Euler 1-step validation',
        paper_role='Strict one-factor generation ablation',
        acceptance_gate=(
            'only time conditioning changes from G2; common validation protocol'
        ),
    )
    d1_parent = by_id['D1I.SOTA.karyoflow']
    d1_parent['server_plan'] = (
        '42=workstation:A5000:0;123=workstation:A4000:1;789=ross:A6000:0'
    )
    by_id['D1I.SOTA.karyoflow_lqcr']['server_plan'] = (
        '42=workstation:A5000:0;123=workstation:A4000:1;789=ross:A6000:0'
    )

    g3 = by_id['D2.ABL.strict.G3']
    g3.update(
        execution_kind='full_train',
        run_count=3,
        status='PLANNED',
        config_state='READY',
        training_seeds='42,123,789',
        method_config=METHODS['strict_g3'],
        matrix_config=D2_GENERATION_MATRIX,
        output_template='work_dirs/v2/d2_generation/strict_g3_rf_shifted_adaln_zero_r50/trainseed_{training_seed}',
        parent_ledger_id='',
        notes='Strict one-factor AdaLN-Zero stage; DPM++ is evaluated separately on a fixed checkpoint.',
        dataset_id=D2_CANONICAL_DATASET,
        parameters='RF; shifted t; AdaLN-Zero; Euler 1-step validation',
        server_plan=(
            '42=ross:A6000:0;123=workstation:A5000:0;789=workstation:A4000:1'
        ),
        paper_role='Cross-cohort strict one-factor generation ablation',
        acceptance_gate=(
            'only time conditioning changes from G2; common validation protocol'
        ),
    )
    for rid in ('D2.ABL.strict.G0', 'D2.ABL.strict.G1', 'D2.ABL.strict.G2'):
        by_id[rid]['dataset_id'] = D2_CANONICAL_DATASET

    for rid in ('D2.INF.solver_steps.fixed1', 'D2.INF.topk_renewal.fixed1'):
        by_id[rid]['parent_ledger_id'] = legacy_parent['ledger_id']
    by_id['D2.DEC.beta_test.fixed1']['parent_ledger_id'] = legacy_lqcr[
        'ledger_id'
    ]
    by_id['D2.DEC.strict_subsets']['parent_ledger_id'] = legacy_lqcr[
        'ledger_id'
    ]

    for rid in ('D2.INF.solver_steps.train3', 'D2.INF.topk_renewal.train3'):
        row = by_id[rid]
        row.update(
            training_seeds='42,123,789',
            method_config=METHODS['karyoflow'],
            matrix_config=D2_MATRIX,
            parent_ledger_id=canonical_parent['ledger_id'],
            notes='Runs on the canonical random-coupling parent triplet.',
            dataset_id=D2_CANONICAL_DATASET,
        )

    distill = by_id['D2.DEP.distill_h3.train3']
    distill.update(
        training_seeds='42,123,789',
        method_config=METHODS['h3_distill'],
        matrix_config=D2_DISTILL_MATRIX,
        parent_ledger_id=canonical_parent['ledger_id'],
        notes='Canonical random-coupling parent-matched students.',
        dataset_id=D2_CANONICAL_DATASET,
    )
    by_id['D2.DEP.GACS']['dataset_id'] = D2_CANONICAL_DATASET

    legacy_speed = by_id['D2.DEP.speed']
    legacy_speed['ledger_id'] = 'D2.DEP.speed.legacy'
    legacy_speed['paper_role'] = 'Historical deployment context only'
    legacy_speed['parent_ledger_id'] = legacy_parent['ledger_id']
    canonical_speed = deepcopy(legacy_speed)
    canonical_speed.update(
        ledger_id='D2.DEP.speed.canonical',
        status='BLOCKED_CHECKPOINTS',
        priority='P0',
        training_seeds='42,123,789',
        config_state='PROTOCOL_READY_PARENT_PENDING',
        method_config=METHODS['karyoflow'],
        matrix_config=D2_MATRIX,
        output_template='results/v2/d2/benchmark/a6000/{variant}',
        database_ids={
            'train_run_ids': [],
            'result_family': 'speed_accuracy_d2_canonical',
            'evidence_artifact_ids': [],
        },
        compatibility_report='',
        parent_ledger_id=canonical_parent['ledger_id'],
        paper_role='Final canonical speed-accuracy figure/table',
        acceptance_gate='three A6000 repeats per final checkpoint; matching test accuracy; params and peak memory',
        notes='Canonical v2 seed 42 is active; seeds 123 and 789 remain planned.',
        server_plan=(
            '42=ross:A6000:0;123=workstation:A5000:0;789=workstation:A4000:1'
        ),
        dataset_id=D2_CANONICAL_DATASET,
    )
    rows.append(canonical_speed)

    templates = [
        (
            'D1I.DATA.test_characterization',
            'D1_INHOUSE1700_V2',
            'Data',
            'test_characterization',
            'analysis',
            'PLANNED',
            'D1 test population, scale CDF, overlap and size bins',
            D1_MATRIX,
            'dataset_characterization_d1i',
        ),
        (
            'D2.DATA.test_characterization',
            D2_CANONICAL_DATASET,
            'Data',
            'test_characterization',
            'analysis',
            'PLANNED',
            'D2 test population, scale CDF, overlap and size bins',
            D2_MATRIX,
            'dataset_characterization_d2',
        ),
        (
            'D1I.DEC.quality_validity',
            'D1_INHOUSE1700_V2',
            'Decision',
            'quality_iou_validity',
            'analysis',
            'BLOCKED_PREDICTIONS',
            'quality-IoU Spearman, MAE/RMSE and reliability bins',
            D1_MATRIX,
            'lqcr_quality_validity_d1i',
        ),
        (
            'D2.DEC.quality_validity',
            D2_CANONICAL_DATASET,
            'Decision',
            'quality_iou_validity',
            'analysis',
            'BLOCKED_PREDICTIONS',
            'quality-IoU Spearman, MAE/RMSE and reliability bins',
            D2_MATRIX,
            'lqcr_quality_validity_d2_canonical',
        ),
        (
            'D2.DEC.strict_subsets.canonical',
            D2_CANONICAL_DATASET,
            'Decision',
            'strict_iou_scale_overlap',
            'analysis',
            'BLOCKED_PREDICTIONS',
            'AP90/AP95, size quartiles, overlap strata and paired image bootstrap',
            D2_MATRIX,
            'conditional_difficult_subset_d2_canonical',
        ),
        (
            'D1I.ANALYSIS.per_class',
            'D1_INHOUSE1700_V2',
            'Decision',
            'per_class_error',
            'analysis',
            'BLOCKED_PREDICTIONS',
            '24-class AP and morphology-group error analysis',
            D1_MATRIX,
            'per_class_d1i_test',
        ),
        (
            'D2.ANALYSIS.per_class',
            D2_CANONICAL_DATASET,
            'Decision',
            'per_class_error',
            'analysis',
            'BLOCKED_PREDICTIONS',
            '24-class AP and morphology-group error analysis',
            D2_MATRIX,
            'per_class_d2_canonical_test',
        ),
    ]
    base = deepcopy(by_id['D1I.DEC.strict_subsets'])
    for (
        rid,
        dataset,
        layer,
        variant,
        kind,
        status,
        parameters,
        matrix,
        family,
    ) in templates:
        item = deepcopy(base)
        item.update(
            ledger_id=rid,
            dataset_id=dataset,
            layer=layer,
            family='Dataset analysis' if '.DATA.' in rid else 'Diagnostic',
            variant=variant,
            execution_kind=kind,
            status=status,
            priority='P0' if '.DATA.' in rid else 'P1',
            replication_unit='deterministic_dataset_analysis'
            if '.DATA.' in rid
            else 'paired_prediction_analysis',
            training_seeds='none' if '.DATA.' in rid else '42,123,789',
            inference_seeds='none' if '.DATA.' in rid else '42',
            run_count=1,
            config_state='PROTOCOL_TO_IMPLEMENT',
            method_config='',
            matrix_config=matrix,
            protocol_config='',
            parameters=parameters,
            server_plan='CPU',
            output_template=f'results/v2/{"d1" if dataset.startswith("D1_") else "d2"}/analysis/{variant}',
            database_ids={
                'train_run_ids': [],
                'result_family': family,
                'evidence_artifact_ids': [],
            },
            compatibility_report='',
            swanlab_project='',
            swanlab_run_template='',
            parent_ledger_id=(
                ''
                if '.DATA.' in rid
                else (
                    'D1I.SOTA.karyoflow_lqcr'
                    if dataset.startswith('D1_')
                    else canonical_lqcr['ledger_id']
                )
            ),
            paper_role=(
                'Dataset table and scale figure'
                if '.DATA.' in rid
                else 'Mechanism/error analysis'
            ),
            acceptance_gate='versioned analysis JSON bound to final test annotation/prediction SHA',
            notes='Required before the manuscript is refreshed.',
        )
        rows.append(item)

    beta = deepcopy(by_id['D1I.DEC.beta_val'])
    beta.update(
        ledger_id='D2.DEC.beta_val.canonical',
        dataset_id=D2_CANONICAL_DATASET,
        status='BLOCKED_PARENT',
        training_seeds='42,123,789',
        method_config=METHODS['karyoflow_lqcr'],
        matrix_config=D2_MATRIX,
        protocol_config='experiments/configs/ablations/d2_lqcr_beta_val.yaml',
        output_template='results/v2/d2/lqcr_beta_val/{parent}/beta_{beta}',
        database_ids={
            'train_run_ids': [],
            'result_family': 'lqcr_beta_d2_canonical_val',
            'evidence_artifact_ids': [],
        },
        parent_ledger_id=canonical_lqcr['ledger_id'],
        paper_role='Canonical D2 validation selection; not a test claim',
        notes='Required before the canonical D2 test evaluation.',
    )
    rows.append(beta)
    for item in rows:
        if item.get('parent_ledger_id') == 'D2.SOTA.karyoflow_train3':
            item['parent_ledger_id'] = legacy_parent['ledger_id']
        elif item.get('parent_ledger_id') == 'D2.SOTA.karyoflow_lqcr_train3':
            item['parent_ledger_id'] = legacy_lqcr['ledger_id']
    return rows


def _normalize_original_d2_routes(rows: list[dict]) -> list[dict]:
    """Restore the publisher-provided D2 split as the paper authority."""
    drop = {
        'D2.SOTA.karyoflow_canonical_train3',
        'D2.SOTA.karyoflow_canonical_lqcr_train3',
        'D2.DEP.speed.canonical',
        'D2.DEC.strict_subsets.canonical',
    }
    rename = {
        'D2.HIST.karyoflow_ot_train3': 'D2.SOTA.karyoflow_train3',
        'D2.HIST.karyoflow_ot_lqcr_train3': 'D2.SOTA.karyoflow_lqcr_train3',
        'D2.DEP.speed.legacy': 'D2.DEP.speed',
        'D2.DEC.beta_val.canonical': 'D2.DEC.beta_val',
    }
    parent_rename = {
        'D2.HIST.karyoflow_ot_train3': 'D2.SOTA.karyoflow_train3',
        'D2.HIST.karyoflow_ot_lqcr_train3': 'D2.SOTA.karyoflow_lqcr_train3',
        'D2.SOTA.karyoflow_canonical_train3': 'D2.SOTA.karyoflow_train3',
        'D2.SOTA.karyoflow_canonical_lqcr_train3': 'D2.SOTA.karyoflow_lqcr_train3',
    }
    normalized = []
    for row in rows:
        if row['ledger_id'] in drop:
            continue
        row['ledger_id'] = rename.get(row['ledger_id'], row['ledger_id'])
        row['parent_ledger_id'] = parent_rename.get(
            row.get('parent_ledger_id', ''), row.get('parent_ledger_id', '')
        )
        if row.get('dataset_id') == D2_CANONICAL_DATASET:
            row['dataset_id'] = 'D2'
        normalized.append(row)

    by_id = {row['ledger_id']: row for row in normalized}
    by_id['D2.SOTA.karyoflow_train3'].update(
        layer='Detector comparison',
        family='SOTA',
        priority='P0',
        paper_role='Original-split D2 main detector',
        notes='Verified three-independent-training-run evidence on the publisher-provided original D2 split.',
    )
    by_id['D2.SOTA.karyoflow_lqcr_train3'].update(
        layer='Decision',
        family='SOTA',
        priority='P0',
        paper_role='Original-split D2 paired LQCR effect',
        notes='Verified parent-matched three-training-run evidence on the publisher-provided original D2 split.',
    )
    d2_progress = _scheduler_d2_sota_progress()
    for variant in D2_LEGACY:
        existing = by_id[f'D2.SOTA.{variant}.existing']
        existing.update(
            layer='Detector comparison',
            family='SOTA',
            priority='P0',
            paper_role='Original-split D2 benchmark point estimate',
            notes='Recorded test result on the publisher-provided original D2 split.',
        )
        supplement = by_id[f'D2.SOTA.{variant}.canonical_train3']
        supplement['ledger_id'] = f'D2.SOTA.{variant}.train3_supplement'
        progress = d2_progress[variant]
        supplement.update(
            dataset_id='D2',
            status=_d2_sota_status(progress),
            priority='P1',
            paper_role='D2 three-training-run robustness supplement',
            swanlab_project='KaryoFlow-D2-Train3-Completion',
            server_plan=(
                'seeds 123/789 on workstation; heavy models on A5000 and '
                'lighter models on A4000; held-out tests and evidence import '
                'centralized on Ross'
            ),
            notes=(
                'The historical seed-42 point and two newly scheduled '
                f'independent runs form train3. New-run completion: '
                f'{progress["completed"]}/{progress["total"]}; '
                f'running={progress["running"]}, queued={progress["queued"]}.'
            ),
        )

    for rid in {
        'D2.ABL.strict.G0',
        'D2.ABL.strict.G1',
        'D2.ABL.strict.G2',
        'D2.ABL.strict.G3',
        'D2.INF.solver_steps.train3',
        'D2.INF.topk_renewal.train3',
        'D2.DEP.distill_h3.train3',
    }:
        by_id[rid].update(
            dataset_id='D2',
            status='DEFERRED_UNTIL_D1_COMPLETE',
            priority='P2',
            notes='D2 three-training-run supplement; do not enqueue until all required D1 work is complete.',
        )

    by_id['D2.DEP.speed'].update(
        layer='Deployment',
        priority='P0',
        paper_role='Original-split D2 speed-accuracy evidence',
        notes='Use original-split accuracy and the unified A6000 latency protocol; no repaired-split rerun.',
    )
    by_id['D2.DEC.beta_val'].update(
        dataset_id='D2',
        status='REVIEW_EXISTING_VALIDATION_SELECTION',
        parent_ledger_id='D2.SOTA.karyoflow_lqcr_train3',
        paper_role='Original-split D2 validation selection; never a test claim',
        notes='Recover and bind existing validation-selection evidence; no retraining is required.',
    )
    for rid in (
        'D2.DATA.test_characterization',
        'D2.DEC.quality_validity',
        'D2.ANALYSIS.per_class',
    ):
        by_id[rid]['dataset_id'] = 'D2'
    by_id['D2.DEC.quality_validity']['parent_ledger_id'] = (
        'D2.SOTA.karyoflow_lqcr_train3'
    )
    by_id['D2.ANALYSIS.per_class']['parent_ledger_id'] = (
        'D2.SOTA.karyoflow_lqcr_train3'
    )
    by_id['D2.DEP.GACS']['dataset_id'] = 'D2'
    return normalized


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def config_for(row: dict) -> tuple[str, str, str, str]:
    """Return method, matrix, protocol, configuration state."""
    rid, dataset, variant = row['ledger_id'], row['dataset_id'], row['variant']
    method = ''
    matrix = (
        D1_MATRIX
        if dataset == 'D1_INHOUSE1700_V2'
        else D2_MATRIX
        if dataset == 'D2'
        else ''
    )
    protocol = PROTOCOLS.get(rid, '')
    state = CONFIG_STATE_OVERRIDE.get(rid, 'READY')

    if rid == 'D1I.SOTA.karyoflow':
        method = METHODS['karyoflow']
    elif rid == 'D1I.SOTA.diffusiondet':
        method = METHODS['diffusiondet']
    elif rid == 'D1I.SOTA.karyoflow_lqcr':
        method = METHODS['karyoflow_lqcr']
    elif rid.startswith('D1I.SOTA.') and variant in METHODS:
        method = METHODS[variant]
    elif rid in {'D1I.ABL.G0', 'D1I.ABL.G1', 'D1I.ABL.G2', 'D1I.ABL.G3'}:
        stage = rid.rsplit('.', 1)[-1].lower()
        method = METHODS[f'strict_{stage}']
        matrix = D1_GENERATION_MATRIX
    elif rid.startswith('D1I.INF.') or rid == 'D1I.DEC.beta_val':
        method = METHODS[
            'karyoflow_lqcr' if rid == 'D1I.DEC.beta_val' else 'karyoflow'
        ]
    elif rid == 'D1I.DEP.GACS':
        method = METHODS['karyoflow']
    elif rid == 'D1I.DEP.distill_h3':
        method = METHODS['h3_distill']
        matrix = D1_DISTILL_MATRIX
    elif rid.startswith('D2.SOTA.karyoflow_lqcr'):
        method = 'experiments/configs/methods/karyoflow_ot_lqcr_legacy.py'
        matrix = D2_HISTORY_MATRIX
        state = 'EXACT'
    elif rid == 'D2.SOTA.karyoflow_train3':
        method = 'experiments/configs/methods/karyoflow_ot_legacy.py'
        matrix = D2_HISTORY_MATRIX
        state = 'EXACT'
    elif rid.startswith('D2.SOTA.') and variant in D2_LEGACY:
        if rid.endswith('.existing'):
            method, history_matrix, _, compatibility = D2_LEGACY[variant]
            matrix = history_matrix
            state = compatibility
        else:
            method = METHODS[variant]
            matrix = D2_SOTA_COMPLETION_MATRIX
            state = 'READY'
    elif rid in {'D2.ABL.strict.G0', 'D2.ABL.strict.G1', 'D2.ABL.strict.G2'}:
        stage = rid.rsplit('.', 1)[-1].lower()
        method = METHODS[f'strict_{stage}']
        matrix = D2_GENERATION_MATRIX
        state = 'READY'
    elif rid == 'D2.ABL.strict.G3':
        method = METHODS['strict_g3']
        matrix = D2_GENERATION_MATRIX
        state = 'READY'
    elif rid == 'D2.ABL.historical_chain':
        protocol = 'experiments/manifests/d2_paper_experiment_inventory.yaml'
        state = 'MIGRATED_MIXED_HISTORY'
    elif rid == 'D2.DEP.distill_h3.existing':
        method = (
            'experiments/configs/methods/karyoflow_ot_h3_student_legacy.py'
        )
        matrix = 'experiments/configs/matrices/d2_h3_student_history.yaml'
        protocol = 'experiments/configs/deployment/h3_distillation_training_archive.yaml'
        state = 'EXACT_INFERENCE_ARCHIVED_TRAINING'
    elif rid == 'D2.DEP.distill_h3.train3':
        method = METHODS['ot_h3_distill']
        matrix = D2_DISTILL_MATRIX
        protocol = (
            'experiments/configs/deployment/h3_distillation_protocol.yaml'
        )
        state = 'READY'
    elif rid == 'D2.DEP.GACS':
        method = 'experiments/configs/methods/karyoflow_ot_legacy.py'
        matrix = D2_HISTORY_MATRIX
        state = 'PROTOCOL_READY'
    elif (
        rid.startswith('D2.INF.')
        or rid.startswith('D2.DEC.')
        or rid == 'D2.DEP.speed'
    ):
        method = 'experiments/configs/methods/karyoflow_ot_legacy.py'
        matrix = D2_HISTORY_MATRIX
        state = (
            'PROTOCOL_READY'
            if rid.endswith('fixed1')
            or rid
            in {
                'D2.DEC.beta_test.fixed1',
                'D2.DEC.strict_subsets',
                'D2.DEP.speed',
            }
            else state
        )
    elif dataset == 'D1_COMPOSITE2200_LEGACY':
        protocol = 'tools/experiment_db/experiments.db'
        state = 'ARCHIVED'

    return method, matrix, protocol, state


def build_rows() -> list[dict]:
    legacy_rows = json.loads(LEGACY.read_text())['experiments']
    rows = []
    for old in legacy_rows:
        row = dict(old)
        rid = row['ledger_id']
        if row['dataset_id'] == 'D1_INHOUSE1700_V1':
            row['dataset_id'] = 'D1_INHOUSE1700_V2'
        method, matrix, protocol, config_state = config_for(row)
        row['legacy_config_path'] = row.pop('config_path')
        row['legacy_config_status'] = row.pop('config_status')
        row['method_config'] = method
        row['matrix_config'] = matrix
        row['protocol_config'] = protocol
        row['config_state'] = config_state
        row['status'] = STATUS_OVERRIDE.get(rid, row['status'])
        row['parameters'] = row.get('parameters', '')
        row['server_plan'] = SERVER_PLAN_OVERRIDE.get(
            rid, row.pop('executor_plan')
        )
        row['output_template'] = row.pop('work_dir_template')
        if row['dataset_id'] == 'D1_INHOUSE1700_V2':
            row['output_template'] = (
                row['output_template']
                .replace('work_dirs/self1700/', 'work_dirs/v2/d1_inhouse1700/')
                .replace('results/self1700/', 'results/v2/d1_inhouse1700/')
            )
        elif rid.endswith('.missing_train2'):
            row['output_template'] = (
                f'work_dirs/v2/d2_taichung/sota_completion/{row["variant"]}/trainseed_{{training_seed}}'
            )
        elif rid.startswith('D2.ABL.strict.G') and rid != 'D2.ABL.strict.G3':
            row['output_template'] = (
                f'work_dirs/v2/d2_taichung/strict_generation/{rid.rsplit(".", 1)[-1].lower()}/trainseed_{{training_seed}}'
            )
        elif rid == 'D2.ABL.strict.G3':
            row['training_seeds'] = '42,123,789'
            row['output_template'] = (
                'work_dirs/v2/d2_taichung/strict_generation/g3/trainseed_{training_seed}'
            )
        elif rid == 'D2.INF.solver_steps.train3':
            row['output_template'] = (
                'results/v2/d2_taichung/inference/solver/{parent}/{solver}_{steps}'
            )
        elif rid == 'D2.INF.topk_renewal.train3':
            row['output_template'] = (
                'results/v2/d2_taichung/inference/topk_renewal/{parent}/{variant}'
            )
        elif rid == 'D2.DEP.distill_h3.train3':
            row['output_template'] = (
                'work_dirs/v2/d2_taichung/deployment/h3/{parent}'
            )
        elif rid == 'D2.DEP.GACS':
            row['output_template'] = (
                'results/v2/d2_taichung/deployment/gacs/{parent}'
            )
        row['database_ids'] = {
            'train_run_ids': []
            if row['dataset_id'] == 'D1_INHOUSE1700_V2'
            else [
                x for x in (row.pop('train_run_ids', '') or '').split(';') if x
            ],
            'result_family': row.pop('result_family', ''),
            'evidence_artifact_ids': [
                x
                for x in (row.pop('evidence_artifact_ids', '') or '').split(
                    ';'
                )
                if x
            ],
        }
        if rid == 'D1I.SOTA.karyoflow':
            row['database_ids']['train_run_ids'] = [
                'd1-inhouse1700-v2__karyoflow-r50__trainseed-42__5102e030f547',
                'd1-inhouse1700-v2__karyoflow-r50__trainseed-123__5102e030f547',
                'd1-inhouse1700-v2__karyoflow-r50__trainseed-789__5102e030f547',
            ]
        elif rid == 'D1I.SOTA.karyoflow_lqcr':
            row['database_ids']['train_run_ids'] = [
                'd1-inhouse1700-v2__karyoflow-lqcr-r50__trainseed-42__bddd7d3d2741',
                'd1-inhouse1700-v2__karyoflow-lqcr-r50__trainseed-123__bddd7d3d2741',
                'd1-inhouse1700-v2__karyoflow-lqcr-r50__trainseed-789__bddd7d3d2741',
            ]
        if rid == 'LEGACY.D1.composite':
            row['database_ids']['result_family'] = (
                'all D1 families except D1_INHOUSE1700_V2'
            )
        if rid == 'D2.ABL.strict.G3':
            row['database_ids']['train_run_ids'] = []
            row['database_ids']['evidence_artifact_ids'] = []
            row['database_ids']['result_family'] = 'strict_generation_d2_test'
        if rid.startswith('D2.SOTA.') and row['variant'] in D2_LEGACY:
            row['compatibility_report'] = D2_LEGACY[row['variant']][2]
        elif rid in {
            'D2.SOTA.karyoflow_train3',
            'D2.SOTA.karyoflow_lqcr_train3',
        }:
            row['compatibility_report'] = (
                'tools/experiment_db/compatibility_reports/d2_mainline_history_v1_index.json'
            )
        elif rid == 'D2.DEP.distill_h3.existing':
            row['compatibility_report'] = (
                'tools/experiment_db/compatibility_reports/d2_head_student_h3.json'
            )
        else:
            row['compatibility_report'] = ''
        if row['dataset_id'] == 'D1_INHOUSE1700_V2':
            row['swanlab_project'] = (
                'KaryoFlow-Self1700-V2'
                if row['execution_kind'] in {'full_train', 'short_train'}
                else ''
            )
            if rid == 'D1I.DEP.distill_h3':
                row['swanlab_project'] = 'KaryoFlow-HeadDistill-D1-V2'
            row['notes'] = NOTES_OVERRIDE.get(
                rid, 'No active v2 run is currently registered.'
            )
        else:
            row['notes'] = NOTES_OVERRIDE.get(rid, row.get('notes', ''))
            if (
                row['dataset_id'] == D2_CANONICAL_DATASET
                and row['execution_kind'] in {'full_train', 'short_train'}
                and row['status'].startswith(('PLANNED', 'BLOCKED'))
            ):
                row['swanlab_project'] = 'KaryoFlow-Dataset2-V2'
                row['swanlab_run_template'] = '{variant}_seed{training_seed}'
                if rid == 'D2.DEP.distill_h3.train3':
                    row['swanlab_project'] = 'KaryoFlow-HeadDistill-D2-V2'

        rows.append(row)
    rows = _normalize_original_d2_routes(_upgrade_scientific_routes(rows))
    for row in rows:
        resolved = _resolved_training_metadata(row)
        row['resolved_training'] = resolved or {}
        if resolved:
            row['parameters'] = _scientific_parameters(
                row['parameters'], has_resolved_training=True
            )
            row['output_template'] = resolved['output_template']
            active_ids = _active_registered_runs(row, resolved)
            if active_ids:
                row['database_ids']['train_run_ids'] = active_ids
    _apply_d1_test_progress(rows)
    return rows


def flatten(row: dict) -> dict:
    return {
        'route_id': row['ledger_id'],
        'dataset_id': row['dataset_id'],
        'layer': row['layer'],
        'family': row['family'],
        'variant': row['variant'],
        'execution_kind': row['execution_kind'],
        'run_count': row['run_count'],
        'training_seeds': row['training_seeds'],
        'inference_seeds': row['inference_seeds'],
        'status': row['status'],
        'priority': row['priority'],
        'replication_unit': row['replication_unit'],
        'config_state': row['config_state'],
        'method_config': row['method_config'],
        'matrix_config': row['matrix_config'],
        'protocol_config': row['protocol_config'],
        'parameters': row['parameters'],
        'resolved_training': json.dumps(
            row.get('resolved_training', {}), sort_keys=True
        ),
        'server_plan': row['server_plan'],
        'output_template': row['output_template'],
        'train_run_ids': ';'.join(row['database_ids']['train_run_ids']),
        'result_family': row['database_ids']['result_family'],
        'evidence_artifact_ids': ';'.join(
            row['database_ids']['evidence_artifact_ids']
        ),
        'compatibility_report': row['compatibility_report'],
        'swanlab_project': row['swanlab_project'],
        'swanlab_run_template': row['swanlab_run_template'],
        'parent_route_id': row['parent_ledger_id'],
        'paper_role': row['paper_role'],
        'acceptance_gate': row['acceptance_gate'],
        'notes': row['notes'],
    }


def write_doc(payload: dict, manifest_sha: str, artifact_id: str) -> None:
    rows = payload['experiments']
    statuses = Counter(r['status'] for r in rows)
    lines = [
        '# KaryoFlow v2 完整实验路线矩阵',
        '',
        '> 这是双数据集实验调度、配置解析、数据库登记和论文数字校对的唯一权威入口。',
        '> YAML 是源文件；本 Markdown、CSV 与 SQLite 表均由生成器派生，禁止手工修改派生文件。',
        '',
        '## 权威身份',
        '',
        f'- YAML：`{OUTPUT.relative_to(ROOT)}`',
        f'- YAML SHA-256：`{manifest_sha}`',
        f'- 数据库 artifact：`{artifact_id}`',
        f'- 论文 claim manifest：`{CLAIM_MANIFEST.relative_to(ROOT)}`（SHA-256 `{payload["claim_manifest_sha256"]}`）',
        f'- 实验组：{len(rows)}；展开运行：{sum(int(r["run_count"]) for r in rows)}。',
        f'- 状态分布：{", ".join(f"{k}={v}" for k, v in sorted(statuses.items()))}。',
        '',
        '## 数据与统计口径',
        '',
        '- `D1_INHOUSE1700_V2`：1190/170/340，纯自建、D2 类别 ID 对齐、group-disjoint 70/10/20 划分。',
        '- `D2`：3500/500/1000，使用作者发布的原始公开划分，以保持与既有 benchmark 和历史权重直接可比。',
        '- repaired-split 不再作为论文数据口径；相关数据、配置、路线和数据库发布记录均不保留。',
        '- 训练复现以不同训练 checkpoint 为统计单位；固定 checkpoint 的推理 seed 不得冒充训练 seed。',
        '- 主精度只允许 held-out test；validation 只用于 checkpoint/超参数选择。',
        '- 效率只允许按 A6000 严格协议比较；服务器名称只存在内部路线矩阵，不进入论文。',
        '',
        '## 路线矩阵',
    ]
    for dataset in ('D1_INHOUSE1700_V2', 'D2', 'D1_COMPOSITE2200_LEGACY'):
        lines += ['', f'### {dataset}', '']
        for layer in (
            'Data',
            'Detector comparison',
            'Generation',
            'Decision',
            'Deployment',
            'Archive',
        ):
            selected = [
                r
                for r in rows
                if r['dataset_id'] == dataset and r['layer'] == layer
            ]
            if not selected:
                continue
            lines += [
                f'#### {layer}',
                '',
                '| ID | Variant | Kind / runs | Seeds | Status | Config state | v2 method / protocol | Server | Output | DB/evidence |',
                '|---|---|---:|---|---|---|---|---|---|---|',
            ]
            for row in selected:
                ref = row['method_config'] or row['protocol_config'] or '—'
                evidence = row['database_ids']['result_family'] or ';'.join(
                    row['database_ids']['evidence_artifact_ids']
                )
                db = f'route:{row["ledger_id"]}' + (
                    f'; {evidence}' if evidence else ''
                )
                seeds = f'train={row["training_seeds"]}; infer={row["inference_seeds"]}'
                lines.append(
                    f'| `{row["ledger_id"]}` | {row["variant"]} | {row["execution_kind"]} / {row["run_count"]} | {seeds} | **{row["status"]}** | {row["config_state"]} | `{ref}` | {row["server_plan"]} | `{row["output_template"]}` | `{db}` |'
                )
            lines.append('')
            for row in selected:
                resolved = row.get('resolved_training') or {}
                resolved_text = (
                    f'；解析配置=batch {resolved["batch_size"]}, '
                    f'{resolved["max_epochs"]} epochs, {resolved["optimizer"]}, '
                    f'lr={resolved["base_lr"]:g}, config_id={resolved["config_id"]}'
                    if resolved
                    else ''
                )
                lines.append(
                    f'- `{row["ledger_id"]}`：科学因素={row["parameters"]}'
                    f'{resolved_text}；验收={row["acceptance_gate"]}；'
                    f'论文用途={row["paper_role"]}；备注={row["notes"]}'
                )
    lines += [
        '',
        '## 当前执行结论',
        '',
        '- D1_INHOUSE1700_V2 的 canonical KaryoFlow 与 parent-matched LQCR 三训练种子 held-out test 已完成并登记。',
        '- D2 使用作者原始公开划分；已完成证据直接作为主结果，确需补充的三训练种子实验统一延后到全部 D1 工作完成后，且未进入当前调度队列。',
        '- D1 严格 G0--G3 的三训练种子均已完成；held-out test、solver/step 和 Top-K/renewal 矩阵由串行队列补齐，状态随证据导入自动更新。',
        '- H3 推理身份可精确复现，但历史蒸馏训练实现仍需恢复；GACS 保持可选部署扩展。',
        '',
    ]
    DOC_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    DOC_OUTPUT.write_text('\n'.join(lines), encoding='utf-8')


def register_db(rows: list[dict], manifest_sha: str, artifact_id: str) -> None:
    con = sqlite3.connect(DB)
    with con:
        con.execute(
            """CREATE TABLE IF NOT EXISTS experiment_route_matrix_v2 (
               route_id TEXT PRIMARY KEY, dataset_id TEXT NOT NULL, layer TEXT NOT NULL,
               family TEXT NOT NULL, variant TEXT NOT NULL, execution_kind TEXT NOT NULL,
               run_count INTEGER NOT NULL, training_seeds TEXT, inference_seeds TEXT,
               status TEXT NOT NULL, priority TEXT NOT NULL, replication_unit TEXT NOT NULL,
               config_state TEXT NOT NULL, method_config TEXT, matrix_config TEXT,
               protocol_config TEXT, server_plan TEXT NOT NULL, output_template TEXT NOT NULL,
               database_refs_json TEXT NOT NULL, swanlab_project TEXT,
               parent_route_id TEXT, paper_role TEXT NOT NULL, acceptance_gate TEXT NOT NULL,
               source_manifest_sha256 TEXT NOT NULL, record_json TEXT NOT NULL)
            """
        )
        claim_sha = sha256(CLAIM_MANIFEST)
        con.execute(
            "DELETE FROM evidence_artifact WHERE artifact_id LIKE 'paper-claim-manifest-v1-%'"
        )
        con.execute(
            """INSERT OR REPLACE INTO evidence_artifact
               (artifact_id,server,path,sha256,kind,generated_at,status,notes)
               VALUES(?,?,?,?,?,NULL,'verified',?)""",
            (
                f'paper-claim-manifest-v1-{claim_sha[:12]}',
                'ross',
                str(CLAIM_MANIFEST.relative_to(ROOT)),
                claim_sha,
                'paper_claim_manifest_v1',
                'Canonical claims that gate manuscript refresh and submission',
            ),
        )
        con.execute('DELETE FROM experiment_route_matrix_v2')
        con.executemany(
            """INSERT INTO experiment_route_matrix_v2 VALUES
               (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            [
                (
                    r['ledger_id'],
                    r['dataset_id'],
                    r['layer'],
                    r['family'],
                    r['variant'],
                    r['execution_kind'],
                    int(r['run_count']),
                    r['training_seeds'],
                    r['inference_seeds'],
                    r['status'],
                    r['priority'],
                    r['replication_unit'],
                    r['config_state'],
                    r['method_config'],
                    r['matrix_config'],
                    r['protocol_config'],
                    r['server_plan'],
                    r['output_template'],
                    json.dumps(r['database_ids'], sort_keys=True),
                    r['swanlab_project'],
                    r['parent_ledger_id'],
                    r['paper_role'],
                    r['acceptance_gate'],
                    manifest_sha,
                    json.dumps(r, sort_keys=True),
                )
                for r in rows
            ],
        )
        con.execute(
            "DELETE FROM evidence_artifact WHERE artifact_id LIKE 'paper-route-matrix-v2-%'"
        )
        con.execute(
            """INSERT OR REPLACE INTO evidence_artifact
               (artifact_id,server,path,sha256,kind,generated_at,status,notes)
               VALUES(?,?,?,?,?,NULL,'verified',?)""",
            (
                artifact_id,
                'ross',
                str(OUTPUT.relative_to(ROOT)),
                manifest_sha,
                'paper_experiment_route_matrix_v2',
                f'{len(rows)} groups; {sum(int(r["run_count"]) for r in rows)} expanded runs',
            ),
        )


def main() -> None:
    rows = build_rows()
    payload = {
        'schema_version': 2,
        'matrix_id': 'karyoflow.paper_experiment_route.v2',
        'authoritative': True,
        'claim_manifest_sha256': sha256(CLAIM_MANIFEST),
        'supersedes': [
            'tools/experiment_db/manifests/master_experiment_ledger_v1.json',
            'tools/experiment_db/exports/master_experiment_ledger.csv',
            'docs/experiments/MASTER_EXPERIMENT_LEDGER.md',
        ],
        'dataset_scope': {
            'D1_INHOUSE1700_V2': {
                'split_images': [1190, 170, 340],
                'role': 'primary in-house cohort',
                'manifest_artifact_id': 'dataset-self1700-48d90fed63ec',
                'manifest_sha256': '48d90fed63ecc107b374a316effc1e5ab0d63b7c4bd9110d33ddd16e1f43146c',
                'annotation_sha256': {
                    'train': 'e62704aa3ce9f7b58170946cca78458ff5331fc4316ba946eace6a1dbc6c8ed1',
                    'val': '9065c7b5fb4df2a9f9c794dfe83c655ddf6695388d2bde562171b208c6d3a5b9',
                    'test': '883696b8e60cc901cfe92b3f009d8c60e7b8cefbb5ba9ce3c742c3720343f08f',
                },
            },
            'D2': {
                'split_images': [3500, 500, 1000],
                'role': 'external Taichung cohort; publisher-provided original benchmark split',
                'annotation_sha256': {
                    'train': '218ae0ebb71ecfb186bdb0872101ac50179c22c69a586cad0366ef10a0d8d5f7',
                    'val': 'bcf0f930dea7380a3d2d5e82b393b7576a5861d416c163ff2930985f9ee7ca10',
                    'test': '110fd2804f435b04a1eee969cb28666a0818b2886040a57cbae2dd05f2767495',
                },
            },
            'D1_COMPOSITE2200_LEGACY': {
                'role': 'archive only; mixed-source and non-comparable'
            },
        },
        'status_policy': {
            'COMPLETED_*': 'evidence exists; scope is qualified by the suffix',
            'PLANNED': 'v2 configuration is ready but no active valid run is registered',
            'QUEUED_*': 'a concrete scheduler task is registered; suffix describes completed or running prerequisites',
            'TRAIN3_COMPLETE_TEST_QUEUED': 'three training runs are registered and their held-out tests are queued',
            'TRAINING_*': 'training remains active; dependent test tasks are already registered',
            'TEST_QUEUED_*': 'held-out-test tasks are registered; suffix describes evidence already complete',
            'DEFERRED_UNTIL_D1_COMPLETE': 'not queued; may start only after every required D1 task is complete',
            'BLOCKED_*': 'a named prerequisite is missing',
            'PARTIAL_LEGACY': 'legacy evidence exists but does not meet the current protocol',
            'ARCHIVED_NONCOMPARABLE': 'retained only for provenance',
        },
        'experiments': rows,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        yaml.safe_dump(
            payload, allow_unicode=True, sort_keys=False, width=120
        ),
        encoding='utf-8',
    )
    manifest_sha = sha256(OUTPUT)
    artifact_id = f'paper-route-matrix-v2-{manifest_sha[:12]}'

    flat = [flatten(r) for r in rows]
    CSV_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with CSV_OUTPUT.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(flat[0]), lineterminator='\n'
        )
        writer.writeheader()
        writer.writerows(flat)
    write_doc(payload, manifest_sha, artifact_id)
    register_db(rows, manifest_sha, artifact_id)
    print(
        json.dumps(
            {
                'status': 'PASS',
                'groups': len(rows),
                'expanded_runs': sum(int(r['run_count']) for r in rows),
                'manifest_sha256': manifest_sha,
                'artifact_id': artifact_id,
            },
            indent=2,
        )
    )


if __name__ == '__main__':
    main()
