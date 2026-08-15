#!/usr/bin/env python3
"""Small dependency-free regression tests for the evidence gate."""

import json
import sqlite3
from copy import deepcopy
from pathlib import Path

import pytest

from tools.experiment_db.import_run_evidence import import_record, sha256_file
from tools.experiment_db.validate_run_evidence import validate

SHA = 'a' * 64


def record():
    artifact = {'path': 'placeholder', 'sha256': SHA}
    return {
        'schema_version': '1.0',
        'record_id': 'd2__model__seed-42',
        'record_type': 'evaluation',
        'dataset_id': 'D2',
        'split': 'test',
        'train_run_id': 'd2__model__trainseed-42__abc123def456',
        'annotation': artifact,
        'num_images': 1000,
        'training_seed': 42,
        'inference_seed': 42,
        'replication_unit': 'independent_training_seed',
        'config': artifact,
        'checkpoint': artifact,
        'source_log': artifact,
        'code': {'git_commit': 'abcdef1', 'dirty': False},
        'evaluation_protocol': {
            'metric_definition': 'COCO bbox mAP@[.50:.95]',
            'max_dets': [1, 10, 100],
            'eval_code_version': 'abcdef1',
            'candidate_count': 500,
            'solver': 'DPM-Solver++',
            'steps': 4,
            'nfe': 4,
        },
        'protocol_sha256': SHA,
        'selection_source': 'validation:best_mAP',
        'status': 'verified',
        'metrics': dict.fromkeys(
            ('mAP', 'AP50', 'AP75', 'AP_S', 'AP_M', 'AP_L'), 0.5
        ),
    }


def test_valid_record_passes():
    base = record()
    assert not validate(base, Path.cwd(), False)


def test_final_accuracy_must_use_test():
    base = record()
    bad = deepcopy(base)
    bad['split'] = 'val'
    assert any(
        'split=test' in error for error in validate(bad, Path.cwd(), False)
    )


def test_six_metrics_are_required():
    base = record()
    bad = deepcopy(base)
    del bad['metrics']['AP_L']
    assert any('AP_L' in error for error in validate(bad, Path.cwd(), False))


def test_paired_intervention_requires_parent_identity():
    base = record()
    bad = deepcopy(base)
    bad['replication_unit'] = 'paired_final_stage_intervention'
    assert any(
        'parent_train_run_id' in error
        for error in validate(bad, Path.cwd(), False)
    )


def test_test_tuning_blocks_final_evidence():
    base = record()
    bad = deepcopy(base)
    bad['test_tuned'] = True
    assert any(
        'test-tuned' in error for error in validate(bad, Path.cwd(), False)
    )


def test_artifact_paths_are_project_relative():
    bad = record()
    bad['checkpoint'] = {'path': '/tmp/model.pth', 'sha256': SHA}
    assert any(
        'project-relative' in error
        for error in validate(bad, Path.cwd(), False)
    )


def prepared_import(tmp_path):
    artifacts = {}
    for name in (
        'annotation.json',
        'config.py',
        'checkpoint.pth',
        'train.log',
    ):
        path = tmp_path / name
        path.write_text(name, encoding='utf-8')
        artifacts[name] = {
            'path': name,
            'sha256': sha256_file(path),
        }
    evidence = record()
    evidence.update(
        {
            'annotation': artifacts['annotation.json'],
            'config': artifacts['config.py'],
            'checkpoint': artifacts['checkpoint.pth'],
            'source_log': artifacts['train.log'],
        }
    )
    evidence_path = tmp_path / 'evidence.json'
    evidence_path.write_text(json.dumps(evidence), encoding='utf-8')

    connection = sqlite3.connect(tmp_path / 'experiments.db')
    schema = Path('tools/experiment_db/schema.sql').read_text(encoding='utf-8')
    connection.executescript(schema)
    connection.execute('PRAGMA foreign_keys=ON')
    connection.execute(
        'INSERT INTO evidence_artifact '
        '(artifact_id,server,path,sha256,kind,status) VALUES (?,?,?,?,?,?)',
        (
            'dataset-manifest',
            'repository',
            'manifest.json',
            SHA,
            'dataset_manifest',
            'verified',
        ),
    )
    connection.execute(
        'INSERT INTO dataset_release '
        '(dataset_id,display_name,root_path,version,image_count,category_count,'
        'construction_protocol_json,manifest_artifact_id,status) '
        'VALUES (?,?,?,?,?,?,?,?,?)',
        (
            'D2',
            'Dataset 2',
            'data/d2',
            '1',
            1000,
            24,
            '{}',
            'dataset-manifest',
            'frozen',
        ),
    )
    connection.execute(
        'INSERT INTO train_run_registry '
        '(train_run_id,dataset_id,method,training_seed,config_path,config_sha256,'
        'dataset_manifest_sha256,git_commit,replication_unit,assigned_executor,'
        'work_dir,status) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
        (
            evidence['train_run_id'],
            'D2',
            'model',
            42,
            'config.py',
            artifacts['config.py']['sha256'],
            SHA,
            'abcdef1',
            'independent_training_seed',
            'gpu',
            'work_dirs/model',
            'trained',
        ),
    )
    connection.commit()
    return connection, evidence, evidence_path


def test_import_is_transactional_and_idempotent(tmp_path):
    connection, evidence, evidence_path = prepared_import(tmp_path)
    import_record(connection, evidence, evidence_path, tmp_path)
    import_record(connection, evidence, evidence_path, tmp_path)
    assert (
        connection.execute(
            'SELECT COUNT(*) FROM controlled_result'
        ).fetchone()[0]
        == 6
    )
    assert (
        connection.execute(
            'SELECT COUNT(*) FROM eval_run_registry'
        ).fetchone()[0]
        == 1
    )
    connection.close()


def test_import_rejects_identity_reuse_with_changed_metric(tmp_path):
    connection, evidence, evidence_path = prepared_import(tmp_path)
    import_record(connection, evidence, evidence_path, tmp_path)
    changed = deepcopy(evidence)
    changed['metrics']['mAP'] = 0.6
    changed_path = tmp_path / 'changed.json'
    changed_path.write_text(json.dumps(changed), encoding='utf-8')
    with pytest.raises(ValueError, match='conflicting immutable'):
        import_record(connection, changed, changed_path, tmp_path)
    connection.close()


def test_import_preserves_additional_strict_iou_metrics(tmp_path):
    connection, evidence, evidence_path = prepared_import(tmp_path)
    evidence['metrics']['AP90'] = 0.4
    evidence['metrics']['AP95'] = 0.3
    evidence_path.write_text(json.dumps(evidence), encoding='utf-8')
    import_record(connection, evidence, evidence_path, tmp_path)
    rows = connection.execute(
        'SELECT metric,value FROM controlled_result '
        "WHERE metric IN ('AP90','AP95') ORDER BY metric"
    ).fetchall()
    assert rows == [('AP90', 0.4), ('AP95', 0.3)]
    connection.close()
