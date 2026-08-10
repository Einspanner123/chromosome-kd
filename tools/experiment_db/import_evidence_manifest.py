"""Idempotently import the curated, source-backed evidence manifest."""

import argparse
import hashlib
import json
import os
import sqlite3


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', default=os.path.join(HERE, 'experiments.db'))
    parser.add_argument('--manifest', default=os.path.join(HERE, 'evidence_manifest.json'))
    args = parser.parse_args()
    manifest = json.load(open(args.manifest))

    # Reuse schema initialization, including legacy evaluation deduplication.
    from build_experiment_db import init_db
    conn = init_db(args.db)

    for item in manifest['artifacts']:
        abs_path = os.path.join(ROOT, item['path'])
        if not os.path.isfile(abs_path):
            raise FileNotFoundError(f"Evidence source missing: {item['path']}")
        conn.execute('''INSERT OR REPLACE INTO evidence_artifact
            (artifact_id, server, path, sha256, kind, generated_at, status, notes)
            VALUES (?, ?, ?, ?, ?, ?, 'verified', ?)''',
            (item['artifact_id'], item['server'], item['path'], sha256(abs_path),
             item['kind'], manifest.get('generated_at'), item.get('notes')))

    # Baselines appear before dependent results in the manifest.
    for row in manifest['results']:
        conn.execute('''INSERT OR REPLACE INTO controlled_result
            (result_id, family, variant, dataset, split, seed, metric, value,
             unit, baseline_result_id, delta, protocol_json, artifact_id,
             evidence_level, paper_eligible, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (row['result_id'], row['family'], row['variant'], row['dataset'],
             row['split'], row['seed'], row['metric'], row['value'],
             row.get('unit', 'absolute'), row.get('baseline_result_id'),
             row.get('delta'), json.dumps(row.get('protocol', {}), sort_keys=True),
             row['artifact_id'], row['evidence_level'],
             int(row.get('paper_eligible', False)), row.get('notes')))

    for finding in manifest['findings']:
        ids = finding.pop('result_ids')
        conn.execute('''INSERT OR REPLACE INTO finding
            (finding_id, title, finding_type, claim, scope, generality_basis,
             status, caveat, primary_metric, benefit, evidence_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (finding['finding_id'], finding['title'], finding['finding_type'],
             finding['claim'], finding['scope'], finding['generality_basis'],
             finding['status'], finding['caveat'], finding.get('primary_metric'),
             finding.get('benefit'), json.dumps(ids)))
        conn.execute('DELETE FROM finding_evidence WHERE finding_id=?',
                     (finding['finding_id'],))
        conn.executemany('INSERT INTO finding_evidence VALUES (?, ?)',
                         [(finding['finding_id'], result_id) for result_id in ids])

    for row in manifest['theory']:
        conn.execute('''INSERT OR REPLACE INTO theory_statement
            (theory_id, statement, assumptions, derivation, predicted_effect,
             empirical_status, source_doc) VALUES (?, ?, ?, ?, ?, ?, ?)''',
            (row['theory_id'], row['statement'], row['assumptions'],
             row['derivation'], row['predicted_effect'], row['empirical_status'],
             row['source_doc']))
    conn.commit()
    print(f"Imported {len(manifest['results'])} results, "
          f"{len(manifest['findings'])} findings, {len(manifest['theory'])} theories")


if __name__ == '__main__':
    main()

