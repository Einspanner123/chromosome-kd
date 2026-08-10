"""Export a human-readable paper evidence register from experiments.db."""

import argparse
import os
import sqlite3


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', default=os.path.join(HERE, 'experiments.db'))
    parser.add_argument('--output', default=os.path.join(ROOT, 'docs', 'PAPER_EVIDENCE_REGISTER.md'))
    args = parser.parse_args()
    conn = sqlite3.connect(args.db)
    results = conn.execute('''SELECT r.result_id,r.family,r.variant,r.dataset,
        r.split,r.seed,r.metric,r.value,r.unit,r.delta,a.server,a.path
        FROM controlled_result r JOIN evidence_artifact a
        ON r.artifact_id=a.artifact_id WHERE r.paper_eligible=1
        ORDER BY r.family,r.dataset,r.result_id''').fetchall()
    findings = conn.execute('''SELECT finding_id,title,status,claim,scope,
        generality_basis,caveat,primary_metric,benefit FROM finding
        ORDER BY finding_id''').fetchall()
    theories = conn.execute('''SELECT theory_id,statement,assumptions,derivation,
        predicted_effect,empirical_status,source_doc FROM theory_statement
        ORDER BY theory_id''').fetchall()

    lines = [
        '# Paper evidence register', '',
        '> Auto-generated from `tools/experiment_db/experiments.db`. Do not edit',
        '> numerical values here; update the manifest/database and regenerate.', '',
        '## Paper-eligible controlled results', '',
        '| ID | Family / variant | Data | Metric | Value | Delta | Source |',
        '|---|---|---|---:|---:|---:|---|']
    for row in results:
        rid, family, variant, dataset, split, seed, metric, value, unit, delta, server, path = row
        delta_text = '' if delta is None else f'{delta:.7g}'
        lines.append(f'| `{rid}` | {family} / {variant} | {dataset} {split}, {seed} | '
                     f'{metric} ({unit}) | {value:.7g} | {delta_text} | `{server}:{path}` |')

    lines += ['', '## Findings and claim boundaries', '']
    for row in findings:
        fid, title, status, claim, scope, basis, caveat, metric, benefit = row
        lines += [f'### {fid}: {title} [{status}]', '', claim, '',
                  f'- Scope: {scope}', f'- Generality basis: {basis}',
                  f'- Caveat: {caveat}']
        if metric:
            lines.append(f'- Primary metric: {metric}')
        if benefit:
            lines.append(f'- Benefit: {benefit}')
        lines.append('')

    lines += ['## Theory statements', '']
    for row in theories:
        tid, statement, assumptions, derivation, prediction, status, source = row
        lines += [f'### {tid}', '', statement, '', f'- Assumptions: {assumptions}',
                  f'- Derivation: {derivation}', f'- Prediction: {prediction}',
                  f'- Empirical status: {status}', f'- Source: `{source}`', '']

    with open(args.output, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines).rstrip() + '\n')
    print(args.output)


if __name__ == '__main__':
    main()

