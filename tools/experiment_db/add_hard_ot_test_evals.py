"""补充 D2 Hard OT 实验记录 + D1/D2 Hard OT test 评估到数据库.

D2 Hard OT 实验在 workstation 上, 本地 work_dirs 未挂载, 需手动入库。
test mAP 由 test.py 评估产生, build_experiment_db.py 仅解析 val, 需手动入库。

用法:
  python tools/experiment_db/add_hard_ot_test_evals.py
"""

import os
import sqlite3
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'experiments.db')


# ─── D2 Hard OT 实验数据 (from workstation train logs) ──────
D2_EXPERIMENTS = [
    {
        'experiment_id': 'work_dirs/hard_ot_24obj_seed42',
        'work_dir': 'work_dirs/hard_ot_24obj_seed42',
        'seed': 42,
        'best_val_mAP': 0.860,
        'best_val_epoch': 89,
        'best_checkpoint': 'best_coco_bbox_mAP_epoch_89.pth',
    },
    {
        'experiment_id': 'work_dirs/hard_ot_24obj_seed123',
        'work_dir': 'work_dirs/hard_ot_24obj_seed123',
        'seed': 123,
        'best_val_mAP': 0.861,
        'best_val_epoch': 59,
        'best_checkpoint': 'best_coco_bbox_mAP_epoch_59.pth',
    },
    {
        'experiment_id': 'work_dirs/hard_ot_24obj_seed789',
        'work_dir': 'work_dirs/hard_ot_24obj_seed789',
        'seed': 789,
        'best_val_mAP': 0.861,
        'best_val_epoch': 98,
        'best_checkpoint': 'best_coco_bbox_mAP_epoch_98.pth',
    },
]

# ─── Test 评估数据 (from test.py 输出) ──────────────────────
# D1 Hard OT (multi_seed_aug, ross A6000, 220 test imgs, 2026-08-04)
# D2 Hard OT (24obj, workstation A5000, 1000 test imgs, 2026-08-04)
TEST_EVALS = [
    # D1 Hard OT
    {
        'experiment_id': 'work_dirs/multi_seed_aug/hard_ot/seed_42',
        'split': 'test',
        'mAP': 0.738, 'AP50': 0.926, 'AP75': 0.820,
        'AP_small': 0.488, 'AP_medium': 0.722, 'AP_large': 0.642,
        'source': 'test_eval',
    },
    {
        'experiment_id': 'work_dirs/multi_seed_aug/hard_ot/seed_123',
        'split': 'test',
        'mAP': 0.735, 'AP50': 0.924, 'AP75': 0.813,
        'AP_small': 0.470, 'AP_medium': 0.720, 'AP_large': 0.654,
        'source': 'test_eval',
    },
    {
        'experiment_id': 'work_dirs/multi_seed_aug/hard_ot/seed_789',
        'split': 'test',
        'mAP': 0.739, 'AP50': 0.929, 'AP75': 0.823,
        'AP_small': 0.518, 'AP_medium': 0.725, 'AP_large': 0.651,
        'source': 'test_eval',
    },
    # D2 Hard OT
    {
        'experiment_id': 'work_dirs/hard_ot_24obj_seed42',
        'split': 'test',
        'mAP': 0.863, 'AP50': 0.988, 'AP75': 0.969,
        'AP_small': 0.562, 'AP_medium': 0.860, 'AP_large': 0.918,
        'source': 'test_eval',
    },
    {
        'experiment_id': 'work_dirs/hard_ot_24obj_seed123',
        'split': 'test',
        'mAP': 0.862, 'AP50': 0.987, 'AP75': 0.966,
        'AP_small': 0.572, 'AP_medium': 0.859, 'AP_large': 0.912,
        'source': 'test_eval',
    },
    {
        'experiment_id': 'work_dirs/hard_ot_24obj_seed789',
        'split': 'test',
        'mAP': 0.861, 'AP50': 0.988, 'AP75': 0.968,
        'AP_small': 0.572, 'AP_medium': 0.859, 'AP_large': 0.916,
        'source': 'test_eval',
    },
]


def main():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # ─── 1. 添加 D2 Hard OT config (复用 D1 hard_ot 的 config 信息) ──
    # 查询 D1 hard_ot config 作为模板
    c.execute('''SELECT config_path, aug_pipeline_hash, coupling_type, ot_epsilon,
                        solver_type, rf_schedule, rf_shift, batch_size, max_epochs,
                        num_classes, num_proposals, sampling_timesteps, has_early_stopping,
                        time_conditioning
                 FROM config WHERE config_path LIKE '%hard_ot%' AND dataset = 'D1'
                 LIMIT 1''')
    d1_cfg = c.fetchone()
    if d1_cfg:
        d2_config_path = 'experiments/configs/multiset/chromo_24obj_hard_ot.py'
        c.execute('''INSERT OR IGNORE INTO config
            (config_path, source_config_path, dataset, data_root, aug_pipeline_hash,
             coupling_type, ot_epsilon, solver_type, rf_schedule, rf_shift,
             batch_size, max_epochs, num_classes, num_proposals, sampling_timesteps,
             has_early_stopping, time_conditioning)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (d2_config_path, d1_cfg[0], 'D2', 'data/24_chromosomes_object/coco/',
             d1_cfg[1], d1_cfg[2], d1_cfg[3], d1_cfg[4], d1_cfg[5], d1_cfg[6],
             d1_cfg[7], d1_cfg[8], d1_cfg[9], d1_cfg[10], d1_cfg[11],
             d1_cfg[12], d1_cfg[13]))
        print(f'  config: {d2_config_path} (D2, dataset=D2)')
    else:
        print('  [WARN] D1 hard_ot config 未找到, 跳过 config 插入')
        d2_config_path = None

    # ─── 2. 添加 D2 Hard OT 实验记录 ──────────────────────────
    print('\n=== D2 Hard OT 实验记录 ===')
    for exp in D2_EXPERIMENTS:
        c.execute('SELECT experiment_id FROM experiment WHERE experiment_id = ?',
                  (exp['experiment_id'],))
        if c.fetchone():
            # 已存在, 更新
            c.execute('''UPDATE experiment
                         SET best_val_mAP=?, best_val_epoch=?, best_checkpoint=?,
                             status='completed', server='workstation',
                             config_path=?
                         WHERE experiment_id=?''',
                      (exp['best_val_mAP'], exp['best_val_epoch'], exp['best_checkpoint'],
                       d2_config_path, exp['experiment_id']))
            print(f'  更新: {exp["experiment_id"]} val={exp["best_val_mAP"]}@ep{exp["best_val_epoch"]}')
        else:
            c.execute('''INSERT INTO experiment
                (experiment_id, work_dir, server, config_path, seed, status,
                 best_val_mAP, best_val_epoch, best_checkpoint)
                VALUES (?, ?, 'workstation', ?, ?, 'completed', ?, ?, ?)''',
                (exp['experiment_id'], exp['work_dir'], d2_config_path, exp['seed'],
                 exp['best_val_mAP'], exp['best_val_epoch'], exp['best_checkpoint']))
            print(f'  新增: {exp["experiment_id"]} val={exp["best_val_mAP"]}@ep{exp["best_val_epoch"]}')

    # ─── 3. 添加 test 评估记录 ─────────────────────────────────
    print('\n=== Test 评估记录 ===')
    for ev in TEST_EVALS:
        # 检查是否已存在 (同 experiment_id + split=test)
        c.execute('''SELECT eval_id FROM evaluation
                     WHERE experiment_id=? AND split='test' AND mAP=?''',
                  (ev['experiment_id'], ev['mAP']))
        if c.fetchone():
            print(f'  跳过 (已存在): {ev["experiment_id"]} test={ev["mAP"]}')
            continue
        c.execute('''INSERT INTO evaluation
            (experiment_id, split, mAP, AP50, AP75, AP_small, AP_medium, AP_large, source)
            VALUES (?, 'test', ?, ?, ?, ?, ?, ?, ?)''',
            (ev['experiment_id'], ev['mAP'], ev['AP50'], ev['AP75'],
             ev['AP_small'], ev['AP_medium'], ev['AP_large'], ev['source']))
        print(f'  新增: {ev["experiment_id"]} test={ev["mAP"]}')

    conn.commit()

    # ─── 4. 验证 ───────────────────────────────────────────────
    print('\n=== 验证 ===')
    print('D1 Hard OT:')
    c.execute('''SELECT e.experiment_id, e.best_val_mAP,
                        (SELECT mAP FROM evaluation WHERE experiment_id=e.experiment_id AND split='test') as test_mAP
                 FROM experiment e
                 WHERE e.experiment_id LIKE '%multi_seed_aug/hard_ot%'
                 ORDER BY e.seed''')
    for r in c.fetchall():
        print(f'  {r[0]}: val={r[1]}, test={r[2]}')

    print('D2 Hard OT:')
    c.execute('''SELECT e.experiment_id, e.best_val_mAP,
                        (SELECT mAP FROM evaluation WHERE experiment_id=e.experiment_id AND split='test') as test_mAP
                 FROM experiment e
                 WHERE e.experiment_id LIKE '%hard_ot_24obj%'
                 ORDER BY e.seed''')
    for r in c.fetchall():
        print(f'  {r[0]}: val={r[1]}, test={r[2]}')

    # 3-seed 统计
    import statistics
    for label, pattern in [('D1', '%multi_seed_aug/hard_ot%'), ('D2', '%hard_ot_24obj%')]:
        c.execute('''SELECT e.best_val_mAP,
                            (SELECT mAP FROM evaluation WHERE experiment_id=e.experiment_id AND split='test')
                     FROM experiment e WHERE e.experiment_id LIKE ? ORDER BY e.seed''', (pattern,))
        rows = c.fetchall()
        vals = [r[0] for r in rows if r[0]]
        tests = [r[1] for r in rows if r[1]]
        if len(vals) >= 2:
            print(f'\n{label} 3-seed val: {vals} → {statistics.mean(vals):.4f} ± {statistics.pstdev(vals):.4f}')
        if len(tests) >= 2:
            print(f'{label} 3-seed test: {tests} → {statistics.mean(tests):.4f} ± {statistics.pstdev(tests):.4f}')

    conn.close()
    print('\n✅ 数据库更新完成')


if __name__ == '__main__':
    main()
