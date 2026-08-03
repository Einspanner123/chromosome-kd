-- 实验数据管理系统 SQLite Schema
-- 用途: 收集所有实验数据 (ross + workstation), 提供结构化查询和文档核对能力

PRAGMA foreign_keys = ON;

-- ─── 增强策略定义 ──────────────────────────────────────────
CREATE TABLE IF NOT EXISTS aug_pipeline (
    pipeline_hash TEXT PRIMARY KEY,         -- train_pipeline 步骤类型序列的 MD5
    pipeline_name TEXT NOT NULL,            -- 'standard' | 'simple' | 'custom'
    has_random_choice_resize BOOLEAN,       -- 标准增强特征: 多尺度 Resize
    has_random_crop BOOLEAN,                -- 标准增强特征: 随机裁剪
    has_random_flip BOOLEAN,
    has_fixed_resize BOOLEAN,               -- 简单增强特征: 固定尺度 Resize
    step_types_json TEXT,                   -- pipeline 步骤类型 JSON 数组
    description TEXT
);

-- ─── 训练配置 (从 dumped config 解析) ──────────────────────
CREATE TABLE IF NOT EXISTS config (
    config_path TEXT PRIMARY KEY,           -- dumped config 相对路径
    source_config_path TEXT,                -- 源配置文件路径 (参考, 可能被修改)
    dataset TEXT,                           -- 'D1' | 'D2' | 'unknown'
    data_root TEXT,                         -- 数据集根目录
    aug_pipeline_hash TEXT,                 -- FK → aug_pipeline
    coupling_type TEXT,                     -- 'hard_ot'|'random'|'sinkhorn_stochastic'|'ot_flow'|'none'
    ot_epsilon REAL,
    ot_matcher TEXT,                        -- 'sinkhorn' | None
    ot_sample BOOLEAN,                      -- multinomial 采样
    coupling_mode TEXT,                     -- 'multinomial'|'argmax'
    time_conditioning TEXT,                 -- 'adaln_zero'|'none'
    solver_type TEXT,                       -- 'heun'|'dpm_solver_pp'|'ddpm'
    rf_schedule TEXT,                       -- 'shifted'|'linear'
    rf_shift REAL,
    batch_size INTEGER,
    max_epochs INTEGER,
    num_classes INTEGER,
    num_proposals INTEGER,
    sampling_timesteps INTEGER,
    has_early_stopping BOOLEAN,
    FOREIGN KEY (aug_pipeline_hash) REFERENCES aug_pipeline(pipeline_hash)
);

-- ─── 实验运行 (每个 seed 一行) ─────────────────────────────
CREATE TABLE IF NOT EXISTS experiment (
    experiment_id TEXT PRIMARY KEY,         -- work_dir 相对路径 (唯一标识)
    name TEXT,                              -- 人类可读名称
    work_dir TEXT NOT NULL,                 -- 相对路径
    server TEXT,                            -- 'ross' | 'workstation' | 'unknown'
    config_path TEXT,                       -- FK → config
    swanlab_project TEXT,
    swanlab_run_id TEXT,
    seed INTEGER,
    status TEXT,                            -- 'completed'|'running'|'failed'|'unknown'
    best_val_mAP REAL,
    best_val_epoch INTEGER,
    best_checkpoint TEXT,
    training_start TEXT,
    training_end TEXT,
    notes TEXT,
    FOREIGN KEY (config_path) REFERENCES config(config_path)
);

-- ─── 评估结果 (val + test, 可多行) ─────────────────────────
CREATE TABLE IF NOT EXISTS evaluation (
    eval_id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT NOT NULL,
    split TEXT NOT NULL,                    -- 'val' | 'test'
    mAP REAL,
    AP50 REAL,
    AP75 REAL,
    AP_small REAL,
    AP_medium REAL,
    AP_large REAL,
    epoch INTEGER,
    checkpoint_path TEXT,
    evaluated_at TEXT,
    source TEXT,                            -- 'training_log'|'test_eval'|'swanlab'
    FOREIGN KEY (experiment_id) REFERENCES experiment(experiment_id)
);

-- ─── 稳定性指标 ────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS stability (
    experiment_id TEXT PRIMARY KEY,
    last_30_std REAL,                       -- 最后 30 epoch mAP 标准差
    last_30_cv REAL,                        -- 变异系数 (std/mean)
    last_30_range REAL,                     -- max - min
    best_1pct_count INTEGER,                -- 最后 30 epoch 中处于 best 1% 内的 epoch 数
    total_epochs INTEGER,                   -- 完成的总 epoch 数
    per_epoch_mAP_json TEXT,                -- 所有 epoch mAP JSON 数组
    FOREIGN KEY (experiment_id) REFERENCES experiment(experiment_id)
);

-- ─── 索引 ──────────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_config_dataset ON config(dataset);
CREATE INDEX IF NOT EXISTS idx_config_coupling ON config(coupling_type);
CREATE INDEX IF NOT EXISTS idx_config_aug ON config(aug_pipeline_hash);
CREATE INDEX IF NOT EXISTS idx_experiment_config ON experiment(config_path);
CREATE INDEX IF NOT EXISTS idx_experiment_server ON experiment(server);
CREATE INDEX IF NOT EXISTS idx_experiment_seed ON experiment(seed);
CREATE INDEX IF NOT EXISTS idx_eval_experiment ON evaluation(experiment_id);
CREATE INDEX IF NOT EXISTS idx_eval_split ON evaluation(split);
