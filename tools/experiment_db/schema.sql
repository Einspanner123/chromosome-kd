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

-- ─── 可追溯证据层 ─────────────────────────────────────────
-- 原始 experiment/evaluation 表用于自动扫描；以下表只保存经过协议核对、
-- 可追溯到具体文件的受控结果。论文数字应优先从 controlled_result 读取。
CREATE TABLE IF NOT EXISTS evidence_artifact (
    artifact_id TEXT PRIMARY KEY,
    server TEXT NOT NULL,
    path TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    kind TEXT NOT NULL,
    generated_at TEXT,
    status TEXT NOT NULL DEFAULT 'verified',
    notes TEXT
);

CREATE TABLE IF NOT EXISTS controlled_result (
    result_id TEXT PRIMARY KEY,
    family TEXT NOT NULL,
    variant TEXT NOT NULL,
    dataset TEXT NOT NULL,
    split TEXT NOT NULL,
    seed TEXT NOT NULL,                    -- integer seed or 'mean(...)'
    metric TEXT NOT NULL,
    value REAL NOT NULL,
    unit TEXT NOT NULL DEFAULT 'absolute',
    baseline_result_id TEXT,
    delta REAL,
    protocol_json TEXT NOT NULL,
    artifact_id TEXT NOT NULL,
    evidence_level TEXT NOT NULL,          -- controlled|diagnostic|running_snapshot
    paper_eligible BOOLEAN NOT NULL DEFAULT 0,
    notes TEXT,
    FOREIGN KEY (baseline_result_id) REFERENCES controlled_result(result_id),
    FOREIGN KEY (artifact_id) REFERENCES evidence_artifact(artifact_id)
);

CREATE TABLE IF NOT EXISTS finding (
    finding_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    finding_type TEXT NOT NULL,             -- positive|negative|engineering|limitation
    claim TEXT NOT NULL,
    scope TEXT NOT NULL,
    generality_basis TEXT NOT NULL,
    status TEXT NOT NULL,                   -- supported|falsified|provisional|pending
    caveat TEXT NOT NULL,
    primary_metric TEXT,
    benefit TEXT,
    evidence_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS finding_evidence (
    finding_id TEXT NOT NULL,
    result_id TEXT NOT NULL,
    PRIMARY KEY (finding_id, result_id),
    FOREIGN KEY (finding_id) REFERENCES finding(finding_id),
    FOREIGN KEY (result_id) REFERENCES controlled_result(result_id)
);

CREATE TABLE IF NOT EXISTS theory_statement (
    theory_id TEXT PRIMARY KEY,
    statement TEXT NOT NULL,
    assumptions TEXT NOT NULL,
    derivation TEXT NOT NULL,
    predicted_effect TEXT NOT NULL,
    empirical_status TEXT NOT NULL,
    source_doc TEXT NOT NULL
);

-- 物理运行粒度使用 server:work_dir，避免两台服务器相同 work_dir 冲突。
CREATE TABLE IF NOT EXISTS run_snapshot (
    run_uid TEXT PRIMARY KEY,
    server TEXT NOT NULL,
    work_dir TEXT NOT NULL,
    seed INTEGER,
    status TEXT NOT NULL,
    current_epoch INTEGER,
    max_epochs INTEGER,
    best_val_mAP REAL,
    best_val_epoch INTEGER,
    observed_at TEXT NOT NULL,
    source_path TEXT,
    notes TEXT,
    UNIQUE(server, work_dir, observed_at)
);

-- ─── 冻结数据集与样本来源 ─────────────────────────────────
CREATE TABLE IF NOT EXISTS dataset_release (
    dataset_id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    root_path TEXT NOT NULL,
    version TEXT NOT NULL,
    image_count INTEGER NOT NULL,
    category_count INTEGER NOT NULL,
    construction_protocol_json TEXT NOT NULL,
    manifest_artifact_id TEXT NOT NULL,
    status TEXT NOT NULL,
    notes TEXT,
    FOREIGN KEY (manifest_artifact_id) REFERENCES evidence_artifact(artifact_id)
);
CREATE TABLE IF NOT EXISTS dataset_split (
    dataset_id TEXT NOT NULL,
    split TEXT NOT NULL,
    image_count INTEGER NOT NULL,
    annotation_count INTEGER NOT NULL,
    annotation_path TEXT NOT NULL,
    annotation_sha256 TEXT NOT NULL,
    PRIMARY KEY (dataset_id, split),
    FOREIGN KEY (dataset_id) REFERENCES dataset_release(dataset_id)
);
CREATE TABLE IF NOT EXISTS dataset_sample_provenance (
    dataset_id TEXT NOT NULL,
    sample_file_sha256 TEXT NOT NULL,
    source_split TEXT NOT NULL,
    split TEXT NOT NULL,
    file_name TEXT NOT NULL,
    group_id TEXT NOT NULL,
    pixel_sha256 TEXT NOT NULL,
    provenance TEXT NOT NULL,
    included BOOLEAN NOT NULL,
    source_dataset_id TEXT,
    source_file_name TEXT,
    match_method TEXT NOT NULL,
    match_score REAL,
    PRIMARY KEY (dataset_id, sample_file_sha256),
    FOREIGN KEY (dataset_id) REFERENCES dataset_release(dataset_id)
);

-- ─── 预注册训练/评估运行 ──────────────────────────────────
CREATE TABLE IF NOT EXISTS train_run_registry (
    train_run_id TEXT PRIMARY KEY,
    dataset_id TEXT NOT NULL,
    method TEXT NOT NULL,
    training_seed INTEGER NOT NULL,
    config_path TEXT NOT NULL,
    config_sha256 TEXT NOT NULL,
    scientific_config_sha256 TEXT,
    dataset_manifest_sha256 TEXT NOT NULL,
    git_commit TEXT NOT NULL,
    replication_unit TEXT NOT NULL,
    parent_train_run_id TEXT,
    assigned_executor TEXT NOT NULL,
    tracker_project TEXT,
    tracker_run_name TEXT,
    tracker_run_id TEXT,
    work_dir TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (dataset_id) REFERENCES dataset_release(dataset_id),
    FOREIGN KEY (parent_train_run_id) REFERENCES train_run_registry(train_run_id)
);
CREATE TABLE IF NOT EXISTS eval_run_registry (
    eval_run_id TEXT PRIMARY KEY,
    train_run_id TEXT NOT NULL,
    split TEXT NOT NULL,
    inference_seed INTEGER NOT NULL,
    annotation_sha256 TEXT NOT NULL,
    protocol_name TEXT NOT NULL,
    protocol_sha256 TEXT NOT NULL,
    selection_source TEXT NOT NULL,
    test_tuned BOOLEAN NOT NULL DEFAULT 0,
    status TEXT NOT NULL,
    evidence_artifact_id TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (train_run_id) REFERENCES train_run_registry(train_run_id),
    FOREIGN KEY (evidence_artifact_id) REFERENCES evidence_artifact(artifact_id)
);
CREATE TABLE IF NOT EXISTS selected_checkpoint (
    train_run_id TEXT PRIMARY KEY,
    checkpoint_path TEXT NOT NULL,
    checkpoint_sha256 TEXT NOT NULL,
    selection_source TEXT NOT NULL,
    selection_metric TEXT NOT NULL,
    selection_value REAL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (train_run_id) REFERENCES train_run_registry(train_run_id)
);

-- ─── 跨数据集实验总账 ──────────────────────────────────────
CREATE TABLE IF NOT EXISTS experiment_ledger (
    ledger_id TEXT PRIMARY KEY,
    dataset_id TEXT NOT NULL,
    layer TEXT NOT NULL,
    family TEXT NOT NULL,
    variant TEXT NOT NULL,
    execution_kind TEXT NOT NULL,
    status TEXT NOT NULL,
    priority TEXT NOT NULL,
    replication_unit TEXT NOT NULL,
    training_seeds TEXT,
    inference_seeds TEXT,
    run_count INTEGER NOT NULL,
    config_path TEXT NOT NULL,
    config_status TEXT NOT NULL,
    parameters_json TEXT NOT NULL,
    executor_plan TEXT NOT NULL,
    work_dir_template TEXT NOT NULL,
    train_run_ids TEXT,
    result_family TEXT,
    evidence_artifact_ids TEXT,
    swanlab_project TEXT,
    swanlab_run_template TEXT,
    parent_ledger_id TEXT,
    paper_role TEXT NOT NULL,
    acceptance_gate TEXT NOT NULL,
    notes TEXT,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
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
CREATE UNIQUE INDEX IF NOT EXISTS idx_eval_identity
    ON evaluation(experiment_id, split, source, IFNULL(epoch, -1), IFNULL(checkpoint_path, ''));
CREATE INDEX IF NOT EXISTS idx_result_family ON controlled_result(family);
CREATE INDEX IF NOT EXISTS idx_result_dataset ON controlled_result(dataset);
CREATE INDEX IF NOT EXISTS idx_result_paper ON controlled_result(paper_eligible);
CREATE INDEX IF NOT EXISTS idx_snapshot_server ON run_snapshot(server);
CREATE INDEX IF NOT EXISTS idx_dataset_provenance
    ON dataset_sample_provenance(dataset_id, provenance, included);
CREATE UNIQUE INDEX IF NOT EXISTS idx_train_run_identity
    ON train_run_registry(dataset_id, method, training_seed, config_sha256);
CREATE INDEX IF NOT EXISTS idx_eval_train_run ON eval_run_registry(train_run_id);
CREATE INDEX IF NOT EXISTS idx_ledger_dataset ON experiment_ledger(dataset_id);
CREATE INDEX IF NOT EXISTS idx_ledger_status ON experiment_ledger(status);
CREATE INDEX IF NOT EXISTS idx_ledger_layer ON experiment_ledger(layer);

-- Historical configuration compatibility is evidence, not an alias. Only an
-- EXACT report may populate canonical_config_id.
CREATE TABLE IF NOT EXISTS config_compatibility (
    report_id TEXT PRIMARY KEY,
    legacy_method_id TEXT NOT NULL,
    legacy_config_path TEXT NOT NULL,
    legacy_config_sha256 TEXT NOT NULL,
    checkpoint_sha256 TEXT NOT NULL,
    target_config_id TEXT NOT NULL,
    target_scientific_config_sha256 TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN (
        'EXACT', 'STRUCTURAL_ONLY', 'INCOMPATIBLE', 'MISSING_EVIDENCE')),
    canonical_config_id TEXT,
    training_seed INTEGER,
    artifact_id TEXT NOT NULL,
    verified_at TEXT NOT NULL,
    notes TEXT,
    CHECK(canonical_config_id IS NULL OR status = 'EXACT'),
    FOREIGN KEY (artifact_id) REFERENCES evidence_artifact(artifact_id)
);
CREATE INDEX IF NOT EXISTS idx_config_compat_target
    ON config_compatibility(target_config_id, status);
CREATE UNIQUE INDEX IF NOT EXISTS idx_config_compat_identity
    ON config_compatibility(
        legacy_config_sha256, checkpoint_sha256, target_config_id);
