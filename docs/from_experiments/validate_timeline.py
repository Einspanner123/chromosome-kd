#!/usr/bin/env python3
"""
MASTER_TIMELINE.md 数据验证脚本

扫描 work_dirs 日志提取真实 mAP，解析文档中的 mAP 声明，交叉校验并输出报告。

用法:
    python validate_timeline.py                    # 完整校验
    python validate_timeline.py --fix              # 输出修复建议
    python validate_timeline.py --json             # JSON 格式输出
"""

import argparse
import hashlib
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


WORK_DIRS_ROOT = os.environ.get(
    "WORK_DIRS_ROOT",
    "/data/linkst/chromosome-kd/work_dirs",
)
TIMELINE_PATH = os.environ.get(
    "TIMELINE_PATH",
    "/home/linkst/workspace/chromosome-kd/docs/from_experiments/MASTER_TIMELINE.md",
)

DIR_NAME_ALIASES = {
    "diffusiondet_baseline": "diffusiondet_baseline",
    "chromodet_baseline": "chromodet_baseline",
    "ldmdet_baseline": "ldmdet_baseline",
    "ldmdet_baseline_fair": "ldmdet_baseline_fair",
    "ldmdet_baseline_step4": "ldmdet_baseline_step4",
    "ldmdet_rf": "ldmdet_rf",
    "ldmdet_rf_shifted_schedule": "ldmdet_rf_shifted_schdule",
    "ldmdet_rf_shifted": "ldmdet_rf_shifted",
    "ldmdet_rf_shifted_schedule_step1": "ldmdet_rf_shifted_schdule_step1",
    "ldmdet_rf_heun_shifted": "ldmdet_rf_heun_shifted",
    "ldmdet_rf_heun_shifted_bs2": "ldmdet_rf_heun_shifted_bs2",
    "ldmdet_rf_heun_shifted_bs2_reproduce": "ldmdet_rf_heun_shifted_bs2_reproduce",
    "ldmdet_rf_heun_shifted_bs2_optimized": "ldmdet_rf_heun_shifted_bs2_optimized",
    "ldmdet_rf_heun_shifted_dist": "ldmdet_rf_heun_shifted_dist",
    "ldmdet_rf_heun_shifted_muon": "ldmdet_rf_heun_shifted_muon",
    "ldmdet_rf_heun_logit_shifted": "ldmdet_rf_heun_logit_shifted",
    "ldmdet_rf_shifted_all": "ldmdet_rf_shifted_all",
    "ldmdet_flowdet_adaln": "ldmdet_flowdet_adaln",
    "ldmdet_flowdet_adaln_cat": "ldmdet_flowdet_adaln_cat",
    "ldmdet_flowdet_adaln_cat_only": "ldmdet_flowdet_adaln_cat_only",
    "ldmdet_flowdet_adaln_obj": "ldmdet_flowdet_adaln_obj",
    "ldmdet_flowdet_adaln_convnext": "ldmdet_flowdet_adaln_convnext",
    "ldmdet_flowdet_convnext": "ldmdet_flowdet_adaln_convnext",
    "ldmdet_flowdet_adaln_crossattn": "ldmdet_flowdet_adaln_crossattn",
    "ldmdet_flowdet_adaln_crossattn_v2": "ldmdet_flowdet_adaln_crossattn_v2",
    "ldmdet_flowdet_adaln_lsas": "ldmdet_flowdet_adaln_lsas",
    "ldmdet_flowdet_ot_coupling": "ldmdet_flowdet_ot_coupling",
    "ldmdet_flowdet_adaln_ot": "ldmdet_flowdet_adaln_ot",
    "ldmdet_flowdet_sinkhorn": "ldmdet_flowdet_sinkhorn",
    "ldmdet_flowdet_adaln_ot_sinkhorn": "ldmdet_flowdet_adaln_ot_sinkhorn",
    "ldmdet_flowdet_adaln_ot_sinkhorn_eps5": "ldmdet_flowdet_adaln_ot_sinkhorn_eps5",
    "ldmdet_flowdet_adaln_ot_sinkhorn_eps10": "ldmdet_flowdet_adaln_ot_sinkhorn_eps10",
    "ldmdet_flowdet_adaln_ot_sinkhorn_eps50": "ldmdet_flowdet_adaln_ot_sinkhorn_eps50",
    "ldmdet_flowdet_adaln_ot_sinkhorn_eps100": "ldmdet_flowdet_adaln_ot_sinkhorn_eps100",
    "ldmdet_flowdet_adaln_ot_sinkhorn_sample_eps05": "ldmdet_flowdet_adaln_ot_sinkhorn_sample_eps05",
    "ldmdet_flowdet_adaln_ot_sinkhorn_sample_eps1": "ldmdet_flowdet_adaln_ot_sinkhorn_sample_eps1",
    "ldmdet_flowdet_adaln_ot_sinkhorn_sample_eps2": "ldmdet_flowdet_adaln_ot_sinkhorn_sample_eps2",
    "ldmdet_flowdet_adaln_ot_sinkhorn_sample_eps3": "ldmdet_flowdet_adaln_ot_sinkhorn_sample_eps3",
    "ldmdet_flowdet_adaln_ot_sinkhorn_sample_eps5": "ldmdet_flowdet_adaln_ot_sinkhorn_sample_eps5",
    "ldmdet_flowdet_adaln_ot_sinkhorn_sample_eps10": "ldmdet_flowdet_adaln_ot_sinkhorn_sample_eps10",
    "ldmdet_flowdet_adaln_ot_sinkhorn_sample_eps50": "ldmdet_flowdet_adaln_ot_sinkhorn_sample_eps50",
    "ldmdet_flowdet_adaln_ot_sinkhorn_sample_eps5_repro": "ldmdet_flowdet_adaln_ot_sinkhorn_sample_eps5_repro",
    "ldmdet_flowdet_adaln_ot_sinkhorn_sample_eps5_seed2": "ldmdet_flowdet_adaln_ot_sinkhorn_sample_eps5_seed2",
    "ldmdet_flowdet_adaln_group_hierarchical": "ldmdet_flowdet_adaln_group_hierarchical",
    "ldmdet_flowdet_adaln_group_hierarchical_stoch": "ldmdet_flowdet_adaln_group_hierarchical_stoch",
    "ldmdet_group_hierarchical_stoch_seed2": "ldmdet_group_hierarchical_stoch_seed2",
    "ldmdet_flowdet_adaln_trd": "ldmdet_flowdet_adaln_trd",
    "ldmdet_flowdet_adaln_trd_only": "ldmdet_flowdet_adaln_trd_only",
    "ldmdet_flowdet_adaln_trd_full": "ldmdet_flowdet_adaln_trd_full",
    "ldmdet_flowdet_adaln_lsas": "ldmdet_flowdet_adaln_lsas",
    "ldmdet_group_hierarchical_trd": "ldmdet_group_hierarchical_trd",
    "ldmdet_flowdet_adaln_stochastic_eps5_trd_cat": "ldmdet_flowdet_adaln_stochastic_eps5_trd_cat",
    "ldmdet_sinkhorn_trd_cat_lsas": "ldmdet_sinkhorn_trd_cat_lsas",
    "ldmdet_flowdet_full": "ldmdet_flowdet_full",
    "ldmdet_flowdet_velocity": "ldmdet_flowdet_velocity",
    "ldmdet_flowdet_structured_noise": "ldmdet_flowdet_structured_noise",
    "ldmdet_flowdet_objectness": "ldmdet_flowdet_objectness",
    "ldmdet_phase1": "ldmdet_phase1",
    "ldmdet_phaseA": "ldmdet_phaseA",
    "ldmdet_phaseB": "ldmdet_phaseB",
    "ldmdet_phaseC": "ldmdet_phaseC",
    "ldmdet_phase1+2": "ldmdet_phase1+2",
    "ldmdet_phase1+2+3": "ldmdet_phase1+2+3",
    "ldmdet_phase1+2_dap": "ldmdet_phase1+2_dap",
    "ldmdet_flowdet_adaln_reflow": "ldmdet_flowdet_adaln_reflow",
    "ldmdet_flowdet_adaln_reflow_v2": "ldmdet_flowdet_adaln_reflow_v2",
    "ldmdet_flowdet_adaln_reflow_v3": "ldmdet_flowdet_adaln_reflow_v3",
    "ldmdet_flowdet_adaln_reflow_v4": "ldmdet_flowdet_adaln_reflow_v4",
    "ldmdet_flowdet_adaln_reflow_v5": "ldmdet_flowdet_adaln_reflow_v5",
    "ldmdet_flowdet_adaln_reflow_v6": "ldmdet_flowdet_adaln_reflow_v6",
    "ldmdet_flowdet_adaln_reflow_det_only": "ldmdet_flowdet_adaln_reflow_det_only",
    "ldmdet_flowdet_adaln_reflow_det_only_30ep": "ldmdet_flowdet_adaln_reflow_det_only_30ep",
    "ldmdet_flowdet_adaln_reflow_det_only_lr1e6": "ldmdet_flowdet_adaln_reflow_det_only_lr1e6",
    "ldmdet_flowdet_adaln_reflow_freeze": "ldmdet_flowdet_adaln_reflow_freeze",
    "ldmdet_flowdet_adaln_reflow_freeze_stage2": "ldmdet_flowdet_adaln_reflow_freeze_stage2",
    "ldmdet_flowdet_adaln_reflow_itd": "ldmdet_flowdet_adaln_reflow_itd",
    "ldmdet_flowdet_adaln_reflow_lr1e6_vel": "ldmdet_flowdet_adaln_reflow_lr1e6_vel",
    "ldmdet_flowdet_adaln_reflow_consistency": "ldmdet_flowdet_adaln_reflow_consistency",
    "ldmdet_flowdet_adaln_reflow_pcgrad": "ldmdet_flowdet_adaln_reflow_pcgrad",
    "ldmdet_flowdet_adaln_reflow_vel_detach": "ldmdet_flowdet_adaln_reflow_vel_detach",
    "ldmdet_flowdet_adaln_reflow_v5_long": "ldmdet_flowdet_adaln_reflow_v5_long",
    "ldmdet_flowdet_adaln_reflow_v5_s03": "ldmdet_flowdet_adaln_reflow_v5_s03",
    "ldmdet_flowdet_adaln_reflow_v5_s07": "ldmdet_flowdet_adaln_reflow_v5_s07",
    "ldmdet_flowdet_adaln_reflow_v5_s10": "ldmdet_flowdet_adaln_reflow_v5_s10",
    "scale_conditioned_sc_loss": "scale_conditioned_sc_loss",
    "scale_conditioned_sc_noise": "scale_conditioned_sc_noise",
    "scale_conditioned_sc_combined": "scale_conditioned_sc_combined",
    "ldmdet_single_chromo_argmax_eps1": "ldmdet_single_chromo_argmax_eps1",
    "ldmdet_single_chromo_argmax_eps5": "ldmdet_single_chromo_argmax_eps5",
    "ldmdet_single_chromo_hard_ot": "ldmdet_single_chromo_hard_ot",
    "ldmdet_single_chromo_random": "ldmdet_single_chromo_random",
    "ldmdet_single_chromo_stoch_eps5": "ldmdet_single_chromo_stoch_eps5",
}

SHORT_NAMES = {
    "group_hierarchical_stoch": "ldmdet_flowdet_adaln_group_hierarchical_stoch",
    "trd_full": "ldmdet_flowdet_adaln_trd_full",
    "adaln": "ldmdet_flowdet_adaln",
    "sinkhorn_sample_eps5": "ldmdet_flowdet_adaln_ot_sinkhorn_sample_eps5",
    "ot_coupling": "ldmdet_flowdet_ot_coupling",
    "ot_sinkhorn": "ldmdet_flowdet_adaln_ot_sinkhorn",
    "trd_only": "ldmdet_flowdet_adaln_trd_only",
    "sinkhorn_sample_eps50": "ldmdet_flowdet_adaln_ot_sinkhorn_sample_eps50",
    "group_hierarchical_trd": "ldmdet_group_hierarchical_trd",
    "stochastic_eps5_trd_cat": "ldmdet_flowdet_adaln_stochastic_eps5_trd_cat",
    "sinkhorn_trd_cat_lsas": "ldmdet_sinkhorn_trd_cat_lsas",
    "ot_sinkhorn_eps5": "ldmdet_flowdet_adaln_ot_sinkhorn_eps5",
    "ot_sinkhorn_eps10": "ldmdet_flowdet_adaln_ot_sinkhorn_eps10",
    "ot_sinkhorn_eps50": "ldmdet_flowdet_adaln_ot_sinkhorn_eps50",
    "ot_sinkhorn_eps100": "ldmdet_flowdet_adaln_ot_sinkhorn_eps100",
    "sc_loss": "scale_conditioned_sc_loss",
    "sc_noise": "scale_conditioned_sc_noise",
    "sc_combined": "scale_conditioned_sc_combined",
    "Reflow v1（从零训练）": "ldmdet_flowdet_adaln_reflow",
    "Reflow v2（微调，val配对）": "ldmdet_flowdet_adaln_reflow_v2",
    "Reflow v3（微调，train配对）": "ldmdet_flowdet_adaln_reflow_v3",
    "Reflow v4（修复velocity_head）": "ldmdet_flowdet_adaln_reflow_v4",
    "Reflow v5（warmup+调参）": "ldmdet_flowdet_adaln_reflow_v5",
    "Reflow v6（第2轮 Reflow）": "ldmdet_flowdet_adaln_reflow_v6",
    "Reflow v6 (2轮 Reflow)": "ldmdet_flowdet_adaln_reflow_v6",
    "Reflow v5 (1轮 Reflow)": "ldmdet_flowdet_adaln_reflow_v5",
    "TRD-Full (无 Reflow)": "ldmdet_flowdet_adaln_trd_full",
    "adaln` (vanilla)": "ldmdet_flowdet_adaln",
    "argmax_eps1": "ldmdet_single_chromo_argmax_eps1",
    "argmax_eps5": "ldmdet_single_chromo_argmax_eps5",
    "hard_ot": "ldmdet_single_chromo_hard_ot",
    "random": "ldmdet_single_chromo_random",
    "stoch_eps5": "ldmdet_single_chromo_stoch_eps5",
}

SKIP_VALIDATION = {
    "TRD-Full (无 Reflow)",
    "1步",
    "2步",
    "4步",
    "8步",
    "repro",
    "seed2",
}


def scan_work_dirs(root: str) -> Dict[str, dict]:
    """扫描 work_dirs，从日志中提取每个实验的最佳 mAP 及其来源。"""
    results = {}
    root_path = Path(root)
    if not root_path.exists():
        print(f"ERROR: work_dirs root not found: {root}", file=sys.stderr)
        return results

    for exp_dir in sorted(root_path.iterdir()):
        if not exp_dir.is_dir():
            continue
        name = exp_dir.name
        all_maps = []

        for log_file in sorted(exp_dir.rglob("*.log")):
            try:
                with open(log_file, "r", errors="replace") as f:
                    for line_no, line in enumerate(f, 1):
                        m = re.search(r"coco/bbox_mAP:\s*(\d+\.\d+)", line)
                        if m:
                            val = float(m.group(1))
                            all_maps.append((val, str(log_file), line_no))
            except Exception:
                continue

        if all_maps:
            best = max(all_maps, key=lambda x: x[0])
            results[name] = {
                "best_mAP": best[0],
                "source_file": best[1],
                "source_line": best[2],
                "all_mAP_count": len(all_maps),
                "all_mAP_values": sorted([x[0] for x in all_maps], reverse=True)[:5],
            }
        else:
            results[name] = {
                "best_mAP": None,
                "source_file": None,
                "source_line": None,
                "all_mAP_count": 0,
                "all_mAP_values": [],
            }

    return results


def resolve_dir_name(doc_name: str) -> Optional[str]:
    """将文档中的实验名映射到 work_dirs 目录名。"""
    if doc_name in DIR_NAME_ALIASES:
        return DIR_NAME_ALIASES[doc_name]
    if doc_name in SHORT_NAMES:
        return SHORT_NAMES[doc_name]
    if doc_name in DIR_NAME_ALIASES.values():
        return doc_name
    return None


def parse_timeline(path: str) -> Tuple[List[dict], set]:
    """解析 MASTER_TIMELINE.md，提取所有 mAP 声明及其位置。

    返回 (claims, mentioned_dirs)，其中 mentioned_dirs 包含文档中
    已提及（包括无 mAP 值）的所有 work_dirs 目录名。
    """
    claims = []
    mentioned_dirs = set()
    with open(path, "r") as f:
        lines = f.readlines()

    for line_no, line in enumerate(lines, 1):
        stripped = line.strip()

        # Pattern 1: Markdown 表格行 | `exp_name` | 0.751 | ... |
        # 必须以 | 开头，且包含至少 3 个 | 分隔符
        if stripped.startswith("|") and stripped.count("|") >= 3:
            cells = [c.strip() for c in stripped.split("|")[1:-1]]
            if len(cells) >= 2 and not all(
                set(c.strip()) <= {"-", ":"} for c in cells
            ):
                exp_cell = cells[0].strip().strip("`").strip("*")
                map_cell = cells[1].strip().strip("*")
                map_match = re.match(r"^(\d+\.\d{3,4})$", map_cell)
                if map_match:
                    map_val = float(map_match.group(1))
                    if 0 <= map_val <= 1.0:
                        claims.append({
                            "line": line_no,
                            "exp_name": exp_cell,
                            "claimed_mAP": map_val,
                            "context": stripped[:120],
                            "type": "table",
                        })
                dir_name = resolve_dir_name(exp_cell)
                if dir_name:
                    mentioned_dirs.add(dir_name)

        # Pattern 2: 目录列表中的 mAP 值
        # 格式: ldmdet_xxx/   - 描述 (0.751) 或 (0.751, 说明) 或 （说明, 0.751；说明）
        list_match = re.match(
            r"^(ldmdet[\w+./-]*)\s*-\s+.*?[\(（].*?0\.(\d{3})",
            stripped,
        )
        if list_match:
            exp_name = list_match.group(1).strip().rstrip("/")
            map_val = float(f"0.{list_match.group(2)}")
            claims.append({
                "line": line_no,
                "exp_name": exp_name,
                "claimed_mAP": map_val,
                "context": stripped[:120],
                "type": "list",
            })
            dir_name = resolve_dir_name(exp_name)
            if dir_name:
                mentioned_dirs.add(dir_name)
            continue

        # Pattern 3: 基线目录列表 (0.xxx)
        # 格式: diffusiondet_baseline/  - 描述 (0.001) 或 （训练失败, 0.001）
        base_match = re.match(
            r"^(diffusiondet[\w+./-]*|chromodet[\w+./-]*)\s*-\s+.*?[\(（].*?0\.(\d{3})",
            stripped,
        )
        if base_match:
            exp_name = base_match.group(1).strip().rstrip("/")
            map_val = float(f"0.{base_match.group(2)}")
            claims.append({
                "line": line_no,
                "exp_name": exp_name,
                "claimed_mAP": map_val,
                "context": stripped[:120],
                "type": "list",
            })
            dir_name = resolve_dir_name(exp_name)
            if dir_name:
                mentioned_dirs.add(dir_name)
            continue

        # Pattern 4: scale_conditioned 目录列表（含内联 mAP）
        # 格式: scale_conditioned_* - ... (sc_loss=0.745, sc_noise=0.738, sc_combined=0.736)
        sc_match = re.match(
            r"^scale_conditioned[\w+./\*-]*\s*-\s+.*?\(sc_\w+=0\.(\d{3})",
            stripped,
        )
        if sc_match:
            for sc_inline in re.finditer(r"(sc_\w+)=0\.(\d{3})", stripped):
                sc_name = sc_inline.group(1)
                sc_val = float(f"0.{sc_inline.group(2)}")
                claims.append({
                    "line": line_no,
                    "exp_name": sc_name,
                    "claimed_mAP": sc_val,
                    "context": stripped[:120],
                    "type": "inline_sc",
                })
                dir_name = resolve_dir_name(sc_name)
                if dir_name:
                    mentioned_dirs.add(dir_name)
            continue

        # Pattern 5: 内联 =0.xxx 格式
        inline_matches = re.finditer(
            r"`(\w[\w+]*)`\s*=\s*0\.(\d{3})",
            stripped,
        )
        for m in inline_matches:
            claims.append({
                "line": line_no,
                "exp_name": m.group(1),
                "claimed_mAP": float(f"0.{m.group(2)}"),
                "context": stripped[:120],
                "type": "inline",
            })
            dir_name = resolve_dir_name(m.group(1))
            if dir_name:
                mentioned_dirs.add(dir_name)

        # Pattern 6: 目录列表中无 mAP 值的条目
        # 格式: ldmdet_xxx/   - 描述（无 (0.xxx) 后缀）
        no_map_match = re.match(
            r"^((?:ldmdet|diffusiondet|chromodet|scale_conditioned)[\w+./-]*)\s*-\s+",
            stripped,
        )
        if no_map_match:
            exp_name = no_map_match.group(1).strip().rstrip("/")
            dir_name = resolve_dir_name(exp_name)
            if dir_name:
                mentioned_dirs.add(dir_name)

        # Pattern 7: 通配符目录名（含内联 mAP）
        # 格式: ldmdet_single_chromo_* - 单染色体实验 (argmax_eps1=0.594, ...)
        # 格式: scale_conditioned_* - ... (sc_loss=0.745, ...)
        wildcard_match = re.match(
            r"^((?:ldmdet|scale_conditioned)[\w]*)\*\s*[-/]?\s*",
            stripped,
        )
        if wildcard_match:
            prefix = wildcard_match.group(1)
            for alias_key, alias_val in DIR_NAME_ALIASES.items():
                if alias_key.startswith(prefix) or alias_val.startswith(prefix):
                    mentioned_dirs.add(alias_val)
            for inline_m in re.finditer(r"(\w+)=0\.(\d{3})", stripped):
                inline_name = inline_m.group(1)
                inline_val = float(f"0.{inline_m.group(2)}")
                dir_name = resolve_dir_name(inline_name)
                if dir_name:
                    claims.append({
                        "line": line_no,
                        "exp_name": inline_name,
                        "claimed_mAP": inline_val,
                        "context": stripped[:120],
                        "type": "inline_wildcard",
                    })
                    mentioned_dirs.add(dir_name)

    return claims, mentioned_dirs


def validate(
    ground_truth: Dict[str, dict],
    claims: List[dict],
) -> List[dict]:
    """交叉校验文档声明与真实数据。"""
    results = []
    seen = set()

    for claim in claims:
        exp_name = claim["exp_name"]
        claimed = claim["claimed_mAP"]

        if exp_name in SKIP_VALIDATION:
            results.append({
                **claim,
                "dir_name": None,
                "actual_mAP": None,
                "status": "SKIP",
                "message": f"'{exp_name}' 需要人工验证（多步推理/特殊条目）",
            })
            continue

        dir_name = resolve_dir_name(exp_name)

        key = (exp_name, dir_name, claim["line"])
        if key in seen:
            continue
        seen.add(key)

        if dir_name is None:
            results.append({
                **claim,
                "dir_name": None,
                "actual_mAP": None,
                "status": "UNKNOWN_ALIAS",
                "message": f"无法映射 '{exp_name}' 到 work_dirs 目录",
            })
            continue

        if dir_name not in ground_truth:
            results.append({
                **claim,
                "dir_name": dir_name,
                "actual_mAP": None,
                "status": "DIR_NOT_FOUND",
                "message": f"work_dirs 中不存在目录 '{dir_name}'",
            })
            continue

        gt = ground_truth[dir_name]
        actual = gt["best_mAP"]

        if actual is None:
            results.append({
                **claim,
                "dir_name": dir_name,
                "actual_mAP": None,
                "status": "NO_LOG",
                "message": f"'{dir_name}' 无有效日志",
            })
            continue

        if abs(claimed - actual) < 0.0005:
            results.append({
                **claim,
                "dir_name": dir_name,
                "actual_mAP": actual,
                "status": "OK",
                "message": "",
                "source": f"{gt['source_file']}:{gt['source_line']}",
            })
        else:
            results.append({
                **claim,
                "dir_name": dir_name,
                "actual_mAP": actual,
                "status": "MISMATCH",
                "message": f"文档={claimed:.3f}, 实际={actual:.3f}, 差={claimed - actual:+.3f}",
                "source": f"{gt['source_file']}:{gt['source_line']}",
            })

    return results


def find_missing_entries(
    ground_truth: Dict[str, dict],
    claims: List[dict],
    mentioned_dirs: Optional[set] = None,
) -> Tuple[List[dict], List[dict]]:
    """找出 work_dirs 中有训练结果但文档中未提及的实验。

    返回 (truly_missing, no_map_value):
    - truly_missing: 文档中完全未提及的目录
    - no_map_value: 文档中已提及但未标注 mAP 值的目录
    """
    claimed_dirs_with_map = set()
    for claim in claims:
        dir_name = resolve_dir_name(claim["exp_name"])
        if dir_name:
            claimed_dirs_with_map.add(dir_name)

    all_mentioned = mentioned_dirs if mentioned_dirs else claimed_dirs_with_map

    skip_dirs = {
        "ablations",
        "chromo_coco_detection",
        "coco_ot_sinkhorn_eps5",
        "coco_random",
        "eval_multistep",
        "gradient_analysis",
        "karyoflow_overfit",
        "karyoflow_overfit2",
        "karyoflow_overfit3",
        "karyoflow_v1",
        "reflow_pairs",
        "ldmdet_phase1+2_eval_T1",
    }

    truly_missing = []
    no_map_value = []
    for dir_name, gt in sorted(ground_truth.items()):
        if dir_name in skip_dirs:
            continue
        if gt["best_mAP"] is None or gt["best_mAP"] <= 0.01:
            continue
        if dir_name in claimed_dirs_with_map:
            continue
        entry = {
            "dir_name": dir_name,
            "best_mAP": gt["best_mAP"],
            "source": f"{gt['source_file']}:{gt['source_line']}",
        }
        if dir_name in all_mentioned:
            no_map_value.append(entry)
        else:
            truly_missing.append(entry)

    return truly_missing, no_map_value


def find_unclaimed_maps(
    ground_truth: Dict[str, dict],
    claims: List[dict],
) -> List[dict]:
    """找出文档中提及但未标注 mAP 的实验（目录列表中无 mAP 值）。"""
    claimed_dirs_with_map = set()
    for claim in claims:
        dir_name = resolve_dir_name(claim["exp_name"])
        if dir_name:
            claimed_dirs_with_map.add(dir_name)

    unclaimed = []
    for dir_name, gt in sorted(ground_truth.items()):
        if dir_name not in claimed_dirs_with_map:
            continue
        if gt["best_mAP"] is not None and gt["best_mAP"] > 0.01:
            unclaimed.append({
                "dir_name": dir_name,
                "best_mAP": gt["best_mAP"],
                "source": f"{gt['source_file']}:{gt['source_line']}",
            })

    return unclaimed


def format_report(
    validation_results: List[dict],
    truly_missing: List[dict],
    no_map_value: List[dict],
    ground_truth: Dict[str, dict],
    show_fix: bool = False,
) -> str:
    """格式化输出报告。"""
    lines = []
    lines.append("=" * 72)
    lines.append("MASTER_TIMELINE.md 数据验证报告")
    lines.append("=" * 72)

    ok_count = sum(1 for r in validation_results if r["status"] == "OK")
    mismatch_count = sum(1 for r in validation_results if r["status"] == "MISMATCH")
    unknown_count = sum(1 for r in validation_results if r["status"] == "UNKNOWN_ALIAS")
    no_log_count = sum(1 for r in validation_results if r["status"] == "NO_LOG")
    dir_missing = sum(1 for r in validation_results if r["status"] == "DIR_NOT_FOUND")
    skip_count = sum(1 for r in validation_results if r["status"] == "SKIP")

    lines.append(f"\n📊 汇总: {len(validation_results)} 条 mAP 声明")
    lines.append(f"  ✅ 正确: {ok_count}")
    lines.append(f"  ❌ 不匹配: {mismatch_count}")
    lines.append(f"  ❓ 未知别名: {unknown_count}")
    lines.append(f"  📭 无日志: {no_log_count}")
    lines.append(f"  🚫 目录不存在: {dir_missing}")
    lines.append(f"  ⏭️  跳过(需人工): {skip_count}")
    lines.append(f"  🟡 已提及缺 mAP: {len(no_map_value)}")
    lines.append(f"  🟠 完全未提及: {len(truly_missing)}")

    if mismatch_count > 0:
        lines.append(f"\n{'=' * 72}")
        lines.append("❌ 数值不匹配")
        lines.append("-" * 72)
        for r in validation_results:
            if r["status"] == "MISMATCH":
                lines.append(
                    f"  L{r['line']:3d} | {r['exp_name']:<45s} | "
                    f"文档={r['claimed_mAP']:.3f} 实际={r['actual_mAP']:.3f} "
                    f"差={r['claimed_mAP'] - r['actual_mAP']:+.3f}"
                )
                lines.append(f"       来源: {r['source']}")
                if show_fix:
                    lines.append(
                        f"       修复: 将 {r['claimed_mAP']:.3f} → {r['actual_mAP']:.3f}"
                    )

    if unknown_count > 0:
        lines.append(f"\n{'=' * 72}")
        lines.append("❓ 无法映射的实验名")
        lines.append("-" * 72)
        for r in validation_results:
            if r["status"] == "UNKNOWN_ALIAS":
                lines.append(f"  L{r['line']:3d} | {r['exp_name']:<45s} | {r['message']}")

    if skip_count > 0:
        lines.append(f"\n{'=' * 72}")
        lines.append(f"⏭️  跳过（需人工验证，{skip_count} 条）")
        lines.append("-" * 72)
        for r in validation_results:
            if r["status"] == "SKIP":
                lines.append(
                    f"  L{r['line']:3d} | {r['exp_name']:<45s} | "
                    f"文档={r['claimed_mAP']:.3f} | {r['message']}"
                )

    if no_map_value:
        lines.append(f"\n{'=' * 72}")
        lines.append(f"🟡 文档已提及但缺 mAP 值（{len(no_map_value)} 个）")
        lines.append("-" * 72)
        for m in no_map_value:
            lines.append(f"  {m['dir_name']:<55s} 实际 mAP={m['best_mAP']:.3f}")
            if show_fix:
                lines.append(f"       建议: 补充 mAP 值 ({m['best_mAP']:.3f})")

    if truly_missing:
        lines.append(f"\n{'=' * 72}")
        lines.append(f"🟠 文档完全未提及的实验（{len(truly_missing)} 个）")
        lines.append("-" * 72)
        for m in truly_missing:
            lines.append(f"  {m['dir_name']:<55s} mAP={m['best_mAP']:.3f}")
            lines.append(f"  {'':55s} 来源: {m['source']}")

    lines.append(f"\n{'=' * 72}")
    lines.append("📋 完整 ground truth（work_dirs 最佳 mAP）")
    lines.append("-" * 72)
    for dir_name in sorted(ground_truth.keys()):
        gt = ground_truth[dir_name]
        if gt["best_mAP"] is not None:
            lines.append(
                f"  {dir_name:<55s} mAP={gt['best_mAP']:.3f}  "
                f"({gt['source_file'].split('/')[-1]}:{gt['source_line']})"
            )
        else:
            lines.append(f"  {dir_name:<55s} (无日志)")

    lines.append(f"\n{'=' * 72}")
    lines.append("✅ 验证通过的所有条目")
    lines.append("-" * 72)
    for r in validation_results:
        if r["status"] == "OK":
            lines.append(
                f"  L{r['line']:3d} | {r['exp_name']:<45s} | "
                f"mAP={r['actual_mAP']:.3f} ← {r['source'].split('/')[-1]}"
            )

    lines.append("")
    lines.append("=" * 72)
    if mismatch_count == 0:
        lines.append("🎉 所有已标注 mAP 数值均与 work_dirs 日志一致！")
    else:
        lines.append(f"⚠️  发现 {mismatch_count} 处数值不匹配，请检查上方详情。")
    lines.append("=" * 72)

    return "\n".join(lines)


MODS_CORE_FILES = [
    "diffusiondet_head.py",
    "loss.py",
    "sinkhorn.py",
    "rectified_flow.py",
    "reflow.py",
    "noise_sampler.py",
    "modules.py",
    "single_head.py",
    "consistency.py",
    "roi_extractor.py",
    "structures.py",
    "utils.py",
]


def _file_hash(p: Path) -> Optional[str]:
    if not p.exists():
        return None
    try:
        return hashlib.md5(p.read_bytes()).hexdigest()
    except Exception:
        return None


def scan_backup_paths(root: str) -> Dict[str, dict]:
    """扫描 work_dirs，为每个实验找到最新备份及差异模块。

    返回 {dir_name: {run_ts, backup_rel_path, config, modified_mods}}。
    """
    root_path = Path(root)
    if not root_path.exists():
        return {}

    ref_dir = root_path / "ldmdet_flowdet_adaln"
    ref_backups = sorted(ref_dir.glob("*/LDMDet_backup"), reverse=True)
    ref_hashes: Dict[str, Optional[str]] = {}
    ref_model_hash: Optional[str] = None
    if ref_backups:
        ref = ref_backups[0]
        for f in MODS_CORE_FILES:
            ref_hashes[f] = _file_hash(ref / "mods" / f)
        ref_model_hash = _file_hash(ref / "model.py")

    results: Dict[str, dict] = {}
    for exp_dir in sorted(root_path.iterdir()):
        if not exp_dir.is_dir():
            continue
        name = exp_dir.name
        backups = sorted(exp_dir.glob("*/LDMDet_backup"), reverse=True)
        if not backups:
            continue
        latest = backups[0]
        run_ts = latest.parent.name

        config_match = None
        config_dir = latest / "configs"
        has_code = any(p.suffix == ".py" for p in latest.rglob("*") if p.is_file())
        if config_dir.exists() and has_code:
            all_configs = sorted(config_dir.rglob("*.py"))
            name_norm = name.replace("+", "_")
            best_match = None
            best_score = -1
            for cfg in all_configs:
                cfg_stem = cfg.stem.replace("+", "_")
                rel = str(cfg.relative_to(config_dir))
                if cfg_stem == name_norm:
                    config_match = f"configs/{rel}"
                    best_match = None
                    break
                if name_norm.startswith(cfg_stem) or cfg_stem.startswith(name_norm):
                    overlap = min(len(name_norm), len(cfg_stem))
                    if overlap > best_score:
                        best_score = overlap
                        best_match = f"configs/{rel}"
            if config_match is None and best_match is not None:
                config_match = best_match
        if config_match is None:
            root_cfg = exp_dir / f"{name}.py"
            if root_cfg.exists():
                config_match = f"../{name}.py"

        modified: List[str] = []
        if not has_code:
            incomplete = True
        else:
            incomplete = False
            model_h = _file_hash(latest / "model.py")
            if model_h and model_h != ref_model_hash:
                modified.append("model.py")
            for f in MODS_CORE_FILES:
                h = _file_hash(latest / "mods" / f)
                if h and h != ref_hashes.get(f):
                    modified.append(f"mods/{f}")

        results[name] = {
            "run_ts": run_ts,
            "backup_rel_path": f"{name}/{run_ts}/LDMDet_backup",
            "config": config_match,
            "modified_mods": modified,
            "incomplete": incomplete,
        }

    return results


def add_backup_paths_to_doc(timeline_path: str, backup_info: Dict[str, dict]) -> str:
    """在 MASTER_TIMELINE.md 的代码块中为实验条目添加备份路径标注。

    格式: 在实验条目下方添加一行
        ↳ <timestamp>/LDMDet_backup/ → configs/xxx.py, mods/yyy.py
    """
    with open(timeline_path, "r") as f:
        lines = f.readlines()

    in_code_block = False
    result: List[str] = []
    i = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.rstrip("\n")

        if stripped.strip().startswith("```"):
            in_code_block = not in_code_block
            result.append(line)
            i += 1
            continue

        if not in_code_block:
            result.append(line)
            i += 1
            continue

        exp_name = None
        dir_name = None

        m_list = re.match(
            r"^(\s*)((?:ldmdet|diffusiondet|chromodet|scale_conditioned)[\w+./-]*)\s*-\s+",
            stripped,
        )
        if m_list:
            exp_name = m_list.group(2).strip().rstrip("/")
            dir_name = resolve_dir_name(exp_name)

        if dir_name is None:
            m_wc = re.match(
                r"^(\s*)((?:ldmdet|scale_conditioned)[\w]*)\*",
                stripped,
            )
            if m_wc:
                prefix = m_wc.group(2)
                wc_indent = m_wc.group(1)
                wc_entries = []
                for alias_val in sorted(set(DIR_NAME_ALIASES.values())):
                    if alias_val.startswith(prefix) and alias_val in backup_info:
                        wc_entries.append(alias_val)
                if wc_entries:
                    result.append(line)
                    already_has_annotation = False
                    if i + 1 < len(lines):
                        next_stripped = lines[i + 1].strip()
                        if next_stripped.startswith("↳"):
                            already_has_annotation = True
                    if not already_has_annotation:
                        for wc_name in wc_entries:
                            info = backup_info[wc_name]
                            parts = []
                            if info["config"]:
                                parts.append(info["config"])
                            parts.extend(info["modified_mods"])
                            if parts:
                                ann = f"{wc_indent}    ↳ {wc_name}/{info['run_ts']}/LDMDet_backup/ → {', '.join(parts)}"
                            else:
                                ann = f"{wc_indent}    ↳ {wc_name}/{info['run_ts']}/LDMDet_backup/"
                            result.append(ann + "\n")
                    i += 1
                    continue

        result.append(line)

        if dir_name and dir_name in backup_info:
            info = backup_info[dir_name]
            indent = m_list.group(1) if m_list else "    "

            if info.get("incomplete"):
                annotation = f"{indent}    ↳ {info['run_ts']}/LDMDet_backup/ (仅文档，无代码备份)"
                if info["config"]:
                    annotation += f"; config→{info['config']}"
            else:
                parts = []
                if info["config"]:
                    parts.append(info["config"])
                parts.extend(info["modified_mods"])
                if parts:
                    annotation = f"{indent}    ↳ {info['run_ts']}/LDMDet_backup/ → {', '.join(parts)}"
                else:
                    annotation = f"{indent}    ↳ {info['run_ts']}/LDMDet_backup/"

            already_has_annotation = False
            if i + 1 < len(lines):
                next_stripped = lines[i + 1].strip()
                if next_stripped.startswith("↳"):
                    already_has_annotation = True

            if not already_has_annotation:
                result.append(annotation + "\n")

        i += 1

    return "".join(result)


def parse_backup_annotations(
    timeline_path: str,
) -> List[dict]:
    """解析文档中已有的 ↳ 标注行，提取结构化信息。"""
    annotations = []
    with open(timeline_path, "r") as f:
        lines = f.readlines()

    in_code_block = False
    last_exp_line = None
    last_exp_name = None
    last_dir_name = None

    for line_no, line in enumerate(lines, 1):
        stripped = line.strip()

        if stripped.startswith("```"):
            in_code_block = not in_code_block
            continue

        if not in_code_block:
            continue

        m_list = re.match(
            r"^(\s*)((?:ldmdet|diffusiondet|chromodet|scale_conditioned)[\w+./-]*)\s*-\s+",
            stripped,
        )
        if m_list:
            last_exp_line = line_no
            last_exp_name = m_list.group(2).strip().rstrip("/")
            last_dir_name = resolve_dir_name(last_exp_name)
            continue

        m_wc = re.match(
            r"^(\s*)((?:ldmdet|scale_conditioned)[\w]*)\*",
            stripped,
        )
        if m_wc:
            last_exp_line = line_no
            last_exp_name = m_wc.group(2).strip()
            last_dir_name = None
            continue

        ann_match = re.match(r"^\s*↳\s+(.+?)/(\d{8}_\d{6})/LDMDet_backup/(?:\s*→\s*(.+))?$", stripped)
        if ann_match:
            ann_dir = ann_match.group(1).strip()
            ann_ts = ann_match.group(2)
            ann_files_str = ann_match.group(3)
            ann_files = []
            if ann_files_str:
                ann_files = [f.strip() for f in ann_files_str.split(",") if f.strip()]

            resolved_dir = resolve_dir_name(ann_dir) or ann_dir
            annotations.append({
                "line": line_no,
                "parent_exp_line": last_exp_line,
                "parent_exp_name": last_exp_name,
                "dir_name": resolved_dir,
                "timestamp": ann_ts,
                "files": ann_files,
                "raw": stripped,
            })
            continue

        ann_match2 = re.match(r"^\s*↳\s+(\d{8}_\d{6})/LDMDet_backup/(?:\s*→\s*(.+))?$", stripped)
        if ann_match2:
            ann_ts = ann_match2.group(1)
            ann_files_str = ann_match2.group(2)
            ann_files = []
            if ann_files_str:
                ann_files = [f.strip() for f in ann_files_str.split(",") if f.strip()]

            annotations.append({
                "line": line_no,
                "parent_exp_line": last_exp_line,
                "parent_exp_name": last_exp_name,
                "dir_name": last_dir_name,
                "timestamp": ann_ts,
                "files": ann_files,
                "raw": stripped,
            })

    return annotations


def validate_backup_paths(
    annotations: List[dict],
    work_dirs_root: str,
    backup_info: Dict[str, dict],
) -> List[dict]:
    """验证文档中已有 ↳ 标注的正确性。"""
    root = Path(work_dirs_root)
    results = []

    for ann in annotations:
        issues = []
        dir_name = ann["dir_name"]
        ts = ann["timestamp"]
        doc_files = ann["files"]

        if dir_name is None:
            results.append({**ann, "status": "UNRESOLVED_DIR", "issues": ["无法解析目录名"]})
            continue

        exp_dir = root / dir_name
        if not exp_dir.exists():
            results.append({**ann, "status": "DIR_NOT_FOUND", "issues": [f"目录不存在: {dir_name}"]})
            continue

        ts_dir = exp_dir / ts / "LDMDet_backup"
        if not ts_dir.exists():
            results.append({**ann, "status": "TS_NOT_FOUND", "issues": [f"时间戳目录不存在: {ts}"]})
            continue

        ts_contents = set()
        for p in ts_dir.rglob("*"):
            if p.is_file():
                rel = str(p.relative_to(ts_dir))
                ts_contents.add(rel)

        has_code = any(
            f.endswith(".py") for f in ts_contents
        )
        if not has_code:
            issues.append("backup 仅含文档，无代码文件")

        doc_config = None
        doc_mods = []
        for f in doc_files:
            if f.startswith("configs/"):
                doc_config = f
            else:
                doc_mods.append(f)

        if doc_files and not has_code:
            issues.append(f"文档列出 {len(doc_files)} 个文件但 backup 无代码")

        if doc_config:
            config_path = ts_dir / doc_config
            if not config_path.exists():
                issues.append(f"配置文件不存在: {doc_config}")

                if dir_name in backup_info:
                    expected = backup_info[dir_name].get("config")
                    if expected:
                        expected_path = ts_dir / expected
                        if expected_path.exists():
                            issues.append(f"  正确配置应为: {expected}")
            else:
                if dir_name in backup_info:
                    expected = backup_info[dir_name].get("config")
                    if expected and expected != doc_config:
                        config_stem = Path(doc_config).stem.replace("+", "_")
                        name_norm = dir_name.replace("+", "_")
                        expected_stem = Path(expected).stem.replace("+", "_")
                        if expected_stem == name_norm and config_stem != name_norm:
                            issues.append(
                                f"配置文件不精确: 文档={doc_config}, "
                                f"更精确匹配={expected}"
                            )
                        elif name_norm.startswith(expected_stem) and not name_norm.startswith(config_stem):
                            issues.append(
                                f"配置文件不精确: 文档={doc_config}, "
                                f"更精确匹配={expected}"
                            )
                        elif expected_stem.startswith(config_stem) and expected_stem != config_stem:
                            exp_overlap = len(expected_stem)
                            doc_overlap = len(config_stem)
                            if exp_overlap > doc_overlap:
                                issues.append(
                                    f"配置文件不精确: 文档={doc_config}, "
                                    f"更精确匹配={expected}"
                                )

        for mod_file in doc_mods:
            mod_path = ts_dir / mod_file
            if not mod_path.exists():
                issues.append(f"文件不存在: {mod_file}")

        if doc_files and has_code:
            actual_configs = set()
            for p in (ts_dir / "configs").rglob("*.py") if (ts_dir / "configs").exists() else []:
                rel = str(p.relative_to(ts_dir))
                actual_configs.add(rel)
            if actual_configs and not doc_config:
                name_norm = dir_name.replace("+", "_")
                matching = [c for c in actual_configs if Path(c).stem.replace("+", "_") == name_norm]
                if matching:
                    issues.append(f"缺少配置文件标注，应包含: {matching[0]}")

        if not issues:
            results.append({**ann, "status": "OK", "issues": []})
        else:
            results.append({**ann, "status": "ISSUE", "issues": issues})

    return results


def format_backup_report(backup_results: List[dict]) -> str:
    """格式化 backup 路径验证报告。"""
    lines = []
    lines.append("=" * 72)
    lines.append("Backup 路径标注验证报告")
    lines.append("=" * 72)

    ok_count = sum(1 for r in backup_results if r["status"] == "OK")
    issue_count = sum(1 for r in backup_results if r["status"] == "ISSUE")
    other_count = len(backup_results) - ok_count - issue_count

    lines.append(f"\n📊 汇总: {len(backup_results)} 条 ↳ 标注")
    lines.append(f"  ✅ 正确: {ok_count}")
    lines.append(f"  ⚠️  有问题: {issue_count}")
    lines.append(f"  ❓ 其他: {other_count}")

    if issue_count > 0:
        lines.append(f"\n{'=' * 72}")
        lines.append(f"⚠️  有问题的标注（{issue_count} 条）")
        lines.append("-" * 72)
        for r in backup_results:
            if r["status"] == "ISSUE":
                dir_name = r["dir_name"] or "?"
                lines.append(
                    f"  L{r['line']:3d} | {dir_name:<50s} | "
                    f"ts={r['timestamp']}"
                )
                for issue in r["issues"]:
                    lines.append(f"       ⚠ {issue}")

    if other_count > 0:
        lines.append(f"\n{'=' * 72}")
        lines.append(f"❓ 无法验证的标注（{other_count} 条）")
        lines.append("-" * 72)
        for r in backup_results:
            if r["status"] not in ("OK", "ISSUE"):
                dir_name = r["dir_name"] or "?"
                lines.append(
                    f"  L{r['line']:3d} | {dir_name:<50s} | "
                    f"状态={r['status']}"
                )
                for issue in r.get("issues", []):
                    lines.append(f"       ⚠ {issue}")

    lines.append("")
    lines.append("=" * 72)
    if issue_count == 0:
        lines.append("🎉 所有 backup 路径标注均正确！")
    else:
        lines.append(f"⚠️  发现 {issue_count} 条标注有问题，请检查上方详情。")
    lines.append("=" * 72)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="验证 MASTER_TIMELINE.md 中的 mAP 数据")
    parser.add_argument("--fix", action="store_true", help="显示修复建议")
    parser.add_argument("--json", action="store_true", help="JSON 格式输出")
    parser.add_argument(
        "--add-backup-paths",
        action="store_true",
        help="为文档中的实验条目添加备份路径标注",
    )
    parser.add_argument(
        "--work-dirs",
        default=WORK_DIRS_ROOT,
        help="work_dirs 根目录路径",
    )
    parser.add_argument(
        "--timeline",
        default=TIMELINE_PATH,
        help="MASTER_TIMELINE.md 文件路径",
    )
    args = parser.parse_args()

    if args.add_backup_paths:
        print("扫描 work_dirs 备份路径...", file=sys.stderr)
        backup_info = scan_backup_paths(args.work_dirs)
        print(f"  找到 {len(backup_info)} 个实验备份", file=sys.stderr)

        print("为文档添加备份路径标注...", file=sys.stderr)
        updated = add_backup_paths_to_doc(args.timeline, backup_info)

        with open(args.timeline, "w") as f:
            f.write(updated)
        print(f"  已写入 {args.timeline}", file=sys.stderr)
        return

    print("扫描 work_dirs 日志...", file=sys.stderr)
    ground_truth = scan_work_dirs(args.work_dirs)
    print(f"  找到 {len(ground_truth)} 个实验目录", file=sys.stderr)

    print("解析 MASTER_TIMELINE.md...", file=sys.stderr)
    claims, mentioned_dirs = parse_timeline(args.timeline)
    print(f"  找到 {len(claims)} 条 mAP 声明，{len(mentioned_dirs)} 个已提及目录", file=sys.stderr)

    print("交叉校验...", file=sys.stderr)
    validation_results = validate(ground_truth, claims)
    truly_missing, no_map_value = find_missing_entries(
        ground_truth, claims, mentioned_dirs
    )

    print("验证 backup 路径标注...", file=sys.stderr)
    backup_info = scan_backup_paths(args.work_dirs)
    annotations = parse_backup_annotations(args.timeline)
    print(f"  找到 {len(annotations)} 条 ↳ 标注", file=sys.stderr)
    backup_results = validate_backup_paths(annotations, args.work_dirs, backup_info)

    if args.json:
        output = {
            "ground_truth": ground_truth,
            "validation": validation_results,
            "truly_missing": truly_missing,
            "no_map_value": no_map_value,
            "backup_validation": backup_results,
        }
        print(json.dumps(output, indent=2, ensure_ascii=False))
    else:
        report = format_report(
            validation_results, truly_missing, no_map_value, ground_truth, args.fix
        )
        print(report)

        backup_report = format_backup_report(backup_results)
        print()
        print(backup_report)

    mismatch_count = sum(1 for r in validation_results if r["status"] == "MISMATCH")
    backup_issue_count = sum(1 for r in backup_results if r["status"] == "ISSUE")
    sys.exit(1 if (mismatch_count > 0 or backup_issue_count > 0) else 0)


if __name__ == "__main__":
    main()
