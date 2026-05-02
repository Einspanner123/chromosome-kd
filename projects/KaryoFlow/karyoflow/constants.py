"""
KaryoFlow 染色体常量定义

标准人类核型: 22 对常染色体 + 1 对性染色体 = 46 条
按 Denver 分类法排列为 A-G 七组 + 性染色体
"""

from typing import Dict, List, Tuple

# ============================================================
# 24 类染色体 (与 COCO 数据集 category 一致)
# ============================================================

CHROMO_CLASSES: Tuple[str, ...] = (
    "A1", "A2", "A3",                                    # Group A: 大, 中着丝粒
    "B4", "B5",                                           # Group B: 大, 亚中着丝粒
    "C6", "C7", "C8", "C9", "C10", "C11", "C12",        # Group C: 中, 亚中着丝粒
    "D13", "D14", "D15",                                  # Group D: 中, 近端着丝粒
    "E16", "E17", "E18",                                  # Group E: 小, 中/亚中着丝粒
    "F19", "F20",                                         # Group F: 小, 中着丝粒
    "G21", "G22",                                         # Group G: 小, 近端着丝粒
    "X", "Y",                                             # 性染色体
)

NUM_CLASSES: int = len(CHROMO_CLASSES)  # 24

# 类别名 → 0-based index
CLASS_TO_INDEX: Dict[str, int] = {c: i for i, c in enumerate(CHROMO_CLASSES)}

# 数据集 category_id (1-based) → 类别名
# 注意: COCO 标注中 category_id 从 1 开始
ID_TO_CLASS: Dict[int, str] = {i + 1: c for i, c in enumerate(CHROMO_CLASSES)}
CLASS_TO_ID: Dict[str, int] = {c: i + 1 for i, c in enumerate(CHROMO_CLASSES)}

# ============================================================
# 标准核型图 46 槽位 (Denver 分类排列)
# ============================================================
# 每个常染色体出现 2 次 (一对同源染色体)
# 性染色体: 正常男性 = X + Y, 正常女性 = X + X
# 约定: 每对中较大的在前 (slot 2i), 较小的在后 (slot 2i+1)

SLOT_ORDER: Tuple[str, ...] = (
    # Group A (slot 0-5)
    "A1", "A1", "A2", "A2", "A3", "A3",
    # Group B (slot 6-9)
    "B4", "B4", "B5", "B5",
    # Group C (slot 10-23)
    "C6", "C6", "C7", "C7", "C8", "C8", "C9", "C9",
    "C10", "C10", "C11", "C11", "C12", "C12",
    # Group D (slot 24-29)
    "D13", "D13", "D14", "D14", "D15", "D15",
    # Group E (slot 30-35)
    "E16", "E16", "E17", "E17", "E18", "E18",
    # Group F (slot 36-39)
    "F19", "F19", "F20", "F20",
    # Group G (slot 40-43)
    "G21", "G21", "G22", "G22",
    # Sex chromosomes (slot 44-45)
    "X", "Y",
)

NUM_SLOTS: int = len(SLOT_ORDER)  # 46

# ============================================================
# 类别 → 槽位映射
# ============================================================

def _build_class_to_slots() -> Dict[str, List[int]]:
    """构建类别名 → 对应槽位索引列表"""
    mapping: Dict[str, List[int]] = {}
    for i, cls_name in enumerate(SLOT_ORDER):
        mapping.setdefault(cls_name, []).append(i)
    return mapping

CLASS_TO_SLOTS: Dict[str, List[int]] = _build_class_to_slots()
# e.g. {'A1': [0, 1], 'A2': [2, 3], ..., 'X': [44], 'Y': [45]}

# ============================================================
# Denver 分组
# ============================================================

DENVER_GROUPS: Dict[str, List[str]] = {
    "A": ["A1", "A2", "A3"],
    "B": ["B4", "B5"],
    "C": ["C6", "C7", "C8", "C9", "C10", "C11", "C12"],
    "D": ["D13", "D14", "D15"],
    "E": ["E16", "E17", "E18"],
    "F": ["F19", "F20"],
    "G": ["G21", "G22"],
    "Sex": ["X", "Y"],
}

# 槽位 → Denver 组名
SLOT_TO_GROUP: Dict[int, str] = {}
for group_name, classes in DENVER_GROUPS.items():
    for cls_name in classes:
        for slot_idx in CLASS_TO_SLOTS[cls_name]:
            SLOT_TO_GROUP[slot_idx] = group_name

# ============================================================
# 同源对定义 (用于数据增强和评估)
# ============================================================
# 每对 = (slot_a, slot_b)，对内顺序可互换

HOMOLOG_PAIRS: List[Tuple[int, int]] = []
for cls_name in CHROMO_CLASSES:
    slots = CLASS_TO_SLOTS[cls_name]
    if len(slots) == 2:
        HOMOLOG_PAIRS.append((slots[0], slots[1]))
# [(0,1), (2,3), ..., (42,43)]  共 22 对常染色体
# X, Y 各只有 1 个 slot，不构成"可交换对"

# ============================================================
# 数据集路径
# ============================================================

DEFAULT_DATA_ROOT = "/data/linkst/datasets/Chromosome20240904_NoAug_NoResize_coco"
