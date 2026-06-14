"""ChromoGen共享常量

所有模块共享的染色体类别定义和常量，避免重复定义。
与LDMDet数据集 (Chromosome20240904_NoAug_NoResize_coco) 一致。
"""

# 24类染色体名称
CHROMO_CLASSES = (
    'A1',
    'A2',
    'A3',
    'B4',
    'B5',
    'C10',
    'C11',
    'C12',
    'C6',
    'C7',
    'C8',
    'C9',
    'D13',
    'D14',
    'D15',
    'E16',
    'E17',
    'E18',
    'F19',
    'F20',
    'G21',
    'G22',
    'X',
    'Y',
)

NUM_CLASSES = len(CHROMO_CLASSES)  # 24

# 染色体分组 (与LDMDet chromo_constants.py一致)
CHROMO_GROUP_OF_CLASS = [
    0,
    0,
    0,  # A1, A2, A3
    1,
    1,  # B4, B5
    2,
    2,
    2,
    2,
    2,
    2,
    2,  # C10, C11, C12, C6, C7, C8, C9
    3,
    3,
    3,  # D13, D14, D15
    4,
    4,
    4,  # E16, E17, E18
    5,
    5,  # F19, F20
    6,
    6,  # G21, G22
    7,
    7,  # X, Y
]

NUM_CHROMO_GROUPS = 8

# COCO category_id → 0-based class index 的映射
# COCO中category_id从1开始，class_idx从0开始
COCO_CAT_ID_TO_CLASS_IDX = {i + 1: i for i in range(NUM_CLASSES)}
