"""染色体分组与配额常量

Class order: A1,A2,A3, B4,B5, C10,C11,C12,C6,C7,C8,C9,
             D13,D14,D15, E16,E17,E18, F19,F20, G21,G22, X,Y

Groups: 0=A, 1=B, 2=C, 3=D, 4=E, 5=F, 6=G, 7=Sex
"""

# class_idx -> group_idx
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

# 每个类别的期望倍性配额。常染色体为二倍体；性染色体平滑为 1，
# 因为 XX/XY/非整倍体应通过观测到的 GT 标签保持软性约束。
CHROMO_QUOTA_OF_CLASS = [
    2,
    2,
    2,  # A1, A2, A3
    2,
    2,  # B4, B5
    2,
    2,
    2,
    2,
    2,
    2,
    2,  # C10, C11, C12, C6, C7, C8, C9
    2,
    2,
    2,  # D13, D14, D15
    2,
    2,
    2,  # E16, E17, E18
    2,
    2,  # F19, F20
    2,
    2,  # G21, G22
    1,
    1,  # X, Y
]

NUM_CHROMO_GROUPS = 8
