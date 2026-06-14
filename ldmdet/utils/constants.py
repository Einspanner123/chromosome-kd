"""染色体分组常量

Class order: A1,A2,A3, B4,B5, C10,C11,C12,C6,C7,C8,C9,
             D13,D14,D15, E16,E17,E18, F19,F20, G21,G22, X,Y

Groups: 0=A, 1=B, 2=C, 3=D, 4=E, 5=F, 6=G, 7=Sex
"""

# class_idx → group_idx
CHROMO_GROUP_OF_CLASS = [
    0, 0, 0,       # A1, A2, A3
    1, 1,           # B4, B5
    2, 2, 2, 2, 2, 2, 2,  # C10, C11, C12, C6, C7, C8, C9
    3, 3, 3,       # D13, D14, D15
    4, 4, 4,       # E16, E17, E18
    5, 5,           # F19, F20
    6, 6,           # G21, G22
    7, 7,           # X, Y
]

NUM_CHROMO_GROUPS = 8
