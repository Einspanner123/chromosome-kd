"""测试 ldmdet.utils.constants — 染色体分组常量"""

from ldmdet.utils.constants import CHROMO_GROUP_OF_CLASS, NUM_CHROMO_GROUPS


class TestChromoConstants:
    def test_group_count(self):
        assert NUM_CHROMO_GROUPS == 8

    def test_group_mapping_length(self):
        """24 个类别 → 24 个分组映射"""
        assert len(CHROMO_GROUP_OF_CLASS) == 24

    def test_group_values_in_range(self):
        """所有 group_idx 在 [0, NUM_CHROMO_GROUPS) 范围内"""
        for g in CHROMO_GROUP_OF_CLASS:
            assert 0 <= g < NUM_CHROMO_GROUPS

    def test_a_group(self):
        """A 组: class 0,1,2 → group 0"""
        assert CHROMO_GROUP_OF_CLASS[0] == 0
        assert CHROMO_GROUP_OF_CLASS[1] == 0
        assert CHROMO_GROUP_OF_CLASS[2] == 0

    def test_sex_group(self):
        """性染色体组: class 22,23 → group 7"""
        assert CHROMO_GROUP_OF_CLASS[22] == 7
        assert CHROMO_GROUP_OF_CLASS[23] == 7

    def test_all_groups_present(self):
        """每个 group 至少有一个 class"""
        groups = set(CHROMO_GROUP_OF_CLASS)
        assert groups == set(range(NUM_CHROMO_GROUPS))
