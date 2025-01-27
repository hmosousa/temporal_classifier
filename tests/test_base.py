import pytest

from src.base import INVERT_RELATION, PointRelation


class TestRelation:
    def test_relation_equality(self):
        r1 = PointRelation(source="start A", target="start B", type="<")
        r2 = PointRelation(source="start A", target="start B", type="<")
        r3 = PointRelation(source="start B", target="start A", type=">")
        r4 = PointRelation(source="start A", target="start B", type=">")

        assert r1 == r2
        assert r1 == r3
        assert r1 != r4

    def test_relation_inversion(self):
        relations = [
            (
                PointRelation(source="start A", target="start B", type="<"),
                PointRelation(source="start B", target="start A", type=">"),
            ),
            (
                PointRelation(source="start X", target="start Y", type=">"),
                PointRelation(source="start Y", target="start X", type="<"),
            ),
            (
                PointRelation(source="start P", target="start Q", type="="),
                PointRelation(source="start Q", target="start P", type="="),
            ),
            (
                PointRelation(source="start M", target="start N", type="-"),
                PointRelation(source="start N", target="start M", type="-"),
            ),
        ]

        for original, expected in relations:
            inverted = ~original
            assert inverted == expected
            assert inverted.source == expected.source
            assert inverted.target == expected.target
            assert inverted.type == expected.type

    def test_relation_invert_relation_consistency(self):
        for rel_type, inverted_type in INVERT_RELATION.items():
            r = PointRelation(source="start A", target="start B", type=rel_type)
            inverted = ~r
            assert inverted.type == inverted_type

    def test_relation_invalid_type(self):
        with pytest.raises(ValueError):
            PointRelation(source="start A", target="start B", type="invalid")

    def test_relation_source_target_swap(self):
        r1 = PointRelation(source="start A", target="start B", type="<")
        r2 = PointRelation(source="start B", target="start A", type=">")
        assert r1 == r2

        r3 = PointRelation(source="start X", target="start Y", type="=")
        r4 = PointRelation(source="start Y", target="start X", type="=")
        assert r3 == r4

        r5 = PointRelation(source="start P", target="start Q", type="-")
        r6 = PointRelation(source="start Q", target="start P", type="-")
        assert r5 == r6

    def test_relation_inequality(self):
        r1 = PointRelation(source="start A", target="start B", type="<")
        r2 = PointRelation(source="start B", target="start C", type="<")
        r3 = PointRelation(source="start A", target="start B", type="=")

        assert r1 != r2
        assert r1 != r3
        assert r2 != r3

    def test_in_list(self):
        r1 = PointRelation(source="start A", target="start B", type="<")
        r2 = PointRelation(source="start B", target="start C", type="<")
        r3 = PointRelation(source="start A", target="start B", type="=")

        relations = [r1, r2, r3]
        assert r1 in relations
        assert r2 in relations
        assert r3 in relations

    def test_hash(self):
        r1 = PointRelation(source="start A", target="start B", type="<")
        r2 = PointRelation(source="start B", target="start A", type=">")
        assert hash(r1) == hash(r2)
