import pytest

from src.base import INVERT_RELATION, PointRelation, Timeline


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


class TestTimeline:
    @pytest.fixture
    def relations(self):
        return [
            PointRelation(source="start A", target="start B", type="<"),
            PointRelation(source="start B", target="start C", type="<"),
            PointRelation(source="start C", target="start D", type="<"),
            PointRelation(source="start E", target="start F", type=">"),
            PointRelation(source="start G", target="start H", type="-"),
        ]

    @pytest.fixture
    def relations_closure(self):
        return [
            PointRelation(source="start A", target="start B", type="<"),
            PointRelation(source="start B", target="start C", type="<"),
            PointRelation(source="start A", target="start C", type="<"),
            PointRelation(source="start C", target="start D", type="<"),
            PointRelation(source="start A", target="start D", type="<"),
            PointRelation(source="start B", target="start D", type="<"),
            PointRelation(source="start E", target="start F", type=">"),
            PointRelation(source="start G", target="start H", type="-"),
        ]

    def test_timeline_equality(self):
        t1 = Timeline(
            relations=[PointRelation(source="end A", target="start B", type="<")]
        )
        t2 = Timeline(
            relations=[PointRelation(source="end A", target="start B", type="<")]
        )
        t3 = Timeline(
            relations=[PointRelation(source="start B", target="end A", type=">")]
        )
        t4 = Timeline(
            relations=[PointRelation(source="start A", target="start B", type="=")]
        )

        assert t1 == t2
        assert t1 == t3
        assert not (t1 == t4)

    def test_timeline_valid_closure(self, relations, relations_closure):
        t = Timeline(relations=relations)
        expected_tc = Timeline(relations=relations_closure)
        tc = t.closure()
        assert tc == expected_tc

    def test_invalid_relations(self):
        relations = [
            PointRelation(source="start A", target="start B", type="<"),
            PointRelation(source="start B", target="start C", type="<"),
            PointRelation(source="start A", target="start C", type=">"),
            PointRelation(source="start D", target="start E", type=">"),
        ]
        timeline = Timeline(relations=relations)
        assert not timeline.is_valid
        assert len(timeline.invalid_relations) == 6

    def test_entities(self, relations):
        t = Timeline(relations=relations)
        assert t.entities == [
            "start A",
            "start B",
            "start C",
            "start D",
            "start E",
            "start F",
            "start G",
            "start H",
        ]

    def test_possible_relation_pairs(self):
        t = Timeline(
            relations=[
                PointRelation(source="start A", target="start B", type="<"),
                PointRelation(source="start B", target="start C", type="<"),
            ],
        )
        assert t.possible_relation_pairs == [
            ("start A", "start B"),
            ("start A", "start C"),
            ("start B", "start C"),
        ]

    def test_get_item(self):
        t = Timeline(
            relations=[PointRelation(source="start A", target="start B", type="<")]
        )
        assert t["start A", "start B"] == [
            PointRelation(source="start A", target="start B", type="<")
        ]
        assert t["start B", "start A"] == [
            PointRelation(source="start B", target="start A", type=">")
        ]

    def test_get_item_source_target(self):
        t = Timeline(
            relations=[PointRelation(source="start A", target="start B", type="<")]
        )
        rel = t["start A", "start B"][0]
        assert rel.source == "start A"
        assert rel.target == "start B"
        assert rel.type == "<"

    def test_get_item_source_target_swap(self):
        t = Timeline(
            relations=[PointRelation(source="start A", target="start B", type="<")]
        )
        rel = t["start B", "start A"][0]
        assert rel.source == "start B"
        assert rel.target == "start A"
        assert rel.type == ">"

    def test_len(self, relations):
        t = Timeline(relations=relations)
        assert len(t) == 5

    def test_contains(self):
        relations = [
            PointRelation(source="start A", target="start B", type="<"),
        ]
        t = Timeline(relations)
        assert PointRelation(source="start A", target="start B", type="<") in t
        assert PointRelation(source="start B", target="start A", type=">") in t
        assert PointRelation(source="start B", target="start C", type="<") not in t

    def test_add_relation(self):
        t = Timeline()

        t.add(PointRelation(source="start A", target="start B", type="<"))
        assert t == Timeline(
            relations=[PointRelation(source="start A", target="start B", type="<")],
        )

        t.add(PointRelation(source="start B", target="start C", type="<"))
        assert t == Timeline(
            relations=[
                PointRelation(source="start A", target="start B", type="<"),
                PointRelation(source="start B", target="start C", type="<"),
            ],
        )

    def test_from_relations(self):
        relations = [
            {"source": "end A", "target": "start B", "type": "<"},
        ]
        t = Timeline().from_relations(relations, compute_closure=True)
        expected_relations = [
            PointRelation(source="end A", target="start B", type="<"),
            PointRelation(source="start A", target="end A", type="<"),
            PointRelation(source="start B", target="end B", type="<"),
        ]
        assert t == Timeline(relations=expected_relations)
