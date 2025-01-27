from typing import Dict, List, Literal

from tieval.links import TLink

RELATIONS = ["<", ">", "=", "-"]
MODEL_RELATIONS = ["<", ">", "="]

N_RELATIONS = len(RELATIONS)

INVERT_RELATION = {
    "<": ">",
    ">": "<",
    "=": "=",
    "-": "-",
}

RELATIONS2ID = {
    ">": 0,
    "<": 1,
    "=": 2,
    "-": 3,
}
ID2RELATIONS = {v: k for k, v in RELATIONS2ID.items()}

MODEL_RELATIONS2ID = {k: RELATIONS2ID[k] for k in MODEL_RELATIONS}
MODEL_ID2RELATIONS = {v: k for k, v in MODEL_RELATIONS2ID.items()}

ENDPOINT_TYPES = ["start", "end"]


class PointRelation:
    def __init__(self, source: str, target: str, type: Literal["<", ">", "=", "-"]):
        if not (source.startswith("start") or source.startswith("end")):
            raise ValueError(
                f"Invalid source: {source}. It must start with 'start' or 'end'."
            )

        if not (target.startswith("start") or target.startswith("end")):
            raise ValueError(
                f"Invalid target: {target}. It must start with 'start' or 'end'."
            )

        if type not in RELATIONS:
            raise ValueError(f"Invalid relation type: {type}")
        self.source = source
        self.target = target
        self.type = type

    def __str__(self) -> str:
        return f"{self.source} {self.type} {self.target}"

    def __repr__(self) -> str:
        return f"Relation({self.source}, {self.target}, {self.type})"

    def __eq__(self, other: "PointRelation") -> bool:
        if (
            self.source == other.source
            and self.target == other.target
            and self.type == other.type
        ):
            return True
        elif (
            self.source == other.target
            and self.target == other.source
            and self.type == INVERT_RELATION[other.type]
        ):
            return True
        return False

    def __ne__(self, other: "PointRelation") -> bool:
        return not self == other

    def __invert__(self) -> "PointRelation":
        return PointRelation(
            source=self.target, target=self.source, type=INVERT_RELATION[self.type]
        )

    def __hash__(self) -> int:
        tmp = sorted([self.source, self.target])
        if tmp[0] == self.source:
            return hash(tuple([self.source, self.target, self.type]))
        else:
            return hash(tuple([self.target, self.source, INVERT_RELATION[self.type]]))

    def to_dict(self) -> Dict:
        return {
            "source": self.source,
            "target": self.target,
            "type": self.type,
        }

    @property
    def source_endpoint(self) -> str:
        return self.source.split(" ")[0]

    @property
    def source_id(self) -> str:
        return self.source.split(" ")[1]

    @property
    def target_endpoint(self) -> str:
        return self.target.split(" ")[0]

    @property
    def target_id(self) -> str:
        return self.target.split(" ")[1]


def tlinks2relations(tlinks):
    relations = []
    for tlink in tlinks:
        if tlink.source.id == tlink.target.id:
            continue

        pr = [r if r is not None else "-" for r in tlink.relation.point.relation]

        relations += [
            PointRelation(
                f"start {tlink.source.id}", f"start {tlink.target.id}", pr[0]
            ),
            PointRelation(f"start {tlink.source.id}", f"end {tlink.target.id}", pr[1]),
            PointRelation(f"end {tlink.source.id}", f"start {tlink.target.id}", pr[2]),
            PointRelation(f"end {tlink.source.id}", f"end {tlink.target.id}", pr[3]),
            PointRelation(f"start {tlink.source.id}", f"end {tlink.source.id}", "<"),
            PointRelation(f"start {tlink.target.id}", f"end {tlink.target.id}", "<"),
        ]

    return set(relations)


def tlink_to_point_relations(tlink: TLink) -> List[PointRelation]:
    pr = [r if r is not None else "-" for r in tlink.relation.point.relation]

    return [
        PointRelation(f"start {tlink.source_id}", f"start {tlink.target_id}", pr[0]),
        PointRelation(f"start {tlink.source_id}", f"end {tlink.target_id}", pr[1]),
        PointRelation(f"end {tlink.source_id}", f"start {tlink.target_id}", pr[2]),
        PointRelation(f"end {tlink.source_id}", f"end {tlink.target_id}", pr[3]),
        PointRelation(f"start {tlink.source_id}", f"end {tlink.source_id}", "<"),
        PointRelation(f"start {tlink.target_id}", f"end {tlink.target_id}", "<"),
    ]
