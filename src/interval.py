from typing import Dict, List, Literal

import numpy as np
from tieval.temporal_relation import _INTERVAL_TO_POINT_RELATION, PointRelation

from src.base import ID2RELATIONS, RELATIONS2ID

PAIRS = [
    ("start_source", "start_target"),
    ("start_source", "end_target"),
    ("end_source", "start_target"),
    ("end_source", "end_target"),
]

PAIRS_TO_IDX = {
    ("start_source", "start_target"): 0,
    ("start_source", "end_target"): 1,
    ("end_source", "start_target"): 2,
    ("end_source", "end_target"): 3,
}


def get_interval_relation(
    y_prob: np.ndarray,
    unique_labels: List[str],
    strategy: Literal["high_to_low", "most_likely"],
) -> str:
    """Get the interval relation from a list of predictions.

    y_prob: The probability predicted for each point relation and class.
    unique_labels: The unique labels of the interval relations.
    strategy: The strategy to use to convert the point relations to an interval relation.
    """

    interval_to_point_relation = {
        label: _INTERVAL_TO_POINT_RELATION[label] for label in unique_labels
    }

    point_to_interval_relation = {
        point: interval for interval, point in interval_to_point_relation.items()
    }

    match strategy:
        case "high_to_low":
            return _high_to_low(y_prob, point_to_interval_relation)
        case "most_likely":
            return _most_likely(y_prob, point_to_interval_relation)
        case _:
            raise ValueError(f"Invalid strategy: {strategy}")


def _high_to_low(
    y_prob: np.ndarray,
    point_to_interval_relation: Dict[PointRelation, str],
) -> str:
    """Try to form an interval relation with the highest confidence point relation."""

    point_relations = [
        {"label": ID2RELATIONS[y_pred], "score": y_prob[i, y_pred].item()}
        for i, y_pred in enumerate(y_prob.argmax(axis=1))
    ]

    # Add the entity pair to the predictions
    for pair, pred in zip(PAIRS, point_relations):
        pred["pair"] = pair

    # Sort by confidence
    point_relations = sorted(point_relations, key=lambda x: x["score"], reverse=True)

    # Get the interval relation
    running_relation = [None, None, None, None]  # ss_st, ss_et, es_st, es_et
    while point_relations:
        for pred in point_relations:
            relation = point_relations[0]["label"]
            if relation == "-":
                relation = None  # in tieval the "-" relation is None

            idx = PAIRS_TO_IDX[pred["pair"]]
            running_relation[idx] = relation
            point_relation = PointRelation(*running_relation)

            if point_relation in point_to_interval_relation:
                interval_relation = point_to_interval_relation[point_relation]
                return interval_relation
        point_relations.pop(0)
    return None


def _most_likely(
    y_prob: np.ndarray,
    point_to_interval_relation: Dict[PointRelation, str],
) -> str:
    """Try to form an interval relation with the highest confidence point relation."""

    highest_prob, interval_relation = 0.0, None
    for relation, relation_name in point_to_interval_relation.items():
        relation_probs = [
            y_prob[row_idx, RELATIONS2ID[point_relation]]
            for row_idx, point_relation in enumerate(relation.relation)
            if point_relation is not None
        ]
        relation_prob = np.prod(relation_probs) if relation_probs else 0.0
        if relation_prob > highest_prob:
            highest_prob = relation_prob
            interval_relation = relation_name
    return interval_relation
