from typing import Dict, List, Literal

import numpy as np
import torch
from tieval.temporal_relation import _INTERVAL_TO_POINT_RELATION, PointRelation
from transformers import AutoTokenizer, pipeline

from src.base import INVERT_POINT_RELATION, MODEL_ID2RELATIONS, MODEL_RELATIONS2ID
from src.model.classifier import ContextClassifier

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


class System:
    def __init__(
        self,
        model_name: str,
        revision: str = "main",
        strategy: str = "most_likely",
        interval_labels: List[str] = ["BEFORE", "AFTER", "SIMULTANEOUS"],
        double_inference: bool = True,
    ):
        tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        model = ContextClassifier.from_pretrained(model_name, revision=revision)
        self.pipe = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            revision=revision,
        )
        self.label2id = self.pipe.model.config.label2id
        self.id2label = self.pipe.model.config.id2label
        self.strategy = strategy
        self.interval_labels = interval_labels
        self.double_inference = double_inference

    def inference(self, text: str) -> str:
        """Infer the interval relation for a given text.

        Args:
            text (str): The text with the entities tagged with <source></source> and <target></target>.

        Returns:
            str: The inferred interval relation.
        """
        texts = [
            text.replace("<source>", f"<{pair[0]}>")
            .replace("</source>", f"</{pair[0]}>")
            .replace("<target>", f"<{pair[1]}>")
            .replace("</target>", f"</{pair[1]}>")
            for pair in PAIRS
        ]

        if self.double_inference:
            inv_texts = [
                text.replace("<source>", f"<{pair[1]}>")
                .replace("</source>", f"</{pair[1]}>")
                .replace("<target>", f"<{pair[0]}>")
                .replace("</target>", f"</{pair[0]}>")
                for pair in PAIRS
            ]
            texts += inv_texts

        # Get the model's prediction
        point_preds = self.pipe(
            texts,
            batch_size=2 * len(texts) if self.double_inference else len(texts),
            top_k=len(self.label2id),
        )

        if self.double_inference:
            y_prob = np.zeros((len(texts), len(self.label2id)))
            for idx, pred in enumerate(point_preds):
                for label_pred in pred:
                    y_prob[idx, self.label2id[label_pred["label"]]] = label_pred[
                        "score"
                    ]

            inverse_columns = [
                self.label2id[INVERT_POINT_RELATION[self.id2label[col]]]
                for col in range(y_prob.shape[1])
            ]
            y_prob = (
                y_prob[: len(texts) // 2] * y_prob[len(texts) // 2 :, inverse_columns]
            )
        else:
            y_prob = np.zeros((len(texts), len(self.label2id)))
            for idx, pred in enumerate(point_preds):
                for label_pred in pred:
                    y_prob[idx, self.label2id[label_pred["label"]]] = label_pred[
                        "score"
                    ]

        interval_relation = get_interval_relation(
            y_prob, self.interval_labels, self.strategy
        )
        return interval_relation

    def __call__(self, texts: List[str]) -> List[str]:
        return [self.inference(text) for text in texts]


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
        {"label": MODEL_ID2RELATIONS[y_pred], "score": y_prob[i, y_pred].item()}
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
            y_prob[row_idx, MODEL_RELATIONS2ID[point_relation]]
            for row_idx, point_relation in enumerate(relation.relation)
            if point_relation is not None
        ]
        relation_prob = np.prod(relation_probs) if relation_probs else 0.0
        if relation_prob > highest_prob:
            highest_prob = relation_prob
            interval_relation = relation_name
    return interval_relation
