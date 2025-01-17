from typing import Dict

import datasets
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import classification_report

from src.base import Timeline


def accuracy(predicted_timeline: Timeline, true_timeline: Timeline) -> float:
    if len(true_timeline) == 0:
        return 1.0

    intersection = true_timeline & predicted_timeline
    return len(intersection) / len(true_timeline)


def precision(predicted_timeline: Timeline, true_timeline: Timeline) -> float:
    if len(true_timeline) == 0:
        return 1.0

    # get all the entity pairs in the true timeline
    predicted_rel_in_true = Timeline()
    for relation in true_timeline.relations:
        rels = predicted_timeline[relation.source, relation.target]
        for rel in rels:
            predicted_rel_in_true.add(rel)

    if len(predicted_rel_in_true) == 0:
        return 0.0

    result = len(true_timeline & predicted_rel_in_true) / len(predicted_rel_in_true)
    return result


def recall(predicted_timeline: Timeline, true_timeline: Timeline) -> float:
    # get all the entity pairs in the true timeline
    predicted_rel_in_true = Timeline()
    for relation in true_timeline.relations:
        rels = predicted_timeline[relation.source, relation.target]
        for rel in rels:
            predicted_rel_in_true.add(rel)

    if len(true_timeline) == 0:
        return 1.0

    result = len(true_timeline & predicted_rel_in_true) / len(true_timeline)
    return result


def f1(predicted_timeline: Timeline, true_timeline: Timeline) -> float:
    p = precision(predicted_timeline, true_timeline)
    r = recall(predicted_timeline, true_timeline)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def evaluate(predicted_timeline: Timeline, true_timeline: Timeline) -> Dict[str, float]:
    return {
        "accuracy": accuracy(predicted_timeline, true_timeline),
        "precision": precision(predicted_timeline, true_timeline),
        "recall": recall(predicted_timeline, true_timeline),
        "f1": f1(predicted_timeline, true_timeline),
    }


def compute_confidence_intervals(
    dataset: datasets.Dataset, n_trials: int = 1_000, labels: list[str] = None
):
    """Compute confidence intervals with bootstrap
    Args:
        dataset: The dataset to compute the confidence intervals for. Assumes the dataset has "label" and "pred" columns.
        n_trials: The number of trials to run.
        labels: The labels to compute the confidence intervals for.
    """
    reports = {
        "accuracy": [],
        "macro avg": {
            "precision": [],
            "recall": [],
            "f1-score": [],
        },
        "weighted avg": {
            "precision": [],
            "recall": [],
            "f1-score": [],
        },
    }
    for _ in range(n_trials):
        # Randomly sample with replacement
        sample = dataset.select(
            np.random.choice(len(dataset), size=len(dataset), replace=True)
        )
        # Compute metrics
        report_trial = classification_report(
            y_true=sample["label"],
            y_pred=sample["pred"],
            output_dict=True,
            zero_division=0.0,
            labels=labels,
        )
        reports["accuracy"].append(report_trial["accuracy"])
        reports["macro avg"]["precision"].append(report_trial["macro avg"]["precision"])
        reports["macro avg"]["recall"].append(report_trial["macro avg"]["recall"])
        reports["macro avg"]["f1-score"].append(report_trial["macro avg"]["f1-score"])
        reports["weighted avg"]["precision"].append(
            report_trial["weighted avg"]["precision"]
        )
        reports["weighted avg"]["recall"].append(report_trial["weighted avg"]["recall"])
        reports["weighted avg"]["f1-score"].append(
            report_trial["weighted avg"]["f1-score"]
        )
    confidence_intervals = {
        "accuracy": {
            "lower": np.percentile(reports["accuracy"], 2.5).item(),
            "upper": np.percentile(reports["accuracy"], 97.5).item(),
        },
        "macro avg": {
            "precision": {
                "lower": np.percentile(reports["macro avg"]["precision"], 2.5).item(),
                "upper": np.percentile(reports["macro avg"]["precision"], 97.5).item(),
            },
            "recall": {
                "lower": np.percentile(reports["macro avg"]["recall"], 2.5).item(),
                "upper": np.percentile(reports["macro avg"]["recall"], 97.5).item(),
            },
            "f1-score": {
                "lower": np.percentile(reports["macro avg"]["f1-score"], 2.5).item(),
                "upper": np.percentile(reports["macro avg"]["f1-score"], 97.5).item(),
            },
        },
        "weighted avg": {
            "precision": {
                "lower": np.percentile(
                    reports["weighted avg"]["precision"], 2.5
                ).item(),
                "upper": np.percentile(
                    reports["weighted avg"]["precision"], 97.5
                ).item(),
            },
            "recall": {
                "lower": np.percentile(reports["weighted avg"]["recall"], 2.5).item(),
                "upper": np.percentile(reports["weighted avg"]["recall"], 97.5).item(),
            },
            "f1-score": {
                "lower": np.percentile(reports["weighted avg"]["f1-score"], 2.5).item(),
                "upper": np.percentile(
                    reports["weighted avg"]["f1-score"], 97.5
                ).item(),
            },
        },
    }
    return confidence_intervals


def compute_loss_func(outputs, inputs, num_items_in_batch=None):
    loss = F.binary_cross_entropy_with_logits(
        input=outputs.logits,
        target=inputs,
        reduction="mean",
    )
    return loss
