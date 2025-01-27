import random
from typing import Callable

import numpy as np
import sklearn
import torch.nn.functional as F


def compute_metrics_with_vague(
    y_true,
    y_pred,
    labels=None,
    zero_division=0.0,
):
    # Compute the metrics as in https://cogcomp.seas.upenn.edu/papers/NingSuRo19.pdf
    if labels[-1] != "-":
        raise ValueError(
            f"The last label is {labels[-1]} not '-' (none relation). This method is only for the case where the last label is none relation."
        )

    confusion_matrix = sklearn.metrics.confusion_matrix(
        y_true, y_pred, labels=labels
    )  # rows are true, columns are pred
    if confusion_matrix.sum().item() == 0:
        accuracy = zero_division
    else:
        accuracy = (confusion_matrix.diagonal().sum() / confusion_matrix.sum()).item()

    pred_cm = confusion_matrix[:, :3]
    if pred_cm.sum().item() == 0:
        precision = zero_division
    else:
        precision = (pred_cm.diagonal().sum() / pred_cm.sum()).item()

    true_cm = confusion_matrix[:3, :]
    if true_cm.sum().item() == 0:
        recall = zero_division
    else:
        recall = (true_cm.diagonal().sum() / true_cm.sum()).item()

    if precision + recall == 0:
        f1 = zero_division
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1-score": f1,
    }


def compute_metrics(
    y_true,
    y_pred,
    labels=None,
    zero_division=0.0,
):
    report = sklearn.metrics.classification_report(
        y_true, y_pred, labels=labels, zero_division=zero_division, output_dict=True
    )
    return {
        "accuracy": report["accuracy"]
        if "accuracy" in report
        else report["micro avg"]["f1-score"],
        "precision": report["macro avg"]["precision"],
        "recall": report["macro avg"]["recall"],
        "f1-score": report["macro avg"]["f1-score"],
    }


def compute_confidence_intervals(
    y_true,
    y_pred,
    n_trials: int = 1_000,
    labels: list[str] = None,
    compute_metrics_func: Callable = compute_metrics,
):
    """Compute confidence intervals with bootstrap
    Args:
        y_true: The true labels.
        y_pred: The predicted labels.
        n_trials: The number of trials to run.
        labels: The labels to compute the confidence intervals for.
    """
    metrics = {"accuracy": [], "precision": [], "recall": [], "f1-score": []}
    for _ in range(n_trials):
        # Randomly sample with replacement
        sample = random.choices(list(zip(y_true, y_pred)), k=len(y_true))
        sample_y_true, sample_y_pred = zip(*sample)

        # Compute metrics
        sample_metrics = compute_metrics_func(
            y_true=sample_y_true,
            y_pred=sample_y_pred,
            labels=labels,
        )
        for metric, value in sample_metrics.items():
            metrics[metric].append(value)

    confidence_intervals = {
        "accuracy": {
            "lower": np.percentile(metrics["accuracy"], 2.5).item(),
            "upper": np.percentile(metrics["accuracy"], 97.5).item(),
        },
        "precision": {
            "lower": np.percentile(metrics["precision"], 2.5).item(),
            "upper": np.percentile(metrics["precision"], 97.5).item(),
        },
        "recall": {
            "lower": np.percentile(metrics["recall"], 2.5).item(),
            "upper": np.percentile(metrics["recall"], 97.5).item(),
        },
        "f1-score": {
            "lower": np.percentile(metrics["f1-score"], 2.5).item(),
            "upper": np.percentile(metrics["f1-score"], 97.5).item(),
        },
    }
    return confidence_intervals


def compute_loss_func(outputs, inputs, num_items_in_batch=None):
    loss = F.cross_entropy(
        input=outputs.logits,
        target=inputs,
        reduction="mean",
    )
    return loss
