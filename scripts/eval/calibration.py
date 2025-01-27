import json
import logging
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from fire import Fire
from sklearn.calibration import calibration_curve

from src.constants import CACHE_DIR, IMGS_DIR
from src.data import load_dataset
from src.model import load_model

logging.basicConfig(level=logging.INFO)


def expected_calibration_error(y_true, y_prob, n_bins=5):
    # uniform binning approach with n_bins number of bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    # get max probability per sample i
    confidences = np.max(y_prob, axis=1)
    # get predictions from confidences (positional in this case)
    predicted_label = np.argmax(y_prob, axis=1)

    # get a boolean list of correct/false predictions
    accuracies = predicted_label == y_true

    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # determine if sample is in bin m (between bin lower & upper)
        in_bin = np.logical_and(
            confidences > bin_lower.item(), confidences <= bin_upper.item()
        )
        # can calculate the empirical probability of a sample falling into bin m: (|Bm|/n)
        prob_in_bin = in_bin.mean()

        if prob_in_bin.item() > 0:
            # get the accuracy of bin m: acc(Bm)
            accuracy_in_bin = accuracies[in_bin].mean()
            # get the average confidence of bin m: conf(Bm)
            avg_confidence_in_bin = confidences[in_bin].mean()
            # calculate |acc(Bm) - conf(Bm)| * (|Bm|/n) for bin m and add to the total ECE
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prob_in_bin
    return ece.item()


def main(
    model_name: str = "hugosousa/smol-135-fe09845a",
    revision: str = "main",
    dataset_name: Literal[
        "point_tempeval",
        "timeset",
        "matres",
        "point_tddiscourse",
        "point_timebank_dense",
    ] = "point_tempeval",
    batch_size: int = 512,
    n_bins: int = 20,
    strategy: Literal["uniform", "quantile"] = "quantile",
):
    """Evaluate a model with a given configuration.

    Args:
        model_name: The HuggingFace name of the model to evaluate.
        revision: The revision of the model to evaluate.
        dataset_name: The name of the dataset to evaluate on.
        strategy: The strategy to use for calibration.
    """
    IMGS_DIR.mkdir(parents=True, exist_ok=True)

    logging.info("Running calibration with the following parameters:")
    logging.info(f"  model_name: {model_name}")
    logging.info(f"  revision: {revision}")
    logging.info(f"  dataset_name: {dataset_name}")
    logging.info(f"  batch_size: {batch_size}")
    logging.info(f"  n_bins: {n_bins}")
    logging.info(f"  strategy: {strategy}")
    logging.info(f"Loading dataset {dataset_name}")

    cachepath = CACHE_DIR / "results" / "point" / dataset_name / f"{model_name}.json"

    dataset = load_dataset(dataset_name, split="test")

    classifier = load_model("classifier", model_name, revision)

    id2label = classifier.model.config.id2label
    label2id = classifier.model.config.label2id
    unique_labels = list(id2label.values())

    if cachepath.exists():
        logging.info(f"Loading predictions from {cachepath}")
        with open(cachepath, "r") as f:
            preds = json.load(f)
    else:
        logging.info(f"Loading model {model_name}")

        logging.info("Getting predictions")
        preds = classifier(
            dataset["text"], batch_size=batch_size, top_k=len(unique_labels)
        )

        logging.info(f"Saving predictions to {cachepath}")
        cachepath.parent.mkdir(parents=True, exist_ok=True)
        with open(cachepath, "w") as f:
            json.dump(preds, f)

    # Convert pipeline predictions to a matrix of probabilities
    y_prob = np.zeros((len(dataset), len(unique_labels)))
    for i, p in enumerate(preds):
        for pred in p:
            y_prob[i, label2id[pred["label"]]] = pred["score"]

    # Convert dataset labels to label idxs
    y_true = np.array([label2id[label] for label in dataset["label"]])

    ece = expected_calibration_error(y_true, y_prob, n_bins=n_bins)
    logging.info(f"ECE: {ece}")

    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    for label_idx, label in id2label.items():
        # transform dataset to binary classification
        y_true_binary = (y_true == label_idx).astype(int)
        y_prob_binary = y_prob[:, label_idx]
        prob_true, prob_pred = calibration_curve(
            y_true_binary,
            y_prob_binary,
            n_bins=n_bins,
            strategy=strategy,
        )
        plt.plot(
            prob_pred,
            prob_true,
            label=f"{label}",
            linewidth=2,
            linestyle="-",
            marker="o",
            markersize=6,
        )
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title(f"ECE: {ece:.4f}")
    plt.grid()
    plt.legend()
    fout = IMGS_DIR / f"calibration_{model_name.split('/')[-1]}_{dataset_name}.png"
    plt.savefig(fout, bbox_inches="tight", dpi=300)
    plt.close()


if __name__ == "__main__":
    Fire(main)
