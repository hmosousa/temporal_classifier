import json
import logging
from typing import Literal

import numpy as np

from fire import Fire
from sklearn.metrics import classification_report

from src.constants import RESULTS_DIR
from src.data import load_dataset
from src.interval import get_interval_relation, PAIRS

from src.model import load_model
from src.model.majority import MajorityClassifier
from src.model.random import RandomClassifier
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)


def main(
    model_name: str = "majority",
    revision: str = "main",
    dataset_name: Literal[
        "interval_tempeval", "interval_tddiscourse"
    ] = "interval_tempeval",
    strategy: Literal["high_to_low", "most_likely"] = "most_likely",
):
    """Evaluate a model with a given configuration.

    Args:
        model_name: The HuggingFace name of the model to evaluate.
        dataset_name: The name of the dataset to evaluate on.
        strategy: The strategy to use to convert the point relations to an interval relation.
    """
    logging.info(f"Loading dataset {dataset_name}")
    dataset = load_dataset(dataset_name, split="test")

    all_labels = dataset["label"]
    unique_interval_labels = list(set(all_labels))

    logging.info(f"Loading model {model_name}")
    if model_name == "random":
        classifier = RandomClassifier(unique_interval_labels)
    elif model_name == "majority":
        classifier = MajorityClassifier(all_labels)
    else:
        classifier = load_model("classifier", model_name, revision)
        label2id = classifier.model.config.label2id

    logging.info("Getting predictions")
    labels, preds = [], []
    for example in tqdm(dataset):
        if model_name in ["random", "majority"]:
            interval_relation = classifier([example["text"]])[0]["label"]
        else:
            # Generate the text for each point the model has to classify
            texts = []
            for pair in PAIRS:
                text = (
                    example["text"]
                    .replace("<source>", f"<{pair[0]}>")
                    .replace("</source>", f"</{pair[0]}>")
                    .replace("<target>", f"<{pair[1]}>")
                    .replace("</target>", f"</{pair[1]}>")
                )
                texts.append(text)

            # Get the model's prediction
            point_preds = classifier(texts, batch_size=len(texts), top_k=len(label2id))

            y_prob = np.zeros((len(texts), len(label2id)))
            for idx, pred in enumerate(point_preds):
                for label_pred in pred:
                    y_prob[idx, label2id[label_pred["label"]]] = label_pred["score"]

            interval_relation = get_interval_relation(
                y_prob, unique_interval_labels, strategy
            )

        labels.append(example["label"])
        preds.append(interval_relation if interval_relation is not None else "None")

    dataset = dataset.add_column("pred", preds)

    logging.info("Calculating metrics for all text types")
    report = classification_report(
        y_true=labels,
        y_pred=preds,
        output_dict=True,
        zero_division=0.0,
        labels=list(unique_interval_labels),
    )

    # Predictions by entity type
    unique_types = list(set(dataset["type"]))
    for type in unique_types:
        type_dataset = dataset.filter(lambda x: x["type"] == type)
        type_labels = type_dataset["label"]
        type_preds = type_dataset["pred"]
        type_report = classification_report(
            y_true=type_labels,
            y_pred=type_preds,
            output_dict=True,
            zero_division=0.0,
            labels=list(unique_interval_labels),
        )
        type_report["support"] = len(type_dataset)
        report[type] = type_report

    logging.info("Saving results")
    model_id = model_name.split("/")[-1]
    outpath = RESULTS_DIR / "interval" / dataset_name / strategy / f"{model_id}.json"
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with open(outpath, "w") as f:
        json.dump(report, f, indent=4)


if __name__ == "__main__":
    Fire(main)
