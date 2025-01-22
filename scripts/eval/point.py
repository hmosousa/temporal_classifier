import json
import logging
from typing import Literal

from fire import Fire
from sklearn.metrics import classification_report

from src.base import RELATIONS
from src.constants import RESULTS_DIR
from src.data import load_dataset
from src.metrics import compute_confidence_intervals, compute_metrics
from src.model import load_model, MajorityClassifier, RandomClassifier

logging.basicConfig(level=logging.INFO)


TEXT_TYPES = ["ss", "ee", "se", "es"]


def add_text_type(example: dict):
    if "<start_source>" in example["text"] and "<start_target>" in example["text"]:
        example["type"] = "ss"  # start-start
    elif "<end_source>" in example["text"] and "<end_target>" in example["text"]:
        example["type"] = "ee"  # end-end
    elif "<start_source>" in example["text"] and "<end_target>" in example["text"]:
        example["type"] = "se"  # start-end
    elif "<end_source>" in example["text"] and "<start_target>" in example["text"]:
        example["type"] = "es"  # end-start
    else:
        raise ValueError(
            f"Text does not contain a valid entity pair: {example['text']}"
        )
    return example


def main(
    model_name: str = "majority",
    revision: str = "main",
    dataset_name: Literal[
        "point_tempeval",
        "timeset",
        "matres",
        "point_tddiscourse",
        "point_timebank_dense",
    ] = "timeset",
    batch_size: int = 32,
    confidence: bool = True,
):
    """Evaluate a model with a given configuration.

    Args:
        model_name: The HuggingFace name of the model to evaluate.
        revision: The revision of the model to evaluate.
        dataset_name: The name of the dataset to evaluate on.
        config_name: The name of the config to evaluate on.
        batch_size: The batch size to use for evaluation.
        confidence: Whether to calculate confidence intervals.
    """
    logging.info("Running eval with the following parameters:")
    logging.info(f"  model_name: {model_name}")
    logging.info(f"  revision: {revision}")
    logging.info(f"  dataset_name: {dataset_name}")
    logging.info(f"  batch_size: {batch_size}")
    logging.info(f"  confidence: {confidence}")

    logging.info(f"Loading dataset {dataset_name}")
    dataset = load_dataset(dataset_name, split="test")
    dataset = dataset.map(add_text_type)

    logging.info(f"Loading model {model_name}")
    if model_name == "random":
        classifier = RandomClassifier(RELATIONS)
    elif model_name == "majority":
        classifier = MajorityClassifier(dataset["label"])
    else:
        classifier = load_model("classifier", model_name, revision)

    logging.info("Getting predictions")
    preds = classifier(dataset["text"], batch_size=batch_size)
    preds = [p["label"] for p in preds]
    dataset = dataset.add_column("pred", preds)

    logging.info("Calculating metrics")
    metrics = compute_metrics(dataset["label"], dataset["pred"], labels=RELATIONS)
    if confidence:
        metrics["confidence"] = compute_confidence_intervals(
            dataset["label"], dataset["pred"], labels=RELATIONS
        )

    logging.info("Calculating metrics for each label")
    per_label = classification_report(
        y_true=dataset["label"],
        y_pred=dataset["pred"],
        output_dict=True,
        zero_division=0.0,
        labels=RELATIONS,
    )
    per_label.pop("accuracy", None)
    per_label.pop("micro avg", None)
    per_label.pop("macro avg", None)
    per_label.pop("weighted avg", None)
    metrics["per_label"] = per_label

    logging.info("Calculating metrics for each text type")
    type_metrics = {}
    dataset_text_types = list(set(dataset["type"]))
    dataset_text_types.sort()
    for text_type in dataset_text_types:
        dataset_type = dataset.filter(lambda x: x["type"] == text_type)
        type_metrics[text_type] = compute_metrics(
            dataset_type["label"], dataset_type["pred"], labels=RELATIONS
        )
        if confidence:
            type_metrics[text_type]["confidence"] = compute_confidence_intervals(
                dataset_type["label"], dataset_type["pred"], labels=RELATIONS
            )
    metrics["pre_type"] = type_metrics

    logging.info("Saving results")
    model_id = model_name.split("/")[-1]
    outpath = RESULTS_DIR / "point" / dataset_name / f"{model_id}.json"
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with open(outpath, "w") as f:
        json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    Fire(main)
