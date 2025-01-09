import json
import logging
from typing import Literal

import torch
from fire import Fire
from sklearn.metrics import classification_report

from src.base import RELATIONS
from src.constants import RESULTS_DIR
from src.data import load_dataset
from src.model.majority import MajorityClassifier
from src.model.random import RandomClassifier
from transformers import pipeline

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
    model_name: str = "random",
    revision: str = "main",
    dataset_name: Literal["temporal_questions", "timeset"] = "temporal_questions",
    config_name: str = None,
    batch_size: int = 512,
    verbose: bool = False,
):
    """Evaluate a model with a given configuration.

    Args:
        model_name: The HuggingFace name of the model to evaluate.
        revision: The revision of the model to evaluate.
        dataset_name: The name of the dataset to evaluate on.
    """
    logging.info("Running eval with the following parameters:")
    logging.info(f"  model_name: {model_name}")
    logging.info(f"  revision: {revision}")
    logging.info(f"  dataset_name: {dataset_name}")
    logging.info(f"  config_name: {config_name}")
    logging.info(f"  batch_size: {batch_size}")
    logging.info(f"  verbose: {verbose}")

    logging.info(f"Loading dataset {dataset_name}")
    dataset = load_dataset(dataset_name, split="test", config=config_name)
    dataset = dataset.map(add_text_type)

    logging.info(f"Loading model {model_name}")
    if model_name == "random":
        classifier = RandomClassifier(RELATIONS)
    elif model_name == "majority":
        classifier = MajorityClassifier(dataset["label"])
    else:
        classifier = pipeline(
            "text-classification",
            model=model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            revision=revision,
        )

    logging.info("Getting predictions")
    preds = classifier(dataset["text"], batch_size=batch_size)
    preds = [p["label"] for p in preds]
    dataset = dataset.add_column("pred", preds)

    logging.info("Calculating metrics for all text types")
    report = classification_report(
        y_true=dataset["label"],
        y_pred=dataset["pred"],
        output_dict=True,
        zero_division=0.0,
        labels=RELATIONS,
    )
    if verbose:
        print(classification_report(dataset["label"], dataset["pred"], digits=4))

    logging.info("Calculating metrics for each text type")
    report["type"] = {}
    for text_type in TEXT_TYPES:
        dataset_type = dataset.filter(lambda x: x["type"] == text_type)
        report_type = classification_report(
            y_true=dataset_type["label"],
            y_pred=dataset_type["pred"],
            output_dict=True,
            zero_division=0.0,
            labels=RELATIONS,
        )
        if verbose:
            print(
                classification_report(
                    dataset_type["label"], dataset_type["pred"], digits=4
                )
            )

        report[text_type] = report_type

    logging.info("Saving results")
    model_id = model_name.split("/")[-1]
    outpath = RESULTS_DIR / "point" / dataset_name / f"{model_id}.json"
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with open(outpath, "w") as f:
        json.dump(report, f, indent=4)


if __name__ == "__main__":
    Fire(main)
