import json
import logging
from typing import Literal

import numpy as np
from fire import Fire
from sklearn.metrics import classification_report

from src.constants import CACHE_DIR, RESULTS_DIR
from src.data import load_dataset
from src.model import load_model
from src.model.majority import MajorityClassifier
from src.model.random import RandomClassifier
from src.system import System
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)


def main(
    model_name: str = "<hf_user>/smol-135-0dd0da37",
    revision: str = "aadb93fe6ee0272e40eac187511de316afa94b5b",
    dataset_name: Literal[
        "interval_tempeval", "interval_tddiscourse"
    ] = "interval_tempeval",
    strategy: Literal["high_to_low", "most_likely"] = "most_likely",
    sample: bool = False,
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
    unique_interval_labels = sorted(list(set(all_labels)))

    is_interval_model = "interval" in model_name

    logging.info(f"Loading model {model_name}")
    if model_name == "random":
        classifier = RandomClassifier(unique_interval_labels)
    elif model_name == "majority":
        classifier = MajorityClassifier(all_labels)
    elif is_interval_model:
        classifier = load_model(
            model_name="classifier", model_path=model_name, revision=revision
        )
    else:
        classifier = System(model_name, revision, strategy, unique_interval_labels)

    cachepath = CACHE_DIR / "results" / "interval" / dataset_name / f"{model_name}.json"

    if cachepath.exists():
        logging.info(f"Loading predictions from {cachepath}")
        with open(cachepath, "r") as f:
            cached_preds = json.load(f)
        preds = cached_preds
    else:
        logging.info("Getting predictions")
        labels, preds = [], []
        for example in tqdm(dataset):
            if model_name in ["random", "majority"]:
                interval_relation = classifier([example["text"]])[0]["label"]
            elif is_interval_model:
                interval_relation = classifier([example["text"]], top_k=None)[0][0][
                    "label"
                ]
            else:
                if not sample:
                    pred = classifier([example["text"]])[0]
                    if pred is not None:
                        interval_relation = pred[0]["label"]
                    else:
                        interval_relation = None
                else:
                    interval_relations = classifier([example["text"]])[0]
                    probs = np.array(
                        [relation["score"] for relation in interval_relations]
                    )
                    interval_relation = np.random.choice(
                        interval_relations, size=1, p=probs
                    )[0]["label"]

            labels.append(example["label"])
            preds.append(interval_relation if interval_relation is not None else "None")

        logging.info(f"Saving predictions to {cachepath}")
        cachepath.parent.mkdir(parents=True, exist_ok=True)
        with open(cachepath, "w") as f:
            json.dump(preds, f)

    dataset = dataset.add_column("pred", preds)

    logging.info("Calculating metrics for all text types")
    report = classification_report(
        y_true=labels,
        y_pred=preds,
        output_dict=True,
        zero_division=0.0,
        labels=unique_interval_labels,
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
            labels=unique_interval_labels,
        )
        type_report["support"] = len(type_dataset)
        report[type] = type_report

    model_id = model_name.split("/")[-1]
    outpath = RESULTS_DIR / "interval" / dataset_name / strategy / f"{model_id}.json"
    outpath.parent.mkdir(parents=True, exist_ok=True)
    logging.info(f"Saving results to {outpath}")
    with open(outpath, "w") as f:
        json.dump(report, f, indent=4)


if __name__ == "__main__":
    Fire(main)
