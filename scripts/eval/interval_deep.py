import json
import logging

import datasets
from fire import Fire
from sklearn.metrics import classification_report

from src.constants import RESULTS_DIR
from src.data import load_dataset
from src.data.tempeval import INTERVAL_RELATIONS
from src.model.majority import MajorityClassifier
from src.model.random import RandomClassifier
from src.system import System
from tieval.closure import temporal_closure
from tieval.links import TLink
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)

datasets.utils.logging.set_verbosity_error()


def main(
    model_name: str = "majority",
    revision: str = "main",
    dataset_name: str = "interval_tempeval",
    strategy: str = "most_likely",
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
        classifier = System(model_name, revision, strategy, unique_interval_labels)

    logging.info("Getting predictions")
    docs = set(dataset["doc"])
    doc_datasets = []
    for doc in tqdm(docs):
        doc_dataset = dataset.filter(lambda x: x["doc"] == doc)
        tlinks = {}
        for example in doc_dataset:
            source, target = example["source"], example["target"]
            if (source, target) in tlinks or (target, source) in tlinks:
                # The relation has already been inferred
                continue

            if model_name in ["random", "majority"]:
                interval_relation = classifier([example["text"]])[0]["label"]
            else:
                interval_relation = classifier([example["text"]])[0]

            tlink = TLink(
                source=example["source"],
                target=example["target"],
                relation=interval_relation,
            )
            tlinks[(source, target)] = tlink
            tlinks[(target, source)] = tlink

            inferred_tlinks = temporal_closure(set(tlinks.values()))
            for tlink in inferred_tlinks:
                if tlink.relation.interval in INTERVAL_RELATIONS:
                    if tlink.relation.interval == "AFTER":
                        tlink
                    tlinks[tlink.source, tlink.target] = tlink
                    tlinks[tlink.target, tlink.source] = ~tlink

        doc_pred = [
            tlinks[example["source"], example["target"]].relation.interval
            for example in doc_dataset
        ]
        doc_dataset = doc_dataset.add_column("pred", doc_pred)
        doc_datasets.append(doc_dataset)

    dataset = datasets.concatenate_datasets(doc_datasets)

    logging.info("Calculating metrics for all text types")
    report = classification_report(
        y_true=dataset["label"],
        y_pred=dataset["pred"],
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
    outpath = (
        RESULTS_DIR / "interval_deep" / dataset_name / strategy / f"{model_id}.json"
    )
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with open(outpath, "w") as f:
        json.dump(report, f, indent=4)


if __name__ == "__main__":
    Fire(main)
