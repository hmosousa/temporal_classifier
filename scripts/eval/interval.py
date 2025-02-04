import json
import logging
from typing import Literal

import numpy as np

import tieval.datasets
import tieval.temporal_relation
import torch
from fire import Fire
from sklearn.metrics import classification_report

from src.base import RELATIONS
from src.constants import RESULTS_DIR
from src.data.utils import get_tlink_context
from src.interval import get_interval_relation, PAIRS
from src.model.majority import MajorityClassifier
from src.model.random import RandomClassifier
from tqdm import tqdm
from transformers import pipeline

logging.basicConfig(level=logging.INFO)


def main(
    model_name: str = "hugosousa/smol-135-tq",
    revision: str = "main",
    dataset_name: Literal["tempeval_3", "tddiscourse"] = "tddiscourse",
    verbose: bool = False,
    strategy: Literal["high_to_low", "most_likely"] = "most_likely",
):
    """Evaluate a model with a given configuration.

    Args:
        model_name: The HuggingFace name of the model to evaluate.
        dataset_name: The name of the dataset to evaluate on.
        strategy: The strategy to use to convert the point relations to an interval relation.
    """
    logging.info(f"Loading dataset {dataset_name}")
    corpus = tieval.datasets.read(dataset_name)

    all_labels = [tl.relation.interval for doc in corpus.documents for tl in doc.tlinks]
    unique_labels = list(set(all_labels))
    test_docs = corpus.test

    logging.info(f"Loading model {model_name}")
    if model_name == "random":
        classifier = RandomClassifier(unique_labels)
    elif model_name == "majority":
        classifier = MajorityClassifier(all_labels)
    else:
        classifier = pipeline(
            "text-classification",
            model=model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            revision=revision,
        )

    logging.info("Getting predictions")
    labels, preds = [], []
    for doc in tqdm(test_docs):
        for tlink in doc.tlinks:
            if model_name in ["random", "majority"]:
                interval_relation = classifier([doc.text])[0]["label"]
            else:
                context = get_tlink_context(doc, tlink)

                # Generate the text for each point the model has to classify
                texts = []
                for pair in PAIRS:
                    text = (
                        context.replace(f"<{tlink.source.id}>", f"<{pair[0]}>")
                        .replace(f"</{tlink.source.id}>", f"</{pair[0]}>")
                        .replace(f"<{tlink.target.id}>", f"<{pair[1]}>")
                        .replace(f"</{tlink.target.id}>", f"</{pair[1]}>")
                    )
                    texts.append(text)

                # Get the model's prediction
                point_preds = classifier(
                    texts, batch_size=len(texts), top_k=len(RELATIONS)
                )

                y_prob = np.zeros((len(texts), len(RELATIONS)))
                for idx, pred in enumerate(point_preds):
                    for label_pred in pred:
                        y_prob[idx, RELATIONS.index(label_pred["label"])] = label_pred[
                            "score"
                        ]

                interval_relation = get_interval_relation(
                    y_prob, unique_labels, strategy
                )

            labels.append(tlink.relation.interval)
            preds.append(interval_relation if interval_relation is not None else "None")

    logging.info("Calculating metrics for all text types")
    report = classification_report(
        y_true=labels,
        y_pred=preds,
        output_dict=True,
        zero_division=0.0,
        labels=list(unique_labels),
    )
    if verbose:
        print(classification_report(labels, preds, digits=4))

    logging.info("Saving results")
    model_id = model_name.split("/")[-1]
    outpath = RESULTS_DIR / "interval" / dataset_name / f"{model_id}.json"
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with open(outpath, "w") as f:
        json.dump(report, f, indent=4)


if __name__ == "__main__":
    Fire(main)
