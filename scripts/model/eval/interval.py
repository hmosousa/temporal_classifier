import copy
import json
import logging
from typing import Literal

import tieval.datasets
import tieval.temporal_relation
import torch
from fire import Fire
from sklearn.metrics import classification_report

from src.base import get_interval_relation, PAIRS
from src.constants import RESULTS_DIR
from src.data.utils import add_tags
from src.model.majority import MajorityClassifier
from src.model.random import RandomClassifier
from tqdm import tqdm
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


def get_tlink_context(doc, tlink):
    """Get the context of a tlink. The context are the sentences that contain the entities of the tlink."""
    entities_map = {ent.id: ent for ent in doc.entities + [doc.dct]}

    if (
        tlink.source.id not in entities_map
        or tlink.target.id not in entities_map
        or tlink.source.id == tlink.target.id
    ):
        return

    has_dct = False
    if tlink.source.is_dct:
        entities = [tlink.target]
        has_dct = True
    elif tlink.target.is_dct:
        entities = [tlink.source]
        has_dct = True
    else:
        entities = [tlink.source, tlink.target]

    offsets = [idx for ent in entities for idx in ent.offsets]

    min_offset = min(offsets)
    max_offset = max(offsets)

    # Get the sentences that contain the entities
    sentences = []
    min_sent_offset = None
    for sent in doc.sentences:
        s_sent, e_sent = sent.offsets
        if (
            s_sent <= min_offset <= e_sent
            or min_offset <= s_sent <= e_sent <= max_offset
            or s_sent <= max_offset <= e_sent
        ):
            sentences.append(str(sent))
            if min_sent_offset is None or s_sent < min_sent_offset:
                min_sent_offset = s_sent
    context = " ".join(sentences)

    # Update entity offsets of the current context
    for idx, ent in enumerate(entities):
        ent_ = copy.deepcopy(ent)
        s_ent, e_ent = ent.offsets
        ent_.offsets = [s_ent - min_sent_offset, e_ent - min_sent_offset]
        entities[idx] = ent_

    if has_dct:
        context = add_tags(context, entities, doc.dct)
    else:
        context = add_tags(context, entities)

    return context


def main(
    model_name: str = "hugosousa/smol-135-tq",
    revision: str = "main",
    dataset_name: Literal["temporal_questions", "timeset"] = "tempeval_3",
    batch_size: int = 512,
    verbose: bool = False,
):
    """Evaluate a model with a given configuration.

    Args:
        model_name: The HuggingFace name of the model to evaluate.
        dataset_name: The name of the dataset to evaluate on.
    """
    logging.info(f"Loading dataset {dataset_name}")
    corpus = tieval.datasets.read(dataset_name)

    train_labels = [
        tl.relation.interval for doc in corpus.documents for tl in doc.tlinks
    ]
    unique_labels = list(set(train_labels))
    test_docs = corpus.test

    logging.info(f"Loading model {model_name}")
    if model_name == "random":
        classifier = RandomClassifier(unique_labels)
    elif model_name == "majority":
        classifier = MajorityClassifier(train_labels)
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
                point_preds = classifier(texts, batch_size=len(texts))
                interval_relation = get_interval_relation(point_preds)

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
