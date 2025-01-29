"""Generate the .tml files to be used with the original SemEval evaluation script.

Before running this script one needs to download the original TempEval-3. There is script in the `scripts/utils` directory to do this.

$ sh scripts/utils/semeval.sh
"""

import collections
import logging
import xml.etree.ElementTree as ET
from typing import Literal

import numpy as np
from fire import Fire

from src.constants import ROOT_DIR
from src.data import load_dataset
from src.interval import get_interval_relation, PAIRS
from src.model import load_model
from src.model.majority import MajorityClassifier
from src.model.random import RandomClassifier
from tieval.temporal_relation import TemporalRelation
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)

TE3TOOLKIT_DIR = ROOT_DIR / "tempeval3_toolkit"
PLATINUM_DATASET_PATH = TE3TOOLKIT_DIR / "te3-platinum"


def write_tlinks(preds_per_doc: dict, dataset_name, strategy, model_id):
    for doc, tlinks in preds_per_doc.items():
        # Format the predicted tlinks
        pred_tlink_dict = {}
        for source, target, interval in tlinks:
            pred_tlink_dict[(source, target)] = interval
            pred_tlink_dict[(target, source)] = (~TemporalRelation(interval)).interval

        original_doc_path = PLATINUM_DATASET_PATH / f"{doc}.tml"
        with open(original_doc_path, "r") as f:
            gold = f.read()

        gold_outpath = TE3TOOLKIT_DIR / "gold" / f"{doc}.tml"
        if not gold_outpath.exists():
            gold_outpath.parent.mkdir(parents=True, exist_ok=True)
            with open(gold_outpath, "w") as f:
                f.write(gold)

        lines = []
        for line in gold.split("\n"):
            if not line.startswith("<TLINK "):
                lines.append(line)
            else:
                # read line with xml parser
                tree = ET.fromstring(line)
                source = (
                    tree.attrib["eventInstanceID"]
                    if "eventInstanceID" in tree.keys()
                    else tree.attrib["timeID"]
                )
                target = (
                    tree.attrib["relatedToEventInstance"]
                    if "relatedToEventInstance" in tree.keys()
                    else tree.attrib["relatedToTime"]
                )
                if (source, target) in pred_tlink_dict:
                    pred_relation = pred_tlink_dict[(source, target)]
                elif (target, source) in pred_tlink_dict:
                    pred_relation = pred_tlink_dict[(target, source)]
                else:
                    pred_relation = "None"
                tree.attrib["relType"] = pred_relation
                lines.append(ET.tostring(tree, encoding="unicode"))

        # Save the new document
        pred_outpath = TE3TOOLKIT_DIR / "results" / strategy / model_id / f"{doc}.tml"
        pred_outpath.parent.mkdir(parents=True, exist_ok=True)
        with open(pred_outpath, "w") as f:
            f.write("\n".join(lines))


def main(
    model_name: str = "majority",
    revision: str = "main",
    strategy: Literal["high_to_low", "most_likely"] = "most_likely",
):
    """Evaluate a model with a given configuration.

    Args:
        model_name: The HuggingFace name of the model to evaluate.
        strategy: The strategy to use to convert the point relations to an interval relation.
    """
    dataset_name = "interval_tempeval"
    logging.info("Loading dataset interval_tempeval")
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
    preds_per_doc = collections.defaultdict(list)
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
        preds_per_doc[example["doc"]].append(
            (example["source"], example["target"], interval_relation)
        )

    model_id = model_name.split("/")[-1]
    write_tlinks(preds_per_doc, dataset_name, strategy, model_id)


if __name__ == "__main__":
    Fire(main)
