from typing import Literal

import datasets
import tieval.datasets

from src.base import Timeline

from src.data.utils import get_tlink_context


def load_interval_tempeval(
    split: Literal["train", "test"],
    **kwargs,
) -> datasets.Dataset:
    """Load TempEval-3 dataset."""
    corpus = tieval.datasets.read("tempeval_3")

    if split == "train":
        docs = corpus.train
    elif split == "test":
        docs = corpus.test
    else:
        raise ValueError(f"Invalid split: {split}")

    examples = []
    for doc in docs:
        for tlink in set(doc.tlinks):
            if tlink.source.id == tlink.target.id:
                continue

            context = get_tlink_context(doc, tlink)
            srcid = tlink.source.id
            tgtid = tlink.target.id
            context = (
                context.replace(f"<{srcid}>", "<source>")
                .replace(f"</{srcid}>", "</source>")
                .replace(f"<{tgtid}>", "<target>")
                .replace(f"</{tgtid}>", "</target>")
            )

            examples.append({"text": context, "label": tlink.relation.interval})
    return datasets.Dataset.from_list(examples)


def load_point_tempeval(
    split: Literal["train", "test"],
    **kwargs,
) -> datasets.Dataset:
    """Load TempEval-3 dataset."""
    corpus = tieval.datasets.read("tempeval_3")

    if split == "train":
        docs = corpus.train
    elif split == "test":
        docs = corpus.test
    else:
        raise ValueError(f"Invalid split: {split}")

    examples = []
    for doc in docs:
        for tlink in set(doc.tlinks):
            context = get_tlink_context(doc, tlink)
            timeline = Timeline(tlinks=[tlink]).to_dict()
            for relation in timeline["relations"]:
                src_endpoint, srcid = relation["source"].split(" ")
                tgt_endpoint, tgtid = relation["target"].split(" ")
                if srcid == tgtid:
                    continue

                if src_endpoint == "start":
                    new_src_tags = "<start_source>", "</start_source>"
                else:
                    new_src_tags = "<end_source>", "</end_source>"

                if tgt_endpoint == "start":
                    new_tgt_tags = "<start_target>", "</start_target>"
                else:
                    new_tgt_tags = "<end_target>", "</end_target>"

                text = (
                    context.replace(f"<{srcid}>", new_src_tags[0])
                    .replace(f"</{srcid}>", new_src_tags[1])
                    .replace(f"<{tgtid}>", new_tgt_tags[0])
                    .replace(f"</{tgtid}>", new_tgt_tags[1])
                )
                examples.append({"text": text, "label": relation["type"]})

    return datasets.Dataset.from_list(examples)
