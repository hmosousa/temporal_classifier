from typing import Literal

import datasets
import tieval.datasets

from src.base import tlink_to_point_relations
from src.data.utils import (
    get_tlink_context,
    INTERVAL_EXPECTED_TAGS,
    POINT_EXPECTED_TAGS,
)


def load_interval_tempeval(
    split: Literal["train", "valid", "test"],
    **kwargs,
) -> datasets.Dataset:
    """Load TempEval-3 dataset."""
    corpus = tieval.datasets.read("tempeval_3")

    if split in ["train", "valid"]:
        n_train_docs = int(len(corpus.train) * 0.8)
        if split == "train":
            docs = corpus.train[:n_train_docs]
        else:
            docs = corpus.train[n_train_docs:]
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
            text = (
                context.replace(f"<{srcid}>", "<source>")
                .replace(f"</{srcid}>", "</source>")
                .replace(f"<{tgtid}>", "<target>")
                .replace(f"</{tgtid}>", "</target>")
            )

            tag_count = sum(1 for tag in INTERVAL_EXPECTED_TAGS if tag in text)
            if tag_count != 4:
                continue

            text = text.replace("\n", " ").strip()
            examples.append(
                {"doc": doc.name, "text": text, "label": tlink.relation.interval}
            )
    return datasets.Dataset.from_list(examples)


def load_point_tempeval(
    split: Literal["train", "valid", "test"],
    **kwargs,
) -> datasets.Dataset:
    """Load TempEval-3 dataset."""
    corpus = tieval.datasets.read("tempeval_3")

    if split in ["train", "valid"]:
        n_train_docs = int(len(corpus.train) * 0.8)
        if split == "train":
            docs = corpus.train[:n_train_docs]
        else:
            docs = corpus.train[n_train_docs:]
    elif split == "test":
        docs = corpus.test
    else:
        raise ValueError(f"Invalid split: {split}")

    examples = []
    for doc in docs:
        for tlink in set(doc.tlinks):
            context = get_tlink_context(doc, tlink)
            relations = tlink_to_point_relations(tlink)
            for relation in relations:
                if relation.source_id == relation.target_id:
                    continue

                if relation.source_endpoint == "start":
                    new_src_tags = "<start_source>", "</start_source>"
                else:
                    new_src_tags = "<end_source>", "</end_source>"

                if relation.target_endpoint == "start":
                    new_tgt_tags = "<start_target>", "</start_target>"
                else:
                    new_tgt_tags = "<end_target>", "</end_target>"

                text = (
                    context.replace(f"<{relation.source_id}>", new_src_tags[0])
                    .replace(f"</{relation.source_id}>", new_src_tags[1])
                    .replace(f"<{relation.target_id}>", new_tgt_tags[0])
                    .replace(f"</{relation.target_id}>", new_tgt_tags[1])
                )

                tag_count = sum(1 for tag in POINT_EXPECTED_TAGS if tag in text)
                if tag_count != 4:
                    continue

                text = text.replace("\n", " ").strip()
                examples.append({"doc": doc.name, "text": text, "label": relation.type})

    return datasets.Dataset.from_list(examples)
