from typing import Literal

import datasets
import tieval.datasets

from src.data.utils import get_tlink_context, POINT_EXPECTED_TAGS

# Map from MATRES relations to this project point relations
_RELATIONS_TO_POINT = {
    "BEFORE": "<",
    "AFTER": ">",
    "SIMULTANEOUS": "=",
    "VAGUE": "-",
}


def load_matres(
    split: Literal["train", "valid", "test"],
    **kwargs,
) -> datasets.Dataset:
    corpus = tieval.datasets.read("matres")
    if split in ["train", "valid"]:
        n_train_docs = int(len(corpus.train) * 0.9)
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
        for tlink in doc.tlinks:
            context = get_tlink_context(doc, tlink)

            # Matres only has start-start annotations
            text = (
                context.replace(f"<{tlink.source.id}>", "<start_source>")
                .replace(f"</{tlink.source.id}>", "</start_source>")
                .replace(f"<{tlink.target.id}>", "<start_target>")
                .replace(f"</{tlink.target.id}>", "</start_target>")
            )
            text = text.replace("\n", " ").strip()

            tag_count = sum(1 for tag in POINT_EXPECTED_TAGS if tag in text)
            if tag_count != 4:
                continue

            label = _RELATIONS_TO_POINT[tlink.relation.interval]
            examples.append({"doc": doc.name, "text": text, "label": label})

    return datasets.Dataset.from_list(examples)
