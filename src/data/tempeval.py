from typing import Literal

import datasets
import tieval.datasets

from src.data.utils import get_tlink_context


def load_tempeval(
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
        for tlink in doc.tlinks:
            context = get_tlink_context(doc, tlink)
            examples.append({"text": context, "label": tlink.relation.interval})

    return datasets.Dataset.from_list(examples)
