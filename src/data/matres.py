from typing import Literal

import datasets
import tieval.datasets

from src.data.utils import get_tlink_context

# Map from MATRES relations to this project point relations
_RELATIONS_TO_POINT = {
    "BEFORE": "<",
    "AFTER": ">",
    "SIMULTANEOUS": "=",
    "VAGUE": "-",
}


def load_matres(
    split: Literal["train", "test"],
    **kwargs,
) -> datasets.Dataset:
    """Format MATRES the same way as TemporalQuestions."""
    corpus = tieval.datasets.read("matres")
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

            # Matres only has start-start annotations
            text = (
                context.replace(f"<{tlink.source.id}>", "<start_source>")
                .replace(f"</{tlink.source.id}>", "</start_source>")
                .replace(f"<{tlink.target.id}>", "<start_target>")
                .replace(f"</{tlink.target.id}>", "</start_target>")
            )
            label = _RELATIONS_TO_POINT[tlink.relation.interval]
            examples.append({"text": text, "label": label})

    return datasets.Dataset.from_list(examples)
