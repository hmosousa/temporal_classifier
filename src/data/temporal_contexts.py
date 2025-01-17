from typing import Literal

import datasets


def load_temporal_contexts(
    split: Literal["train", "valid", "test"],
) -> datasets.Dataset:
    """Used to train classification models."""
    dataset = datasets.load_dataset("hugosousa/TemporalContexts", split=split)
    return dataset
