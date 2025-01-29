from typing import Literal

import datasets


def load_synthetic(split: Literal["train", "valid"]) -> datasets.Dataset:
    """The combination of the manual and synthetic temporal questions datasets."""

    dataset = datasets.load_dataset(
        "hugosousa/SyntheticTemporalContexts", "super_clean", split=split
    )

    return dataset
