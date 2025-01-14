from typing import Literal

import datasets


def load_synthetic_temporal_questions(
    split: Literal["train", "valid"],
) -> datasets.Dataset:
    """Used to train classification models."""
    dataset = datasets.load_dataset(
        "hugosousa/SyntheticTemporalQuestions", "super_clean", split=split
    )
    return dataset
