from typing import Literal

import datasets


def load_temporal_questions(
    split: Literal["train", "valid", "test"], config: Literal["closure", "raw"]
) -> datasets.Dataset:
    """Used to train classification models."""
    dataset = datasets.load_dataset("hugosousa/TemporalQuestions", config, split=split)
    return dataset
