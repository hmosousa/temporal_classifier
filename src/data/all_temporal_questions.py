from typing import Literal

import datasets


def load_all_temporal_questions(
    split: Literal["train", "valid", "test"],
) -> datasets.Dataset:
    """The combination of the manual and synthetic temporal questions datasets."""
    manual = datasets.load_dataset("hugosousa/TemporalQuestions", split=split)

    if split != "test":  # Synthetic dataset is only available for train and valid
        synthetic = datasets.load_dataset(
            "hugosousa/SyntheticTemporalQuestions", "clean", split=split
        )

        dataset = datasets.concatenate_datasets([manual, synthetic])
    else:
        dataset = manual

    return dataset
