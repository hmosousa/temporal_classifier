"""Generate Temporal Questions dataset from TemporalEval3 corpus."""

import logging
from collections import Counter
from pathlib import Path

import datasets
import fire
import pandas as pd

from src.constants import HF_TOKEN
from src.data import load_dataset

ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"


def validate_dataset(examples: datasets.Dataset) -> list[dict]:
    """Drop any relation that appears more than once.
    Most likely a mistake in the dataset."""
    text_counter = Counter([example["text"] for example in examples])
    duplicates = [text for text, count in text_counter.items() if count > 1]
    examples = [example for example in examples if example["text"] not in duplicates]
    return examples


def main():
    """Generate TemporalQuestions dataset from TemporalEval3 corpus.

    Args:
        dataset_name: Name of the dataset to use.
        n_valid_samples: Number of samples to use for validation.
        closure: Whether to compute temporal closure or not.
        just_sentences: Whether to use just the sentences that contain the temporal entities as context or not.
    """
    dataset_names = [
        "point_tempeval",
        "matres",
        "timeset",
        "point_timebank_dense",
        "point_tddiscourse",
    ]

    train_datasets = []
    valid_datasets = []
    test_datasets = []
    for dataset in dataset_names:
        if dataset != "timeset":  # Timeset doesnt have a train split
            trainset = load_dataset(dataset, "train")
            trainset = trainset.add_column("dataset", [dataset] * len(trainset))
            train_datasets.append(trainset)

        validset = load_dataset(dataset, "valid")
        validset = validset.add_column("dataset", [dataset] * len(validset))
        valid_datasets.append(validset)

        testset = load_dataset(dataset, "test")
        testset = testset.add_column("dataset", [dataset] * len(testset))
        test_datasets.append(testset)

    train_dataset = datasets.concatenate_datasets(train_datasets)
    train_df = train_dataset.to_pandas()
    train_df = train_df[["dataset", "doc", "text", "label"]]

    valid_dataset = datasets.concatenate_datasets(valid_datasets)
    valid_df = valid_dataset.to_pandas()
    valid_df = valid_df[["dataset", "doc", "text", "label"]]

    test_dataset = datasets.concatenate_datasets(test_datasets)
    test_df = test_dataset.to_pandas()
    test_df = test_df[["dataset", "doc", "text", "label"]]

    logging.info(
        "Dropping from train and valid any doc that appears in testset",
    )
    train_df = train_df[~train_df["doc"].isin(test_df["doc"])]
    valid_df = valid_df[~valid_df["doc"].isin(test_df["doc"])]

    logging.info("Moving from valid to train any doc that appears in trainset")
    train_df = pd.concat([train_df, valid_df[valid_df["doc"].isin(train_df["doc"])]])
    valid_df = valid_df[~valid_df["doc"].isin(train_df["doc"])]

    logging.info("Dropping duplicates from trainset, validset")
    train_df.drop_duplicates(subset="text", keep="last", inplace=True)
    valid_df.drop_duplicates(subset="text", keep="last", inplace=True)

    logging.info("Pushing to hub")
    trainset = datasets.Dataset.from_pandas(train_df, preserve_index=False)
    validset = datasets.Dataset.from_pandas(valid_df, preserve_index=False)
    testset = datasets.Dataset.from_pandas(test_df, preserve_index=False)

    trainset.push_to_hub("hugosousa/TemporalContexts", split="train", token=HF_TOKEN)
    validset.push_to_hub("hugosousa/TemporalContexts", split="valid", token=HF_TOKEN)
    testset.push_to_hub("hugosousa/TemporalContexts", split="test", token=HF_TOKEN)


if __name__ == "__main__":
    fire.Fire(main)
