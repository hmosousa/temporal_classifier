"""Generate Temporal Questions dataset from TemporalEval3 corpus."""

from collections import Counter
from pathlib import Path

import datasets
import fire

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
    train_texts_count = Counter(train_dataset["text"])
    duplicate_train_texts = {
        text for text, count in train_texts_count.items() if count > 1
    }
    duplicates_dataset = train_dataset.filter(
        lambda x: x["text"] in duplicate_train_texts
    )
    duplicates_dataset.to_pandas().to_csv("duplicates_train.csv", index=False)

    valid_dataset = datasets.concatenate_datasets(valid_datasets)
    valid_texts_count = Counter(valid_dataset["text"])
    duplicate_valid_texts = {
        text for text, count in valid_texts_count.items() if count > 1
    }
    duplicates_dataset = valid_dataset.filter(
        lambda x: x["text"] in duplicate_valid_texts
    )
    duplicates_dataset.to_pandas().to_csv("duplicates_valid.csv", index=False)

    test_dataset = datasets.concatenate_datasets(test_datasets)
    test_texts_count = Counter(test_dataset["text"])
    duplicate_test_texts = {
        text for text, count in test_texts_count.items() if count > 1
    }
    duplicates_dataset = test_dataset.filter(
        lambda x: x["text"] in duplicate_test_texts
    )
    duplicates_dataset.to_pandas().to_csv("duplicates_test.csv", index=False)

    # # Stratified split into train and validation
    # train_examples, valid_examples = train_test_split(
    #     dev_examples,
    #     test_size=n_valid_samples,
    #     random_state=42,
    #     stratify=[example["label"] for example in dev_examples],
    #     shuffle=True,
    # )

    # logging.info("Pushing to hub")
    # trainset = datasets.Dataset.from_list(train_examples)
    # validset = datasets.Dataset.from_list(valid_examples)
    # testset = datasets.Dataset.from_list(test_examples)

    # config = "closure" if closure else "raw"
    # trainset.push_to_hub(
    #     "hugosousa/TemporalQuestions", config_name=config, split="train", token=HF_TOKEN
    # )
    # validset.push_to_hub(
    #     "hugosousa/TemporalQuestions", config_name=config, split="valid", token=HF_TOKEN
    # )
    # testset.push_to_hub(
    #     "hugosousa/TemporalQuestions", config_name=config, split="test", token=HF_TOKEN
    # )


if __name__ == "__main__":
    fire.Fire(main)
