from datasets import Dataset

from src.data import load_dataset, load_synthetic


def test_load_synthetic_train():
    trainset = load_synthetic("train")

    assert isinstance(trainset, Dataset)
    assert set(trainset.column_names) == {"text", "label"}
    assert len(trainset) == 441_931


def test_load_synthetic_valid():
    validset = load_synthetic("valid")

    assert isinstance(validset, Dataset)
    assert set(validset.column_names) == {"text", "label"}
    assert len(validset) == 5_000


def test_load_dataset():
    trainset = load_dataset("synthetic", "train")
    assert len(trainset) == 441_931

    validset = load_dataset("synthetic", "valid")
    assert len(validset) == 5_000
