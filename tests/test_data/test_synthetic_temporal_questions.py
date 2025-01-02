from datasets import Dataset

from src.data import load_dataset, load_synthetic_temporal_questions


def test_load_synthetic_temporal_questions_train():
    trainset = load_synthetic_temporal_questions("train")

    assert isinstance(trainset, Dataset)
    assert set(trainset.column_names) == {"text", "label"}
    assert len(trainset) == 441_741


def test_load_synthetic_temporal_questions_valid():
    validset = load_synthetic_temporal_questions("valid")

    assert isinstance(validset, Dataset)
    assert set(validset.column_names) == {"text", "label"}
    assert len(validset) == 5_000


def test_load_dataset():
    trainset = load_dataset("synthetic_temporal_questions", "train")
    assert len(trainset) == 441_741

    validset = load_dataset("synthetic_temporal_questions", "valid")
    assert len(validset) == 5_000
