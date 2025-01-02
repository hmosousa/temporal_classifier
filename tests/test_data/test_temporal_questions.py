from datasets import Dataset

from src.data import load_dataset, load_temporal_questions


def test_load_temporal_questions_train():
    trainset = load_temporal_questions("train")

    assert isinstance(trainset, Dataset)
    assert set(trainset.column_names) == {"text", "label"}
    assert len(trainset) == 226_676


def test_load_temporal_questions_valid():
    validset = load_temporal_questions("valid")

    assert isinstance(validset, Dataset)
    assert set(validset.column_names) == {"text", "label"}
    assert len(validset) == 5_000


def test_load_temporal_questions_test():
    testset = load_temporal_questions("test")

    assert isinstance(testset, Dataset)
    assert set(testset.column_names) == {"text", "label"}
    assert len(testset) == 3_716


def test_load_dataset():
    trainset = load_dataset("temporal_questions", "train")
    assert len(trainset) == 226_676

    validset = load_dataset("temporal_questions", "valid")
    assert len(validset) == 5_000

    testset = load_dataset("temporal_questions", "test")
    assert len(testset) == 3_716
