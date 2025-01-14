from datasets import Dataset

from src.data import load_all_temporal_questions, load_dataset


def test_load_all_temporal_questions_train():
    trainset = load_all_temporal_questions("train")

    assert isinstance(trainset, Dataset)
    assert set(trainset.column_names) == {"text", "label"}
    assert len(trainset) == 38_796 + 441_931


def test_load_all_temporal_questions_valid():
    validset = load_all_temporal_questions("valid")

    assert isinstance(validset, Dataset)
    assert set(validset.column_names) == {"text", "label"}
    assert len(validset) == 10_000


def test_load_all_temporal_questions_test():
    testset = load_all_temporal_questions("test")

    assert isinstance(testset, Dataset)
    assert set(testset.column_names) == {"text", "label"}
    assert len(testset) == 3_716


def test_load_all_temporal_questions_dataset():
    trainset = load_dataset("all_temporal_questions", "train")
    assert len(trainset) == 38_796 + 441_931

    validset = load_dataset("all_temporal_questions", "valid")
    assert len(validset) == 10_000

    testset = load_dataset("all_temporal_questions", "test")
    assert len(testset) == 3_716
