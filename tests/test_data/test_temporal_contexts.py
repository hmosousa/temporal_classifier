from datasets import Dataset

from src.data import load_dataset, load_temporal_contexts


def test_load_temporal_questions_train():
    trainset = load_temporal_contexts("train")

    assert isinstance(trainset, Dataset)
    assert set(trainset.column_names) == {"dataset", "doc", "text", "label"}
    assert len(trainset) == 112_069


def test_load_temporal_questions_valid():
    validset = load_temporal_contexts("valid")

    assert isinstance(validset, Dataset)
    assert set(validset.column_names) == {"dataset", "doc", "text", "label"}
    assert len(validset) == 2_902


def test_load_temporal_questions_test():
    testset = load_temporal_contexts("test")

    assert isinstance(testset, Dataset)
    assert set(testset.column_names) == {"dataset", "doc", "text", "label"}
    assert len(testset) == 25_393


def test_load_dataset_raw():
    trainset = load_dataset("temporal_contexts", "train")
    assert len(trainset) == 112_069

    validset = load_dataset("temporal_contexts", "valid")
    assert len(validset) == 2_902

    testset = load_dataset("temporal_contexts", "test")
    assert len(testset) == 25_393
