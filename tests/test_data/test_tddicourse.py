from datasets import Dataset

from src.data import load_dataset, load_tddiscourse


def test_load_tddiscourse_train():
    trainset = load_tddiscourse("train")

    assert isinstance(trainset, Dataset)
    assert set(trainset.column_names) == {"text", "label"}
    assert len(trainset) == 4_650


def test_load_tddiscourse_test():
    testset = load_tddiscourse("test")

    assert isinstance(testset, Dataset)
    assert set(testset.column_names) == {"text", "label"}
    assert len(testset) == 1_500


def test_load_dataset_tddiscourse():
    testset = load_dataset("tddiscourse", "test")
    trainset = load_dataset("tddiscourse", "train")

    assert isinstance(testset, Dataset)
    assert isinstance(trainset, Dataset)

    assert set(testset.column_names) == {"text", "label"}
    assert set(trainset.column_names) == {"text", "label"}

    assert len(testset) == 1_500
    assert len(trainset) == 4_650
