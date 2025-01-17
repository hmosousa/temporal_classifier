from datasets import Dataset

from src.data import load_dataset, load_interval_tddiscourse


def test_load_interval_tddiscourse_train():
    trainset = load_interval_tddiscourse("train")

    assert isinstance(trainset, Dataset)
    assert set(trainset.column_names) == {"text", "label"}
    assert len(trainset) == 4_000


def test_load_interval_tddiscourse_valid():
    validset = load_interval_tddiscourse("valid")

    assert isinstance(validset, Dataset)
    assert set(validset.column_names) == {"text", "label"}
    assert len(validset) == 650


def test_load_interval_tddiscourse_test():
    testset = load_interval_tddiscourse("test")

    assert isinstance(testset, Dataset)
    assert set(testset.column_names) == {"text", "label"}
    assert len(testset) == 1_500


def test_load_dataset_interval_tddiscourse():
    trainset = load_dataset("interval_tddiscourse", "train")
    validset = load_dataset("interval_tddiscourse", "valid")
    testset = load_dataset("interval_tddiscourse", "test")

    assert isinstance(trainset, Dataset)
    assert isinstance(validset, Dataset)
    assert isinstance(testset, Dataset)

    assert set(trainset.column_names) == {"text", "label"}
    assert set(validset.column_names) == {"text", "label"}
    assert set(testset.column_names) == {"text", "label"}

    assert len(trainset) == 4_000
    assert len(validset) == 650
    assert len(testset) == 1_500
