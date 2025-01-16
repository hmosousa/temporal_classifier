from datasets import Dataset

from src.data import load_dataset, load_timebank_dense


def test_load_timebank_dense_train():
    trainset = load_timebank_dense("train")

    assert isinstance(trainset, Dataset)
    assert set(trainset.column_names) == {"text", "label"}
    assert len(trainset) == 8_949


def test_load_timebank_dense_valid():
    validset = load_timebank_dense("valid")

    assert isinstance(validset, Dataset)
    assert set(validset.column_names) == {"text", "label"}
    assert len(validset) == 1_067


def test_load_timebank_dense_test():
    testset = load_timebank_dense("test")

    assert isinstance(testset, Dataset)
    assert set(testset.column_names) == {"text", "label"}
    assert len(testset) == 2_699


def test_load_dataset_timebank_dense():
    trainset = load_dataset("timebank_dense", "train")
    validset = load_dataset("timebank_dense", "valid")
    testset = load_dataset("timebank_dense", "test")

    assert isinstance(trainset, Dataset)
    assert isinstance(validset, Dataset)
    assert isinstance(testset, Dataset)

    assert set(trainset.column_names) == {"text", "label"}
    assert set(validset.column_names) == {"text", "label"}
    assert set(testset.column_names) == {"text", "label"}

    assert len(trainset) == 8_949
    assert len(validset) == 1_067
    assert len(testset) == 2_699
