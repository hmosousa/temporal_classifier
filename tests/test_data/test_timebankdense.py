from datasets import Dataset

from src.data import (
    load_dataset,
    load_interval_timebank_dense,
    load_point_timebank_dense,
)


def test_load_interval_timebank_dense_train():
    trainset = load_interval_timebank_dense("train")

    assert isinstance(trainset, Dataset)
    assert set(trainset.column_names) == {"doc", "text", "label"}
    assert len(trainset) == 8_949


def test_load_interval_timebank_dense_valid():
    validset = load_interval_timebank_dense("valid")

    assert isinstance(validset, Dataset)
    assert set(validset.column_names) == {"doc", "text", "label"}
    assert len(validset) == 1_067


def test_load_interval_timebank_dense_test():
    testset = load_interval_timebank_dense("test")

    assert isinstance(testset, Dataset)
    assert set(testset.column_names) == {"doc", "text", "label"}
    assert len(testset) == 2_699


def test_load_dataset_interval_timebank_dense():
    trainset = load_dataset("interval_timebank_dense", "train")
    validset = load_dataset("interval_timebank_dense", "valid")
    testset = load_dataset("interval_timebank_dense", "test")

    assert isinstance(trainset, Dataset)
    assert isinstance(validset, Dataset)
    assert isinstance(testset, Dataset)

    assert set(trainset.column_names) == {"doc", "text", "label"}
    assert set(validset.column_names) == {"doc", "text", "label"}
    assert set(testset.column_names) == {"doc", "text", "label"}

    assert len(trainset) == 8_949
    assert len(validset) == 1_067
    assert len(testset) == 2_699


def test_load_point_timebank_dense_train():
    trainset = load_point_timebank_dense("train")

    assert isinstance(trainset, Dataset)
    assert set(trainset.column_names) == {"doc", "text", "label"}
    assert len(trainset) == 4 * 8_949


def test_load_point_timebank_dense_valid():
    validset = load_point_timebank_dense("valid")

    assert isinstance(validset, Dataset)
    assert set(validset.column_names) == {"doc", "text", "label"}
    assert len(validset) == 4 * 1_067


def test_load_point_timebank_dense_test():
    testset = load_point_timebank_dense("test")

    assert isinstance(testset, Dataset)
    assert set(testset.column_names) == {"doc", "text", "label"}
    assert len(testset) == 4 * 2_699


def test_load_dataset_point_timebank_dense():
    trainset = load_dataset("point_timebank_dense", "train")
    validset = load_dataset("point_timebank_dense", "valid")
    testset = load_dataset("point_timebank_dense", "test")

    assert isinstance(trainset, Dataset)
    assert isinstance(validset, Dataset)
    assert isinstance(testset, Dataset)

    assert set(trainset.column_names) == {"doc", "text", "label"}
    assert set(validset.column_names) == {"doc", "text", "label"}
    assert set(testset.column_names) == {"doc", "text", "label"}

    assert len(trainset) == 4 * 8_949
    assert len(validset) == 4 * 1_067
    assert len(testset) == 4 * 2_699
