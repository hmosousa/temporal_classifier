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
    assert len(trainset) == 4_616


def test_load_interval_timebank_dense_valid():
    validset = load_interval_timebank_dense("valid")

    assert isinstance(validset, Dataset)
    assert set(validset.column_names) == {"doc", "text", "label"}
    assert len(validset) == 637


def test_load_interval_timebank_dense_test():
    testset = load_interval_timebank_dense("test")

    assert isinstance(testset, Dataset)
    assert set(testset.column_names) == {"doc", "text", "label"}
    assert len(testset) == 1552


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

    assert len(trainset) == 4_616
    assert len(validset) == 637
    assert len(testset) == 1552


def test_load_point_timebank_dense_train():
    trainset = load_point_timebank_dense("train")

    assert isinstance(trainset, Dataset)
    assert set(trainset.column_names) == {"doc", "text", "label"}
    assert len(trainset) == 4 * 4_616


def test_load_point_timebank_dense_valid():
    validset = load_point_timebank_dense("valid")

    assert isinstance(validset, Dataset)
    assert set(validset.column_names) == {"doc", "text", "label"}
    assert len(validset) == 4 * 637


def test_load_point_timebank_dense_test():
    testset = load_point_timebank_dense("test")

    assert isinstance(testset, Dataset)
    assert set(testset.column_names) == {"doc", "text", "label"}
    assert len(testset) == 4 * 1552


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

    assert len(trainset) == 4 * 4_616
    assert len(validset) == 4 * 637
    assert len(testset) == 4 * 1552
