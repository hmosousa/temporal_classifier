from datasets import Dataset

from src.data import load_dataset, load_interval_tempeval, load_point_tempeval


def test_load_interval_tempeval_train():
    trainset = load_interval_tempeval("train")

    assert isinstance(trainset, Dataset)
    assert set(trainset.column_names) == {"text", "label"}
    assert len(trainset) == 10_951


def test_load_interval_tempeval_test():
    testset = load_interval_tempeval("test")

    assert isinstance(testset, Dataset)
    assert set(testset.column_names) == {"text", "label"}
    assert len(testset) == 929


def test_load_dataset_interval_tempeval():
    testset = load_dataset("interval_tempeval", "test")
    trainset = load_dataset("interval_tempeval", "train")

    assert isinstance(testset, Dataset)
    assert isinstance(trainset, Dataset)

    assert set(testset.column_names) == {"text", "label"}
    assert set(trainset.column_names) == {"text", "label"}

    assert len(testset) == 929
    assert len(trainset) == 10_951


def test_load_point_tempeval_train():
    trainset = load_point_tempeval("train")

    assert isinstance(trainset, Dataset)
    assert set(trainset.column_names) == {"text", "label"}
    assert len(trainset) == 4 * 10_951


def test_load_point_tempeval_test():
    testset = load_point_tempeval("test")

    assert isinstance(testset, Dataset)
    assert set(testset.column_names) == {"text", "label"}
    assert len(testset) == 4 * 929


def test_load_dataset_point_tempeval():
    testset = load_dataset("point_tempeval", "test")
    trainset = load_dataset("point_tempeval", "train")

    assert isinstance(testset, Dataset)
    assert isinstance(trainset, Dataset)

    assert set(testset.column_names) == {"text", "label"}
    assert set(trainset.column_names) == {"text", "label"}

    assert len(testset) == 4 * 929
    assert len(trainset) == 4 * 10_951
