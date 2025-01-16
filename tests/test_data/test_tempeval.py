from datasets import Dataset

from src.data import load_dataset, load_tempeval


def test_load_tempeval_train():
    trainset = load_tempeval("train")

    assert isinstance(trainset, Dataset)
    assert set(trainset.column_names) == {"text", "label"}
    assert len(trainset) == 10_952


def test_load_tempeval_test():
    testset = load_tempeval("test")

    assert isinstance(testset, Dataset)
    assert set(testset.column_names) == {"text", "label"}
    assert len(testset) == 929


def test_load_dataset_tempeval():
    testset = load_dataset("tempeval", "test")
    trainset = load_dataset("tempeval", "train")

    assert isinstance(testset, Dataset)
    assert isinstance(trainset, Dataset)

    assert set(testset.column_names) == {"text", "label"}
    assert set(trainset.column_names) == {"text", "label"}

    assert len(testset) == 929
    assert len(trainset) == 10_952
