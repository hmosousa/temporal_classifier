from datasets import Dataset

from src.data import load_dataset, load_matres


def test_load_matres_train():
    trainset = load_matres("train")

    assert isinstance(trainset, Dataset)
    assert set(trainset.column_names) == {"doc", "text", "label"}
    assert len(trainset) == 10_091


def test_load_matres_valid():
    validset = load_matres("valid")

    assert isinstance(validset, Dataset)
    assert set(validset.column_names) == {"doc", "text", "label"}
    assert len(validset) == 2_576


def test_load_matres_test():
    testset = load_matres("test")

    assert isinstance(testset, Dataset)
    assert set(testset.column_names) == {"doc", "text", "label"}
    assert len(testset) == 837


def test_load_dataset_matres():
    testset = load_dataset("matres", "test")
    validset = load_dataset("matres", "valid")
    trainset = load_dataset("matres", "train")

    assert isinstance(testset, Dataset)
    assert isinstance(validset, Dataset)
    assert isinstance(trainset, Dataset)

    assert set(testset.column_names) == {"doc", "text", "label"}
    assert set(validset.column_names) == {"doc", "text", "label"}
    assert set(trainset.column_names) == {"doc", "text", "label"}

    assert len(testset) == 837
    assert len(trainset) == 10_091
    assert len(validset) == 2_576
