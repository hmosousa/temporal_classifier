from datasets import Dataset

from src.data import load_dataset, load_interval_meantime, load_point_meantime


def test_load_interval_meantime_train():
    trainset = load_interval_meantime("train")

    assert isinstance(trainset, Dataset)
    assert set(trainset.column_names) == {"doc", "text", "label"}
    assert len(trainset) == 1232


def test_load_interval_meantime_valid():
    validset = load_interval_meantime("valid")

    assert isinstance(validset, Dataset)
    assert set(validset.column_names) == {"doc", "text", "label"}
    assert len(validset) == 160


def test_load_interval_meantime_test():
    testset = load_interval_meantime("test")

    assert isinstance(testset, Dataset)
    assert set(testset.column_names) == {"doc", "text", "label"}
    assert len(testset) == 361


def test_load_dataset_interval_meantime():
    testset = load_dataset("interval_meantime", "test")
    validset = load_dataset("interval_meantime", "valid")
    trainset = load_dataset("interval_meantime", "train")

    assert isinstance(testset, Dataset)
    assert isinstance(validset, Dataset)
    assert isinstance(trainset, Dataset)

    assert set(testset.column_names) == {"doc", "text", "label"}
    assert set(validset.column_names) == {"doc", "text", "label"}
    assert set(trainset.column_names) == {"doc", "text", "label"}

    assert len(testset) == 361
    assert len(validset) == 160
    assert len(trainset) == 1232


def test_load_point_tempeval_train():
    trainset = load_point_meantime("train")

    assert isinstance(trainset, Dataset)
    assert set(trainset.column_names) == {"doc", "text", "label"}
    assert len(trainset) == 4 * 1232


def test_load_point_meantime_valid():
    validset = load_point_meantime("valid")

    assert isinstance(validset, Dataset)
    assert set(validset.column_names) == {"doc", "text", "label"}
    assert len(validset) == 4 * 160


def test_load_point_meantime_test():
    testset = load_point_meantime("test")

    assert isinstance(testset, Dataset)
    assert set(testset.column_names) == {"doc", "text", "label"}
    assert len(testset) == 4 * 361


def test_load_dataset_point_tempeval():
    testset = load_dataset("point_meantime", "test")
    validset = load_dataset("point_meantime", "valid")
    trainset = load_dataset("point_meantime", "train")

    assert isinstance(testset, Dataset)
    assert isinstance(validset, Dataset)
    assert isinstance(trainset, Dataset)

    assert set(testset.column_names) == {"doc", "text", "label"}
    assert set(validset.column_names) == {"doc", "text", "label"}
    assert set(trainset.column_names) == {"doc", "text", "label"}

    assert len(testset) == 4 * 361
    assert len(validset) == 4 * 160
    assert len(trainset) == 4 * 1232


def test_load_point_meantime_train_closure():
    trainset = load_point_meantime("train", closure=True)

    assert isinstance(trainset, Dataset)
    assert set(trainset.column_names) == {"doc", "text", "label"}
    assert len(trainset) == 12260


def test_load_point_meantime_valid_closure():
    validset = load_point_meantime("valid", closure=True)

    assert isinstance(validset, Dataset)
    assert set(validset.column_names) == {"doc", "text", "label"}
    assert len(validset) == 1494


def test_load_point_meantime_test_closure():
    testset = load_point_meantime("test", closure=True)

    assert isinstance(testset, Dataset)
    assert set(testset.column_names) == {"doc", "text", "label"}
    assert len(testset) == 4117


def test_load_dataset_point_meantime_closure():
    testset = load_dataset("point_meantime", "test", closure=True)
    validset = load_dataset("point_meantime", "valid", closure=True)
    trainset = load_dataset("point_meantime", "train", closure=True)

    assert isinstance(testset, Dataset)
    assert isinstance(validset, Dataset)
    assert isinstance(trainset, Dataset)

    assert set(testset.column_names) == {"doc", "text", "label"}
    assert set(validset.column_names) == {"doc", "text", "label"}
    assert set(trainset.column_names) == {"doc", "text", "label"}

    assert len(testset) == 4117
    assert len(validset) == 1494
    assert len(trainset) == 12260
