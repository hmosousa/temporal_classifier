from datasets import Dataset

from src.data import load_dataset, load_interval_tempeval, load_point_tempeval


def test_load_interval_tempeval_train():
    trainset = load_interval_tempeval("train")

    assert isinstance(trainset, Dataset)
    assert set(trainset.column_names) == {
        "doc",
        "text",
        "label",
        "source",
        "target",
        "type",
    }
    assert len(trainset) == 9_217


def test_load_interval_tempeval_valid():
    validset = load_interval_tempeval("valid")

    assert isinstance(validset, Dataset)
    assert set(validset.column_names) == {
        "doc",
        "text",
        "label",
        "source",
        "target",
        "type",
    }
    assert len(validset) == 1_734


def test_load_interval_tempeval_test():
    testset = load_interval_tempeval("test")

    assert isinstance(testset, Dataset)
    assert set(testset.column_names) == {
        "doc",
        "text",
        "label",
        "source",
        "target",
        "type",
    }
    assert len(testset) == 929


def test_load_dataset_interval_tempeval():
    testset = load_dataset("interval_tempeval", "test")
    validset = load_dataset("interval_tempeval", "valid")
    trainset = load_dataset("interval_tempeval", "train")

    assert isinstance(testset, Dataset)
    assert isinstance(validset, Dataset)
    assert isinstance(trainset, Dataset)

    assert set(testset.column_names) == {
        "doc",
        "text",
        "label",
        "source",
        "target",
        "type",
    }
    assert set(validset.column_names) == {
        "doc",
        "text",
        "label",
        "source",
        "target",
        "type",
    }
    assert set(trainset.column_names) == {
        "doc",
        "text",
        "label",
        "source",
        "target",
        "type",
    }

    assert len(testset) == 929
    assert len(validset) == 1_734
    assert len(trainset) == 9_217


def test_load_interval_tempeval_train_closure():
    trainset = load_interval_tempeval("train", closure=True)

    assert isinstance(trainset, Dataset)
    assert set(trainset.column_names) == {
        "doc",
        "text",
        "label",
        "source",
        "target",
        "type",
    }
    assert len(trainset) == 60428


def test_load_interval_tempeval_valid_closure():
    validset = load_interval_tempeval("valid", closure=True)

    assert isinstance(validset, Dataset)
    assert set(validset.column_names) == {
        "doc",
        "text",
        "label",
        "source",
        "target",
        "type",
    }
    assert len(validset) == 8104


def test_load_interval_tempeval_test_closure():
    testset = load_interval_tempeval("test", closure=True)

    assert isinstance(testset, Dataset)
    assert set(testset.column_names) == {
        "doc",
        "text",
        "label",
        "source",
        "target",
        "type",
    }
    assert len(testset) == 3_454


def test_load_dataset_interval_tempeval_closure():
    testset = load_dataset("interval_tempeval", "test", closure=True)
    validset = load_dataset("interval_tempeval", "valid", closure=True)
    trainset = load_dataset("interval_tempeval", "train", closure=True)

    assert isinstance(testset, Dataset)
    assert isinstance(validset, Dataset)
    assert isinstance(trainset, Dataset)

    assert set(testset.column_names) == {
        "doc",
        "text",
        "label",
        "source",
        "target",
        "type",
    }
    assert set(validset.column_names) == {
        "doc",
        "text",
        "label",
        "source",
        "target",
        "type",
    }
    assert set(trainset.column_names) == {
        "doc",
        "text",
        "label",
        "source",
        "target",
        "type",
    }

    assert len(testset) == 929
    assert len(validset) == 1_734
    assert len(trainset) == 9_217


def test_load_point_tempeval_train():
    trainset = load_point_tempeval("train")

    assert isinstance(trainset, Dataset)
    assert set(trainset.column_names) == {
        "doc",
        "text",
        "label",
        "source",
        "target",
        "type",
    }
    assert len(trainset) == 4 * 9_217


def test_load_point_tempeval_valid():
    validset = load_point_tempeval("valid")

    assert isinstance(validset, Dataset)
    assert set(validset.column_names) == {
        "doc",
        "text",
        "label",
        "source",
        "target",
        "type",
    }
    assert len(validset) == 4 * 1_734


def test_load_point_tempeval_test():
    testset = load_point_tempeval("test")

    assert isinstance(testset, Dataset)
    assert set(testset.column_names) == {
        "doc",
        "text",
        "label",
        "source",
        "target",
        "type",
    }
    assert len(testset) == 4 * 929


def test_load_dataset_point_tempeval():
    testset = load_dataset("point_tempeval", "test")
    validset = load_dataset("point_tempeval", "valid")
    trainset = load_dataset("point_tempeval", "train")

    assert isinstance(testset, Dataset)
    assert isinstance(validset, Dataset)
    assert isinstance(trainset, Dataset)

    assert set(testset.column_names) == {
        "doc",
        "text",
        "label",
        "source",
        "target",
        "type",
    }
    assert set(validset.column_names) == {
        "doc",
        "text",
        "label",
        "source",
        "target",
        "type",
    }
    assert set(trainset.column_names) == {
        "doc",
        "text",
        "label",
        "source",
        "target",
        "type",
    }

    assert len(testset) == 4 * 929
    assert len(validset) == 4 * 1_734
    assert len(trainset) == 4 * 9_217


def test_load_point_tempeval_train_closure():
    trainset = load_point_tempeval("train", closure=True)

    assert isinstance(trainset, Dataset)
    assert set(trainset.column_names) == {
        "doc",
        "text",
        "label",
        "source",
        "target",
        "type",
    }
    assert len(trainset) == 339_950


def test_load_point_tempeval_valid_closure():
    validset = load_point_tempeval("valid", closure=True)

    assert isinstance(validset, Dataset)
    assert set(validset.column_names) == {
        "doc",
        "text",
        "label",
        "source",
        "target",
        "type",
    }
    assert len(validset) == 40_189


def test_load_point_tempeval_test_closure():
    testset = load_point_tempeval("test", closure=True)

    assert isinstance(testset, Dataset)
    assert set(testset.column_names) == {
        "doc",
        "text",
        "label",
        "source",
        "target",
        "type",
    }
    assert len(testset) == 19_504


def test_load_dataset_point_tempeval_closure():
    testset = load_dataset("point_tempeval", "test", closure=True)
    validset = load_dataset("point_tempeval", "valid", closure=True)
    trainset = load_dataset("point_tempeval", "train", closure=True)

    assert isinstance(testset, Dataset)
    assert isinstance(validset, Dataset)
    assert isinstance(trainset, Dataset)

    assert set(testset.column_names) == {
        "doc",
        "text",
        "label",
        "source",
        "target",
        "type",
    }
    assert set(validset.column_names) == {
        "doc",
        "text",
        "label",
        "source",
        "target",
        "type",
    }
    assert set(trainset.column_names) == {
        "doc",
        "text",
        "label",
        "source",
        "target",
        "type",
    }

    assert len(testset) == 19_504
    assert len(validset) == 40_189
    assert len(trainset) == 339_950
