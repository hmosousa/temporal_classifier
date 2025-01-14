from datasets import Dataset

from src.data import load_dataset, load_temporal_questions


def test_load_temporal_questions_train_raw():
    trainset = load_temporal_questions("train", "raw")

    assert isinstance(trainset, Dataset)
    assert set(trainset.column_names) == {"text", "label"}
    assert len(trainset) == 38_796


def test_load_temporal_questions_valid_raw():
    validset = load_temporal_questions("valid", config="raw")

    assert isinstance(validset, Dataset)
    assert set(validset.column_names) == {"text", "label"}
    assert len(validset) == 5_000


def test_load_temporal_questions_test_raw():
    testset = load_temporal_questions("test", "raw")

    assert isinstance(testset, Dataset)
    assert set(testset.column_names) == {"text", "label"}
    assert len(testset) == 3_716


def test_load_dataset_raw():
    trainset = load_dataset("temporal_questions", "train", config="raw")
    assert len(trainset) == 38_796

    validset = load_dataset("temporal_questions", "valid", config="raw")
    assert len(validset) == 5_000

    testset = load_dataset("temporal_questions", "test", config="raw")
    assert len(testset) == 3_716


def test_load_temporal_questions_train_closure():
    trainset = load_temporal_questions("train", "closure")

    assert isinstance(trainset, Dataset)
    assert set(trainset.column_names) == {"text", "label"}
    assert len(trainset) == 252_508


def test_load_temporal_questions_valid_closure():
    validset = load_temporal_questions("valid", "closure")

    assert isinstance(validset, Dataset)
    assert set(validset.column_names) == {"text", "label"}
    assert len(validset) == 5_000


def test_load_temporal_questions_test_closure():
    testset = load_temporal_questions("test", "closure")

    assert isinstance(testset, Dataset)
    assert set(testset.column_names) == {"text", "label"}
    assert len(testset) == 12_160


def test_load_dataset_closure():
    trainset = load_dataset("temporal_questions", "train", config="closure")
    assert len(trainset) == 252_508

    validset = load_dataset("temporal_questions", "valid", config="closure")
    assert len(validset) == 5_000

    testset = load_dataset("temporal_questions", "test", config="closure")
    assert len(testset) == 12_160
