# from datasets import Dataset

# from src.data import load_dataset, load_matres


# def test_load_matres_train():
#     trainset = load_matres("train")

#     assert isinstance(trainset, Dataset)
#     assert set(trainset.column_names) == {"text", "label"}
#     assert len(trainset) == 1_000


# def test_load_matres_test():
#     testset = load_matres("test")

#     assert isinstance(testset, Dataset)
#     assert set(testset.column_names) == {"text", "label"}
#     assert len(testset) == 1_000


# def test_load_dataset_matres():
#     testset = load_dataset("matres", "test")
#     trainset = load_dataset("matres", "train")

#     assert isinstance(testset, Dataset)
#     assert isinstance(trainset, Dataset)

#     assert set(testset.column_names) == {"text", "label"}
#     assert set(trainset.column_names) == {"text", "label"}

#     assert len(testset) == 1_000
#     assert len(trainset) == 1_000
