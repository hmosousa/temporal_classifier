from src.data import load_dataset
from src.data.utils import augment_dataset


trainset = load_dataset("point_tempeval", "train")
inverse = augment_dataset(trainset)
closure = augment_dataset(inverse)
inverse_closure = augment_dataset(closure)

trainset = trainset.to_pandas()
inverse = inverse.to_pandas()
closure = closure.to_pandas()
inverse_closure = inverse_closure.to_pandas()

trainset_count = trainset[["label", "type"]].value_counts().unstack()
inverse_count = inverse[["label", "type"]].value_counts().unstack()
closure_count = closure[["label", "type"]].value_counts().unstack()
inverse_closure_count = inverse_closure[["label", "type"]].value_counts().unstack()


print(trainset_count[["e-e", "e-t", "dct-e", "t-t", "dct-t"]])
print(inverse_count[["e-e", "e-t", "dct-e", "t-t", "dct-t"]])
print(closure_count[["e-e", "e-t", "dct-e", "t-t", "dct-t"]])
print(inverse_closure_count[["e-e", "e-t", "dct-e", "t-t", "dct-t"]])
