from src.data import load_dataset
from src.data.utils import augment_dataset


def add_text_type(example: dict):
    if "<start_source>" in example["text"] and "<start_target>" in example["text"]:
        example["type"] = "ss"  # start-start
    elif "<end_source>" in example["text"] and "<end_target>" in example["text"]:
        example["type"] = "ee"  # end-end
    elif "<start_source>" in example["text"] and "<end_target>" in example["text"]:
        example["type"] = "se"  # start-end
    elif "<end_source>" in example["text"] and "<start_target>" in example["text"]:
        example["type"] = "es"  # end-start
    else:
        raise ValueError(
            f"Text does not contain a valid entity pair: {example['text']}"
        )
    return example


trainset = load_dataset("point_tempeval", "train")
inverse = augment_dataset(trainset)
closure = augment_dataset(inverse)
inverse_closure = augment_dataset(closure)

trainset = trainset.map(add_text_type)
inverse = inverse.map(add_text_type)
closure = closure.map(add_text_type)
inverse_closure = inverse_closure.map(add_text_type)

trainset = trainset.to_pandas()
inverse = inverse.to_pandas()
closure = closure.to_pandas()
inverse_closure = inverse_closure.to_pandas()

trainset_count = trainset[["label", "type"]].value_counts().unstack()
inverse_count = inverse[["label", "type"]].value_counts().unstack()
closure_count = closure[["label", "type"]].value_counts().unstack()
inverse_closure_count = inverse_closure[["label", "type"]].value_counts().unstack()


print(trainset_count[["ss", "se", "es", "ee"]])
print(inverse_count[["ss", "se", "es", "ee"]])
print(closure_count[["ss", "se", "es", "ee"]])
print(inverse_closure_count[["ss", "se", "es", "ee"]])
