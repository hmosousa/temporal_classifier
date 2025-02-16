import json

from sklearn.metrics import confusion_matrix
from src.constants import CACHE_DIR
from src.data import load_dataset

trainset = load_dataset("interval_tempeval", "test")

interval_model_path = (
    CACHE_DIR
    / "results"
    / "interval"
    / "interval_tempeval"
    / "hugosousa"
    / "smol-360-interval-a-5f554f47.json"
)
interval_model_preds = json.load(open(interval_model_path, "r"))

point_model_path = (
    CACHE_DIR
    / "results"
    / "interval"
    / "interval_tempeval"
    / "hugosousa"
    / "smol-360-a-4a820490.json"
)
point_model_preds = json.load(open(point_model_path, "r"))

true = trainset["label"]

labels = list(set(true))

cm_interval = confusion_matrix(true, interval_model_preds, labels=labels)
print(cm_interval)

cm_point = confusion_matrix(true, point_model_preds, labels=labels)
print(cm_point)

print(cm_interval - cm_point)
