import random
from typing import List


class RandomClassifier:
    def __init__(self, labels: List[str]):
        self.labels = list(set(labels))

    def __call__(self, texts: List[str], *args, **kwargs):
        """Predict a random label for each text. Set to have the same syntax as huggingface pipeline."""
        return [{"label": random.choice(self.labels)} for _ in texts]
