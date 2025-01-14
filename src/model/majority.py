from collections import Counter
from typing import List


class MajorityClassifier:
    def __init__(self, labels: List[str]):
        counter = Counter(labels)
        most_common = counter.most_common(1)
        self.most_common_label = most_common[0][0]

    def __call__(self, texts: List[str], *args, **kwargs):
        """Predict the most common label for each text. Set to have the same syntax as huggingface pipeline."""
        return [{"label": self.most_common_label} for _ in texts]
