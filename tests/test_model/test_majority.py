from src.model.majority import MajorityClassifier


class TestMajorityClassifier:
    def test_majority_classifier_initialization(self):
        labels = ["<", ">", "<", "="]
        classifier = MajorityClassifier(labels)
        assert classifier.most_common_label == "<"

    def test_majority_classifier_predictions(self):
        labels = ["<", ">", "<", "="]
        classifier = MajorityClassifier(labels)

        texts = ["This is great!", "This is terrible.", "Just ok"]
        predictions = classifier(texts)

        assert len(predictions) == 3
        assert all(pred["label"] == "<" for pred in predictions)
        assert all(isinstance(pred, dict) for pred in predictions)
