from src.model.random import RandomClassifier


class TestRandomClassifier:
    def test_random_classifier_initialization(self):
        labels = ["<", ">", "<", "="]
        classifier = RandomClassifier(labels)
        assert all(label in classifier.labels for label in ["<", ">", "="])

    def test_random_classifier_predictions(self):
        labels = ["<", ">", "<", "="]
        classifier = RandomClassifier(labels)

        texts = ["This is great!", "This is terrible.", "Just ok"]
        predictions = classifier(texts)

        assert len(predictions) == 3
        assert all(pred["label"] in labels for pred in predictions)
        assert all(isinstance(pred, dict) for pred in predictions)
