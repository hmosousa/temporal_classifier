import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression


class CalibratedClassifier:
    def __init__(self, model):
        self.model = model
        self.label2id = self.model.model.config.label2id
        self.id2label = self.model.model.config.id2label
        self.label_regressors = {}

    def fit(self, texts, labels):
        preds = self.model(texts, top_k=len(self.label2id))
        y_prob = self._pred_to_prob(preds)

        # Convert dataset labels to label idxs
        y_true = np.array([self.label2id[label] for label in labels])

        self.label_regressors = {}
        for label_id in self.label2id.values():
            label_y_true = (y_true == label_id).astype(int)
            label_y_pred = y_prob[:, label_id]
            label_pred_prob, label_true_prob = calibration_curve(
                label_y_true, label_y_pred, n_bins=10, strategy="uniform"
            )
            label_regressor = self.isotonic_calibrator(label_pred_prob, label_true_prob)
            self.label_regressors[label_id] = label_regressor

    def isotonic_calibrator(self, pred_prob, true_prob):
        regressor = IsotonicRegression(out_of_bounds="clip")
        regressor.fit(pred_prob, true_prob)
        return regressor

    def __call__(self, texts, batch_size=16, top_k=None):
        """Uses the calibrated regressors to calibrate the predictions. Inputs and outputs like hugginface pipeline object."""
        preds = self.model(texts, batch_size=batch_size, top_k=top_k)
        y_prob = self._pred_to_prob(preds)
        for label_id in self.label2id.values():
            y_prob[:, label_id] = self.label_regressors[label_id].predict(
                y_prob[:, label_id]
            )

        # apply softmax to get probabilities
        # y_prob = np.exp(y_prob) / np.sum(np.exp(y_prob), axis=1, keepdims=True)
        calibrated_preds = self._prob_to_pred(y_prob)
        return calibrated_preds

    def _pred_to_prob(self, preds):
        y_prob = np.zeros((len(preds), len(self.label2id)))
        for i, p in enumerate(preds):
            for pred in p:
                y_prob[i, self.label2id[pred["label"]]] = pred["score"]
        return y_prob

    def _prob_to_pred(self, y_prob):
        preds = [
            [
                {"label": self.id2label[label_id], "score": prob.item()}
                for label_id, prob in enumerate(probs)
            ]
            for probs in y_prob
        ]
        return preds


def expected_calibration_error(y_true, y_prob, n_bins=5):
    # uniform binning approach with n_bins number of bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    # get max probability per sample i
    confidences = np.max(y_prob, axis=1)
    # get predictions from confidences (positional in this case)
    predicted_label = np.argmax(y_prob, axis=1)

    # get a boolean list of correct/false predictions
    accuracies = predicted_label == y_true

    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # determine if sample is in bin m (between bin lower & upper)
        in_bin = np.logical_and(
            confidences > bin_lower.item(), confidences <= bin_upper.item()
        )
        # can calculate the empirical probability of a sample falling into bin m: (|Bm|/n)
        prob_in_bin = in_bin.mean()

        if prob_in_bin.item() > 0:
            # get the accuracy of bin m: acc(Bm)
            accuracy_in_bin = accuracies[in_bin].mean()
            # get the average confidence of bin m: conf(Bm)
            avg_confidence_in_bin = confidences[in_bin].mean()
            # calculate |acc(Bm) - conf(Bm)| * (|Bm|/n) for bin m and add to the total ECE
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prob_in_bin
    return ece.item()
