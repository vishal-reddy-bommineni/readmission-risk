"""Calibrators with one interface (.fit / .predict), so train.py and app.py can use either."""
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


class PlattCalibrator:
    """Logistic regression on the model's log-odds. Strictly monotonic, so it keeps the
    model's ranking (AUROC / PR-AUC unchanged) and only rescales the probabilities."""
    name = "Platt"

    def fit(self, p, y):
        self.lr = LogisticRegression(C=1e6).fit(_logit(p), y)
        return self

    def predict(self, p):
        return self.lr.predict_proba(_logit(p))[:, 1]


class IsotonicCalibrator:
    """Step function fitted to the data. More flexible, but turns many different scores into
    the same value (ties), which can hurt ranking at the top of the list."""
    name = "Isotonic"

    def fit(self, p, y):
        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4).fit(p, y)
        return self

    def predict(self, p):
        return self.iso.predict(p)
