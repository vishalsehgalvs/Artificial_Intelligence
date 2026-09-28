"""Evaluate thresholds and calibration: python labs/evaluation.py."""

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def run():
    data = load_breast_cancer()
    train_x, test_x, train_y, test_y = train_test_split(
        data.data, data.target, test_size=0.25, stratify=data.target, random_state=42
    )
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=42))
    model.fit(train_x, train_y)
    probabilities = model.predict_proba(test_x)[:, 1]

    print("Positive class prevalence:", round(test_y.mean(), 3))
    for threshold in (0.3, 0.5, 0.7):
        predictions = (probabilities >= threshold).astype(int)
        print(
            "Threshold",
            threshold,
            "accuracy",
            round(accuracy_score(test_y, predictions), 3),
            "precision",
            round(precision_score(test_y, predictions), 3),
            "recall",
            round(recall_score(test_y, predictions), 3),
        )

    baseline = np.ones_like(test_y)
    assert accuracy_score(test_y, baseline) == test_y.mean()
    assert confusion_matrix(test_y, (probabilities >= 0.5).astype(int)).sum() == len(test_y)
    brier = brier_score_loss(test_y, probabilities)
    observed, predicted = calibration_curve(test_y, probabilities, n_bins=5, strategy="quantile")
    print("Brier score (lower is better):", round(brier, 3))
    print("Calibration bins observed:", [round(value, 3) for value in observed])
    print("Calibration bins predicted:", [round(value, 3) for value in predicted])
    print("Choose a threshold from validation costs, then report this test result once.")


if __name__ == "__main__":
    run()