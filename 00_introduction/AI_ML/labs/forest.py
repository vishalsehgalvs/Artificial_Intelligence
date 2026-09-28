"""Run with: python labs/forest.py (after installing course dependencies)."""

from sklearn.datasets import load_breast_cancer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split


def run():
    data = load_breast_cancer()
    train_x, test_x, train_y, test_y = train_test_split(
        data.data, data.target, test_size=0.2, stratify=data.target, random_state=42
    )
    baseline = DummyClassifier(strategy="most_frequent").fit(train_x, train_y)
    search = GridSearchCV(
        RandomForestClassifier(n_estimators=60, random_state=42, n_jobs=1),
        {"max_depth": [3, None], "min_samples_leaf": [1, 4]},
        scoring="balanced_accuracy",
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        n_jobs=1,
    )
    search.fit(train_x, train_y)
    predictions = search.predict(test_x)
    mistakes = (predictions != test_y).nonzero()[0]
    matrix = confusion_matrix(test_y, predictions, labels=[0, 1])
    assert matrix.sum() == len(test_y)
    print("Baseline test balanced accuracy:", round(balanced_accuracy_score(test_y, baseline.predict(test_x)), 3))
    print("Best training-only CV score:", round(search.best_score_, 3), search.best_params_)
    print("Forest test balanced accuracy:", round(balanced_accuracy_score(test_y, predictions), 3))
    print("Rows = actual 0/1, columns = predicted 0/1:\n", matrix)
    for index in mistakes[:5]:
        print("Mistake on held-out row", index, "actual", test_y[index], "predicted", predictions[index])
    print("Total mistakes:", len(mistakes))


if __name__ == "__main__":
    run()