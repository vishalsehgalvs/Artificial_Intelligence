# Project 1: sort support messages fairly

**Start with:** [ML lesson](../../docs/03-machine-learning.md) and [classification notebook](../../notebooks/02-ml/classification.ipynb). No account or model download required. Use the bundled scikit-learn data in the notebook as a stand-in for messages; never deploy it for medical decisions.

## The question

You receive labeled old examples and want to classify new ones. Is a trained model measurably better than always guessing the most frequent label? Imagine explaining your result to someone who has never heard of logistic regression.

1. Copy the notebook into a new working notebook or run it as-is. Write down the number of training and held-out examples before fitting anything.
2. Print the baseline balanced accuracy. Why would raw accuracy be misleading if one label is rare?
3. Fit the scaler-plus-logistic pipeline. Compare five training-only validation scores with the held-out test score. Keep the test set untouched while deciding what to try.
4. Change only one thing, such as a regularization setting. Measure training-only cross-validation again. Keep a two-row experiment log: setting, validation result, likely explanation.
5. Report a confusion matrix, one limitation of this dataset, and why a high score is not permission to make real-world clinical decisions.

**Example report format:** `Baseline balanced accuracy: ..., trained model: ...; most costly error: ...; next evaluation needed: ...`. Do not claim a fixed expected score; library versions and splits can vary.

**Self-review (10 points):** fair split (2), preprocessing fitted only within folds (2), baseline comparison (2), clear error interpretation (2), stated real-world limitation (2). A reproducible command from the root is `python -m jupyter notebook` after following [setup](../../docs/00-start-here.md).
