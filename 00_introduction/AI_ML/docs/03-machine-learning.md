# 3. Machine learning: from baselines to reliable evaluation

**Goal:** learn what a model can predict, how to check it fairly and when a simple rule is enough. Labs: [classification](../notebooks/02-ml/classification.ipynb), [clustering](../notebooks/02-ml/clustering.ipynb) and [threshold evaluation](../labs/evaluation.py).

## Start with a pile of messages

Suppose we want to sort 100 incoming support messages into "urgent" and "routine." We already know the correct label for some old messages. A **model** is a rule the computer adjusts using those labeled examples. We might record features such as message length and whether it contains "can't log in." The label we want to predict is the **target**.

1. Set aside 20 messages and do not use them to choose the rule. This is the **test set**, like an unopened exam.
2. On the other 80 (**training set**), try the simplest rule: always guess "routine." This is the **baseline**. If 75 of the 80 are routine, the rule gets 75/80 right but misses every urgent case.
3. Train another rule on the 80 examples. Change its settings using a separate **validation** portion of those 80, or divide them into several practice folds (**cross-validation**). Do not peek at the 20 exam messages to pick settings.
4. Compare on the 20 test messages _once_ after deciding what to use. Report both the baseline and new rule, plus the kinds of mistakes each makes.

Why so careful? If tomorrow's ticket resolution is included as an input to predict whether today's ticket is urgent, the model is reading the answer after the fact. That is **leakage**. A **pipeline** means putting the preparation steps and learning rule in one package so each practice fold learns only from its own training messages. The notebook uses a built-in dataset to demonstrate the same procedure; it is not a medical tool.

## What kind of answer do you need?

If you need a number, such as delivery time, use **regression**. If you need a category, such as urgent/routine, use **classification**. If you have no labels and only want to explore similar-looking groups, use **clustering**. The last task does not automatically uncover true categories: two similar messages may belong to different business issues.

For a concrete metric, suppose the model marks 10 tickets urgent. Six really are urgent and four are routine. **Precision** is $6/10=60\%$: of those we alerted on, how many deserved it? If there were 12 urgent tickets total, **recall** is $6/12=50\%$: how many urgent tickets did we find? A false alarm and a missed urgent case need different treatment; "accuracy" hides that choice.

## Later: define the task precisely

Regression predicts a number; classification predicts a category or probability; clustering groups unlabeled rows; ranking orders items. Define the prediction _time_, population, target and cost of mistakes. A good baseline might be the majority class, mean target or a simple logistic model. A flexible model that barely beats it may not justify its complexity.

Split data **before fitting any learned preprocessing**. A test set is held back for one final estimate. Use a validation set or cross-validation for tuning. Stratification keeps class proportions similar; group splits keep entities (patients, devices) out of both train and test; time splits respect chronology. Put imputation, encoding, scaling and feature selection inside a `Pipeline` / `ColumnTransformer` so each CV fold learns transformations from its own training portion. Duplicates, post-event variables and target-derived aggregate statistics leak information even with a pipeline.

## Later: model families and when to try them

Before reading the comparison table, try each idea with a few imaginary examples:

1. **Linear regression** draws a best-fit straight line. For orders of one, two and three apples costing $2, $4 and $6, the line `price = 2 * apples` predicts $8 for four. If discounts start above three apples, a straight line may fail. **Ridge** discourages excessively large line coefficients; **Lasso** can set some to zero when many features are unnecessary.
2. **Logistic regression** uses a weighted score but turns it into a number from 0 to 1, which can be used as an estimated chance of "urgent." At an estimated chance of 0.8 and a decision threshold of 0.5, we alert; at threshold 0.9 we do not. The threshold is a business decision, not an intrinsic property of the model.
3. **Nearest neighbors (kNN)** asks the closest old examples to vote. If the three most similar labeled messages are urgent, urgent, routine, the majority vote is urgent. If "message length" ranges into thousands but "keyword present" is only 0 or 1, raw distance can mostly reflect length. Put features on comparable scales when that makes sense.
4. **A decision tree** asks questions in sequence: "Contains 'locked out'?" If yes, perhaps urgent; if no, ask something else. A tree can memorize unusual wording in the old messages. A **random forest** trains many slightly different trees and combines their votes to reduce that instability. **Boosting** grows a series of small models, each trying to correct shortcomings of the current combined prediction; evaluate it, don't assume more rounds help.

### Build, tune and diagnose a forest

Run `python labs/forest.py` from the course root after [setup](00-start-here.md). The script uses bundled numeric examples (not real patient care) and hides 20% as a final exam. It compares an always-pick-the-common-class baseline with 60-tree forests. `max_depth` limits how many questions a tree can ask; `min_samples_leaf` stops it making rules from very few examples. Three training-only folds choose among four settings; **the held-out test set is not used for this choice**. The printed confusion matrix has actual class in each row and predicted class in each column: off-diagonal counts are mistakes. The listed row numbers refer to positions in the held-out test subset, not original dataset IDs. Ask: is one class missed more often? Is training-only CV much better than the final test? Do not keep retuning after reading the final test score; use a fresh test set for another experiment. Try `max_depth=[2, 3, None]` and compare training-only CV before deciding anything about the held-out result. 5. **Naive Bayes** combines evidence such as word frequencies under a simplifying independence assumption. A **support vector machine** chooses a separating boundary with a large margin, and a kernel can make that boundary curved. **Gaussian processes** describe a range of plausible functions with uncertainty but become computationally expensive as data grows. 6. **k-means clustering**, without urgent/routine labels, chooses a number of centers and repeatedly assigns messages to nearby centers. A group of similar wording does not necessarily correspond to a useful support category. **PCA** instead rotates numeric data to directions capturing the most variation; it is not a classifier.

The lesson is not "use all six." Start with the simplest relevant baseline, note where it fails, then choose a more flexible model for a measurable reason. Revisit the table below after running the two [ML notebooks](../notebooks/README.md).

| Family                       | Intuition / representative algorithm       | Strength                                     | Typical failure                                      |
| ---------------------------- | ------------------------------------------ | -------------------------------------------- | ---------------------------------------------------- |
| Linear regression            | Minimize squared residuals $\|Xw-y\|^2$    | Interpretable numeric baseline               | Nonlinear relations or outliers                      |
| Ridge / Lasso / Elastic Net  | Penalize $\|w\|_2^2$ / $\|w\|_1$ / both    | Control variance; Lasso selects coefficients | Scaling changes penalty effects                      |
| Logistic regression          | $P(y=1\mid x)=\sigma(w^Tx+b)$              | Fast probability baseline                    | Unmodeled interactions; uncalibrated after shifts    |
| kNN                          | Vote among nearest rows                    | Simple local patterns                        | Distance fails in high dimensions; scaling essential |
| Naive Bayes / LDA / QDA      | Model class distributions                  | Small-data or text baseline                  | Strong distribution/independence assumptions         |
| SVM / kernel models          | Seek large margin; kernel changes geometry | Medium-size nonlinear data                   | Expensive training, scaling and tuning needed        |
| Trees                        | Repeated feature-based splits              | Rules and nonlinear interactions             | Deep trees overfit                                   |
| Random forests / bagging     | Average varied trees                       | Robust tabular baseline                      | Larger model, less transparent                       |
| Gradient boosting / AdaBoost | Correct errors in stages                   | Strong tabular performance                   | Sensitive to tuning, leakage and validation          |
| Gaussian processes           | Prior over functions, posterior with data  | Uncertainty on small datasets                | Poor scaling with dataset size                       |

Special cases include robust/quantile regression, SGD for streaming or sparse inputs, multi-label/multi-output prediction, and calibrated classification. Choose based on data size, latency, feature type, interpretability and the error costs, not a leaderboard. For example, if 95% of tickets are routine, 95% accuracy from predicting "routine" every time detects nothing; compare recall and precision for the rare class.

## Later: evaluation without self-deception

For regression use MAE for understandable absolute error, RMSE when large errors matter more, and inspect residuals by group. For classification: $\text{precision}=TP/(TP+FP)$, $\text{recall}=TP/(TP+FN)$; $F_1$ balances them. A threshold turns a probability into a decision; tune that threshold on validation data, not the held-out test. ROC-AUC compares ranking across thresholds but may conceal poor precision for rare positives; PR-AUC and confusion matrices often help more. Log loss and calibration curves assess probabilities rather than only labels. Bootstrap intervals, learning curves and repeated splits expose uncertainty. Fix the random seed for reproducibility, _not_ as a substitute for robust evaluation.

### Practice thresholds and calibration

Run `python labs/evaluation.py` after the forest lab. It fits a scaled logistic baseline, reports precision and recall at three thresholds, and prints calibration bins. A lower threshold usually catches more positives but also raises false alarms; the best threshold depends on the cost of each mistake. Choose it using training or validation data, then report the untouched test result once. The **Brier score** averages squared probability errors, so it evaluates confidence as well as the final class. A calibration bin whose predicted probabilities average 0.8 should contain roughly 80% positives; a difference is evidence of miscalibration, not proof that the model is useless.

### Cross-validation and feature preparation

Use ordinary shuffled $k$-fold cross-validation when rows are independent. Use stratified folds when class proportions matter, group folds when several rows belong to the same person or device, and time-based splits when the future must never train the past. Put imputation, scaling, encoding and feature selection inside the pipeline so each fold learns them from its training rows only. One-hot encoding is a safe starting point for unordered categories; ordinal encoding asserts an order that may not exist. Target encoding can be powerful but is especially prone to leakage unless it is fitted within each training fold. Feature engineering should be justified by a measurable improvement on a representative held-out set, not by a more complicated feature list.

High train accuracy and low validation accuracy suggests overfitting (variance); both low suggests underfitting (bias), though label noise and distribution shift can mimic either. Inspect slices by subgroup, acquisition time and missingness; evaluate fairness harms with domain experts rather than declaring a model fair from one statistic. Feature importance and SHAP-style explanations describe fitted associations, not causal effects.

## Later: beyond supervised learning

| Task                     | Common tools                                                  | Evaluation caution                                               |
| ------------------------ | ------------------------------------------------------------- | ---------------------------------------------------------------- |
| Clustering               | k-means, agglomerative, DBSCAN/HDBSCAN, Gaussian mixtures     | Silhouette alone cannot tell whether groups are meaningful       |
| Dimensionality reduction | PCA, SVD, t-SNE, UMAP, autoencoders                           | Fit transforms on training data; 2D plots distort distances      |
| Anomaly detection        | Isolation forest, one-class SVM, robust covariance            | Rare labels and drifting definitions make false positives costly |
| Recommenders             | Popularity, content similarity, matrix factorization, ranking | Offline clicks contain exposure and position bias                |
| Time series              | Naive last-value/seasonal, lag regression, ARIMA, boosting    | Use rolling-origin validation; never use future lags             |
| Semi-/self-supervised    | Pseudo-labels, contrastive pretraining, masked objectives     | Monitor feedback loops and data contamination                    |
| RL                       | Value methods, policy gradient, actor-critic                  | Offline reward is not proof of real-world safety                 |

## Practical workflow

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

data = load_breast_cancer()
train_x, test_x, train_y, test_y = train_test_split(
    data.data, data.target, stratify=data.target, test_size=0.2, random_state=42
)
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
model.fit(train_x, train_y)
print(balanced_accuracy_score(test_y, model.predict(test_x)))
```

This bundled dataset is for education, **not** clinical decision-making. In a real application, split by patient and institution, check label quality and calibration and involve clinicians.

**Exercise:** Why does `StandardScaler().fit(data.data)` before the split produce an optimistic estimate? **Answer:** even without labels it uses test-set feature statistics, violating the assumption that the model has never seen the test distribution. See the [scikit-learn user guide](https://scikit-learn.org/stable/supervised_learning.html) and [source ledger](sources.md); continue with [Neural networks](04-neural-networks.md).
