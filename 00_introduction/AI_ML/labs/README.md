# Three next-step algorithm labs

From the course root, after installing the dependencies in [setup](../docs/00-start-here.md), run one command at a time:

| Command                     | What to look for                                              | Read first                                         |
| --------------------------- | ------------------------------------------------------------- | -------------------------------------------------- |
| `python labs/forest.py`     | Baseline, best training-only tuning result, held-out mistakes | [Machine learning](../docs/03-machine-learning.md) |
| `python labs/evaluation.py` | Threshold tradeoffs, Brier score, calibration bins            | [Machine learning](../docs/03-machine-learning.md) |
| `python labs/cnn.py`        | Ten output scores per image, held-out accuracy, wrong guesses | [Neural networks](../docs/04-neural-networks.md)   |
| `python labs/diffusion.py`  | Early/late noise-prediction loss, 1D generated values         | [Neural networks](../docs/04-neural-networks.md)   |

The data is bundled or generated; no model files, API keys, or GPU required. PyTorch is installed through the course extra; use its [official install guide](https://pytorch.org/get-started/locally/) if the platform needs a different build. The forest's clinical dataset is strictly educational. CNN images are tiny digits; diffusion produces numbers rather than pictures. Compare the observed results with the lesson's expectations; random training results can vary by machine.
