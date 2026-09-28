# Project 2: debug a tiny neural network

**Start with:** [neural-network lesson](../../docs/04-neural-networks.md) and [PyTorch notebook](../../notebooks/03-deep-learning/mlp.ipynb). The notebook makes two random inputs and labels a point by whether their product is positive. This is toy data, not a real image task.

1. Predict what a positive product means when both numbers are negative. **Answer:** their product is positive, so the target is class 1.
2. Run the notebook. Write down the logit array shape and held-out accuracy. A **logit** is a raw score before it is turned into a probability; `argmax` picks the larger score.
3. Change the hidden width from 16 to 2. Re-run from the beginning with the same seed. Record whether the last training loss and validation accuracy change. Do not infer a law from one run.
4. Deliberately change the learning rate to an extreme value. If training becomes unstable, restore it and explain how a too-large step can overshoot the target.
5. To diagnose memorization, make a small labeled training subset and compare its accuracy with accuracy on the untouched held-out set. Explain why the gap matters.

**Submit:** a short table of your two settings and their results, one sentence about the error pattern, and a proposed next experiment. **Self-review (10 points):** can explain the target rule (2), compare settings fairly (2), use untouched validation data (2), explain a training failure (2), acknowledge toy-data limits (2). No GPU is needed.
