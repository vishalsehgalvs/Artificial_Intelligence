# 4. Neural networks: functions learned in layers

**Goal:** describe a forward pass, derive one backward update, and recognize unstable or overfit training. Lab: [CPU PyTorch network](../notebooks/03-deep-learning/mlp.ipynb).

## First: one decision is not always enough

Think of sorting a picture into "cat" or "not cat." One rule might notice pointed ears, another whiskers, and a third combines their evidence. A **neuron** is a little adjustable calculator: multiply its inputs by weights, add them, then pass the result through an **activation** that may keep or squash it. A **layer** is a group of these calculators; the result of one layer feeds the next. No part is literally a brain cell.

Here is a tiny two-layer example with just numbers. The first calculator receives $x=3$, multiplies by $w=2$ and gets $6$. ReLU, a common activation, keeps positive numbers and replaces negatives with zero, so its output is still $6$. A second calculator multiplies by $v=0.5$ and predicts $3$. If the true answer is $4$, the error is $3-4=-1$ and squared-error loss is $1$. This computation from input to prediction is the **forward pass**.

To learn, ask which knobs contributed to the error. **Backpropagation** carries information about the loss backward through those same calculations; it does not send the data backward in time. For this example, changing $w$ changes the first result, which changes the final result. The derivative of loss with respect to $w$ is $2(3-4)\times0.5\times3=-3$. Subtracting a small learning-rate multiple of $-3$ _increases_ $w$, nudging the prediction toward $4$. The [notebook](../notebooks/03-deep-learning/mlp.ipynb) lets PyTorch calculate gradients for many knobs automatically.

**Pause:** If we remove every activation, can stacking lots of multiply-and-add layers learn a bend that one such layer cannot? **No.** Without nonlinear steps, they collapse into another multiply-and-add rule.

## Later: the notation behind the story

A perceptron forms a weighted sum $z=w^Tx+b$ and applies a nonlinearity. Stacking layers lets an MLP express nonlinear boundaries: $h=\mathrm{ReLU}(W_1x+b_1)$, $\hat y=W_2h+b_2$. Without nonlinearities, stacked linear layers collapse to one linear transformation. Sigmoid maps a score to $(0,1)$ but saturates for large inputs; ReLU is cheap but a unit can become inactive. GELU and SiLU are smooth alternatives used in modern networks.

## Next: loss, gradients and updates in code

The loss compares prediction and target: MSE for numeric regression; cross-entropy for classification. For two classes, the model can produce two raw scores called _logits_: `[1.2, 0.3]` chooses the first class because 1.2 is larger. Use these raw logits with PyTorch's `CrossEntropyLoss`; it handles the probability normalization internally. The following five lines form **one practice round**: clear old adjustments, make predictions, measure mistakes, calculate each knob's share of the mistake, then adjust the knobs. `loss.backward()` calculates the gradients; `optimizer.step()` applies them. Always zero gradients before the next batch:

```python
optimizer.zero_grad()
logits = model(features)
loss = criterion(logits, targets)
loss.backward()
optimizer.step()
```

For $h=wx$, $\hat y=vh$ and $L=(\hat y-y)^2$, chain rule gives $\partial L/\partial w=2(\hat y-y)vx$. The output error flows backward through the multiplication by $v$. In a complete training loop: set seed, split data, fit preprocessing on train only, iterate mini-batches, evaluate under `model.eval()` and `torch.no_grad()`, save the best checkpoint on validation, then measure on test once.

## What can go wrong while training?

Imagine practicing only the exact flashcards you will be examined on. You can memorize them without understanding the topic. In ML, **overfitting** looks similar: training loss keeps falling but errors on new, held-out examples grow. Plot training and validation results after each round (**epoch**). If both are bad, consider a model that is too simple, poor features or a faulty training loop. If only the held-out results are bad, inspect leakage, data differences and overfitting before adding more layers.

Poor feature scale, learning rate, initialization and numerical instability cause exploding/vanishing gradients or NaNs. SGD is a simple baseline; Adam maintains moving moments and often trains faster initially but still needs tuning. Weight decay penalizes large weights; dropout randomly removes activations during training; early stopping halts when validation quality stops improving. Batch normalization uses batch statistics, while layer normalization normalizes features per item and is common in transformers. Never estimate validation statistics by fitting on validation labels. Plot **both** train and validation losses, not only final accuracy.

## Later: which architecture fits the structure?

CNNs reuse small filters across images, exploiting local spatial structure. RNNs process sequences with recurrent state, but long-distance dependencies can be hard to learn. LSTM and GRU gates help retain information. Transformers use attention over tokens and dominate many text applications; they trade recurrent sequential processing for heavier attention/memory. Embeddings map discrete tokens to dense learned vectors. Autoencoders compress and reconstruct; VAEs learn a probabilistic latent representation; GANs pit a generator against a discriminator; diffusion models learn iterative denoising. These solve different objectives: reconstruction, generation, classification, or forecasting. Transfer learning starts with pretrained weights and adapts a head or a subset of layers; check data and license suitability.

### Build a small CNN

Run `python labs/cnn.py` from the course root. Each image is an 8 by 8 grid of brightness numbers. A 3 by 3 convolution filter checks the same small neighborhood everywhere, like sliding a magnifying glass over the image. ReLU keeps positive signals, pooling shrinks each feature map from 8 by 8 to 4 by 4, and the final layer scores ten digits. Training uses only the training portion; the printed held-out accuracy and wrong (actual, guessed) pairs reveal mistakes. Try fewer training epochs: does held-out accuracy fall? An accuracy on these tiny digits says nothing about real handwriting from other sources.

Change the number of filters or remove pooling, predict the tensor shape before running, and explain what information was lost or retained. Sequence models do not have a bundled core lab yet, so use a small thought experiment: train on prefixes of `1, 2, 3, ...` and compare a short-memory RNN with an LSTM on long prefixes. The important diagnostic is not only final accuracy; inspect whether gradients and validation performance survive as the dependency gets farther back. For production sequence work, compare this baseline with attention rather than assuming an LSTM is always better.

Transfer learning is an optional extension when a suitable pretrained model and data license are available: freeze the feature extractor, replace the task-specific head, train the head first, then unfreeze a small final block only if validation data supports it. Compare against a randomly initialized model with the same split. A pretrained model is not automatically appropriate; check its training domain, license, input preprocessing and whether its data overlaps your evaluation set.

### Build a tiny diffusion model, not an image generator

Run `python labs/diffusion.py`. Our training numbers are clustered near -2 or +2. At a randomly selected noise level we mix a clean number with random noise; a tiny network sees the noisy value **and** the noise level, and tries to predict the added noise. We measure mean squared error between predicted and actual noise. During sampling, start with pure noise, predict and remove a little noise at each of 20 steps, then print 10 resulting numbers. The lab checks that average training loss falls and samples are finite, **not** that every output belongs to a cluster. Plot a histogram or compare counts near -2 and +2 as an extra quality check. Real image diffusion needs spatial networks, much more data and compute; this 1D example teaches the denoising loop, not realistic picture generation.

## Scaling without confusing it with learning

Mixed precision may accelerate supported GPU hardware but can change numerical behavior. Distributed training partitions data or model parameters; it is not necessary for these labs. More parameters do not guarantee better outcomes on a small dataset. A CPU model with 16 hidden units can demonstrate backprop exactly as a large model does.

**Exercises:** (1) Why can a two-layer linear network be replaced by one layer? **Answer:** matrix multiplication composes linearly. (2) Train loss falls while validation loss rises: what next? **Answer:** verify the split, then try early stopping/regularization or more representative data. (3) Why not call `softmax` before `CrossEntropyLoss`? **Answer:** the loss expects logits and handles the stable normalization internally.

Continue to [Transformers](05-transformers-llms.md). Reference: [PyTorch tutorial](https://docs.pytorch.org/tutorials/beginner/basics/intro.html).
