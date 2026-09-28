# Notebook path: one small experiment at a time

Open a notebook in VS Code, select your **AI ML Course** kernel, read its first text cell and run code cells from top to bottom with Shift+Enter. A cell is just a box of text or runnable Python. If an `assert` fails, compare the printed values with your prediction before changing more code. Notebooks never write output files or require a login.

| Order | Notebook                                                  | Before you run it, predict...                   | Requirement                     |
| ----- | --------------------------------------------------------- | ----------------------------------------------- | ------------------------------- |
| 1     | [Gradient descent](01-foundations/gradient_descent.ipynb) | Will the prediction move closer to 5?           | NumPy, CPU                      |
| 2     | [Two ways to search](01-foundations/search.ipynb)         | Will the shortest route also be cheapest?       | Python only                     |
| 3     | [Classification](02-ml/classification.ipynb)              | Can a majority-class rule miss every rare case? | scikit-learn, CPU               |
| 4     | [Clustering](02-ml/clustering.ipynb)                      | Can unlabeled data form groups?                 | scikit-learn, CPU               |
| 5     | [Tiny neural net](03-deep-learning/mlp.ipynb)             | Can a model combine two inputs?                 | PyTorch, CPU                    |
| 6     | [Three-token attention](04-transformers/attention.ipynb)  | Will the first token see the third?             | NumPy, CPU                      |
| 7     | [Offline retrieval](05-rag-evaluation/retrieval.ipynb)    | What if no note matches the question?           | scikit-learn, CPU               |
| 8     | [Read-only tool](06-agents-mcp/agent.ipynb)               | What happens when a tool name is not allowed?   | Python only                     |
| 9     | [Plain Python vs graph](07-frameworks/graph.ipynb)        | Does a graph change the answer?                 | Python only; optional LangGraph |

Notebooks 1-9 use no paid APIs or model files. For each, read its matching [lesson](../README.md) first. If an optional library is missing, skip its cell and finish the plain-Python part. The local-model experiment is described in the [Ollama lesson](../docs/09-local-ai-and-emerging-tools.md), not a mandatory notebook.
