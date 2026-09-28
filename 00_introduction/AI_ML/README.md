# Practical AI: from foundations to agents

A Python-first course that starts with a fruit-shop price guess, then gradually builds toward neural networks and AI agents. You do not need to know what a tensor, transformer or protocol is yet; each is introduced when it becomes useful. Start at [Setup and learning guide](docs/00-start-here.md). Core labs use bundled or generated data, run on a local CPU, and do not require API keys. GPU, downloads, hosted APIs, and account-based services are optional.

**How to read this course:** First read the story and predict the result; next run the tiny example; only then read the formula or advanced comparison. If a section says "later" or "optional," skip it on your first pass. Each lesson ends with answers so you can check your understanding without searching the internet.

## Route through the course

| Stage | Read                                                                                                                                               | Build                                                  | Checkpoint                                          |
| ----- | -------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------ | --------------------------------------------------- |
| 0     | [Setup and learning guide](docs/00-start-here.md)                                                                                                  | Python environment                                     | Run a notebook cell                                 |
| 1     | [Math and data foundations](docs/01-foundations.md), [Classical AI](docs/02-classical-ai.md)                                                       | Gradient descent and graph search                      | Explain the difference between search and learning  |
| 2     | [Machine learning](docs/03-machine-learning.md)                                                                                                    | Leakage-safe classifier, clustering and evaluation lab | Defend a baseline and its metric                    |
| 3     | [Neural networks](docs/04-neural-networks.md)                                                                                                      | CPU PyTorch training loop                              | Diagnose overfitting                                |
| 4     | [Transformers and LLMs](docs/05-transformers-llms.md), [AI applications](docs/06-ai-applications.md)                                               | Attention and offline retrieval                        | Trace one attention shape and one retrieval failure |
| 5     | [Agents and MCP](docs/07-agents-mcp.md)                                                                                                            | Read-only tool simulation; optional protocol tutorial  | Identify an unsafe tool call                        |
| 6     | [LangChain, LangGraph, LangSmith](docs/08-langchain-langgraph-langsmith.md), [Local AI and emerging tools](docs/09-local-ai-and-emerging-tools.md) | Deterministic graph, optional Ollama                   | Compare direct Python with a framework              |
| 7     | [What to watch next](docs/10-next-topics.md), [Revision guide](docs/11-revision-guide.md)                                                          | Capstone projects                                      | Explain the tradeoffs in a design review            |

Start with the [notebooks index](notebooks/README.md) after reading the first chapter. Finish with [tabular ML](projects/01-tabular-ml/README.md), [neural-network debugging](projects/02-tiny-neural-network/README.md), and [offline retrieval and agents](projects/03-local-rag-agent/README.md). The [source ledger](docs/sources.md) dates the primary references and calls out changing APIs.

After the starter notebooks, try the small scripts in [the lab guide](labs/README.md): train/tune/diagnose a forest, classify tiny digits with a CNN, and generate one-dimensional samples by iterative denoising. These extend specific algorithm families; they do not claim to cover every published algorithm.

## What you will actually do

Read the [plain-English topic map](docs/topic-map.md) first if the names in the table above feel unfamiliar. Then work one row at a time:

1. **Guess a price:** If two apples cost $4, guess the price of three. Adjust a wrong guess by a small amount. This introduces data, loss and learning.
2. **Find a route:** Compare the trip with the fewest stops to the cheapest trip. This is AI search, where the rules are known rather than learned.
3. **Sort examples:** Hold back some labeled examples as a fair test. Compare a trained classifier with an always-pick-the-most-common-label baseline.
4. **Train a small network:** Connect little adjustable calculators; watch training mistakes fall, then check whether new examples also improve.
5. **Look up evidence:** See which earlier word a token can use, then find the right shop note before answering a question. No large model download is required.
6. **Offer a safe tool:** Let an assistant request a read-only policy lookup. Check its arguments in ordinary code before letting anything execute. Compare the same fixed workflow in plain Python and optionally LangGraph.

At each stage ask: **What was the input? What was predicted? How did we check? When might this answer be wrong?** The worked notebooks teach one idea at a time; the later parts of each chapter name other algorithms you can return to after the example makes sense.

## What comprehensive means here

Major algorithm families receive intuition, relevant equations, code and failure modes. Specialist variants are compared in selection tables rather than pretending that every published algorithm fits in one course. Stable ideas are taught in the core; rapidly changing provider integrations are isolated as optional labs. Expect roughly 8-12 weeks at 5-7 hours per week, plus time for the projects.

## Entry check

You are ready for stage 1 if you can add and multiply numbers and follow a short Python example. You do **not** need to know what a train/test split is yet. If Python itself is new, read the short code and the printed answers together; use [Start here](docs/00-start-here.md) to set up your editor. At each checkpoint, guess the output before running the cell, then compare it with the answer.
