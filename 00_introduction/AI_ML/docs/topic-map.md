# The whole course in everyday language

You do not need to learn these names at once. Read one section, try its example, and follow its link only when the example makes sense. A model is a program whose adjustable parts are learned from examples; an agent is an application that can choose from allowed actions. They are not the same thing.

## 1. AI: choosing a useful action

Imagine finding your way home. If you know every road and fare, you can search for the cheapest route without training a model. If travel times are unknown, you might learn them from past journeys first. *Artificial intelligence* is the broad field of building systems that reason, search, learn or act. A hard rule, a probability estimate and a language model solve different problems. **Try:** Which route wins if one has two roads costing 10 total and another has three roads costing 6? Cheapest is the second, shortest by number of roads is the first. Go to [classical AI](02-classical-ai.md) and its [search lab](../notebooks/01-foundations/search.ipynb).

## 2. ML: learning from past examples

To spot urgent support messages, give a program old messages with known labels. Hold some back as a surprise exam. A model that always predicts "routine" can look accurate if urgent messages are rare; compare the number of real urgent messages found too. Numeric answers such as a house price are *regression*; yes/no labels are *classification*. Grouping messages with no labels is *clustering*, which discovers similarity, not necessarily meaning. **Try:** If 6 of 10 urgent alerts are right, precision is 6/10. If there were 12 urgent messages in all, recall is 6/12. Read [ML](03-machine-learning.md) and run [classification](../notebooks/02-ml/classification.ipynb). Revisit trees, forests, boosting, nearest neighbors, Bayes, SVMs and PCA in the comparison tables only after the basic test makes sense.

## 3. Neural nets: layers of adjustable calculators

One calculator multiplies inputs, adds a number and passes the result on. Many connected calculators form layers. Training measures how wrong an answer was, then nudges their weights to improve it. A network can memorize its practice examples: always check held-out answers. Image filters (CNNs), sequence memory (RNNs/LSTMs) and attention (transformers) use different structures for different inputs. **Try:** If the network guesses 3 and the answer is 4, squared error is $(3-4)^2=1$. Read [neural networks](04-neural-networks.md) and run the [small CPU model](../notebooks/03-deep-learning/mlp.ipynb).

## 4. Transformers: deciding what earlier words matter

In "the ball rolled because it was round," which earlier word helps explain "it"? Attention mixes information from relevant tokens (text pieces). While predicting the next piece, the model must not peek at later pieces; a mask prevents cheating. Large language models learn patterns from many sequences, then generate new tokens one after another. They can be fluent and still wrong. Model size, context length, fine-tuning, quantization and GPU memory are tradeoffs, not guarantees of quality. **Try:** Predict which cells in a three-token attention table must be zero when future words are hidden; then run the [attention lab](../notebooks/04-transformers/attention.ipynb). See [transformers](05-transformers-llms.md).

## 5. Applications: find the right note before answering

A shop has separate notes on shipping and returns. A customer asks about shipping: find that note, answer with its source ID, and say "I don't know" if no note covers the question. Searching the notes is *retrieval*; writing from them is *generation*. Together they can form RAG. Measure whether the correct note was found separately from whether the answer follows it. Do not treat an instruction written inside a retrieved page as permission to use tools. **Try:** Ask about opening hours when the notes contain no hours. The right answer is uncertainty, not a guess. Run [offline retrieval](../notebooks/05-rag-evaluation/retrieval.ipynb) and read [AI applications](06-ai-applications.md).

## 6. Agents and MCP: asking to use a tool

An assistant asks `lookup_policy("refund")`; your app checks the tool name and input, then returns a fixed note. The assistant does not get to grant itself permission to delete files. A *workflow* follows predefined steps; an *agent* may choose a next step from permitted options. MCP is a common connection protocol between an application and tool/data providers, not a model or a safety system. **Try:** Would you allow a webpage to add a new `delete_files` command? No: webpages are untrusted data. Start with the [read-only simulation](../notebooks/06-agents-mcp/agent.ipynb), then read [agents and MCP](07-agents-mcp.md). The simulation is not a live MCP server.

## 7. Frameworks and local models: optional tools

LangChain helps connect models and tools; LangGraph helps run branching, stateful steps; LangSmith helps inspect and compare runs, usually with an account. First write `classify -> look up -> answer` in normal Python so you can tell what a framework adds. Ollama can serve a model on your own machine **or** use a cloud model: check where requests actually go. OpenClaw can connect assistants to messaging accounts and actions, so treat it as an optional security-sensitive experiment. Jev is an early-access typed-decision service; its claimed performance is not a substitute for your own tests. Read [frameworks](08-langchain-langgraph-langsmith.md), [local models](09-local-ai-and-emerging-tools.md), and the dated [source ledger](sources.md).

**Your next move:** return to the [course route](../README.md), begin at stage 1, and record one wrong prediction from each notebook. Understanding *why* it was wrong is the skill you are building.