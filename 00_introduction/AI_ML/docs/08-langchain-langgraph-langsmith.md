# 8. LangChain, LangGraph and LangSmith

**Goal:** choose the smallest orchestration layer needed, and evaluate behavior before enabling hosted tracing. Lab: [pure-Python and graph workflow](../notebooks/07-frameworks/graph.ipynb) (LangGraph portion optional).

## First: sort, look up, answer

A shop assistant can follow three steps: read a customer's question, decide whether it asks about delivery or returns, look up the matching note, then answer. We can write that as three ordinary Python functions. If there is no matching note, return "I don't know." **You do not need a framework to do this.**

Suppose we later add a branch: "if a refund exceeds $100, pause for approval; otherwise answer." Draw boxes for actions and arrows for what runs next. That picture is a **graph**. Its **state** is the little bundle of information passed between boxes, such as `{question, topic, answer}`. LangGraph is a library for building and running such stateful graphs. The [framework notebook](../notebooks/07-frameworks/graph.ipynb) compares normal functions with the equivalent graph. Learn plain Python first, then use a graph when branching, resuming or approval becomes hard to manage.

LangChain helps connect models and tools and offers a ready-made agent loop; LangSmith helps inspect what happened and compare answers against a test set. Neither makes an incorrect answer correct by itself. Begin with a paper test: two shipping questions and one unanswerable question, each with an expected result. Count correct answers locally before considering hosted tracing.

## Later: choosing among the three products

| Product   | Core responsibility                                                 | Good reason to use it                                                                 |
| --------- | ------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| LangChain | Common model/tool interfaces, `create_agent` harness and middleware | Several providers or reusable tool-calling agent components                           |
| LangGraph | StateGraph, edges, conditional routing, persistence and interrupts  | Multi-step stateful process with resumability/human approval                          |
| LangSmith | Traces, datasets, evaluators and experiments                        | Team debugging and comparing agent versions; requires account/key for hosted features |

Start with plain Python: `classify(query)` then `lookup(topic)` then `answer(excerpt)`. A graph adds typed state, nodes that return state updates, and edges that decide what runs next. A conditional edge chooses a path based on state; it is _not_ a guarantee that a model's text is valid. Compile the graph, invoke it with input, and test both matching and no-match routes. Persistence uses a checkpoint store and thread identity so interrupted work can resume; use an interrupt before a side-effecting node and inspect its state before allowing continuation. Avoid replaying non-idempotent actions on recovery.

LangChain's `create_agent` composes models, tools and instructions and is built on LangGraph. Its current [overview](https://docs.langchain.com/oss/python/langchain/overview) demonstrates the interface; provider-specific calls need installed integration packages and a model/key. LangGraph's [overview](https://docs.langchain.com/oss/python/langgraph/overview) shows `StateGraph`, `START` and `END`. An offline graph with deterministic nodes needs no LLM, so it is an excellent first test. When no graph behavior is needed, keep the plain Python pipeline.

LangSmith [evaluation](https://docs.langchain.com/langsmith/evaluation) separates offline curated dataset experiments from monitoring production traces. Start with a local list of `(question, expected_source)` pairs and score exact source IDs; this still works without a hosted account. Then optionally send synthetic trace data to LangSmith for comparison, redacting secrets and checking organization retention policies first. LLM-as-judge is not ground truth; compare it with human adjudications and inspect disagreements.

**Exercise:** A tool must ask for approval, resume tomorrow and never run the payment twice. Choose plain Python or graph with checkpoints? **Graph with a durable checkpoint and explicit interrupt**, plus idempotency key/transaction controls outside the graph. Do not confuse checkpointing with financial exactly-once execution.

Next: [Local AI](09-local-ai-and-emerging-tools.md). Recheck upstream APIs before installing optional extras with `python -m pip install -e ".[agents]"`.
