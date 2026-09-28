# 6. AI applications: retrieval, evaluation and safety

**Goal:** build a local retrieval baseline and separate retrieval errors from generation errors. Lab: [offline retrieval](../notebooks/05-rag-evaluation/retrieval.ipynb).

## First: answer from three notes

Imagine a shop has three notes: "Delivery takes three days," "Returns are accepted within 30 days," and "Passwords must not be shared." Someone asks, "How long does delivery take?" A simple program looks for the most relevant note, then answers **"Three days [delivery note]."** If asked, "When is the store open?", it should say **"I cannot find that in these notes"** instead of guessing.

Finding the note is **retrieval**. Writing an answer from it is **generation**. Putting them together is retrieval-augmented generation (**RAG**). The [offline notebook](../notebooks/05-rag-evaluation/retrieval.ipynb) deliberately uses a fixed answer instead of a language model, so you can inspect what was found without confusing a search mistake with a writing mistake.

Try two failure cases: (1) The returns note was found for the delivery question: retrieval failed. (2) The delivery note was found but the answer says seven days: generation failed. Keep the note's ID next to the answer so a reader can verify it. Later, compare whether a more elaborate search actually fixes these cases.

## Later: build a stronger search

For a question-answering tool over documents: collect permitted text, chunk it without losing provenance, index it, retrieve candidates, optionally rerank, and then answer with cited source IDs. A sparse TF-IDF vector stores term weights; cosine similarity compares vector direction. An embedding model can match semantic paraphrases but introduces model/license/download choices and may miss exact identifiers. Hybrid retrieval combines keyword and semantic candidates; a reranker rescores a short list. For a single short corpus, keyword search plus a refusal when evidence is weak can be stronger than an elaborate pipeline.

Retrieval-augmented generation (RAG) is not a guarantee of truth. A generator can misread a relevant document, fabricate a citation or follow instructions in retrieved text. Treat source text as _data_, not as developer instructions. Keep origin IDs, chunk boundaries and a strict instruction hierarchy. Return "I do not know from these documents" when evidence is absent; do not turn a high similarity score into a factual confidence value.

## Measure each stage separately

Make a tiny quiz: three questions that have answers in the notes and one that does not. For each, write down the expected note ID **before** trying the search. Count how often the correct note was retrieved, then separately count how often the final answer agrees with that note. A model that sounds fluent while quoting the wrong note has still failed.

Write a small set of questions with expected source IDs and acceptable answers, including unanswered questions and malicious documents. Retrieval recall@$k$ = proportion of relevant source IDs appearing in top $k$. Generation faithfulness asks whether every factual statement is supported by retrieved evidence; answer correctness asks if it answers the question; citation accuracy checks whether IDs actually support the claim. Also measure latency, cost, regressions and abstention rate. Split eval questions away from any examples used to tune prompts or thresholds. For nondeterministic LLM answers, run repeated trials and inspect failures; a judge model is useful but must itself be calibrated against human review.

For a stronger retrieval experiment, keep the corpus and questions fixed and compare three candidates: TF-IDF, dense embeddings and a simple hybrid union. Measure recall@$1$ and recall@$3$ against expected source IDs, then inspect failures involving exact identifiers, spelling changes and paraphrases. Dense retrieval needs a model download and can introduce licensing, memory and domain-shift concerns; it is not automatically better. Preserve document IDs and chunk text so every score can be traced back to evidence.

Prompts specify goal, boundaries, available tools and expected format. JSON/schema-constrained output helps machines consume responses but does not ensure correct facts. Function calling is a _request_ to execute code in your application, not proof the call is safe. Validate arguments, enforce authorization and use read-only tools by default. Consider prompt injection, sensitive data exfiltration, data poisoning, copyright/license limits, disparate error rates and human appeal paths for consequential uses. Log the minimum necessary data and use opt-in redaction before vendor tracing.

## Other modalities and applications

Vision classification/detection/segmentation, speech recognition and synthesis, text-to-image/video generation, forecasting, recommenders and robotics have different metrics and consent requirements. A multimodal model may process images and text together, but evaluation still needs task-specific labels and adversarial cases. When a deterministic rule already solves a task reliably, prefer it over a model.

**Exercise:** Your system retrieves the correct policy paragraph but answers using a conflicting older paragraph. Is this mainly a retrieval failure? **No:** relevant evidence reached the generator; inspect ranking, prompt/context assembly and answer faithfulness. What if the relevant paragraph never appears in top 5? **Retrieval** is the first suspect.

Next: [Agents and MCP](07-agents-mcp.md). See [OpenAI tool guidance](https://developers.openai.com/api/docs/guides/tools) and [LangSmith evaluation concepts](https://docs.langchain.com/langsmith/evaluation).
