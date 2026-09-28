# 9. Local models, Ollama, OpenClaw and Jev

**Goal:** distinguish offline inference from cloud APIs and assess whether a tool belongs in a local workflow. All product labs here are optional; the [offline retrieval notebook](../notebooks/05-rag-evaluation/retrieval.ipynb) teaches the underlying idea without a model download.

## First: where does the answer come from?

If you write a question on paper and ask a friend in the same room, your note has not gone to another company. If you send it through a website, it has. Local AI asks the same practical question: **which computer receives my text, and which computer does the work?** A model file stored and run on your own machine can answer without sending prompts to a cloud provider, but an app with a local-looking interface might still call an online service.

Ollama is software that can manage and run local model files, and can also work with cloud models. A **model** is the trained file that produces responses, not the Ollama app itself. First inspect your available memory and free disk space; then check the model's size, license and whether its tag is local. You can learn this course without downloading any model. OpenClaw is a different kind of tool: it connects an assistant to messaging and other actions, which adds account permissions and security risks. Jev is a specialized early-access decision model, not something you need to train or install for these lessons.

**Pause:** You open a local app but it sends requests to a cloud model. Are your questions guaranteed to remain on your computer? **No.** Follow the network destination, not the app's name.

## Later: sizing and running a local model

Ollama can run open-weight models on your computer or use cloud-hosted models: **using the Ollama command does not by itself mean data stays local**. Review model license, tag/size, quantization, context length, RAM/VRAM and provenance before downloading. Quantized weights use less memory but can change quality. Long context increases KV-cache memory. On Windows use the [official quickstart](https://docs.ollama.com/quickstart), choose a model that fits your machine, verify it is tagged local, and only then opt in to a pull. The currently documented `gemma4:e2b` download is about 7.2 GB with roughly 8 GB VRAM recommended; it is intentionally _not_ required.

After opting in and verifying the model tag in the library:

```powershell
ollama pull gemma4:e2b
ollama run gemma4:e2b
```

A local HTTP request to `http://localhost:11434/api/chat` with a selected local model differs from a request to `https://ollama.com/api/chat` or another vendor endpoint. Bind local services carefully; do not expose inference servers publicly without authentication. Model downloads, license checks, speed comparisons, embedding APIs and tool calls are exercises for a machine with adequate resources. If memory is insufficient, keep using the offline TF-IDF demo or select a demonstrably smaller licensed model from the current catalog.

## OpenClaw is an optional gateway, not a beginner prerequisite

The [Ollama integration](https://docs.ollama.com/integrations/openclaw) describes OpenClaw as a personal AI assistant connecting messaging services to an agent gateway. The guide documents `ollama launch openclaw` and recommends at least a 64k-token context for local models. **Do not run it just to study this course.** Before experimenting, read its current security notice, use a separate account and test workspace, disable write/shell/network tools unless necessary, restrict channel access, require action approvals, keep real credentials out of prompts, and plan how to stop its gateway. Messaging service setup and Ollama web search may need sign-in/network access. No automatic installation or personal account integration is included here.

## Jev and typed decisions

[TypeSafe's docs](https://docs.typesafe.ai/) describe Jev, an early-access model for typed choice, score and confidence-style decisions given a state. This is conceptually different from asking a generative model to output JSON: a schema constrains **format**, not the correctness or calibration of the answer. Compare a simple three-way routing problem: a deterministic keyword rule, a supervised classifier's `predict_proba`, and a hypothetical typed model's probabilities. For 100 cases assigned confidence near 0.8, check how many were actually right; if only 50, the score is not calibrated. Route uncertain cases to human review. The vendor's latency/accuracy claims are not independently established here, and API access is not required.

**Exercise:** A local UI uses a cloud-tagged model through Ollama. Is it private/offline? **No**: follow the request destination, not the name of the launcher. What would justify a decision model instead of a rule? **Repeated ambiguous cases with labeled outcomes, measured improvements and an acceptable cost/privacy profile.**

Continue to [What to watch next](10-next-topics.md) and the [projects](../projects/03-local-rag-agent/README.md).
