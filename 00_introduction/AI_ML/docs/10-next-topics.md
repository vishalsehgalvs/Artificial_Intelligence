# 10. Watchlist: important directions, not a hype checklist

Checked 2026-09-28. Revisit the [source ledger](sources.md) before investing in fast-changing products.

## Read a new AI announcement like a product label

Suppose a new model claims to be "twice as fast." Before learning its API, ask: **fast at what task, on whose computer, compared with what, and does it keep my data on my machine?** Imagine comparing two delivery services: "faster" is not useful unless we know the route, price and reliability. The same habit applies to AI. The table below lists topics worth following _after_ the core lessons, not prerequisites to starting them.

| Direction                             | Learn now                                  | A question to test in practice                                              |
| ------------------------------------- | ------------------------------------------ | --------------------------------------------------------------------------- |
| Reasoning/test-time compute           | Deliberation latency and verification      | Does extra inference improve _your_ task's accuracy enough to justify cost? |
| Small/on-device and multimodal models | Quantization, hardware and privacy         | Does the local model pass the same evals as your baseline?                  |
| Efficient attention and context       | KV cache, grouped-query attention, memory  | What happens to latency and correctness as context grows?                   |
| Agent skills/tool catalogs            | Scoped capabilities, explicit permissions  | Can a tool read or write more than its task requires?                       |
| Fine-tuning and synthetic data        | LoRA, contamination and held-out evals     | Does training beat retrieval plus a better baseline?                        |
| Interpretability and uncertainty      | Probes, calibration, counterfactual tests  | Do explanations survive a distribution shift?                               |
| Protocol interoperability             | MCP versioning, authentication and consent | Does the host remain safe with an untrusted server?                         |

Do not equate an announcement with a reproducible result. Record model ID, source, evaluation set, version/date, hardware, license and data egress before adopting a new technique. Track changes in [PyTorch](https://docs.pytorch.org/tutorials/), [OpenAI](https://developers.openai.com/api/docs/guides/agents), [Anthropic](https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview), [MCP](https://modelcontextprotocol.io/specification/2025-11-25) and [Ollama](https://docs.ollama.com/) upstream docs.

**Try it:** A model advertisement reports 95% accuracy but never says what questions were asked. Can you decide whether it will answer your policy questions correctly? **No.** First create your own small, held-out set of policy questions and compare it with your existing baseline.
