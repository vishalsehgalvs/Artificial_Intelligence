# 5. Transformers and large language models

**Goal:** calculate attention weights for a short sequence and explain how an LLM is trained and decoded. Lab: [attention with shape assertions](../notebooks/04-transformers/attention.ipynb).

## First: what does "it" refer to?

Read: "The dog chased the ball because **it** rolled away." To understand "it," you look back at "dog" and "ball" and give more weight to the useful word. **Attention** is a learned way for a model to decide how much information each earlier word contributes to the current word. It is a calculation, not human understanding.

A computer first splits text into pieces called **tokens** (a token can be a whole word or part of one). It represents each token by a list of numbers called an **embedding**. Suppose the current token gives two earlier tokens attention weights $1/3$ and $2/3$, whose one-number values are $3$ and $0$. The mixed result is $(1/3)\times3+(2/3)\times0=1$. The weights add up to one: this is like mixing two ingredients in a recipe. Real models use many dimensions and learn the weights from data.

For text generation, a model predicts the _next_ token. While learning to predict word three, it must not peek at word four. A **causal mask** crosses out future positions before attention weights are calculated. The [three-token notebook](../notebooks/04-transformers/attention.ipynb) makes that rule visible in a small table. **Pause:** If the future word is visible while training but hidden when generating, can the training score be trusted? **No**: the model cheated by seeing the answer.

## Later: attention as a matrix calculation

Tokenizers divide text into token IDs, often subwords rather than words. The same visible word may split differently depending on tokenizer and surrounding text. IDs index embedding vectors, giving $X\in\mathbb R^{B\times T\times D}$. Position information is necessary because attention alone has no inherent ordering: learned position embeddings, sinusoidal encodings or rotary position embeddings (RoPE) modify how positions are represented.

Think of a query as "what am I looking for?", a key as "what information do I match?", and a value as "what information do I take away?" These are not hand-written search terms: the model learns three numerical projections from each token. Self-attention projects inputs to queries $Q=XW_Q$, keys $K=XW_K$ and values $V=XW_V$. Each row of $QK^T$ gives match scores from one token to all tokens it is allowed to see. Softmax turns those scores into nonnegative weights that add to one. For one head:

$$\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}+M\right)V.$$

For $T=3$, $QK^T$ is $[3,3]$: each query scores each key. $d_k$ is the size of a key vector; dividing by its square root helps keep scores manageable. $M$ is a mask: it sets disallowed _future_ positions to $-\infty$ before softmax, so token 2 cannot see token 3. Padding masks suppress placeholder positions. With score row $[0,\ln 2]$, softmax weights are $[1/3,2/3]$; if values are $[3,0]$, the output is 1. A masked position receives weight zero, not a small learned score. The formula describes **one attention head**, not a complete LLM.

Multihead attention projects into several smaller subspaces, concatenates results and mixes them with an output projection. Grouped-query attention (GQA) shares key/value heads across query heads to reduce inference memory. Feed-forward layers act per token, while attention exchanges information across tokens. Residual connections plus normalization stabilize deep stacks. Decoder-only models predict the next token; encoder-only models learn bidirectional representations; encoder-decoder models read input then generate output.

## Next: training and generating are different jobs

Think of pretraining as practicing "complete the next piece" on many texts, then instruction tuning as practicing response examples. During **inference**, the trained model generates one new token, adds it to the context and repeats. A **context window** is how much input/history fits in one request; a larger window is not a promise that every detail will be used well. Read the rest of this section after you can explain that loop in your own words.

Pretraining often minimizes next-token cross-entropy on large corpora; masked-token and sequence-to-sequence objectives are alternatives. Supervised instruction tuning (SFT) teaches example response patterns; preference optimization (including RLHF and direct preference methods) influences behavior but does not prove factuality. During generation, greedy choice, temperature, top-$k$ and top-$p$ change diversity. A context window bounds the tokens available to one call; long inputs increase cost and can degrade retrieval of relevant details. KV caching reuses past attention keys/values during autoregressive generation, improving speed but consuming memory. Attention's naive computation uses $O(T^2)$ score storage; implementations can reduce memory without making all long-context tradeoffs disappear.

Mixture-of-experts (MoE) routes each token through a subset of expert blocks; it can increase parameter count without invoking every expert each time. Multimodal models integrate image, audio or video representations with text, with distinct tokenization, latency and safety limits. Fine-tuning all weights is expensive; LoRA learns low-rank adapters. Quantization lowers weight precision for inference, trading memory/latency against quality. Distillation teaches a smaller model from a larger one. None of these removes the need for evals on your task and population.

### A practical token and adaptation checklist

Before calling a real model, tokenize representative short, long, multilingual and code-like inputs with that model's tokenizer. Record token counts, special tokens, truncation and padding; character count is not a reliable proxy for token count. Never silently truncate system instructions or retrieved evidence. For adaptation, distinguish **prompting** (no weight changes), **SFT** (learn from input/response examples), preference optimization (learn from chosen-versus-rejected behavior) and retrieval (keep knowledge outside the weights). Start with retrieval or a prompt change when the problem is missing evidence; consider fine-tuning only when repeated style or task behavior justifies a training set and a held-out evaluation.

For deployment, compare the same held-out prompts before and after LoRA or quantization. Track answer quality, refusal behavior, token latency, peak memory and license constraints. A smaller or cheaper model is an engineering tradeoff, not an accuracy guarantee. The attention notebook is the core hands-on exercise; tokenizer and fine-tuning experiments are optional extensions because their packages and model downloads change quickly.

**Exercises:** (1) Why does attention need a mask for next-token training? **Answer:** to prevent future tokens leaking into the prediction. (2) If $Q$ is $[B,H,T,d]$, what is $QK^T$? **Answer:** $[B,H,T,T]$. (3) Does a 128k context guarantee accurate use of all 128k tokens? **No**; context capacity and effective retrieval are different.

Next: [AI applications](06-ai-applications.md). References: [original transformer paper](https://arxiv.org/abs/1706.03762) and [PyTorch basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html). Check the [source ledger](sources.md) before using a provider-specific model.
