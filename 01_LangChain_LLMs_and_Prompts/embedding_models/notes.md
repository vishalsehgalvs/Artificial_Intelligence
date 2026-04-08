# Embedding Models — Notes

## What's in this folder

| File                        | What it does                                                                 |
| --------------------------- | ---------------------------------------------------------------------------- |
| `text_to_vectors.py`        | Converts sentences into vectors and compares them by meaning                 |
| `find_similar_documents.py` | Stores documents in a vector database and searches by meaning — not keywords |

---

## What is an Embedding?

An embedding is a way of turning **text into numbers** — specifically, a long list of decimal numbers called a **vector**.

The magic is that **meaning is preserved in those numbers**. Two sentences that mean similar things end up with vectors that are mathematically close to each other.

```
"I love dogs"     → [0.21, -0.45, 0.83, 0.12, ...]
"I adore puppies" → [0.22, -0.43, 0.81, 0.14, ...]   ← very similar!

"The stock market crashed" → [-0.91, 0.34, -0.22, 0.67, ...]  ← very different
```

Humans look at words. Computers look at numbers. Embeddings are the bridge.

---

## Why Do We Need Embeddings?

Three main use cases:

### 1. Semantic Search

Traditional search looks for exact word matches. Semantic search finds results that mean the same thing even if the words are different.

```
Query: "How do I make my code run faster?"

Keyword search finds:  docs containing "make", "code", "run", "faster"
Semantic search finds: "Performance optimisation techniques", "Profiling Python code",
                        "Reducing time complexity" → these don't share words but share meaning
```

### 2. RAG (Retrieval-Augmented Generation)

The most important use case. You store your documents as embeddings in a vector database. When a user asks a question, you find the most relevant chunks and feed them to the LLM.

```mermaid
flowchart LR
    A[User Question] --> B[Embed the question]
    B --> C[Search vector DB\nfind closest document chunks]
    C --> D[Send question + chunks to LLM]
    D --> E[LLM answers using YOUR documents ✅]
```

### 3. Clustering / Recommendations

Group similar items together or recommend similar content based on meaning rather than metadata.

---

## How Similarity is Measured

The most common way is **cosine similarity**:

- Score of **1.0** = identical meaning
- Score of **0.8–0.9** = very similar
- Score of **0.5** = somewhat related
- Score near **0** = unrelated

Think of each vector as an arrow pointing in space. Cosine similarity measures the angle between two arrows — the smaller the angle, the more similar the meaning.

---

## What is a Vector Database?

A regular database stores text and numbers in rows and columns.
A vector database stores **embeddings** and is optimised to answer:
_"Which of these million vectors is closest to this query vector?"_ — very fast.

| Database     | Type                     | Notes                                  |
| ------------ | ------------------------ | -------------------------------------- |
| **FAISS**    | Local library (Facebook) | Fast, free, runs in memory or on disk  |
| **Chroma**   | Local or cloud           | Popular for LangChain RAG projects     |
| **Pinecone** | Cloud service            | Managed, scales to millions of vectors |
| **Weaviate** | Self-hosted or cloud     | Open source, good for production       |

For learning, **FAISS** is the go-to — it's free, local, and needs no account.

---

## Embedding Providers Compared

| Provider        | Model                    | Free?                      | Dimensions | Notes                      |
| --------------- | ------------------------ | -------------------------- | ---------- | -------------------------- |
| **Google**      | `models/embedding-001`   | ✅ Free tier               | 768        | Good general purpose       |
| **OpenAI**      | `text-embedding-3-small` | ❌ Paid (~$0.02/1M tokens) | 1536       | Very popular, high quality |
| **HuggingFace** | `all-MiniLM-L6-v2`       | ✅ Always free             | 384        | Runs locally, no API key   |
| **Ollama**      | `nomic-embed-text`       | ✅ Always free             | 768        | Runs locally, needs Ollama |

**Dimensions** = the length of the vector (number of numbers). More dimensions usually means more nuance captured, but more storage and compute.

---

## Using Free Local Embeddings (No API Key)

If you don't want to use any API, `sentence-transformers` from HuggingFace runs entirely on your machine:

```python
from langchain_huggingface import HuggingFaceEmbeddings

# Model downloads automatically on first run (~90MB)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector = embeddings.embed_query("What is deep learning?")
print(len(vector))  # 384 dimensions
```

This is the cheapest and most private option for learning and small projects.

---

## Key Vocabulary

| Term                         | Plain English                                                                           |
| ---------------------------- | --------------------------------------------------------------------------------------- |
| **Vector**                   | A list of numbers representing meaning                                                  |
| **Embedding**                | The process of converting text into a vector                                            |
| **Vector store / Vector DB** | A database optimised for storing and searching vectors                                  |
| **Cosine similarity**        | A score (0 to 1) measuring how similar two vectors are                                  |
| **Semantic search**          | Finding results by meaning, not exact word match                                        |
| **RAG**                      | Retrieval-Augmented Generation — using a vector DB to give LLMs access to your own data |
| **Dimensions**               | How many numbers are in each vector                                                     |
| **FAISS**                    | A fast local vector search library (Facebook AI Similarity Search)                      |
