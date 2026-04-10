# Part 2 — RAG, Vector Databases & Document Loaders

Part 1 taught you how to talk to an AI model — how to send it a message, shape a prompt, and get structured output back. This part goes a level deeper.

Here you'll learn how to connect an AI model to your _own_ data — documents, PDFs, notes — so it can answer questions based on content it was never trained on. The technology behind this is called **RAG**, and it's one of the most useful things you can build with AI today.

---

## What Document Loaders are and how they load PDFs into LangChain

Let's start with a basic problem: you have a PDF — maybe a textbook, a report, a set of lecture notes. You want the AI to read it and answer questions about it. But an LLM can't just open a file the way you double-click it. It only understands text that's been handed to it programmatically.

That's where **Document Loaders** come in.

A Document Loader is a tool that reads a file and converts it into a standard format that LangChain can work with. It extracts the text from the file and wraps it in a simple structure called a `Document` object.

Every `Document` has two parts:

- **page_content** — the actual text from that page
- **metadata** — extra info attached to it, like which page it came from or what file it was in

```python
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("my_notes.pdf")
documents = loader.load()

# documents is now a list — one item per page of the PDF
print(documents[0].page_content)   # text from page 1
print(documents[0].metadata)       # {"source": "my_notes.pdf", "page": 0}
```

LangChain has loaders for almost every format you'd want:

| Format            | Loader           |
| ----------------- | ---------------- |
| PDF               | `PyPDFLoader`    |
| Plain text        | `TextLoader`     |
| Website / URL     | `WebBaseLoader`  |
| CSV / spreadsheet | `CSVLoader`      |
| Word document     | `Docx2txtLoader` |

The loader handles the messy part — decoding the file format — so the rest of your pipeline always works with plain text in the same predictable structure.

---

## How Text Splitters break large documents into manageable chunks

Once you've loaded a document, you run into a new problem: the document is probably too big to work with in one piece.

LLMs have a **context window** — a hard limit on how much text they can read at once. Think of it like short-term memory. Even if a model can handle 100,000 words, a hefty textbook might have 500,000. It simply won't fit.

Even if it did fit, sending the entire book with every single question would be:

- **Expensive** — you pay per word (token) with most API providers
- **Slow** — more text means longer wait times
- **Less accurate** — models tend to lose focus when given too much text at once

The solution is to cut the document into smaller pieces called **chunks**. Later, instead of sending the whole document, you find only the chunks that are actually relevant to the question and send just those.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,       # each chunk is ~500 characters
    chunk_overlap=50      # 50 characters of overlap between chunks
)

chunks = splitter.split_documents(documents)
# chunks is now a much longer list of smaller pieces
```

**Why chunk_overlap?** When you cut text at an exact boundary, you sometimes split a sentence or idea in half. The overlap means each chunk starts a little bit before where the previous one ended — so no thought gets completely lost at the seam.

**Chunk size matters a lot:**

- Too large → each chunk contains multiple unrelated topics, search becomes imprecise
- Too small → each chunk loses context (a sentence with "it" but no mention of what "it" refers to)

For most documents, 400–800 characters per chunk with a 10–15% overlap is a good starting point.

---

## Understanding Embeddings and semantic meaning of text

Here's the core question: once you've split the document into hundreds of chunks, how do you find the right one when someone asks a question? You can't just search for matching keywords — the question might use completely different words than the answer.

The solution is **embeddings**.

An embedding is a way of turning a piece of text into a list of numbers. Not random numbers — numbers that capture the _meaning_ of the text. The clever part is this: **text that means similar things gets similar numbers**.

```
"I love dogs"        → [0.21, -0.45, 0.83, 0.12, ...]
"I adore puppies"    → [0.22, -0.43, 0.81, 0.14, ...]   ← very similar numbers

"The stock market crashed"  → [-0.91, 0.34, -0.22, 0.67, ...]  ← very different
```

Think of it like placing every sentence on a giant map where meaning determines location. Sentences that talk about the same idea end up close together on the map. Sentences about completely different things end up far apart.

When you search for something, you convert your question into an embedding too — and then look for the chunks whose numbers are mathematically closest to your question's numbers. That's how you find relevant content without needing exact keyword matches.

```python
from langchain_google_genai import GoogleGenerativeAIEmbeddings

embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Convert a sentence to a vector of numbers
vector = embedding_model.embed_query("What is backpropagation?")
# vector looks like: [0.04, -0.12, 0.77, 0.23, ...] — usually 768 numbers
```

This is the technology that makes AI search actually _understand_ your question rather than just hunting for matching words.

---

## How Vector Databases store and retrieve information

So now you have hundreds of chunks, each converted into a list of numbers. You need somewhere smart to store all of these — somewhere that can quickly find the most similar ones when a new question comes in.

That's what a **Vector Database** does.

A regular database (like MySQL or SQLite) is great at finding exact matches: "give me all rows where name = 'Alice'". But it has no concept of _similarity_. It can't answer "give me the chunks that are most similar in meaning to this question."

A Vector Database is built specifically for this. You store your chunk vectors in it, and it gives you back the most similar ones using clever search algorithms — not one-by-one comparison, but fast approximate search that works even with millions of entries.

```python
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Store all your chunks in the vector database
vectorstore = FAISS.from_documents(chunks, embedding=embedding_model)

# Save to disk so you don't have to rebuild it every time
vectorstore.save_local("faiss_index")

# Later, load it back
vectorstore = FAISS.load_local("faiss_index", embeddings=embedding_model,
                                allow_dangerous_deserialization=True)
```

Two popular options used in this repo:

**FAISS** (by Meta) — runs entirely on your machine, extremely fast, no server needed. Perfect for local projects and learning.

**Chroma** — open source, can run locally or as a server, popular in the LangChain world. Easy to persist to disk.

Both do the same job: store your chunk vectors and let you search them by meaning.

---

## What Retrievers are and how they fetch relevant context

Once your chunks are stored in a vector database, you need a way to ask: _"given this question, which chunks are most relevant?"_

That's a **Retriever**. It's the bridge between the question and the stored knowledge.

The default retriever does a simple similarity search — find the top 4 chunks whose meaning is closest to the question:

```python
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

results = retriever.invoke("What is gradient descent?")
# Returns the 4 most relevant chunks from your document
```

But there are smarter retrieval strategies too:

**MMR (Max Marginal Relevance)** — the default similarity search sometimes returns 4 chunks that all say basically the same thing. MMR adds a "diversity" requirement: each retrieved chunk must be relevant _and_ different from the ones already selected. This gives the AI a broader view of the topic.

```python
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 4, "fetch_k": 20}
)
```

**MultiQuery Retriever** — rewrites your question in several different ways and runs a separate search for each version, then combines the results. This helps catch relevant chunks that might have been missed because the wording didn't quite match.

For most use cases, the default similarity retriever with `k=4` is a solid starting point. The advanced strategies help when your document is repetitive or when users tend to ask vague, short questions.

---

## How all components connect to form a complete RAG pipeline

Let's zoom out and see how everything fits together. RAG has two distinct phases:

### Phase 1 — Loading and storing your material (done once)

This runs when you first set up the system with a new document. You won't re-run this every time someone asks a question.

```
Your PDF
    ↓
Document Loader   — reads the file, gives you text + metadata
    ↓
Text Splitter     — breaks the text into ~500 character chunks
    ↓
Embedding Model   — converts each chunk into a list of numbers
    ↓
Vector Database   — stores (numbers + original text) together on disk
```

### Phase 2 — Answering questions (runs every time a user asks something)

```
User's Question
    ↓
Embedding Model   — converts the question into numbers too
    ↓
Retriever         — finds the 4 chunks most similar to the question
    ↓
Build a Prompt    — "Use the following notes to answer: [chunks] \n\nQuestion: [question]"
    ↓
LLM               — reads the prompt and generates an answer
    ↓
Answer shown to user
```

In code, LangChain's `RetrievalQA` chain wires Phase 2 together:

```python
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

answer = qa_chain.invoke("What is backpropagation?")
print(answer["result"])
```

That's the complete RAG pipeline. Five components, two phases, one coherent system.

---

## How AI systems answer questions using external knowledge

Here's the fundamental thing that makes RAG powerful — and how it's different from just asking an LLM a question directly.

When you ask an LLM something without RAG, it can only answer based on what it learned during training. That training data has a cutoff date. It doesn't know about your company's internal docs, your personal notes, the latest research paper, or anything that happened after it was trained. If you ask about your specific textbook, it simply doesn't know.

With RAG, the AI's answer is grounded in _your_ actual content. Here's what's happening under the hood:

1. Your question comes in
2. The system searches your stored document chunks for the most relevant pieces
3. Those pieces get included in the prompt sent to the LLM: _"Here is the relevant context from the document. Use it to answer the question."_
4. The LLM reads both your question and the retrieved context, and generates an answer based on that specific material

The LLM is not making things up from training memory — it's reading the relevant paragraphs from _your_ document and synthesising an answer from them. This is why RAG is called Retrieval-**Augmented** Generation: the retrieval augments (improves and grounds) the generation.

**Why not just fine-tune the LLM on your data instead?**

Fine-tuning means retraining the model on your data so it "remembers" it permanently. But fine-tuning is expensive (needs a GPU and hours of compute), static (if your document updates, you retrain from scratch), and can sometimes make the model hallucinate more confidently. RAG avoids all of that — your knowledge base can be updated any time without touching the model at all.

**The practical takeaway:** RAG is how you build AI systems that know about _your_ domain, _your_ documents, and _your_ data — without needing to train a new model or pay for a fine-tune. It's the go-to approach for real-world AI applications that need to answer questions from specific knowledge sources.

The code in this folder shows working examples of each part of this pipeline end to end:

| File                        | What it does                                                                                      |
| --------------------------- | ------------------------------------------------------------------------------------------------- |
| `load_and_split_pdf.py`     | Loads a PDF using `PyPDFLoader` and splits it into chunks                                         |
| `load_and_split_text.py`    | Same thing but for plain `.txt` files                                                             |
| `split_text_into_chunks.py` | Experiments with chunk size and overlap to see how splitting behaves                              |
| `load_webpage.py`           | Loads content from a live website URL instead of a file                                           |
| `vector_store_db.py`        | Takes documents, converts them to embeddings, stores them in Chroma, and runs a similarity search |

`Generative_AI_part2_notes.pdf` contains the handwritten and typed notes for this whole section — useful as a companion reference alongside the code.

---

## How similarity search actually works — Cosine Similarity explained simply

When you ask a question, the system converts it into a vector (a list of numbers). It also has all your document chunks stored as vectors. Now it needs to find which chunks are "closest" in meaning to your question.

But how do you measure "closeness" between two lists of numbers?

The most common method is called **cosine similarity**. Here's the intuition:

Imagine every vector as an arrow pointing in some direction in space. Two arrows that point in almost the same direction are "similar" in meaning. Two arrows pointing in completely different directions are unrelated.

Cosine similarity measures the **angle between two arrows**. A small angle = high similarity. A large angle = low similarity.

```
         ↑  "I love dogs"
         |   \
         |    \  ← small angle = very similar
         |     ↘
         |      "I adore puppies"
         |
         |
         |                        → "The stock market crashed"
         |____________________________________

Small angle between "I love dogs" and "I adore puppies" → cosine similarity close to 1.0
Large angle between "I love dogs" and "The stock market crashed" → cosine similarity close to 0.0
```

The score goes from **0 to 1**:

- `1.0` = identical meaning
- `0.5` = somewhat related
- `0.0` = completely unrelated

So when you search, you're not matching keywords — you're finding the chunks whose arrows point in the most similar direction to your question's arrow. That's why "backpropagation" and "how neural networks learn" can match each other, even though they don't share a single word.

---

## Why a regular database can't do this fast enough

Imagine you have 100,000 chunks stored. A regular database like SQL would need to compare your question's vector against every single one of those 100,000 chunks, one by one, to find the closest match.

This is called an **O(n)** operation — the time it takes grows linearly with the number of items. 100,000 chunks? 100,000 comparisons. 10 million chunks? 10 million comparisons. With vectors that are 768 numbers long, that's a massive amount of math for every single question.

```
Regular Search (O(n)) — checks every single item:

Question vector ──┬──► Compare with chunk #1 ... score: 0.21
                  ├──► Compare with chunk #2 ... score: 0.67
                  ├──► Compare with chunk #3 ... score: 0.12
                  ├──► Compare with chunk #4 ... score: 0.89  ← best so far
                  ├──► Compare with chunk #5 ... score: 0.34
                  ├──► ...
                  └──► Compare with chunk #100,000 ... score: 0.11

Total: 100,000 comparisons every time someone asks a question. Way too slow.
```

SQL and traditional databases were never designed for this kind of math. They're built for exact matches ("find rows where name = Alice"), not for "find something approximately similar to this 768-dimensional arrow."

---

## How Vector Databases solve the speed problem — ANN Algorithms

Vector databases don't do an exhaustive comparison against every chunk. Instead, they use clever shortcut algorithms called **Approximate Nearest Neighbor (ANN)** algorithms. The word "approximate" is key — they sacrifice a tiny bit of accuracy to gain massive speed.

Think of it like this: if you're looking for a coffee shop in a new city, you don't check every building in the city one by one. You look at your neighbourhood first — because coffee shops near you are far more likely to be relevant than ones on the other side of town. ANN does the same thing with vectors.

There are three main approaches:

---

### HNSW — Hierarchical Navigable Small World

Think of this like a **multi-floor building**. The top floor has a rough map of all the neighbourhoods in the city. The bottom floor has every single house.

When searching, you start at the top floor, quickly navigate to the right neighbourhood, then descend to lower floors to find the exact closest match.

```
HNSW Structure (multi-layer graph):

Layer 2 (sparse overview):     A ──────────────────── E
                                         |
Layer 1 (medium detail):       A ──── C ──── E
                                |          |
Layer 0 (all chunks):          A ─ B ─ C ─ D ─ E ─ F ─ G

Search process:
  → Start at Layer 2, jump to the rough area
  → Drop to Layer 1, narrow it down
  → Drop to Layer 0, find the exact best matches
  → Done in a few dozen comparisons instead of thousands
```

FAISS uses HNSW. This is why FAISS can search millions of vectors in milliseconds.

---

### IVF — Inverted File Index

Think of this like a **library organised into sections**. Instead of searching every shelf, you first figure out which section your topic belongs to, then only search that section.

IVF divides all your chunk vectors into clusters during setup. Each cluster has a central point called a **centroid**. When a question comes in:

1. Find the centroids closest to the question vector (just a few comparisons)
2. Search only the chunks inside those matching clusters
3. Return the best results

```
IVF — Cluster-based search:

During setup:
  All 100,000 chunks → grouped into 256 clusters
  Each cluster gets a centroid (average center point)

During search:
  Question vector
       ↓
  Compare with 256 centroids only (fast!)
       ↓
  Find top 3 matching clusters
       ↓
  Only search chunks inside those 3 clusters (~1,200 chunks)
       ↓
  Return top results

Instead of 100,000 comparisons → only ~1,200. 80x faster.
```

---

### PQ — Product Quantization

This one is about **compressing the vectors themselves** so comparisons are cheaper.

Each vector is 768 numbers long. PQ splits it into smaller sub-vectors and replaces each sub-section with a shortcode from a pre-built lookup table. Now instead of comparing 768 numbers, you're comparing a handful of shortcodes — much faster arithmetic.

```
Original vector (768 numbers):
[0.21, -0.45, 0.83, 0.12, ... 768 numbers total]

After PQ compression (8 shortcodes):
[code_14, code_7, code_31, code_2, code_19, code_5, code_28, code_11]

Comparison is now 8 integer lookups instead of 768 floating-point multiplications.
```

PQ is often combined with IVF — this combination (IVF + PQ) is what powers large-scale production vector search at companies like Meta and Spotify.

---

### Speed comparison at a glance

```
Method              How it works                        Speed vs accuracy
──────────────────────────────────────────────────────────────────────────
Brute force (O(n))  Compare everything                  Exact, but very slow
HNSW                Multi-layer graph navigation        Very fast, ~99% accurate
IVF                 Search only nearest clusters        Fast, ~95% accurate
PQ                  Compressed vector comparisons       Fastest, minor accuracy loss
IVF + PQ (combined) Cluster search + compression        Best trade-off at scale
```

For learning and small projects (a few thousand chunks), brute force is totally fine. For production apps with millions of chunks, you'd use HNSW or IVF+PQ.

---

## Normal Database vs Vector Database — side by side

It helps to see exactly how these two are different:

```
                    Normal Database (SQL)         Vector Database
──────────────────────────────────────────────────────────────────────
What it stores      Text, numbers, dates          Vectors (lists of numbers)
How it searches     Exact match (name = "Alice")  Similarity match (nearest vectors)
Best question       "Find all orders from May"    "Find chunks about neural networks"
Can it understand   No — keyword only             Yes — meaning-based matching
  meaning?
Speed trick         Index on specific columns     ANN algorithms (HNSW, IVF, PQ)
Example tools       MySQL, PostgreSQL, SQLite      FAISS, Chroma, Annoy, Pinecone
```

You can think of it this way: a normal database finds things by their **label**. A vector database finds things by their **meaning**.

---

## Types of Vector Stores — which one to use

The three most commonly used vector stores in the LangChain and LlamaIndex world are:

### FAISS (by Meta)

- Runs **entirely on your machine** — no server, no internet needed
- Blazing fast — built in C++ with Python wrappers
- Supports both brute-force and HNSW indexing
- Save to disk with `save_local()`, reload with `load_local()`
- Best for: learning, local prototypes, offline use

```python
from langchain_community.vectorstores import FAISS

vectorstore = FAISS.from_documents(chunks, embedding=embedding_model)
vectorstore.save_local("my_faiss_index")
```

### Chroma

- Open source, easy to set up, can run locally or as a persistent server
- Very popular in the LangChain community
- Automatically persists data to a folder — just point it at a directory
- Best for: projects where you want persistence without much setup

```python
from langchain_community.vectorstores import Chroma

vectorstore = Chroma.from_documents(
    chunks,
    embedding=embedding_model,
    persist_directory="./chroma_db"
)
```

### Annoy (by Spotify)

- Stands for "Approximate Nearest Neighbors Oh Yeah" (yes, really)
- Uses tree-based indexing — builds multiple random projection trees
- Read-only after indexing (you can't add new items after building)
- Best for: music recommendation-style use cases, read-heavy workloads

```
Quick pick guide:

Just learning / building locally?       → FAISS
Need persistence with an easy setup?    → Chroma
Read-heavy, won't add new data later?   → Annoy
Production at large scale?              → Pinecone or Weaviate (managed cloud services)
```

All three are free, open source, and work directly with LangChain and LlamaIndex out of the box. For everything in this repo, FAISS and Chroma are the go-to choices.
