# Document Loaders, Text Splitters, Vector DBs, RAG & LangChain

This is **Part 2** of the Generative AI course by [Sheryians AI School](https://sheryians.com). Part 1 covered the basics — LLMs, API keys, chat models, Hugging Face, prompts, and structured output. Now we go deeper and build something real.

---

## Quick recap of Part 1

Before diving in, here's what was covered in Part 1 so you know what's assumed:

- What LLMs are and how to use API keys
- Working with Chat Models (like GPT, Claude)
- Using Hugging Face models and running models locally
- How messages and prompts work
- Getting Structured Output from LLMs

If any of those feel unclear, go back to Part 1 first. This part builds directly on top of them.

---

## What are we building?

We're building **CourseMate AI** — an AI-powered study assistant that lets students _chat with their own study material_.

The idea is simple: you upload a PDF (a textbook chapter, lecture notes, research paper, anything), and then just ask questions about it in plain English. The system reads the material, understands it, and answers your questions — like having a tutor who has already read everything.

**Example use cases:**

- "Summarise chapter 3 of my ML notes"
- "What did the lecture say about backpropagation?"
- "Explain the part about cosine similarity in my own notes"
- "What are the differences between CNN and RNN from this textbook?"

The core technology making all of this possible is called **RAG — Retrieval-Augmented Generation**.

---

## What is RAG and why does it exist?

You might wonder — can't we just send the whole PDF to the LLM and ask questions? The answer is: not really, and here's why.

LLMs have a **context window** — a hard limit on how much text they can process in one go. Think of it like short-term memory. GPT-4 can handle maybe 128,000 tokens at a stretch, which sounds like a lot, but a single 200-page textbook can easily be 500,000+ tokens. That just won't fit.

Even if it did fit, stuffing an entire book into every prompt is:

- **Expensive** — you're charged per token
- **Slow** — more tokens = more time to process
- **Less accurate** — LLMs tend to "lose focus" when given too much text at once, a problem known as the "lost in the middle" issue

RAG solves all of this. Instead of sending the whole book, it finds only the _most relevant_ paragraphs for each question and sends just those. The LLM gets focused, useful context and gives a much better answer.

**RAG = teaching the AI to retrieve before it answers.**

---

## Why not just fine-tune the LLM instead?

Fine-tuning means training the model on your data so it "remembers" it permanently. Sounds good, but:

- Fine-tuning is **expensive** — needs a GPU and hours of training
- Fine-tuning is **static** — once trained, outdated information is baked in. Update the textbook? Retrain the whole model.
- Fine-tuned models can **hallucinate** more — they think they "know" the answer but might blend things incorrectly

RAG avoids all of this. You can swap, update, or add new documents any time without touching the model at all. It's dynamic, cheap, and far more practical for most real-world use cases.

---

## How RAG works — the 10-step plan

RAG has two distinct phases: first you process and store all the material (done once), then you handle queries (done every time a user asks something). Here's the full pipeline:

### Step 1 — User uploads study material

The user provides their learning material — could be a PDF, Word document, plain text file, scraped lecture notes, or a research paper. This is the "knowledge base" that CourseMate AI will learn from.

### Step 2 — Document Loading

The raw file can't be understood by code directly. A **Document Loader** reads the file and converts it into a standard document object — basically structured text with some metadata attached (like page number, filename, source URL, etc.).

LangChain has loaders for almost every format: `PyPDFLoader` for PDFs, `TextLoader` for plain text, `WebBaseLoader` for scraping web pages, `CSVLoader` for spreadsheets, and many more.

```python
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("machine_learning_notes.pdf")
documents = loader.load()
# documents is now a list of Document objects, one per page
```

Each `Document` object has two fields:

- `page_content` — the actual text on that page
- `metadata` — extra info like `{ "source": "ml_notes.pdf", "page": 4 }`

### Step 3 — Text Splitting (Chunking)

The loaded documents are often large. A 50-page PDF has thousands of words. We need to break this down into smaller, manageable **chunks** before we can work with them.

This is one of the most important steps in the whole pipeline — the quality of your chunks directly affects the quality of your answers. More on this in a dedicated section below.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)
# chunks is now a much longer list of smaller Document pieces
```

### Step 4 — Embedding Generation

Each chunk of text gets passed through an **Embedding Model**, which converts it into a vector — a list of decimal numbers like `[0.23, -0.81, 0.44, 0.07, ...]`.

This vector is a numerical representation of the _meaning_ of that text. Pieces of text that mean similar things will have vectors that are close to each other mathematically. This is what allows semantic search to work.

```python
from langchain_openai import OpenAIEmbeddings

embedding_model = OpenAIEmbeddings()
# This is called internally by the vector store — you rarely call it manually
```

### Step 5 — Vector Database Storage

The embeddings (vectors) get stored in a **Vector Database** along with the original text and metadata. This database is optimised specifically for finding similar vectors quickly. The data now "lives" here and can be queried any time.

```python
from langchain_community.vectorstores import Chroma

vectorstore = Chroma.from_documents(chunks, embedding=embedding_model, persist_directory="./chroma_db")
```

### Step 6 — User asks a question

The user types something like: _"What is backpropagation?"_

### Step 7 — Query Embedding

The user's question gets converted into a vector using the **same embedding model** used to embed the chunks. This is crucial — both the query and the chunks must live in the same "vector space" for comparison to make sense.

### Step 8 — Similarity Search

The vector store searches through all stored chunk vectors and finds the ones that are mathematically closest to the query vector. "Closest" here means they have a similar meaning — not the same words, but the same _idea_.

### Step 9 — Retriever

The top-K most similar chunks get selected (usually top 3 to 5). These are the pieces of your study material that are most likely to contain the answer.

### Step 10 — LLM generates the answer

The original question + the retrieved chunks get combined into a prompt and sent to the LLM. The LLM now has focused, relevant context and generates a proper answer grounded in _your_ material.

```python
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
answer = qa_chain.invoke("What is backpropagation?")
```

---

## What are Embeddings, really?

This concept confuses a lot of people, so let's spend more time on it.

Imagine you could place every sentence in the English language on a big map. Sentences that mean similar things are placed close together. Sentences that mean completely different things are placed far apart.

- "The dog ran fast" and "The dog sprinted quickly" would be right next to each other
- "The dog ran fast" and "Photosynthesis happens in leaves" would be very far apart

An **embedding** is that position on the map — stored as a long list of numbers. The map has hundreds or thousands of "directions" instead of just left/right/up/down like a normal map, but conceptually it works exactly the same way.

When you compare embeddings mathematically, you're answering the question: _"How similar is the meaning of these two pieces of text?"_

That's why embeddings are the backbone of semantic search. You're not matching keywords — you're matching _meaning_.

---

## Text Splitting — a deep dive

### Why it matters so much

The chunk size you choose has a huge impact on how well your RAG system works:

- **Too large** → each chunk contains multiple topics, the embedding becomes blurry and imprecise, retrieval quality drops
- **Too small** → each chunk loses context (e.g., a pronoun without the noun it refers to), the answer can be confusing

There's no single perfect size — it depends on your content. General guidelines:

- **500–1000 characters** works well for most documents
- **chunk_overlap of 10–15%** of chunk size helps avoid cutting ideas in half

### Chunk Overlap

When you split text, you can allow a small overlap between consecutive chunks. For example, if chunk 1 ends with "...the loss function is minimised using", chunk 2 might start with those same words. This way, no sentence gets completely lost at a boundary.

```
Chunk 1: [...paragraph A... partial sentence...]
Chunk 2: [...partial sentence continued... paragraph B...]
```

### 3 ways to split text

| Method                       | How it works                                                                                                    | When to use it                                        |
| ---------------------------- | --------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------- |
| **Character-Based**          | Cuts at every N characters. Simple and fast, but mindlessly cuts at any point, even mid-word                    | Quick prototypes, very large datasets                 |
| **Token-Based**              | Counts tokens (the way LLMs read text) instead of characters. More aligned with model limits                    | When you need to stay precisely within a token budget |
| **Semantic / Meaning-Based** | Splits at natural idea boundaries — paragraphs, topic changes, sentence endings. Keeps complete thoughts intact | Production systems where quality matters most         |

LangChain's `RecursiveCharacterTextSplitter` is the most commonly used and tries to split on `\n\n`, then `\n`, then spaces, then characters — in that order. It prefers clean splits but falls back to dirtier ones if needed.

---

## Vector Stores — what they are and how they work

### The problem with normal databases

Imagine you have 1,00,000 chunks, each stored as a 512-number vector. Now a user asks a question, and you convert it to a vector too. To find the most relevant chunks, you need to compare the query vector to all 1,00,000 stored vectors.

A normal database (MySQL, PostgreSQL, MongoDB) would do this by scanning every single row — one by one. With 1,00,000 vectors, that means 1,00,000 individual comparisons for every single question a user asks. The more data you have, the slower it gets — it never improves. That's just not practical.

### How Vector Stores solve this

Vector databases use clever shortcut algorithms called **ANN (Approximate Nearest Neighbor)**. The word "approximate" is key — they don't promise to find the _absolute_ closest match 100% of the time, but they find something _very close_ in a tiny fraction of the time. And for our use case, close enough is more than good enough.

The three main ANN algorithms:

#### HNSW — Hierarchical Navigable Small World

Think of it like a layered map. At the top layer, there are only a few widely spaced vectors. At the bottom, all vectors exist. When you search, you start at the top, find the rough direction, then zoom into finer and finer layers. Like zooming into Google Maps.

#### IVF — Inverted File Index

Divide all your embeddings into clusters (like neighbourhoods). Each cluster has a centroid — the "average" point at the centre. When a query comes in:

1. Compare the query to all centroids (just 5–10, not 1,00,000)
2. Identify the nearest centroid (nearest neighbourhood)
3. Search only within that neighbourhood

**Concrete example:**

- 1,00,000 embeddings split into 5 clusters of 20,000 each
- Query matches Cluster 3's centroid
- Search only 20,000 vectors instead of 1,00,000
- **Result: 5× faster search**

#### PQ — Product Quantization

Instead of storing full high-precision vectors (which take lots of memory), PQ compresses them into a much smaller representation. It loses a tiny bit of precision but dramatically reduces memory usage and speeds up comparisons.

### Normal DB vs Vector DB — side by side

| Feature          | Normal Database                            | Vector Database                                    |
| ---------------- | ------------------------------------------ | -------------------------------------------------- |
| What it stores   | Structured rows, text, numbers             | Embeddings (high-dimensional vectors)              |
| How it searches  | Exact match — row by row                   | Similarity search — ANN algorithms                 |
| Example query    | `WHERE movie = "Interstellar"`             | "find movies about space exploration"              |
| Index type       | B-Tree, Hash index                         | HNSW, IVF, PQ                                      |
| Best used for    | Banking systems, user profiles, e-commerce | AI search, RAG systems, recommendations            |
| Matching method  | Equality checks, range filters             | Cosine similarity, dot product, Euclidean distance |
| Handles meaning? | No — only exact values                     | Yes — finds semantically similar content           |

### Popular Vector Stores

**Chroma**

- Open source, very popular in the LangChain ecosystem
- Runs locally or as a server
- Great for prototyping and small-to-medium projects
- Easy to persist to disk: `persist_directory="./chroma_db"`

**FAISS (Facebook AI Similarity Search)**

- Built by Meta, extremely fast
- Runs entirely in memory (or can be saved/loaded from disk)
- Industry standard for high-performance vector search
- No server needed — works as a Python library

**Annoy (Approximate Nearest Neighbours Oh Yeah)**

- Built by Spotify for music recommendation
- Very memory-efficient
- Read-heavy workloads (great for inference, not great for frequent updates)

---

## Retrievers — picking the right chunks

A **Retriever** is the component that takes the user's question and pulls the most relevant chunks from the vector store. It's the bridge between the stored knowledge and the LLM.

### There are two broad categories of retrievers:

**1. By Data Source**
Instead of searching your own vector store, these retrievers pull from external knowledge sources like Wikipedia, Arxiv (academic papers), or PubMed (medical research). Useful when you need real-time or specialised external knowledge.

**2. By Retrieval Strategy**
This is what we focus on in CourseMate AI. These retrievers all work on your local vector store but use different strategies to select the best chunks.

---

### Strategy 1 — Similarity Search (the default)

The most straightforward approach. The query vector is compared to all stored chunk vectors, and the top-K closest ones are returned.

**How "closeness" is measured:**

**Cosine Similarity** (most common)

- Measures the angle between two vectors
- 1.0 = identical direction (same meaning), 0.0 = completely unrelated, -1.0 = opposite meaning
- Works regardless of vector length, so it's robust

**Dot Product**

- Multiplies corresponding elements of two vectors and sums them
- Faster to compute than cosine similarity
- Best used when vectors are already normalised (length = 1)

**Euclidean Distance**

- Straight-line distance between two points in vector space
- Smaller distance = more similar
- More sensitive to the magnitude of vectors, less common for text

```python
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)
```

---

### Strategy 2 — MMR (Max Marginal Relevance)

**The problem with pure similarity search:**

When you ask "What is gradient descent?", pure similarity search might return:

- Chunk 1: "Gradient descent minimises the loss function by adjusting weights"
- Chunk 2: "Gradient descent is used to reduce loss in neural networks"
- Chunk 3: "Loss minimisation via gradient descent updates model parameters"
- Chunk 4: "Gradient descent works by computing gradients of the loss"

All four are highly relevant to the query — but they're all saying basically the same thing. You've used up all 4 of your retrieval slots with redundant information. That wastes your context window and token budget, and gives the LLM nothing new to work with.

**How MMR fixes this:**

MMR scores each candidate chunk on two things simultaneously:

- **Relevance score** — how similar is this chunk to the query?
- **Novelty score** — how different is this chunk from the ones already selected?

It picks chunks that are both relevant _and_ diverse. The result is 4 chunks that cover the topic from different angles — much richer context for the LLM.

```python
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 4, "fetch_k": 20, "lambda_mult": 0.5}
)
# lambda_mult: 0 = max diversity, 1 = max relevance, 0.5 = balanced
```

**When to use MMR:** When your document has a lot of repetitive or overlapping content, or when the same concept is explained in multiple similar ways across the text.

---

### Strategy 3 — MultiQuery Retriever

**The problem with single-query search:**

The embedding of a query is sensitive to exact wording. The same question phrased differently produces a different vector, which might retrieve different (and sometimes better) results.

**Real example:**

- Your query: _"What is gradient descent?"_
- This chunk would match well: _"Gradient descent is an optimisation algorithm that minimises loss"_
- But this chunk might NOT match well enough: _"Neural networks are trained using algorithms like gradient descent to adjust weights iteratively"_

The second chunk is absolutely relevant, but the wording is so different that pure vector similarity might rank it lower.

**How MultiQuery Retriever fixes this:**

It uses the LLM itself to automatically generate several rephrased versions of your original question:

Original: "What is gradient descent?"

Generated variations:

- "Explain the gradient descent algorithm"
- "How does gradient descent work in machine learning?"
- "What is the role of gradient descent in training neural networks?"
- "Optimization algorithm that minimizes loss function"

Each variation gets its own similarity search. All results are combined and deduplicated. The final set of chunks has much broader coverage.

```python
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(),
    llm=llm
)
```

**Full pipeline:**

```
User's Question
       ↓
LLM generates 3–5 rephrased variations of the question
       ↓
Similarity search is run for EACH variation separately
       ↓
All retrieved chunks are combined
       ↓
Duplicates are removed
       ↓
Final set of diverse, relevant chunks → sent to LLM
```

**When to use MultiQuery:** When users ask short or ambiguous questions, or when you need high recall (catching every potentially relevant chunk matters more than speed).

---

## Similarity metrics — which one to use and why

You've seen these terms pop up a few times. Here's a clearer breakdown:

### Cosine Similarity

Measures the angle between two vectors, not their absolute distance. This makes it great for text because a short sentence and a long paragraph covering the same topic will still have vectors pointing in roughly the same direction.

- Range: -1 to +1 (practically 0 to 1 for text)
- 1.0 = identical meaning, 0.0 = completely unrelated
- **Most commonly used for text similarity**

### Dot Product

You multiply each pair of numbers from the two vectors together, then add them all up. It's faster to compute than cosine similarity. The catch: it works best when vectors are already scaled to the same length. If they are, you get basically the same result as cosine anyway.

### Euclidean Distance

This is just the straight-line distance between two points — the kind of distance you'd measure with a ruler. Unlike cosine, it cares about the size of the vector, not just its direction. So two sentences that mean exactly the same thing but differ in length might look "far apart" in Euclidean terms even though they shouldn't be.

- Not great for text for that reason
- More useful in image or number-based tasks

---

## The full RAG architecture — both phases together

### Phase 1 — Loading and storing your material (done once, upfront)

```
PDF / Document
      ↓
  Document Loader   ← reads the file, extracts text + metadata
      ↓
  Text Splitter     ← breaks text into small chunks (e.g. 500 chars each)
      ↓
    Chunks[]        ← list of smaller Document objects
      ↓
 Embedding Model    ← each chunk → vector [0.23, -0.81, 0.44, ...]
      ↓
  Vector Store      ← stores (vector, original text, metadata) together
                       persisted to disk for later use
```

### Phase 2 — Querying (runs every time a user asks a question)

```
  User's Question
        ↓
  Embedding Model   ← question → vector
        ↓
  Similarity Search ← compare question vector to all stored chunk vectors
        ↓
  Top-K Chunks      ← most relevant pieces of the uploaded material
        ↓
  Build Prompt      ← "Answer using this context: [chunks] \n\n Question: [question]"
        ↓
      LLM           ← reads the prompt, generates a grounded answer
        ↓
    Final Answer
```

The key insight: the LLM never reads the whole PDF. It only sees the ~4 most relevant chunks. Everything else is handled by the retrieval system.

---

## What is LangChain and why are we using it?

LangChain is basically the glue that holds everything together. Without it, you'd have to manually connect the document loader, text splitter, embedding model, vector store, retriever, and LLM — each with their own completely different format and API. That's a lot of annoying plumbing code.

LangChain gives you one consistent way to talk to all of them, so you can focus on what you're building instead of how things connect.

**What LangChain handles in this project:**

- `Document Loaders` — standardised way to load any file format
- `Text Splitters` — clean splitting with overlap control
- `Embeddings` — unified interface for OpenAI, Hugging Face, Cohere, etc.
- `Vector Stores` — single API to switch between Chroma, FAISS, Pinecone, etc.
- `Retrievers` — similarity, MMR, MultiQuery all plug in the same way
- `Chains` — connect retriever + prompt + LLM into one callable pipeline

---

## Common mistakes to avoid

**1. Chunk size too large**
Bigger chunks = blurry embeddings = poor retrieval. If answers feel vague or off-topic, try reducing chunk size.

**2. No chunk overlap**
Without overlap, ideas that span a chunk boundary get lost. Always set some overlap (10–15% of chunk size is a good starting point).

**3. Using the wrong embedding model for retrieval**
Whatever model you used to turn chunks into vectors — you _must_ use that exact same model to convert the user's question into a vector too. If you mix models, the vectors live in completely different "worlds" and the similarity scores become meaningless garbage.

**4. Forgetting metadata**
Chunk metadata (page number, source file, section title) is extremely useful. Without it, you can't tell the LLM _where_ the answer came from, and users can't verify it.

**5. Not saving the vector store to disk**
If you don't persist the vector store, every time your app restarts you have to re-embed every single chunk from scratch. That costs time and money (embedding API calls aren't free). Always save it to disk and just load it on startup.

---

## Tech stack

| Component        | Tool                                                  |
| ---------------- | ----------------------------------------------------- |
| Framework        | LangChain                                             |
| Document Loading | `PyPDFLoader`, `TextLoader`, `WebBaseLoader`          |
| Text Splitting   | `RecursiveCharacterTextSplitter`                      |
| Embedding Models | OpenAI `text-embedding-ada-002`, Hugging Face models  |
| Vector Store     | Chroma (local), FAISS, Annoy                          |
| LLM              | GPT-4o, GPT-4o-mini, or local models via Hugging Face |
| Language         | Python                                                |

---

## A closer look at Document Loaders

So far we've talked about document loading at a high level. Here's what's actually happening under the hood when you use one in LangChain.

### Why do we even need a Document Loader?

A PDF is just a binary file. A `.txt` is just raw bytes. Your Python code can't understand any of that directly — it needs to convert those files into a consistent format that the rest of the pipeline (splitter, embedder, vector store) can work with.

That's what Document Loaders do. They read the file and hand you back a list of `Document` objects — each one containing the text from that page/section plus some extra information about where it came from.

This is also the whole point of RAG in one sentence: **the LLM gets useful information that it was never trained on.** It didn't read your textbook during training — but through RAG, you're feeding it that knowledge at the moment it needs to answer.

### How to use one in code

```python
from langchain_community.document_loaders import TextLoader

loader = TextLoader("my_notes.txt")
documents = loader.load()
```

Every document object you get back looks like this:

```python
Document(
    page_content="This is the actual text on the page...",
    metadata={"source": "my_notes.txt", "page": 0}
)
```

- **`page_content`** — the text itself, the thing that eventually becomes a chunk and gets embedded
- **`metadata`** — useful context like the filename, page number, or URL. This travels with the chunk all the way to the final answer, so you can always tell the user _where_ the information came from

### `load()` vs `lazy_load()`

There are two ways to load documents:

**`load()`** — reads everything at once and returns all documents in memory. Fine for small files.

**`lazy_load()`** — reads and yields documents one at a time, without loading the whole file into memory first. Use this when you're dealing with a really large document and don't want your app to freeze or crash.

```python
# lazy_load processes one page at a time — much gentler on memory
for doc in loader.lazy_load():
    process(doc)
```

### Loading PDFs specifically — PyPDF

For PDFs, LangChain uses a library called **PyPDF** under the hood. The loader is called `PyPDFLoader` and it gives you one `Document` per page.

```python
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("lecture_notes.pdf")
pages = loader.load()

print(pages[0].page_content)   # text from page 1
print(pages[0].metadata)       # {"source": "lecture_notes.pdf", "page": 0}
```

PyPDF works well for most standard PDFs. If you're dealing with scanned PDFs (images of text rather than actual text), you'd need an OCR-based loader instead — but that's a more advanced setup.

---

_Part of the Generative AI learning series — Sheryians AI School, Part 2_
