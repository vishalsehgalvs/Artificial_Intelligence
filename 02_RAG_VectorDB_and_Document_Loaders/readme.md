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

The code in this folder (`load_and_split_pdf.py`, `load_and_split_text.py`, `split_text_into_chunks.py`, and `load_webpage.py`) shows working examples of each part of this pipeline end to end.
