# Vectorless RAG & PageIndex — An Honest Breakdown

> Everyone is talking about Vectorless RAG like it is going to replace traditional RAG forever. This guide breaks down exactly how both work, when to use which, and gives you the honest verdict nobody else is giving.

---

## What we cover

- How traditional RAG actually works (quick recap)
- The real problems with vector-based retrieval
- How PageIndex builds a tree without embeddings
- When Vectorless RAG wins — and when it completely fails
- Final verdict — which one should YOU use

---

## Quick Recap — How Traditional RAG Works

Before we talk about what Vectorless RAG is, let's make sure the baseline is crystal clear.

In regular (vector-based) RAG, you do two things:

**Step 1 — Build the knowledge base once:**
You take your documents, split them into small chunks, convert each chunk into a list of numbers (embeddings), and store all those number-lists in a vector database.

**Step 2 — Answer questions at runtime:**
When someone asks a question, you convert the question into numbers too, find the chunks whose numbers are closest to the question's numbers, and hand those chunks to the LLM as context.

```
TRADITIONAL RAG — Full Picture
═══════════════════════════════════════════════════════════════

SETUP (done once):

  Your Document
       │
       ▼
  ┌─────────────┐
  │ Text        │  "How neural networks learn..." → chunk #1
  │ Splitter    │  "Backpropagation adjusts..."   → chunk #2
  └─────────────┘  "Gradient descent is..."       → chunk #3
       │
       ▼
  ┌─────────────────┐
  │ Embedding       │  chunk #1 → [0.21, -0.45, 0.83, ...]
  │ Model           │  chunk #2 → [0.19, -0.41, 0.79, ...]
  └─────────────────┘  chunk #3 → [-0.33, 0.71, 0.12, ...]
       │
       ▼
  ┌─────────────────┐
  │ Vector          │  Stores all the number-lists so they
  │ Database        │  can be searched quickly later
  └─────────────────┘

QUERY (runs every time someone asks something):

  User asks: "How does backpropagation work?"
       │
       ▼
  ┌─────────────────┐
  │ Embedding       │  question → [0.18, -0.39, 0.80, ...]
  │ Model           │
  └─────────────────┘
       │
       ▼
  ┌─────────────────┐
  │ Vector Search   │  Find the 4 chunks with the most
  │ (Similarity)    │  similar number-lists
  └─────────────────┘
       │
       ▼
  ┌─────────────────┐
  │ LLM             │  "Here are some notes. Answer the
  │                 │   question using them."
  └─────────────────┘
       │
       ▼
  Answer to the user
```

This works really well. It is the backbone of nearly every AI document chat app you've seen. But it has some real problems that people don't talk about enough.

---

## The Real Problems With Vector-Based Retrieval

Traditional RAG is powerful — but it is not perfect. Here are the genuine pain points you will run into the moment you move beyond simple demos.

### Problem 1 — Embeddings are not always semantically accurate

Embeddings are trained on general internet text. They are very good at capturing general meaning. But if your document uses specific domain language — legal terms, niche medical vocabulary, internal company jargon — the embedding model might not understand it properly.

```
Example — legal contract:
  Your document says: "force majeure clause exempts performance"
  User asks: "what happens if the contract can't be fulfilled?"

  ┌─────────────────────────────────────────────────────────┐
  │ Embedding model sees these as different topics.         │
  │ "force majeure" ≠ "can't be fulfilled" in vector space  │
  │ → Retrieves the WRONG chunks                            │
  └─────────────────────────────────────────────────────────┘
```

The AI says "I don't know" — not because the answer isn't in your document, but because the retriever couldn't find it.

### Problem 2 — Chunking destroys context

When you split a 100-page document into 500-character chunks, you are severing natural relationships. A conclusion in chapter 8 might only make sense if you read the setup in chapter 2. But the retriever returns isolated chunks, not the full thread.

```
Original document flow:
  Chapter 2: "The patient has condition X."
  Chapter 5: "Treatment Y was applied."
  Chapter 8: "The patient recovered."   ← this is the retrieved chunk

  The LLM only gets: "The patient recovered."
  It has no idea WHAT they recovered from, or HOW.

  Result: Incomplete, potentially misleading answer
```

### Problem 3 — Embeddings cost money and time

Every time you update your document, you need to:
1. Re-chunk the changed sections
2. Re-embed all the new chunks (API call = money)
3. Update the vector database

For a document that changes frequently — a live knowledge base, a product manual that gets updated weekly — this becomes expensive and slow.

### Problem 4 — Retrieval is a black box

When traditional RAG returns the wrong answer, it is very hard to debug. Why did the retriever pick chunk #47 instead of chunk #12? The similarity scores are just numbers. There is no clear human-readable reason. This makes it hard to improve.

```
Traditional RAG debug experience:

  User: "Why is the answer wrong?"
  You:  "Let me check... the retriever returned these chunks..."
        Chunk #47: similarity 0.71
        Chunk #83: similarity 0.68
        Chunk #12: similarity 0.65  ← the RIGHT chunk, but ranked 3rd

  Why was 0.71 > 0.65 here? Hard to tell. The numbers don't explain themselves.
```

These four problems are why people started asking: **what if we just didn't use embeddings at all?**

---

## What Vectorless RAG Is

Vectorless RAG is a retrieval approach that finds relevant content from your documents **without converting anything to vectors or using an embedding model at all**.

Instead of turning text into numbers and doing similarity math, it indexes your document as a **structured tree** and uses the LLM itself (or lightweight keyword logic) to navigate that tree and find relevant sections.

The most well-known implementation of this is called **PageIndex**.

---

## How PageIndex Works — Building a Tree Without Embeddings

PageIndex reads your document and builds a hierarchical summary tree — like a table of contents that goes several layers deep, with each layer being a more detailed breakdown of the layer above it.

Here is the full picture:

```
PAGEINDEX — Building the Tree (setup phase):

  Your Document (e.g., 50-page PDF)
       │
       ▼
  ┌──────────────────────────────────────────────────────────┐
  │  LEVEL 3 — Full document summary (1 node)                │
  │  "This document covers neural networks, training         │
  │   methods, and modern architectures."                    │
  └──────────────────────────────────────────────────────────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
  │ LEVEL 2     │  │ LEVEL 2     │  │ LEVEL 2     │
  │ Section A   │  │ Section B   │  │ Section C   │
  │ summary     │  │ summary     │  │ summary     │
  └─────────────┘  └─────────────┘  └─────────────┘
        │                │                │
    ┌───┴───┐        ┌───┴───┐        ┌───┴───┐
    ▼       ▼        ▼       ▼        ▼       ▼
  ┌───┐  ┌───┐    ┌───┐  ┌───┐    ┌───┐  ┌───┐
  │L1 │  │L1 │    │L1 │  │L1 │    │L1 │  │L1 │
  │   │  │   │    │   │  │   │    │   │  │   │
  └───┘  └───┘    └───┘  └───┘    └───┘  └───┘
  Actual paragraphs / pages from the document

  Each Level 1 node = raw text from your document
  Level 2 = LLM-generated summaries of groups of Level 1 nodes
  Level 3 = LLM-generated summary of all Level 2 summaries
```

Now when someone asks a question, the system **walks down the tree** from top to bottom — at each level deciding which branch to follow based on which summary best matches the question. It only reads the full raw text when it reaches the bottom.

```
PAGEINDEX — Answering a Question (query phase):

  User asks: "How does backpropagation work?"
       │
       ▼
  Check LEVEL 3 summary:
  "This document covers neural networks, training methods, and architectures."
  → Yes, "training methods" sounds relevant. Keep going.
       │
       ▼
  Check LEVEL 2 summaries (3 options):
    A: "Introduction to neurons and layers"       ← probably not
    B: "Training, loss functions, backpropagation"  ← YES, this one
    C: "Modern architectures like CNNs and RNNs"  ← probably not
       │  (only follow branch B)
       ▼
  Check LEVEL 1 nodes under branch B (5 pages):
    B1: "Loss functions and how they measure error"    ← maybe
    B2: "Gradient descent step by step"                ← yes
    B3: "Backpropagation: the full algorithm"           ← YES
    B4: "Learning rate and convergence"                 ← maybe
       │
       ▼
  Read raw text from B2 + B3 (only 2 pages retrieved)
       │
       ▼
  ┌─────────────────┐
  │ LLM             │  Answers the question using
  │                 │  only those 2 pages as context
  └─────────────────┘
       │
       ▼
  Answer to the user
```

No embeddings. No vector math. Just structured navigation through summaries.

---

## Vectorless RAG vs Traditional RAG — Side by Side

```
                    TRADITIONAL RAG          VECTORLESS RAG (PageIndex)
                    ──────────────────────   ──────────────────────────
Retrieval method    Embedding similarity     Tree navigation + summaries
Embedding model     Required                 Not needed
Vector database     Required                 Not needed
Setup cost          Medium (embed all docs)  Higher (LLM builds summaries)
Update cost         Re-embed changed chunks  Re-summarise changed sections
Retrieval style     "Find similar numbers"   "Navigate down the right branch"
Handles long docs   Struggles (context gap)  Better (top-down structure)
Domain-specific     Can miss jargon          Less affected by vocab mismatch
Debuggability       Hard (black box scores)  Easier (can read the summaries)
Speed at query      Very fast                Slightly slower (multi-step)
Cost per query      Low                      Higher (LLM calls at each level)
```

---

## When Vectorless RAG Wins

There are specific scenarios where ditching vectors is genuinely the right call:

### Long, structured documents

If your document is 200+ pages with a clear structure — a textbook, a legal agreement, a technical manual — PageIndex is excellent. The tree mirrors the document's own structure and navigates it intelligently.

```
200-page Legal Contract:

  Traditional RAG struggle:
    Clause on page 3 references a definition on page 87.
    Chunking breaks that link. Retriever returns page 3 chunk
    without the page 87 context. Answer is incomplete.

  PageIndex advantage:
    Tree is built around sections of the contract.
    Query walks down to "Definitions" section when needed.
    Full relevant section is retrieved together.
```

### Domain-specific or unusual vocabulary

If your documents use niche language that general embedding models don't handle well, tree navigation based on LLM-generated summaries is more reliable — because the LLM generating those summaries already understands the domain language.

### When you can't afford a vector database

PageIndex only needs file storage for the tree structure. No Chroma, no FAISS, no Pinecone subscription. For budget-conscious setups or local environments, this matters.

### When explainability matters

If you need to audit which part of the document an answer came from — for compliance, legal review, or debugging — the tree path is easy to log and read. You can literally print "the answer came from Section B → Sub-section B3 → page 47."

---

## When Vectorless RAG Completely Fails

This is the part people skip. Vectorless RAG is not magic. There are situations where it is clearly the wrong choice.

### Short or unstructured documents

If your document has no logical hierarchy — just a flat list of facts, a set of FAQs, a collection of short product descriptions — there is no meaningful tree to build. PageIndex's tree navigation becomes arbitrary and unreliable.

```
Example — 500 product descriptions:

  Each product is 3–4 lines. There is no chapter structure.
  No nested sections. Just flat data.

  PageIndex builds a tree anyway, but the summaries are vague:
    Level 2: "Products related to electronics"
    Level 2: "Products related to clothing"
    ... but a lot of products don't fit neatly into one category

  The navigation misfires. Traditional RAG (with embeddings)
  handles this better because it compares each product description
  directly to the query.
```

### High-volume, real-time search

Every query in PageIndex requires multiple LLM calls — one at each level of the tree. At scale (thousands of queries per minute), this gets expensive and slow fast.

```
Cost comparison for 10,000 daily queries:

  Traditional RAG:
    → 1 embedding call per query (cheap)
    → 1 vector search (no LLM cost)
    → 1 LLM call to generate answer
    Total LLM calls: 10,000

  PageIndex (3-level tree):
    → 1 LLM call at Level 3
    → 1 LLM call at Level 2
    → 1 LLM call at Level 1
    → 1 LLM call to generate answer
    Total LLM calls: 40,000 — 4x more expensive
```

### When you need broad coverage across many documents

Traditional RAG with a vector database can search across thousands of documents in milliseconds. PageIndex builds a tree per document — if you have 10,000 documents, navigating all those trees for a single query is not practical without significant engineering.

### When your questions are vague or conversational

Vector similarity handles fuzzy, natural-language questions well — "tell me something about training" will still surface relevant chunks. Tree navigation needs to pick the right branch at each level, and vague questions can lead the navigator down the wrong branch with no recovery mechanism.

---

## Final Verdict — Which One Should YOU Use?

Here is the honest answer: **they are tools for different jobs**, and picking the right one depends on your specific situation.

```
DECISION GUIDE:

  Start here:
       │
       ▼
  Is your document long (100+ pages) and
  clearly structured (chapters, sections)?
       │
      YES ──────────────────────► Consider PageIndex
       │                          (but check cost below)
       NO
       │
       ▼
  Do you have many documents (100+)
  to search across simultaneously?
       │
      YES ──────────────────────► Traditional RAG
       │
       NO
       │
       ▼
  Is your vocabulary very domain-specific
  and do embeddings keep missing the right chunks?
       │
      YES ──────────────────────► PageIndex is worth trying
       │
       NO
       │
       ▼
  Do you need fast, cheap, high-volume queries?
       │
      YES ──────────────────────► Traditional RAG
       │
       NO
       │
       ▼
  Are you building a local prototype with no
  vector DB infrastructure?
       │
      YES ──────────────────────► PageIndex (simpler setup)
       │
       NO
       ▼
  Probably Traditional RAG — it is battle-tested
  and works well for the majority of use cases
```

### The plain English summary

**Use Traditional RAG when:**
- You have many documents or a large unstructured knowledge base
- Your queries need to be fast and cheap at scale
- Your vocabulary is general enough that embeddings understand it
- You are building something that needs to go live quickly

**Use Vectorless RAG / PageIndex when:**
- You have one (or a few) long, well-structured documents
- You need to explain exactly where the answer came from
- Embeddings keep failing you because of niche vocabulary
- You want to avoid the vector database infrastructure entirely
- You are working in a constrained local environment

**The honest verdict:**
Traditional RAG is not going anywhere. For most real-world applications — especially anything at scale — it is still the right default. Vectorless RAG and PageIndex are a genuine improvement for a specific type of problem: deep, structured, single-document retrieval where explainability and vocabulary precision matter.

The hype makes it sound like Vectorless RAG replaces everything. It does not. What it does is solve a real problem that traditional RAG genuinely struggles with — and for those specific cases, it solves it well.

Know the problem you are solving first. Then pick the tool.

---

## At a Glance — The Complete Comparison

```
┌──────────────────────┬────────────────────────┬────────────────────────┐
│ Factor               │ Traditional RAG         │ Vectorless RAG         │
├──────────────────────┼────────────────────────┼────────────────────────┤
│ Best for             │ Many docs, scale        │ Long structured docs   │
│ Embedding model      │ Required                │ Not needed             │
│ Vector database      │ Required                │ Not needed             │
│ Setup complexity     │ Medium                  │ Medium-High            │
│ Query cost           │ Low                     │ Higher (multi-LLM)     │
│ Query speed          │ Very fast               │ Slower                 │
│ Long doc handling    │ Struggles               │ Excellent              │
│ Many docs search     │ Excellent               │ Impractical            │
│ Domain jargon        │ Can struggle            │ Handles better         │
│ Debuggability        │ Hard                    │ Easy                   │
│ Infrastructure need  │ Vector DB required      │ File storage only      │
│ Maturity             │ Battle-tested           │ Newer, less proven     │
└──────────────────────┴────────────────────────┴────────────────────────┘
```

---

_This topic is covered as a conceptual deep-dive. The code examples for Vectorless RAG / PageIndex will be added as hands-on files in a future update._
