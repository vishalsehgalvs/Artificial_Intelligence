# mmr_retrievers.py — Compare regular similarity search vs MMR retrieval
# -----------------------------------------------------------------------
# This is a side-by-side demo to show the difference between two retrieval strategies:
#
#   Similarity Search — returns the top 3 chunks most similar to your question.
#                        Problem: if your docs are repetitive, you might get
#                        3 chunks all saying basically the same thing.
#
#   MMR (Max Marginal Relevance) — picks chunks that are relevant AND different
#                        from each other. So instead of 3 near-identical results,
#                        you get a more varied, well-rounded picture of the topic.
#
# Both retrievers use Chroma + HuggingFace embeddings (no API key needed).

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Our mini knowledge base — notice that 3 out of 5 docs say nearly the same thing
# about gradient descent. This is on purpose to show the difference between the two strategies.
docs = [
    Document(page_content="Gradient descent is an optimization algorithm used in machine learning."),
    Document(page_content="Gradient descent minimizes the loss function."),
    Document(page_content="Gradient descent is an optimization that minimizes the loss function."),
    Document(page_content="Neural networks use gradient descent for training."),
    Document(page_content="Support Vector Machines are supervised learning algorithms.")
]

# Local embeddings — downloads a small model, no API key needed
embeddings = HuggingFaceEmbeddings()

# Build the vector store
vectorstore = Chroma.from_documents(docs, embeddings)

# --- Strategy 1: Regular similarity search ---
# Just finds the 3 most similar. Watch how they overlap a lot.
similarity_retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

print("\n===== Similarity Search Results =====")
print("(likely to return repetitive/overlapping results)\n")

similarity_docs = similarity_retriever.invoke("What is gradient descent?")

for doc in similarity_docs:
    print(doc.page_content)

# --- Strategy 2: MMR retrieval ---
# Finds relevant AND varied results — notice the last result is more different.
mmr_retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3}
)

print("\n===== MMR Results =====")
print("(picks relevant results but avoids repeating the same idea)\n")

mmr_docs = mmr_retriever.invoke("What is gradient descent?")

for doc in mmr_docs:
    print(doc.page_content)
