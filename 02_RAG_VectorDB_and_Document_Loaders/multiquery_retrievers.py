# multiquery_retrievers.py — Search with multiple question phrasings at once
# --------------------------------------------------------------------------
# The problem with regular retrieval: if you phrase your question slightly wrong,
# you might miss relevant chunks that would have matched a different wording.
#
# MultiQueryRetriever fixes this by:
#   1. Taking your original question
#   2. Using an LLM to rewrite it in several different ways
#   3. Running a separate search for each version
#   4. Combining all the results (removing duplicates)
#
# The result is a richer, more complete set of relevant documents
# — catching things that one phrasing alone would have missed.

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv

load_dotenv()

# Our small knowledge base
docs = [
    Document(page_content="Gradient descent is an optimization algorithm used in machine learning."),
    Document(page_content="Gradient descent minimizes the loss function."),
    Document(page_content="Gradient descent is an optimization that minimizes the loss function."),
    Document(page_content="Neural networks use gradient descent for training."),
    Document(page_content="Support Vector Machines are supervised learning algorithms.")
]

# Local embeddings — no API key needed
embeddings = HuggingFaceEmbeddings()

# Build the vector store
vectorstore = Chroma.from_documents(docs, embeddings)

# Base retriever — a standard similarity search retriever
retriever = vectorstore.as_retriever()

# The LLM that will rephrase the question into multiple versions
llm = ChatMistralAI(model="mistral-small-latest")

# Wrap the base retriever with MultiQueryRetriever
# When you call .invoke(), it automatically generates multiple query variants,
# searches with each one, and returns the combined unique results
multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=retriever,
    llm=llm
)

query = "What is gradient descent?"

# This single call actually fires multiple searches behind the scenes
docs = multi_query_retriever.invoke(query)

print("\nRetrieved Documents:\n")

for doc in docs:
    print(doc.page_content)
