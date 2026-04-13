# vector_store_db.py — Store documents in a vector database and search by meaning
# ---------------------------------------------------------------------------------
# This demo shows the two main ways to search a Chroma vector database:
#
#   1. similarity_search — ask a question, get back the most relevant documents
#   2. as_retriever      — same idea, but wrapped as a proper LangChain retriever
#                          (the format used in full RAG pipelines)
#
# The search doesn't look for exact word matches — it finds documents that
# are most SIMILAR IN MEANING to your question, even if they use different words.

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

from langchain_core.documents import Document

# Our tiny pretend knowledge base — in a real app these would come from a PDF or website
# metadata keeps track of where each document came from
docs = [
    Document(page_content="Python is widely used in Artificial Intelligence.", metadata={"source": "AI_book"}),
    Document(page_content="Pandas is used for data analysis in Python.", metadata={"source": "DataScience_book"}),
    Document(page_content="Neural networks are used in deep learning.", metadata={"source": "DL_book"}),
]

# Create the embedding model — this turns text into vectors (lists of numbers)
embedding_model = OpenAIEmbeddings()

# Build the Chroma vector database from our documents and save it to disk
# persist_directory means it's saved as a folder — no need to rebuild next time
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embedding_model,
    persist_directory="chroma-db"
)

# --- Approach 1: Direct similarity search ---
# Ask a plain question, get back the top 2 most similar documents
result = vectorstore.similarity_search("what is used for data analysis?", k=2)

for r in result:
    print(r.page_content)
    print(r.metadata)  # shows which source document this came from

# --- Approach 2: Using a Retriever ---
# A retriever is just a neater wrapper around similarity_search.
# It's the format that LangChain RAG chains and pipelines expect.
retriver = vectorstore.as_retriever()

docs = retriver.invoke("Explain deep learning")

for d in docs:
    print(d.page_content)