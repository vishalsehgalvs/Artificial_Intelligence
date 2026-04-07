# similarity_search.py — Finding the most relevant documents using embeddings
# ----------------------------------------------------------------------------
# This is the heart of how RAG (Retrieval-Augmented Generation) works.
# We embed a set of documents, store them in a vector database (FAISS),
# and then search that database using a question.
#
# The search doesn't look for exact word matches —
# it finds documents that are most SIMILAR IN MEANING to the question.
#
# Requirements: pip install faiss-cpu langchain-community

from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# --- Our "knowledge base" ---
# In a real app these would come from PDFs, databases, websites, etc.
documents = [
    Document(page_content="Python is a popular programming language used in AI and data science."),
    Document(page_content="LangChain is a framework for building applications with large language models."),
    Document(page_content="Neural networks are inspired by the structure of the human brain."),
    Document(page_content="RAG stands for Retrieval-Augmented Generation — LLMs that look up facts before answering."),
    Document(page_content="The Transformer architecture was introduced in the paper 'Attention Is All You Need'."),
    Document(page_content="Embeddings convert text into numerical vectors that capture semantic meaning."),
    Document(page_content="FAISS is a library by Facebook for fast similarity search across vectors."),
    Document(page_content="GPT stands for Generative Pre-trained Transformer."),
    Document(page_content="Overfitting happens when a model memorises training data but fails on new data."),
    Document(page_content="Streamlit is a Python library for building data apps and AI demos quickly."),
]

# Create embedding model
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Build a FAISS vector store from the documents
# Under the hood: each document gets embedded → stored in the vector DB
print("Building vector store...")
vector_store = FAISS.from_documents(documents, embeddings)
print(f"Stored {len(documents)} documents in the vector DB\n")

# --- Perform similarity searches ---
queries = [
    "What is LangChain?",
    "How do neural networks work?",
    "What is RAG?",
]

for query in queries:
    print(f"🔍 Query: '{query}'")
    results = vector_store.similarity_search(query, k=2)  # top 2 most relevant
    for i, doc in enumerate(results, 1):
        print(f"   Result {i}: {doc.page_content}")
    print()

# --- Save and reload the vector store (so you don't re-embed every time) ---
vector_store.save_local("faiss_index")
print("✅ Vector store saved to 'faiss_index' folder")

# To reload later:
# loaded_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
