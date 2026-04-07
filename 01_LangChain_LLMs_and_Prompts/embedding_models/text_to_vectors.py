# embeddings.py — Converting text into numbers (vectors)
# -------------------------------------------------------
# Embeddings turn words/sentences into lists of numbers that capture meaning.
# Two sentences that mean similar things will produce vectors that are "close"
# to each other in mathematical space — even if they use different words.
#
# This is the backbone of:
#   - Semantic search (find documents by meaning, not just keyword match)
#   - RAG (Retrieval-Augmented Generation)
#   - Recommendation systems
#   - Document clustering

from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Create the embedding model
# This model converts text → fixed-size list of numbers (vectors)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# --- Example 1: Embed a single query ---
query = "What is machine learning?"
vector = embeddings.embed_query(query)

print(f"Query: '{query}'")
print(f"Vector length: {len(vector)} numbers")
print(f"First 5 values: {vector[:5]}")
print()

# --- Example 2: Embed multiple documents at once ---
documents = [
    "Machine learning is a branch of AI where computers learn from data.",
    "Deep learning uses neural networks with many layers.",
    "Paris is the capital city of France.",
    "The Eiffel Tower is located in Paris.",
    "Supervised learning requires labeled training examples.",
]

doc_vectors = embeddings.embed_documents(documents)

print(f"Embedded {len(doc_vectors)} documents")
print(f"Each vector has {len(doc_vectors[0])} dimensions")
print()

# --- Example 3: Similarity search by hand ---
# Cosine similarity: how similar are two vectors?
# 1.0 = identical meaning, 0.0 = completely unrelated, -1.0 = opposite
import numpy as np

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

query_vec = embeddings.embed_query("What is deep learning?")

print("Similarity scores (query: 'What is deep learning?'):")
for doc, doc_vec in zip(documents, doc_vectors):
    score = cosine_similarity(query_vec, doc_vec)
    print(f"  {score:.3f} | {doc}")


# ---------------------------------------------------------------
# ALTERNATIVE PROVIDERS (comment/uncomment as needed):

# --- OpenAI embeddings (paid, very popular) ---
# from langchain_openai import OpenAIEmbeddings
# embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# --- HuggingFace local embeddings (free, runs on your machine) ---
# from langchain_huggingface import HuggingFaceEmbeddings
# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# No API key needed — model downloads automatically (~90MB)
# ---------------------------------------------------------------
