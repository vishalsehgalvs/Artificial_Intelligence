# arixv_retrievers.py — Pull real research papers from arXiv as a knowledge source
# ----------------------------------------------------------------------------------
# arXiv (arxiv.org) is a free archive of millions of academic research papers
# on topics like AI, physics, mathematics, and more.
#
# LangChain has a built-in ArxivRetriever that fetches papers directly from arXiv
# — no PDF download needed, no vector database needed.
# You just give it a search term and it returns the actual papers as Documents.
#
# This is great for quickly querying cutting-edge research without any setup.

from langchain_community.retrievers import ArxivRetriever

# Create the retriever
# load_max_docs=2: only fetch the top 2 papers (keep it fast for demo)
# load_all_available_meta=True: also grab metadata like title and authors
retriever = ArxivRetriever(
    load_max_docs=2,
    load_all_available_meta=True
)

# Search arXiv — this goes live to arxiv.org and fetches real papers
docs = retriever.invoke("large language models")

# Print the results — each doc is a real research paper
for i, doc in enumerate(docs):
    print(f"\nResult {i+1}")
    print("Title:", doc.metadata.get("Title"))
    print("Authors:", doc.metadata.get("Authors"))
    print("Summary:", doc.page_content)  # the abstract / summary of the paper
