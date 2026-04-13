# load_webpage.py — Load content from a live webpage
# ---------------------------------------------------
# Instead of reading a file stored on your computer, this fetches content
# directly from a URL on the internet.
#
# WebBaseLoader visits the page, pulls all the HTML, and strips the tags
# so you're left with just the plain readable text — same clean format
# as loading a PDF or text file.
#
# This is useful when you want to build a RAG system that answers questions
# about a website, blog post, or documentation page.

from langchain_community.document_loaders import WebBaseLoader

url = "https://www.apple.com/in/macbook-pro/"

# Fetch the webpage and extract just the visible text
data = WebBaseLoader(url)
docs = data.load()

# Print what was extracted — it'll be all the readable text from that page
print(docs[0].page_content)