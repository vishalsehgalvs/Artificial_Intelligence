# split_text_into_chunks.py — Experiment with splitting text into chunks
# -----------------------------------------------------------------------
# This is a simple experiment — no LLM, no embeddings.
# The whole point is just to see what the text looks like after it's been
# chopped into smaller pieces.
#
# CharacterTextSplitter cuts text purely by character count.
# Unlike the smarter RecursiveCharacterTextSplitter, it doesn't try to
# split on sentences or paragraphs — it just counts characters.

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

# Set up the splitter
# separator="": no special character to split on — just go by length
# chunk_size=1000: aim for ~1000 characters per chunk
# chunk_overlap=1: tiny 1-character overlap so nothing is completely lost at the cut
splitter = CharacterTextSplitter(
    separator="",
    chunk_size=1000,
    chunk_overlap=1
)

# Load the text file
data = TextLoader("document loaders/notes.txt")
docs = data.load()

# Split into chunks
chunks = splitter.split_documents(docs)

# Print each chunk with visible spacing so you can clearly see
# where one chunk ends and the next one begins
for i in chunks:
    print(i.page_content)
    print()
    print()
    print()
