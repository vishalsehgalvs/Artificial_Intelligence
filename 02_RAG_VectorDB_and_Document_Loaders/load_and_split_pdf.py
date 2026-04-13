# load_and_split_pdf.py — Load a PDF and break it into smaller chunks
# ---------------------------------------------------------------------
# PDFs are too big to send to an LLM all at once — there's a limit on how much
# text a model can read in a single go.
# So the trick is: load the PDF, then cut it into small overlapping pieces
# called "chunks". Later, only the relevant chunks get sent to the model.

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load the PDF — we get back a list with one Document object per page
data = PyPDFLoader("document loaders/GRU.pdf")
docs = data.load()

# Split all the pages into smaller chunks
# chunk_size=1000: aim for roughly 1000 characters per chunk
# chunk_overlap=10: share 10 characters between neighbouring chunks
#                   so a sentence doesn't get cut off right at the edge
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=10
)

chunks = splitter.split_documents(docs)

# Print the text from the very first chunk just to check what we got
print(chunks[0].page_content)