# create_database.py — One-time setup: read a PDF and save it as a vector database
# ----------------------------------------------------------------------------------
# Run this script ONCE before you run rag_application.py.
# It reads a PDF, cuts it into chunks, converts each chunk into a vector (embedding),
# and saves everything into a Chroma database folder called "chroma_db".
#
# Once this is done, rag_application.py can load that saved database and
# answer questions from it — without having to re-process the PDF every time.

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

# Step 1 — Load the PDF and extract text, one Document object per page
data = PyPDFLoader("document loaders/deeplearning.pdf")
docs = data.load()

# Step 2 — Split all the pages into overlapping chunks
# chunk_size=1000: each chunk is roughly 1000 characters
# chunk_overlap=200: 200 characters overlap between neighbouring chunks
#                    so a sentence or idea doesn't get cut off right at the edge
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(docs)

# Step 3 — Create the embedding model that will convert text into vectors
embedding_model = OpenAIEmbeddings()

# Step 4 — Build the Chroma vector database and save it to disk
# Every chunk gets embedded and stored inside the "chroma_db" folder
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="chroma_db"
)
