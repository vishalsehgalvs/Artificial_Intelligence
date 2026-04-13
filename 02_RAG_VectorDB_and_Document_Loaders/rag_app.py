# rag_app.py — Full RAG pipeline with a browser interface (Streamlit)
# ---------------------------------------------------------------------
# This is the browser version of the RAG app.
# Upload any PDF, click a button to build the vector database,
# then ask questions about it and get answers grounded in the document.
#
# Run with: streamlit run rag_app.py
#
# How it works:
#   Part 1 (setup):  Upload PDF → split into chunks → embed → save to Chroma DB
#   Part 2 (QA):     Load saved DB → retrieve relevant chunks → send to LLM → display answer

import streamlit as st
from dotenv import load_dotenv
import tempfile
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# -------------------- Page Setup --------------------
st.set_page_config(page_title="RAG Book Assistant")

st.title("📚 RAG Book Assistant")
st.write("Upload a PDF and ask questions from the document")

# -------------------- File Upload --------------------
uploaded_file = st.file_uploader("Upload a PDF book", type="pdf")

if uploaded_file:

    # Streamlit uploads are in-memory — we need to save to a real temp file
    # so PyPDFLoader can open it from disk
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name

    st.success("PDF uploaded successfully!")

    # -------------------- Build the Vector Database --------------------
    if st.button("Create Vector Database"):

        with st.spinner("Processing document..."):

            # Step 1 — Load the PDF and extract text page by page
            loader = PyPDFLoader(file_path)
            docs = loader.load()

            # Step 2 — Split all pages into overlapping chunks
            # chunk_size=1000: roughly 1000 characters per chunk
            # chunk_overlap=200: shared overlap so ideas don't get cut at the edge
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            chunks = splitter.split_documents(docs)

            # Step 3 — Embed all chunks and store them in Chroma
            # persist_directory saves the database to disk so it survives page reruns
            embeddings = OpenAIEmbeddings()
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory="chroma_db"
            )
            vectorstore.persist()

        st.success("Vector database created!")

# -------------------- Q&A Section --------------------
# This section only appears once a chroma_db folder exists on disk.
# That folder is created by the "Create Vector Database" step above.
if os.path.exists("chroma_db"):

    # Load the saved vector database back into memory
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings
    )

    # MMR retriever picks the 4 most relevant AND varied chunks
    # fetch_k=10: look at 10 candidates, then keep the best 4 diverse ones
    # lambda_mult=0.5: balance between relevance (1.0) and diversity (0.0)
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 10,
            "lambda_mult": 0.5
        }
    )

    # Set up the LLM
    llm = ChatMistralAI(model="mistral-small-2506")

    # The prompt instructs the AI to ONLY use the retrieved context
    # and not make things up from its own training data
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document."
"""
            ),
            (
                "human",
                """Context:
{context}

Question:
{question}
"""
            )
        ]
    )

    st.divider()
    st.subheader("Ask Questions From the Book")

    query = st.text_input("Enter your question")

    if query:

        # Find the most relevant chunks from the database for this question
        docs = retriever.invoke(query)

        # Combine them all into one block of context text
        context = "\n\n".join(
            [doc.page_content for doc in docs]
        )

        # Fill in the prompt with context and question, then call the LLM
        final_prompt = prompt.invoke({
            "context": context,
            "question": query
        })

        response = llm.invoke(final_prompt)

        st.write("### AI Answer")
        st.write(response.content)