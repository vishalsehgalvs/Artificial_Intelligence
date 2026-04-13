# rag_application.py — Full RAG pipeline running in the terminal
# ---------------------------------------------------------------
# This is the terminal/CLI version of the RAG app.
# Run create_database.py FIRST to build the vector database.
# Then run this file — it loads the saved database and lets you
# ask questions about the document in a back-and-forth loop.
#
# What happens each time you type a question:
#   1. Your question gets turned into a vector (embedding)
#   2. The retriever searches the database for the 4 most relevant chunks
#   3. Those chunks + your question are combined into a prompt
#   4. The LLM reads the prompt and gives an answer grounded in the document
#
# Type 0 to exit.

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# --- Load the saved vector database from disk ---
# This is the database that create_database.py built.
# We pass the same embedding model so the vectors stay comparable.
embedding_model = OpenAIEmbeddings()

vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_model
)

# --- Set up the retriever ---
# MMR (Max Marginal Relevance) picks results that are both relevant AND varied
# — so you don't get 4 chunks all saying the exact same thing.
# fetch_k=10: consider 10 candidates then pick the best 4 diverse ones
# lambda_mult=0.5: halfway between pure relevance (1.0) and pure diversity (0.0)
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "fetch_k": 10,
        "lambda_mult": 0.5
    }
)

# --- Set up the LLM ---
llm = ChatMistralAI(model="mistral-small-2506")

# --- Build the prompt template ---
# {context} will be filled with the retrieved document chunks
# {question} will be filled with whatever the user asked
# The system instruction tells the AI to only use the provided context
# so it doesn't make things up from its training data
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

print("Rag system created ")
print("press 0 to exit ")

# --- Main question-answer loop ---
while True:
    query = input("You : ")
    if query == "0":
        break

    # Find the most relevant chunks from the database for this question
    docs = retriever.invoke(query)

    # Combine all retrieved chunks into one big block of context text
    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    # Fill in the prompt template with the context and the question
    final_prompt = prompt.invoke({
        "context": context,
        "question": query
    })

    # Send the filled prompt to the LLM and print its answer
    response = llm.invoke(final_prompt)

    print(f"\n AI: {response.content}")
