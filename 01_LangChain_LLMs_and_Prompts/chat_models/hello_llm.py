# chat.py — Your first "Hello World" with an LLM
# -----------------------------------------------
# This is the simplest possible way to call an LLM using LangChain.
# You create a model, send it a message, and print the reply.
# No memory, no history — just a single one-shot call.

from dotenv import load_dotenv
load_dotenv()  # loads your API key from the .env file

# --- Using Google Gemini (FREE tier available) ---
from langchain_google_genai import ChatGoogleGenerativeAI

model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",   # free tier model
    temperature=0.7             # 0 = very precise/consistent, 1 = more creative/random
)

response = model.invoke("Write a short poem about artificial intelligence.")

print(response.content)


# ---------------------------------------------------------------
# WANT TO SWITCH PROVIDERS? Just swap the 3 lines above.
# Everything else (model.invoke, response.content) stays the same.
# ---------------------------------------------------------------

# --- Using Mistral (free tier via Mistral API) ---
# from langchain_mistralai import ChatMistralAI
# model = ChatMistralAI(model="mistral-small-latest", temperature=0.7)

# --- Using OpenAI GPT-4o (paid) ---
# from langchain_openai import ChatOpenAI
# model = ChatOpenAI(model="gpt-4o", temperature=0.7)

# --- Using Ollama locally (completely free, no internet) ---
# from langchain_ollama import ChatOllama
# model = ChatOllama(model="llama3")   # run: ollama pull llama3 first
