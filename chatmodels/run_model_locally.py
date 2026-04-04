# localmodel.py — Run a model 100% on your own machine using Ollama
# ------------------------------------------------------------------
# No API key. No internet. No cost. No data sent anywhere.
# Everything runs locally on your laptop.
#
# Setup (one time only):
#   1. Download Ollama → https://ollama.com
#   2. Open a terminal and run: ollama pull llama3 (downloads ~4.7GB)
#   3. Ollama starts a local server on http://localhost:11434
#   4. Run this file — it talks to that local server
#
# Requirements: pip install langchain-ollama

from langchain_ollama import ChatOllama

# This talks to your locally running Ollama server
# Make sure Ollama is running before executing this script
model = ChatOllama(
    model="llama3",         # must be pulled first: ollama pull llama3
    temperature=0.7         # creativity level: 0 = focused, 1 = creative
)

response = model.invoke("What is the difference between AI and machine learning?")

print(response.content)


# ---------------------------------------------------------------
# Other models you can run locally with Ollama:
#   ollama pull mistral        → ~4.1GB, fast and capable
#   ollama pull phi3           → ~2.3GB, very small but surprisingly good
#   ollama pull gemma2         → ~5.4GB, Google's open model
#   ollama pull codellama      → great at writing and explaining code
#
# List all your downloaded models: ollama list
# Start a chat in the terminal:    ollama run llama3
# ---------------------------------------------------------------
