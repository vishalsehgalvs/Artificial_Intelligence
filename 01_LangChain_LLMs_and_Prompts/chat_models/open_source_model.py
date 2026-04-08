# huggingface.py — Using a HuggingFace model via the cloud API
# -------------------------------------------------------------
# HuggingFace hosts thousands of open-source models.
# Instead of downloading them, you can call them via an API endpoint
# just like you would call OpenAI or Gemini.
#
# You'll need a FREE HuggingFace account and API token:
#   → sign up at https://huggingface.co
#   → go to Settings → Access Tokens → New Token (read access is enough)
#   → add HUGGINGFACEHUB_API_TOKEN=your_token to your .env file

from dotenv import load_dotenv
load_dotenv()

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# HuggingFaceEndpoint = runs the model on HuggingFace's servers (cloud)
# repo_id = the model's name on HuggingFace (like its address)
llm = HuggingFaceEndpoint(
    repo_id="microsoft/Phi-3-mini-4k-instruct",  # a small but capable free model
    task="text-generation",
    max_new_tokens=256,
)

# Wrap it in ChatHuggingFace to get the standard LangChain chat interface
model = ChatHuggingFace(llm=llm)

response = model.invoke("Explain what a neural network is in 3 simple sentences.")

print(response.content)


# ----------------------------------------------------------------
# Other popular free models on HuggingFace you can try:
#   "deepseek-ai/DeepSeek-R1"         → reasoning model
#   "Qwen/Qwen2.5-7B-Instruct"        → strong multilingual model
#   "google/gemma-2-2b-it"            → Google's small open model
#   "mistralai/Mistral-7B-Instruct-v0.3"  → Mistral's open version
# ----------------------------------------------------------------
