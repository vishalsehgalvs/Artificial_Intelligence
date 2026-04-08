# Chat Models — Notes

## What's in this folder

| File                        | What it does                                                                    |
| --------------------------- | ------------------------------------------------------------------------------- |
| `hello_llm.py`              | The simplest possible demo — one question, one answer                           |
| `personality_chatbot.py`    | A chatbot with memory and personality modes, runs in the terminal               |
| `personality_chatbot_ui.py` | The same chatbot but with a proper browser interface using Streamlit            |
| `open_source_model.py`      | Using open-source models from HuggingFace instead of paid APIs                  |
| `run_model_locally.py`      | Running a model 100% on your own machine using Ollama — no internet, no API key |

---

## What is a Chat Model?

A chat model is an LLM that's designed to have a **back-and-forth conversation**.
Instead of just completing a piece of text, it understands roles —
who is talking, what was said before, and who said it.

You send it a **list of messages** (the conversation so far), and it sends back the next message.

```
You send:
  [
    SystemMessage("You are a helpful tutor"),
    HumanMessage("What is machine learning?")
  ]

Model replies:
  AIMessage("Machine learning is when computers learn from examples instead of being told the rules...")
```

---

## The Three Message Types

Every conversation is built from just three types of messages:

| Type            | Who Sends It    | Purpose                                          |
| --------------- | --------------- | ------------------------------------------------ |
| `SystemMessage` | You (developer) | Set personality, rules, tone — invisible to user |
| `HumanMessage`  | The user        | What the user typed                              |
| `AIMessage`     | The model       | What the model replied                           |

The **chat history** is just a Python list containing all of these in order.
You append to it as the conversation grows, and send the whole list with every request.

```python
history = [
    SystemMessage("You are a helpful tutor"),
    HumanMessage("What is deep learning?"),
    AIMessage("Deep learning is a type of machine learning that uses neural networks..."),
    HumanMessage("Give me an example"),      # new message
]
response = model.invoke(history)             # model sees everything
```

---

## Temperature — Controlling Creativity

`temperature` is a number between 0 and 1 (sometimes up to 2) that controls
how creative or predictable the model's responses are.

```
temperature = 0.0  →  Very focused. Same question almost always gives same answer.
                       Good for: facts, code, structured output

temperature = 0.7  →  Balanced. Good default for most tasks.

temperature = 1.0  →  Creative and varied. Different answers each time.
                       Good for: stories, brainstorming, poetry, jokes
```

---

## Providers Compared

| Provider       | Model Used             | Free?          | Internet Needed? | Privacy               |
| -------------- | ---------------------- | -------------- | ---------------- | --------------------- |
| Google Gemini  | `gemini-1.5-flash`     | ✅ Free tier   | Yes              | Sent to Google        |
| Mistral        | `mistral-small-latest` | ✅ Free tier   | Yes              | Sent to Mistral       |
| OpenAI         | `gpt-4o`               | ❌ Paid only   | Yes              | Sent to OpenAI        |
| HuggingFace    | Various open models    | ✅ Free tier   | Yes              | Sent to HuggingFace   |
| Ollama (local) | LLaMA 3, Mistral, etc. | ✅ Always free | No               | Stays on your machine |

---

## How to Switch Providers in LangChain

This is LangChain's superpower — one line change, everything else stays the same:

```python
# Google Gemini (free)
from langchain_google_genai import ChatGoogleGenerativeAI
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# Mistral (free tier)
from langchain_mistralai import ChatMistralAI
model = ChatMistralAI(model="mistral-small-latest")

# Ollama local (free, offline)
from langchain_ollama import ChatOllama
model = ChatOllama(model="llama3")

# All three use the same interface:
response = model.invoke("Your question here")
print(response.content)
```

---

## .env File — Keeping API Keys Safe

**Never hardcode API keys in your code.** Use a `.env` file instead.

Create a file called `.env` in the project root:

```
GOOGLE_API_KEY=your_google_key_here
MISTRAL_API_KEY=your_mistral_key_here
OPENAI_API_KEY=your_openai_key_here
```

Then in your Python file:

```python
from dotenv import load_dotenv
load_dotenv()   # reads the .env file and loads all keys as environment variables
```

LangChain automatically picks up the keys from the environment.
The `.env` file is in `.gitignore` so it's never pushed to GitHub.

---

## Streamlit — Making a UI in 10 Minutes

Streamlit turns a Python script into a web app with almost no extra code.

```python
import streamlit as st

st.title("My AI App")
user_input = st.chat_input("Ask something...")

if user_input:
    response = model.invoke(user_input)
    st.write(response.content)
```

Run it: `streamlit run UIchatbot.py`
Opens automatically at: `http://localhost:8501`

The key trick for keeping memory between Streamlit reruns is `st.session_state` —
it persists data within a user's browser session, so appending messages there
means they survive page rerenders.
