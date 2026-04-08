# Part 1 — LangChain, LLMs & Prompt Engineering

This is where everything starts. Before you can build anything with AI, you need to understand what these models actually are, how to talk to them, and how to get useful answers out of them. That's exactly what this part covers — from the very basics all the way to building your first working AI-powered tool.

No fluff, no PhD required. Just plain explanations and working code.

---

## What is Generative AI and how it works

You've probably used Google to search for answers, or Excel to sort numbers. Those tools process information and give you a result — but they don't _create_ anything new.

Generative AI is different. It can actually **create new things** — write an essay, generate code, answer a question you've never asked before, summarise a document, translate text. It produces brand new output every single time.

Here's the simplest way to think about how it works:

These AI models were trained by reading an absolutely enormous amount of text — we're talking billions of web pages, books, articles, code, conversations. While reading all of that, the model's job was to learn one simple thing: _"given everything that came before, what word is most likely to come next?"_

It practised this billions of times. And by doing that, it didn't just learn words — it learned grammar, facts, reasoning, how to write code, how to explain concepts, how conversations work. All of that got baked into the model.

So when you type a question, the model isn't "thinking" the way you do. It's doing a very sophisticated version of autocomplete — predicting the best next word, then the next, then the next, until it has a complete, useful answer.

> **The magic isn't that it "understands" language the way humans do. The magic is that predicting words well enough, at scale, turns out to produce something that looks and feels a lot like understanding.**

---

## Understanding Large Language Models (LLMs)

LLM stands for **Large Language Model**. The name says exactly what it is:

- **Large** — trained on a huge amount of data (billions of pages of text) and has billions of internal settings (called parameters) that store everything it learned
- **Language** — it works with human language — English, Hindi, Spanish, Python code, SQL, all of it
- **Model** — it's a mathematical system (a deep neural network) that learned patterns from data

Popular LLMs you've probably heard of:

| Model         | Made By    | Notes                                        |
| ------------- | ---------- | -------------------------------------------- |
| GPT-4, GPT-4o | OpenAI     | The most well-known, very capable, paid      |
| Claude        | Anthropic  | Great for long documents and reasoning       |
| Gemini        | Google     | Free tier available, what we mostly use here |
| LLaMA         | Meta       | Open source, can run on your own computer    |
| Mistral       | Mistral AI | Lightweight, very fast, free tier available  |

The key thing to know: **all of these work the same way from your perspective.** You send them text, they send text back. The differences are in quality, speed, cost, and what they're good at.

---

## How to use APIs to access AI models

You don't need a supercomputer to use these models. The companies that built them run them on massive servers, and they let you access those servers over the internet through something called an **API**.

Think of an API like a restaurant. The kitchen (the model) is doing all the real work. You just place an order (send your question), and the waiter (the API) brings back your food (the answer). You never go into the kitchen yourself.

Here's how you actually use one:

**Step 1 — Get an API key**

An API key is basically your password. You sign up on the provider's website and they give you a key — a long string of letters and numbers. Every request you make includes this key so they know it's you.

For Google Gemini (which is free): go to [aistudio.google.com](https://aistudio.google.com), sign in, and get your key.

**Step 2 — Store your key safely**

Never paste your API key directly into your code. Instead, store it in a `.env` file:

```
GOOGLE_API_KEY=your_actual_key_here
```

Then load it in Python:

```python
from dotenv import load_dotenv
load_dotenv()  # reads the .env file and makes the key available
```

**Step 3 — Send a message, get a reply**

```python
from langchain_google_genai import ChatGoogleGenerativeAI

model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
response = model.invoke("What is machine learning?")
print(response.content)
```

That's it. Three lines and you're talking to an AI model.

---

## Writing effective prompts

A **prompt** is whatever you send to the model. It could be a question, an instruction, some text to summarise — anything. The model has no idea what you want unless you tell it clearly.

Here's the honest truth: **the quality of your output depends almost entirely on the quality of your prompt.** A vague prompt gets a vague answer. A clear, specific prompt gets exactly what you need.

### What makes a prompt bad

```
"Tell me about Python"
```

Too vague. Python the snake? Python the programming language? A beginner overview or advanced internals? The model will guess — and probably guess wrong for your specific need.

### What makes a prompt good

```
"I'm a Python beginner who knows basic variables and loops.
Explain what list comprehensions are and give me 2 simple examples."
```

You told it: who you are, what you want, what level to pitch it at, and how many examples you want. The answer will be much better.

### Use the system message to set the context

In most AI apps, you split your prompt into two parts:

- **System message** — sets the model's role, tone, and rules (the model behaves like this for the whole conversation)
- **Human message** — the actual question from the user

```python
from langchain_core.messages import SystemMessage, HumanMessage

messages = [
    SystemMessage("You are a patient coding tutor who explains things using everyday analogies."),
    HumanMessage("What is a variable?")
]
response = model.invoke(messages)
```

The system message is like briefing someone before they take a call — you tell them who they're supposed to be.

---

## Structured Input vs Structured Output

### Structured Input

Normally, you just send free text to the model. But sometimes you want to send data in a specific, consistent format — like always including a name, a topic, and a difficulty level in every request.

That's structured input. It makes your prompts predictable and easier to reuse.

### Structured Output

By default, the model replies in plain text. But what if you want the response to come back as proper data — like a Python dictionary or a JSON object — so your code can actually use it?

That's structured output. You tell the model: "Don't just answer in sentences. Fill in this specific form."

**Example — you want this back:**

```json
{
  "name": "Gradient Descent",
  "difficulty": "intermediate",
  "one_line_explanation": "An algorithm that adjusts a model's settings to reduce its mistakes"
}
```

Instead of a paragraph of text, you get clean data your code can read directly. This is incredibly useful in real applications where the AI output feeds into something else — a database, a UI, an email, a report.

---

## Generating clean structured data using schemas

A **schema** is just a blueprint. It says: "the output must have these exact fields, with these types."

In Python, we use **Pydantic** to define schemas. Pydantic checks that the output actually matches what you asked for — it won't let a number sneak in where you expected a string, or let a field be missing.

```python
from pydantic import BaseModel

class JobDetails(BaseModel):
    job_title: str
    company: str
    required_skills: list[str]
    salary: str
```

You pass this schema to the model and LangChain uses it to force the response into that exact shape:

```python
structured_model = model.with_structured_output(JobDetails)
result = structured_model.invoke("Analyse this job posting: [paste job text here]")

print(result.job_title)      # "Senior Data Scientist"
print(result.required_skills) # ["Python", "SQL", "TensorFlow"]
```

No parsing, no string splitting, no guesswork. You get back a Python object with proper fields you can use directly.

This is what you'll use in real-world AI apps: the AI reads messy human text and hands you back clean, structured data.

---

## Prompt Templates and dynamic prompts

Imagine you need to send the same type of prompt 500 times — but with a different name, topic, or language each time. Writing each one by hand would be ridiculous.

A **Prompt Template** is a fill-in-the-blank version of a prompt. You write the template once, then slot in different values each time you use it.

```python
from langchain_core.prompts import ChatPromptTemplate

template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful tutor who explains things simply."),
    ("human", "Explain {topic} as if I'm {level}.")
])

# Use it once:
prompt = template.invoke({"topic": "neural networks", "level": "a complete beginner"})

# Use it again with different values:
prompt = template.invoke({"topic": "recursion", "level": "someone who knows basic Python"})
```

The `{topic}` and `{level}` parts get swapped out automatically each time.

Why does this matter? Because in a real app, those values come from the user, a database, or another part of your code — not from you manually typing them. Templates are what make prompts reusable and programmable.

---

## How LangChain connects everything together

Here's the problem without LangChain: OpenAI, Anthropic, Google, and Mistral all have their own Python libraries, their own way of sending messages, their own way of reading responses. If you build your app using OpenAI's library and then want to switch to Gemini, you'd have to rewrite a chunk of your code.

**LangChain solves this by sitting in the middle.** It gives you one consistent way to talk to any model. You write your code once. If you want to switch from GPT-4 to Gemini, you change one line — the model name — and everything else stays exactly the same.

```python
# Using Gemini
from langchain_google_genai import ChatGoogleGenerativeAI
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# Switch to Mistral — change just this part, nothing else in your code changes
from langchain_mistralai import ChatMistralAI
model = ChatMistralAI(model="mistral-small-latest")
```

Beyond just model switching, LangChain also gives you ready-made tools for:

- **Prompts** — templates, system messages, structured output
- **Chains** — linking steps together (format prompt → call model → parse output)
- **Memory** — keeping track of the conversation history
- **Document loading** — reading PDFs, text files, websites
- **Vector stores** — storing and searching documents by meaning
- **Agents** — giving the model the ability to use tools and take actions

Instead of building all of that from scratch, you use LangChain's components and focus on what your app actually does.

---

## How real-world AI applications are structured

Most AI apps — whether it's a chatbot, a document analyser, a data extractor, or a coding assistant — follow the same basic pattern:

```
User Input
    ↓
Format the Prompt  (apply a template, add context, set the role)
    ↓
Call the LLM  (send to model via LangChain)
    ↓
Process the Output  (parse response, validate schema, extract fields)
    ↓
Return the Result  (show to user, save to database, trigger next step)
```

As apps get more complex, two things get added to this flow:

**Memory** — so the model remembers what was said earlier in the conversation, not just the last message.

**Retrieval** — so the model can read from your own data (PDFs, databases, websites) before answering, rather than relying only on what it learned during training. This is called RAG, and it's what Part 2 covers.

The files in this folder are working examples of each concept — start with `chat_models/hello_llm.py` for the simplest possible demo and work your way up.
