# load_and_split_text.py — Load a plain text file and summarise it with an LLM
# ------------------------------------------------------------------------------
# This loads a .txt file and sends its content to an LLM to get a summary back.
# Uses a custom-hosted model accessed through a company's internal API endpoint
# — same LangChain interface, different server behind it.

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import TextLoader

load_dotenv()

from langchain_core.prompts import ChatPromptTemplate

# Load the text file — gives us one Document object with all the text inside
data = TextLoader("notes.txt")
docs = data.load()

# Build a reusable prompt template
# {data} will be swapped out for the actual text from the file at runtime
template = ChatPromptTemplate.from_messages(
    [("system", "you are an AI that summarises test"),
     ("human", "{data}")]
)

# Custom-hosted model — behaves like OpenAI but runs on a different server
model = ChatOpenAI(
    model="VIO:Gemini 2.5 Pro",
    openai_api_base="https://vio.automotive-wan.com:446",
    default_headers={"useLegacyCompletionsEndpoint": "false", "X-Tenant-ID": "default_tenant"}
)

# Fill in the template with the actual text from the first (and only) document
prompt = template.format_messages(data=docs[0].page_content)

# Send the filled-in prompt to the model and print the summary
result = model.invoke(prompt)
print(result.content)