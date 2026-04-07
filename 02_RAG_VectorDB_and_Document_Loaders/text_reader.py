from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import TextLoader

load_dotenv()
from langchain_core.prompts import ChatPromptTemplate

data = TextLoader("notes.txt")
docs = data.load()
template = ChatPromptTemplate.from_messages(
    [("system", "you are an AI that summarises test"),
     ("human", "{data}")]
)

model = ChatOpenAI(
    model="VIO:Gemini 2.5 Pro",
    openai_api_base="https://vio.automotive-wan.com:446",
    default_headers={"useLegacyCompletionsEndpoint": "false", "X-Tenant-ID": "default_tenant"}
)
prompt = template.format_messages(data= docs[0].page_content)
result = model.invoke(prompt)
print(result.content)