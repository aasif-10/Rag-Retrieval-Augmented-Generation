from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv
load_dotenv()

model = init_chat_model("mistral-small-2506")

embedding_model = MistralAIEmbeddings(model="mistral-embed")

vectorstore = Chroma(  # if db already exists
    embedding_function=embedding_model,
    persist_directory="chroma_db"
)

retriever = vectorstore.as_retriever(
    search_type = "mmr",
    search_kwargs ={
        "k": 4,
        "fetch_k": 10,
        "lamda_mult":0.5 # diversity 0 - 1
    }
)

template = ChatPromptTemplate.from_messages(
    [
        ("system","""You are an AI Assistant.Use only the provided context to answer the question.
        If the answer is not present in the context say : " I could not find the answer"""),
        ("human","Context : {content} \n\n Question : {question}")
    ]
)

print("Rag system created")

query = input("You : ")

docs = retriever.invoke(query)

context = "\n\n".join(
    [doc.page_content for  doc in docs]
)

prompt = template.invoke({
    "context":context,
    "question":query
})

response = model.invoke(prompt)

print("AI: ",response.content)