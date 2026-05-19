from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from dotenv import load_dotenv

load_dotenv()

model = init_chat_model("mistral-small-2506")

data = PyPDFLoader("document-loader/GRU.pdf")
docs = data.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
chunks = text_splitter.split_documents(docs)

embedding_model = MistralAIEmbeddings(model="mistral-embed")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="chroma_db"
)

retriever = vectorstore.as_retriever(
    search_type = "mmr",
    search_kwargs = {"k":3}
)

multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=retriever,
    llm=model
)

query = "What is the title"

docs = multi_query_retriever.invoke(query)

similarity_docs = retriever.invoke("What is the title")

for doc in similarity_docs:
    print(doc.page_content+"\n",)
    print("----------------------------End------------------------------")