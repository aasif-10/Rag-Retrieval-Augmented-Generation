from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

data = PyPDFLoader("document-loader/GRU.pdf")
docs = data.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
chunks = text_splitter.split_documents(docs);

embedding_model = MistralAIEmbeddings(model="mistral-embed")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="chroma_db"
)

