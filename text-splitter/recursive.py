from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

data = PyPDFLoader("document-loader/GRU.pdf")

docs = data.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=10)

chunks = text_splitter.split_documents(docs)

print(len(chunks))