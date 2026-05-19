from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

data = PyPDFLoader("document-loader/GRU.pdf")

docs = data.load()

text_splitter = CharacterTextSplitter(separator="",chunk_size=100, chunk_overlap=0)

chunks = text_splitter.split_documents(docs)

print(len(chunks))