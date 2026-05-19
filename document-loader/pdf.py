from langchain_community.document_loaders import PyPDFLoader

docs = PyPDFLoader("document-loader/GRU.pdf")

data = docs.load()

print(data[0].page_content)