from langchain_community.document_loaders import TextLoader

docs = TextLoader("document-loader/notes.txt")

data = docs.load()

print(data[0].page_content)