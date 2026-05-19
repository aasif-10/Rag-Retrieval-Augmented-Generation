from langchain_community.document_loaders import WebBaseLoader

url = "https://en.wikipedia.org/wiki/Sample"

data = WebBaseLoader(url)

content = data.load()

print(content[0].page_content)