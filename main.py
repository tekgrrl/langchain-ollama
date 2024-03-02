from langchain_community.llms import Ollama
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

import bs4


loader = WebBaseLoader("https://www.gutenberg.org/files/1727/1727-h/1727-h.htm")
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)

oembed = OllamaEmbeddings(base_url='http://localhost:11434', model='nomic-embed-text')
vector_store = Chroma.from_documents(documents=all_splits, embedding=oembed)

question="Who is Neleus and who is in Neleus' family?"
# vector_store.similarity_search(question)


ollama = Ollama(base_url='http://localhost:11434', model='llama2:latest')

qachain = RetrievalQA.from_chain_type(ollama, retriever=vector_store.as_retriever())
print(qachain.invoke({"query": question}))


# ollama = Ollama(base_url='http://localhost:11434', model='llama2:latest')
# print(ollama("Why is the sky blue?"))   