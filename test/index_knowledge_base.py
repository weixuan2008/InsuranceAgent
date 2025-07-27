from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader

from utils.config import get_embedding

loader = TextLoader("../data/faqs.txt", encoding = 'UTF-8')
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

vectorstore = Chroma.from_documents(texts, get_embedding(), persist_directory="../db/chroma_db")
# vectorstore.persist()

print("Knowledge base indexed and saved to 'chroma_db'.")