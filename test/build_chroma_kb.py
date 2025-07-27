from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter

from utils.config import get_llm, get_embedding2, get_embedding

def embed_to_db(file_path: str):
    loader = PyPDFLoader(file_path)

    docs = loader.load()
    print(len(docs))
    print(docs[0].page_content[0:100])
    print(docs[0].metadata)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    # text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=10)

    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(persist_directory="..\db\chroma_db", documents=splits, embedding=get_embedding())
    vectorstore.persist()

    print("Knowledge base indexed and saved to 'chroma_db'.")

def retrieve_from_db(query:str):
    vectorstore = Chroma(persist_directory="..\db\chroma_db", embedding_function=get_embedding())
    results = vectorstore.similarity_search(query=query, k=3)

    print("The retrieved content is:")
    index = 0
    for doc in results:
        index += 1
        print(f"------------------------Result-{index}------------------------")
        print(f"* {doc.page_content} [{doc.metadata}]")

if __name__ == "__main__":
    file_path = "../data/newVHISmedical-en.pdf"
    # embed_to_db(file_path)

    query = "What is the features for plan AVPU?"
    # query = "AVPU 保险产品的特点是什么?"
    retrieve_from_db(query)
    print("Done")