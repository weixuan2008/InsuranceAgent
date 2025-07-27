from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate
from utils.config import get_llm, get_embedding2, get_embedding


class CustomerSupportAgent:
    def __init__(self):
        self.llm = get_llm()
        
        #Loading the Chroma vector store
        self.vectorstore = Chroma(persist_directory="..\db\chroma_db", embedding_function=get_embedding())
        
        self.prompt_template = ChatPromptTemplate.from_template(
            """
            You are a customer support assistant. Use the following context to answer the user's question:
            Context: {context}
            Question: {question}
            Answer:
            """
        )
    
    def handle_query(self, query):
        print("Calling Agent: CustomerSupportAgent (with RAG)")  
        docs = self.vectorstore.similarity_search(query, k=3)  
        context = "\n".join([doc.page_content for doc in docs])
        
        if not context.strip():
            return "I'm sorry, I couldn't find the information you're looking for. Please contact customer support for further assistance."
        
        prompt = self.prompt_template.format(context=context, question=query)
        
        response = self.llm.invoke(prompt)
        return response.content