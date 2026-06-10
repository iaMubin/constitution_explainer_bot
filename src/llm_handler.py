import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def get_rag_chain(retriever):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is missing! Check your .env file.")

    # Upgraded to the active 2026 model with higher TPM limits
    llm = ChatGroq(
        model="llama-3.1-8b-instant", 
        temperature=0.25,
        groq_api_key=api_key
    )
    
    system_prompt = (
        "You are a helpful, friendly, and smart AI Legal Assistant specializing in the Constitution of Bangladesh.\n"
        "Your goal is to help regular people understand the constitution easily. Keep your answers concise, clear, and to the point.\n"
        "Apply common sense to understand acronyms (like 'MP' for Member of Parliament), typos, and layman terms.\n"
        "Use the retrieved context to answer. If the exact phrasing isn't there, but the concept is, explain it simply based on the context.\n"
        "Only say 'I cannot answer this' if the topic is completely absent from the context.\n"
        "Always cite the exact Article numbers at the end.\n\n"
        "Context:\n{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}")
    ])
    
    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])
        
    rag_chain_base = (
        RunnablePassthrough.assign(
            context=(lambda x: x["question"]) | retriever | format_docs,
            history=lambda x: x.get("history", [])[-6:]
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain_base