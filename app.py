import streamlit as st
from dotenv import load_dotenv
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from src.embedder import get_advanced_retriever
from src.llm_handler import get_rag_chain

load_dotenv()

st.set_page_config(page_title="BD Constitution AI", page_icon="⚖️", layout="centered")

@st.cache_resource
def init_pipeline():
    persist_dir = "./vector_store"
    retriever = get_advanced_retriever(persist_dir)
    chain_base = get_rag_chain(retriever)
    return chain_base

@st.cache_resource
def get_memory_store():
    return {}

try:
    rag_chain_base = init_pipeline()
    store = get_memory_store()
except Exception as e:
    st.error(f"System Error: {e}")
    st.stop()

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

rag_chain = RunnableWithMessageHistory(
    rag_chain_base,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history"
)

st.title("⚖️ Constitution AI Assistant")
st.caption("Advanced RAG Agent - Ask anything about the Constitution of Bangladesh")

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.session_id = "client_session_01"

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_query = st.chat_input("Enter your query here (e.g., 'Who appoints the Chief Justice?')...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)
        
    with st.chat_message("assistant"):
        with st.spinner("Analyzing legal context..."):
            config = {"configurable": {"session_id": st.session_state.session_id}}
            response = rag_chain.invoke({"question": user_query}, config=config)
            st.markdown(response)
            
    st.session_state.messages.append({"role": "assistant", "content": response})