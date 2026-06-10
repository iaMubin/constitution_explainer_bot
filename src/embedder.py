import warnings
from sentence_transformers import CrossEncoder
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_core.documents import Document

warnings.filterwarnings("ignore")

def create_vector_store(docs, persist_dir):
    embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
    Chroma.from_documents(
        documents=docs,
        embedding=embedding_model,
        persist_directory=persist_dir
    )

def get_advanced_retriever(persist_dir):
    embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
    vectorstore = Chroma(
        persist_directory=persist_dir, 
        embedding_function=embedding_model
    )
    
    db_data = vectorstore.get()
    all_docs = [
        Document(page_content=txt, metadata=meta)
        for txt, meta in zip(db_data["documents"], db_data["metadatas"])
    ]
    
    if not all_docs:
        raise ValueError("Vector store is empty. Please run main.py first.")

    # Initialize individual retrievers
    bm25_retriever = BM25Retriever.from_documents(all_docs)
    bm25_retriever.k = 10
    
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    
    reranker_model = CrossEncoder("BAAI/bge-reranker-base")
    
    def combine_deduplicate_and_rerank(inputs: dict) -> list:
        query = inputs["query"]
        bm25_docs = inputs["bm25_docs"]
        vector_docs = inputs["vector_docs"]
        
        # Deduplicate based on exact page content
        unique_docs_map = {}
        for doc in bm25_docs + vector_docs:
            if doc.page_content not in unique_docs_map:
                unique_docs_map[doc.page_content] = doc
                
        documents = list(unique_docs_map.values())
        
        if not documents:
            return []
            
        # Score with High-Performance Cross-Encoder
        pairs = [[query, doc.page_content] for doc in documents]
        scores = reranker_model.predict(pairs)
        
        # Inject scores and sort
        for doc, score in zip(documents, scores):
            doc.metadata["rerank_score"] = float(score)
            
        sorted_docs = sorted(documents, key=lambda x: x.metadata["rerank_score"], reverse=True)
        
        # Return top 5 highly relevant chunks
        return sorted_docs[:5]

    # Advanced LCEL Parallel Architecture
    advanced_retriever_chain = (
        RunnableParallel({
            "bm25_docs": bm25_retriever,
            "vector_docs": vector_retriever,
            "query": RunnableLambda(lambda x: x)
        })
        | RunnableLambda(combine_deduplicate_and_rerank)
    )
    
    return advanced_retriever_chain