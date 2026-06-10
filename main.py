import os
import sys

# Force Python to recognize the root directory
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from dotenv import load_dotenv
from src.parser import load_and_smart_chunk_pdf
from src.embedder import create_vector_store

def run_ingestion():
    load_dotenv()
    
    pdf_path = "./data/Constitution.pdf"
    persist_dir = "./vector_store"
    
    if not os.path.exists(pdf_path):
        print(f"Error: Could not find the PDF at {pdf_path}")
        return
        
    print(f"Loading and processing PDF: {pdf_path}")
    docs = load_and_smart_chunk_pdf(pdf_path)
    print(f"Total chunks successfully extracted: {len(docs)}")
    
    print("Initializing embedding model and building Vector Database...")
    create_vector_store(docs, persist_dir)
    print(f"Vector Database finalized and saved to: {persist_dir}")

if __name__ == "__main__":
    run_ingestion()