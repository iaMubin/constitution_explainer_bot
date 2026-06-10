import re
import pdfplumber
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_and_smart_chunk_pdf(file_path: str) -> list[Document]:
    full_text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"
                
    # Clean noise
    full_text = re.sub(r'--- PAGE \d+ ---', '', full_text)
    full_text = re.sub(r'\d{2}/\d{2}/\d{4}', '', full_text)
    full_text = re.sub(r"The Constitution of the People's Republic of Bangladesh", '', full_text)
    
    # Loose semantic split
    raw_chunks = re.split(r'\n(?=\d+[A-Z]?\.\s)', full_text)
    
    # 2026 Defensive Safeguard: Hard limit chunk sizes to fit Groq's low TPM
    safe_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " "]
    )
    
    documents = []
    next_header = ""
    
    for i, chunk in enumerate(raw_chunks):
        chunk = chunk.strip()
        if not chunk:
            continue
            
        current_header = next_header
        
        match = list(re.finditer(r'[.;\]]', chunk))
        if match:
            last_punct_idx = match[-1].end()
            potential_header = chunk[last_punct_idx:].strip()
            
            if len(potential_header) < 200:
                next_header = potential_header.replace('\n', ' ')
                next_header = re.sub(r'\s+', ' ', next_header).strip()
                chunk = chunk[:last_punct_idx]
            else:
                next_header = ""
        else:
            next_header = ""
        
        chunk = chunk.replace('\n', ' ')
        chunk = re.sub(r'\s+', ' ', chunk).strip()
        
        if current_header:
            final_text = f"{current_header}\n{chunk}"
        else:
            final_text = chunk
            
        if len(final_text) > 30: 
            # If a chunk is too big, safely break it down into smaller sub-chunks
            sub_chunks = safe_splitter.split_text(final_text)
            for sub_chunk in sub_chunks:
                documents.append(Document(
                    page_content=sub_chunk, 
                    metadata={"source": file_path, "chunk_id": len(documents)}
                ))
            
    return documents