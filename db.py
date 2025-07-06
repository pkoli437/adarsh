__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import fitz  # PyMuPDF for PDF reading
import pandas as pd
from uuid import uuid4
import chromadb
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

def extract_text_from_pdf(file_path):
    """Extract text from a PDF file"""
    doc = fitz.open(file_path)
    full_text = []

    for page in doc:
        text = page.get_text()
        if text.strip():
            full_text.append(text.strip())

    return "\n".join(full_text)

def process_pdf_and_store(pdf_file_path, persist_directory='RSP', collection_name='test-1'):
    """Process a PDF file, chunk it semantically, and store in ChromaDB"""
    
    print(f"Processing document: {pdf_file_path}")
    
    # Step 1: Extract text from the PDF file
    full_text = extract_text_from_pdf(pdf_file_path)
    print(f"Extracted {len(full_text)} characters of text")
    
    # Step 2: Set up the embedding model
    hf = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
    
    # Step 3: Set up the semantic chunker
    text_splitter = SemanticChunker(
        hf,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=92.0,
        min_chunk_size=3
    )
    
    # Step 4: Split the full text into semantic chunks
    print("Splitting text into semantic chunks...")
    docs = text_splitter.create_documents([full_text])
    print(f"Created {len(docs)} semantic chunks")
    
    # Step 5: Generate embeddings for storing
    print("Generating embeddings...")
    texts = [doc.page_content for doc in docs]
    embeddings = hf.embed_documents(texts)
    
    # Step 6: Store in ChromaDB
    print(f"Storing embeddings in ChromaDB at {persist_directory}")
    client = chromadb.PersistentClient(persist_directory)
    collection = client.get_or_create_collection(collection_name)
    
    # Check if collection is empty before adding
    if not collection.get()['ids']:
        collection.add(
            ids=[str(uuid4()) for _ in docs],
            documents=texts,
            embeddings=embeddings
        )
        print(f"✅ Added {len(docs)} chunks to ChromaDB collection: {collection_name}")
    else:
        print(f"⚠️ Collection already has data. Use a different collection name or clear it first.")
    
    # Create Langchain Chroma instance for easier retrieval
    db = Chroma(collection_name=collection_name, persist_directory=persist_directory, embedding_function=hf)
    
    return db

def perform_search(db, user_query, similarity_threshold=0.5, k=3):
    """Perform similarity search and filter by relevance score"""
    
    print(f"\n🔍 Searching for: '{user_query}'")
    
    # Get results with relevance scores
    result = db.similarity_search_with_relevance_scores(user_query, k=k)
    
    for i, (doc, score) in enumerate(result):
        print(f"\n🔹 Retrieved Chunk {i+1} (Score: {score}):\n{doc.page_content}\n{'-'*80}")
    
    # Filter by threshold
    filtered_results = [(doc, score) for doc, score in result if score > similarity_threshold]
    
    if not filtered_results:
        print(f"❌ No results found with relevance score above {similarity_threshold}")
        return ""
    
    # Combine content from relevant chunks
    context = "\n\n".join(doc.page_content for doc, _ in filtered_results)
    print(f"\n✅ Relevant context ({len(filtered_results)} chunks):")
    
    return context

if __name__ == "__main__":
    # ✅ Example usage
    pdf_file = "/root/local_LLM/SALES_BOT/data/MO_Detailed_Summary.pdf"  # Change to your actual PDF file path
    
    # Process and store PDF document
    db = process_pdf_and_store(pdf_file)
    
    # Query example
    perform_search(db, "Why should I invest now?", similarity_threshold=0.05)



# pip install pymupdf langchain chromadb pandas 
# !pip install --quiet langchain_experimental



