#!/usr/bin/env python3
"""
Standalone script to process PDF files from /knowledgebase directory,
create a vector database using ChromaDB, and retrieve top 3 chunks for user queries.
"""

import os
import logging
import chromadb
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from uuid import uuid4

# Try to import PDF processing libraries
PDF_AVAILABLE = False
USE_PDFPLUMBER = False

try:
    import PyPDF2
    PDF_AVAILABLE = True
    USE_PDFPLUMBER = False
except ImportError:
    try:
        import pdfplumber
        PDF_AVAILABLE = True
        USE_PDFPLUMBER = True
    except ImportError:
        PDF_AVAILABLE = False
        USE_PDFPLUMBER = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables for embedding model and vector DB
_embedding_model = None
_vector_db = None
_chroma_db_dir = "chroma_db"
_vector_db_collection_name = "pdf_docs"


def _get_embedding_model():
    """Get or create the embedding model instance."""
    global _embedding_model
    if _embedding_model is None:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True}
        )
        logger.info(f"Initialized embedding model on device: {device}")
    return _embedding_model


def _get_chroma_db_path():
    """Get the ChromaDB persistence directory path."""
    global _chroma_db_dir
    os.makedirs(_chroma_db_dir, exist_ok=True)
    return _chroma_db_dir


def _extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from a PDF file.
    Tries PyPDF2 first, falls back to pdfplumber if available.
    """
    if not PDF_AVAILABLE:
        raise ImportError(
            "No PDF library available. Please install one: pip install PyPDF2 or pip install pdfplumber"
        )
    
    text = ""
    
    try:
        if USE_PDFPLUMBER:
            # Use pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        else:
            # Use PyPDF2
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        
        logger.info(f"Extracted {len(text)} characters from {os.path.basename(pdf_path)}")
        return text.strip()
    
    except Exception as e:
        logger.error(f"Error extracting text from {pdf_path}: {e}")
        raise


def _chunk_text(text: str, chunk_size: int = 2000, chunk_overlap: int = 500) -> list:
    """
    Split text into chunks using RecursiveCharacterTextSplitter.
    This ensures proper chunking for each PDF file.
    """
    if not text or not text.strip():
        return []
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = text_splitter.split_text(text)
    logger.info(f"Split text into {len(chunks)} chunks")
    return chunks


def _load_pdfs_from_directory(docs_dir: str = "/knowledgebase") -> dict:
    """
    Load all PDF files from the specified directory.
    Returns a dictionary mapping filename to extracted text.
    """
    if not os.path.exists(docs_dir):
        logger.warning(f"Directory {docs_dir} does not exist. Creating it...")
        os.makedirs(docs_dir, exist_ok=True)
        return {}
    
    if not os.path.isdir(docs_dir):
        logger.error(f"{docs_dir} is not a directory")
        return {}
    
    pdf_contents = {}
    pdf_files = [f for f in os.listdir(docs_dir) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        logger.warning(f"No PDF files found in {docs_dir}")
        return pdf_contents
    
    logger.info(f"Found {len(pdf_files)} PDF file(s) in {docs_dir}")
    
    for filename in pdf_files:
        file_path = os.path.join(docs_dir, filename)
        try:
            logger.info(f"Processing PDF: {filename}")
            text = _extract_text_from_pdf(file_path)
            if text and text.strip():
                pdf_contents[filename] = text
                logger.info(f"Successfully processed {filename} ({len(text)} characters)")
            else:
                logger.warning(f"PDF {filename} contains no extractable text")
        except Exception as e:
            logger.error(f"Failed to process {filename}: {e}")
            continue
    
    return pdf_contents


def _get_or_create_vector_store(collection_name: str = "pdf_docs", pdf_contents: dict = None):
    """
    Get existing ChromaDB vector store or create it if it doesn't exist.
    If pdf_contents is provided and the collection is empty, it will populate it.
    Returns the vector store instance.
    """
    global _vector_db
    
    # Return cached vector DB if it exists
    if _vector_db is not None:
        try:
            chroma_db_dir = _get_chroma_db_path()
            client_db = chromadb.PersistentClient(chroma_db_dir)
            try:
                collection = client_db.get_collection(name=collection_name)
                collection_data = collection.get()
                if collection_data['ids']:
                    logger.info(f"Using cached vector DB with {len(collection_data['ids'])} chunks")
                    return _vector_db
            except Exception:
                logger.info("Cached vector DB collection not found, will recreate")
                _vector_db = None
        except Exception as e:
            logger.warning(f"Error verifying cached vector DB: {e}, will recreate")
            _vector_db = None
    
    try:
        # Get embedding model and ChromaDB path
        embedding_model = _get_embedding_model()
        chroma_db_dir = _get_chroma_db_path()
        
        # Initialize ChromaDB client and collection
        client_db = chromadb.PersistentClient(chroma_db_dir)
        collection = client_db.get_or_create_collection(name=collection_name)
        
        # Check if collection already has data
        collection_data = collection.get()
        if collection_data['ids']:
            logger.info(f"ChromaDB collection '{collection_name}' already exists with {len(collection_data['ids'])} chunks. Reusing existing vector store.")
            _vector_db = Chroma(
                collection_name=collection_name,
                persist_directory=chroma_db_dir,
                embedding_function=embedding_model
            )
            return _vector_db
        
        # Collection is empty, need to populate it
        if not pdf_contents or len(pdf_contents) == 0:
            logger.warning("Vector store collection is empty but no PDF content provided. Cannot initialize vector store.")
            return None
        
        logger.info(f"Vector store collection '{collection_name}' is empty. Initializing from PDF files...")
        
        # Process each PDF file separately and chunk it
        all_chunks = []
        chunk_metadata = []  # Store metadata about which file each chunk came from
        
        for filename, text in pdf_contents.items():
            logger.info(f"Chunking PDF: {filename}")
            # Chunk each PDF file separately
            chunks = _chunk_text(text, chunk_size=2000, chunk_overlap=500)
            
            # Add filename metadata to each chunk
            for chunk in chunks:
                all_chunks.append(chunk)
                chunk_metadata.append({"source_file": filename})
            
            logger.info(f"Created {len(chunks)} chunks from {filename}")
        
        if not all_chunks:
            logger.warning("No chunks created from PDF files")
            return None
        
        logger.info(f"Total chunks created: {len(all_chunks)}")
        
        # Generate embeddings and add to collection
        logger.info(f"Generating embeddings and populating ChromaDB collection '{collection_name}' with {len(all_chunks)} chunks...")
        embeddings = embedding_model.embed_documents(all_chunks)
        
        # Add chunks with metadata
        collection.add(
            ids=[str(uuid4()) for _ in all_chunks],
            documents=all_chunks,
            embeddings=embeddings,
            metadatas=chunk_metadata
        )
        logger.info(f"Successfully added {len(all_chunks)} chunks to ChromaDB collection '{collection_name}'")
        
        # Create LangChain Chroma vector store
        _vector_db = Chroma(
            collection_name=collection_name,
            persist_directory=chroma_db_dir,
            embedding_function=embedding_model
        )
        
        return _vector_db
    
    except Exception as e:
        logger.exception(f"Error getting or creating vector store: {e}")
        return None


def _get_top_similar_chunks(vector_db, query: str, k: int = 3):
    """
    Retrieve top k similar chunks from the vector store based on the query.
    Returns a list of tuples: (chunk_text, score, metadata)
    """
    if vector_db is None:
        logger.warning("Vector store is not initialized, returning empty chunks")
        return []
    
    try:
        # Perform similarity search with scores
        results_with_scores = vector_db.similarity_search_with_score(query, k=k)
        
        if not results_with_scores:
            logger.warning("No results from similarity search")
            return []
        
        # Extract chunks with metadata
        chunks = []
        for doc, score in results_with_scores:
            chunk_data = {
                "text": doc.page_content,
                "score": score,
                "metadata": doc.metadata if hasattr(doc, 'metadata') else {}
            }
            chunks.append(chunk_data)
        
        logger.info(f"Retrieved {len(chunks)} similar chunks for query: '{query[:100]}...'")
        return chunks
    
    except Exception as e:
        logger.exception(f"Error retrieving similar chunks: {e}")
        return []


def initialize_vector_db(knowledgebase_dir: str = "/knowledgebase"):
    """
    Initialize the vector database by loading PDFs from the knowledgebase directory.
    """
    logger.info(f"Initializing vector database from {knowledgebase_dir}")
    
    # Load PDFs from directory
    pdf_contents = _load_pdfs_from_directory(docs_dir=knowledgebase_dir)
    
    if not pdf_contents:
        logger.error(f"No PDF files found or processed in {knowledgebase_dir}")
        return False
    
    # Create or get vector store
    vector_db = _get_or_create_vector_store(
        collection_name=_vector_db_collection_name,
        pdf_contents=pdf_contents
    )
    
    if vector_db is None:
        logger.error("Failed to initialize vector store")
        return False
    
    logger.info("Vector database initialized successfully")
    return True


def search_chunks(query: str, top_k: int = 3):
    """
    Search for top k similar chunks based on the query.
    Returns list of chunk dictionaries with text, score, and metadata.
    """
    global _vector_db
    
    # Ensure vector DB is initialized
    if _vector_db is None:
        chroma_db_dir = _get_chroma_db_path()
        embedding_model = _get_embedding_model()
        try:
            _vector_db = Chroma(
                collection_name=_vector_db_collection_name,
                persist_directory=chroma_db_dir,
                embedding_function=embedding_model
            )
            # Verify it has data
            client_db = chromadb.PersistentClient(chroma_db_dir)
            collection = client_db.get_collection(name=_vector_db_collection_name)
            collection_data = collection.get()
            if not collection_data['ids']:
                logger.error("Vector database is empty. Please initialize it first.")
                return []
        except Exception as e:
            logger.error(f"Failed to load vector database: {e}")
            return []
    
    # Get top k chunks
    chunks = _get_top_similar_chunks(_vector_db, query, k=top_k)
    return chunks


def print_chunks(chunks: list):
    """
    Print the retrieved chunks in a formatted way.
    """
    if not chunks:
        print("\n" + "="*80)
        print("No chunks found for the query.")
        print("="*80 + "\n")
        return
    
    print("\n" + "="*80)
    print(f"TOP {len(chunks)} SIMILAR CHUNKS")
    print("="*80)
    
    for i, chunk_data in enumerate(chunks, 1):
        chunk_text = chunk_data.get("text", "")
        score = chunk_data.get("score", 0)
        metadata = chunk_data.get("metadata", {})
        source_file = metadata.get("source_file", "Unknown")
        
        print(f"\n{'='*80}")
        print(f"CHUNK {i} (Similarity Score: {score:.4f})")
        print(f"Source File: {source_file}")
        print(f"Length: {len(chunk_text)} characters")
        print(f"{'='*80}")
        print(chunk_text)
        print(f"{'='*80}\n")
    
    print("="*80)
    print("END OF CHUNKS")
    print("="*80 + "\n")


def main():
    """
    Main function to run the interactive PDF vector search.
    """
    print("="*80)
    print("PDF Vector Search - ChromaDB")
    print("="*80)
    
    # Check if PDF library is available
    if not PDF_AVAILABLE:
        print("\nERROR: No PDF processing library found!")
        print("Please install one of the following:")
        print("  pip install PyPDF2")
        print("  OR")
        print("  pip install pdfplumber")
        return
    
    # Get knowledgebase directory (default: /knowledgebase or ./knowledgebase)
    knowledgebase_dir = os.environ.get("KNOWLEDGEBASE_DIR", None)
    if knowledgebase_dir is None:
        # Try absolute path first, then relative
        if os.path.exists("/knowledgebase"):
            knowledgebase_dir = "/knowledgebase"
        elif os.path.exists("./knowledgebase"):
            knowledgebase_dir = "./knowledgebase"
        elif os.path.exists("knowledgebase"):
            knowledgebase_dir = "knowledgebase"
        else:
            knowledgebase_dir = "/knowledgebase"  # Default, will create if needed
    
    # Check if vector DB exists, if not initialize it
    chroma_db_dir = _get_chroma_db_path()
    client_db = chromadb.PersistentClient(chroma_db_dir)
    try:
        collection = client_db.get_collection(name=_vector_db_collection_name)
        collection_data = collection.get()
        if not collection_data['ids']:
            print(f"\nVector database is empty. Initializing from {knowledgebase_dir}...")
            if not initialize_vector_db(knowledgebase_dir):
                print("Failed to initialize vector database. Exiting.")
                return
        else:
            print(f"\nVector database found with {len(collection_data['ids'])} chunks.")
            print("To reinitialize, delete the 'chroma_db' directory and run again.")
    except Exception:
        print(f"\nVector database not found. Initializing from {knowledgebase_dir}...")
        if not initialize_vector_db(knowledgebase_dir):
            print("Failed to initialize vector database. Exiting.")
            return
    
    # Interactive loop
    print("\n" + "="*80)
    print("Ready to search! Enter your questions (type 'quit' or 'exit' to stop)")
    print("="*80 + "\n")
    
    while True:
        try:
            query = input("Enter your question: ").strip()
            
            if not query:
                continue
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break
            
            print(f"\nSearching for: '{query}'...")
            chunks = search_chunks(query, top_k=3)
            print_chunks(chunks)
        
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            logger.exception(f"Error during search: {e}")
            print(f"\nError: {e}")


if __name__ == "__main__":
    main()

