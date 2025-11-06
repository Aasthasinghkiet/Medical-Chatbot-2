# ✅ Updated imports for LangChain v0.2+ with Gemini embeddings
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from typing import List
from langchain_core.documents import Document
import os


# Extract Data From the PDF File
def load_pdf_file(data):
    """
    Load all PDF files from a directory.
    
    Args:
        data (str): Path to directory containing PDF files
        
    Returns:
        list: List of Document objects
    """
    loader = DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    return documents


def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """
    Given a list of Document objects, return a new list of Document objects
    containing only 'source' in metadata and the original page_content.
    
    Args:
        docs (List[Document]): List of documents to filter
        
    Returns:
        List[Document]: Filtered documents with minimal metadata
    """
    minimal_docs: List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": src}
            )
        )
    return minimal_docs


# Split the Data into Text Chunks
def text_split(extracted_data):
    """
    Split documents into smaller chunks for better embedding and retrieval.
    
    Args:
        extracted_data (list): List of Document objects
        
    Returns:
        list: List of chunked Document objects
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, 
        chunk_overlap=20
    )
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks


# Download the Embeddings from Google Gemini
def download_hugging_face_embeddings():
    """
    Initialize Google Gemini embeddings using the API key.
    Uses 'models/embedding-001' which:
    - Returns 768 dimensions
    - Optimized for semantic search
    - Works great with medical/general text
    - No need to download models locally
    
    Returns:
        GoogleGenerativeAIEmbeddings: Initialized Gemini embeddings model
    """
    # Get API key from environment variable
    google_api_key = os.environ.get("GOOGLE_API_KEY")
    
    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment variables!")
    
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=google_api_key
    )
    
    return embeddings
