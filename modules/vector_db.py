# modules/vector_db.py
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
import faiss
import numpy as np
import re
import os
import json

class Chunk:
    def __init__(self, page_content):
        self.page_content = page_content

def create_directory(directory_path):
    """Create a directory if it doesn't already exist."""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Created directory: {directory_path}")
    else:
        print(f"Directory already exists: {directory_path}")

def save_chunks_to_file(chunks, file_path):
    """Save chunks to a JSON file."""
    # Save each chunk as a dictionary with 'page_content'
    chunk_data = [{"page_content": chunk.page_content} for chunk in chunks]
    with open(file_path, 'w') as f:
        json.dump(chunk_data, f)
    print(f"Chunks saved to {file_path}")

def load_chunks_from_file(file_path):
    """Load chunks from a JSON file."""
    with open(file_path, 'r') as f:
        chunk_contents = json.load(f)
    # Re-create Chunk objects from the loaded data
    chunks = [Chunk(page_content=chunk['page_content']) for chunk in chunk_contents]
    print(f"Chunks loaded from {file_path}")
    return chunks

def save_faiss_index(index, file_path):
    """Save the FAISS index to a file."""
    faiss.write_index(index, file_path)
    print(f"FAISS index saved to {file_path}")

def load_faiss_index(file_path):
    """Load the FAISS index from a file."""
    index = faiss.read_index(file_path)
    print(f"FAISS index loaded from {file_path}")
    return index

def create_or_load_vector_db(pdf_name, data, base_dir="pdf_storage"):
    """
    Creates or loads chunks and FAISS index for a given PDF.
    Args:
        pdf_name: The name of the source PDF file (without extension).
        data: The document content.
        base_dir: Base directory to store PDF-related data.
    Returns:
        index: FAISS index.
        chunks: Loaded or newly created chunks.
    """
    # Create a directory for the PDF
    pdf_dir = os.path.join(base_dir, pdf_name)
    create_directory(pdf_dir)

    # Define file paths for chunks and index
    chunks_file = os.path.join(pdf_dir, "chunks.json")
    index_file = os.path.join(pdf_dir, "index.faiss")

    # Check if the files already exist
    if os.path.exists(chunks_file) and os.path.exists(index_file):
        print(f"Loading existing embeddings and chunks for '{pdf_name}'...")
        chunks = load_chunks_from_file(chunks_file)
        index = load_faiss_index(index_file)
    else:
        print(f"Generating embeddings and chunks for '{pdf_name}'...")
        
        # Split the document into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
        chunks = text_splitter.split_documents(data)
        print(f"Number of Chunks: {len(chunks)}")

        # Clean chunks
        cleaned_chunks = []
        for chunk in chunks:
            cleaned_content = re.sub(r'\n{2,}', '\n', chunk.page_content).strip()
            if cleaned_content:
                chunk.page_content = cleaned_content
                cleaned_chunks.append(chunk)

        # Create embeddings
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        all_embeddings = []
        for chunk in cleaned_chunks:
            embedding = embeddings.embed_documents([chunk.page_content])
            all_embeddings.append(embedding)

        # Ensure all embeddings are of the same dimension
        first_dim = len(all_embeddings[0][0])
        embeddings_matrix = np.vstack(all_embeddings)

        # Create FAISS index
        index = faiss.IndexFlatL2(first_dim)
        index.add(embeddings_matrix)

        # Save chunks and FAISS index
        save_chunks_to_file(cleaned_chunks, chunks_file)
        save_faiss_index(index, index_file)

    return index, chunks
