# modules/vector_db.py

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
import faiss
import numpy as np
import gc

def create_vector_db(data):
    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=50)
    chunks = text_splitter.split_documents(data)
    print(f"Total number of chunks: {len(chunks)}")
    
    # Initialize embedding model
    embeddings = OllamaEmbeddings(model="nomic-embed-text", show_progress=True)

    # Store embeddings
    all_embeddings = []
    for i, chunk in enumerate(chunks):
        # Print the content of each chunk
        # print(f"Chunk {i + 1} content: {chunk.page_content}")  # Assuming chunk has page_content attribute
        try:
            # Compute embeddings for each chunk
            embedding = embeddings.embed_documents([chunk])
            all_embeddings.append(embedding)
            print(f"Processed chunk {i + 1}, embedding dimension: {len(embedding[0])}\n")
        except Exception as e:
            print(f"Error processing chunk {i + 1}: {e}\n")

    # Ensure all embeddings have the same dimensionality
    first_embedding_dim = len(all_embeddings[0][0])
    for i, embedding in enumerate(all_embeddings):
        if len(embedding[0]) != first_embedding_dim:
            print(f"Dimension mismatch at chunk {i + 1}. Expected {first_embedding_dim}, got {len(embedding[0])}")
            return None

    # Create FAISS index
    d = first_embedding_dim  # Use the dimension of the first embedding
    index = faiss.IndexFlatL2(d)  # Using L2 distance (Euclidean)

    # Convert embeddings list to numpy array
    embeddings_matrix = np.vstack(all_embeddings)
    
    # Add embeddings to FAISS index
    index.add(embeddings_matrix)
    print("All embeddings added to FAISS index.")

    return index, chunks

def search_faiss(index, query_embedding, k):
    """Search in FAISS for top-k results"""
    distances, indices = index.search(np.array([query_embedding]), k)
    return distances, indices

def delete_vector_db(index):
    """Delete or reset the FAISS index to free memory"""
    del index  # This deletes the FAISS index from memory
    gc.collect()  # Trigger garbage collection to free memory
    print("FAISS index deleted.")

def save_faiss_index(index, file_path):
    """Save the FAISS index to a file."""
    faiss.write_index(index, file_path)
    print(f"FAISS index saved to {file_path}")

def load_faiss_index(file_path):
    """Load the FAISS index from a file."""
    index = faiss.read_index(file_path)
    print(f"FAISS index loaded from {file_path}")
    return index
