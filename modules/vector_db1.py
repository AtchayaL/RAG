from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
import faiss
import numpy as np
import gc
import pandas as pd  # Import pandas for CSV handling

# Define a custom class for documents with metadata
class Document:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata if metadata is not None else {}  # Default to an empty dictionary if no metadata

def create_vector_db(data):
    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=50)
    chunks = text_splitter.split_documents(data)
    
    # Filter out chunks that are completely blank or contain only whitespace
    cleaned_chunks = [Document(chunk.page_content.replace('\n\n', '').strip()) for chunk in chunks if chunk.page_content.strip()]
    
    print(f"Total number of cleaned chunks: {len(cleaned_chunks)}")
    
    # Initialize embedding model
    embeddings = OllamaEmbeddings(model="nomic-embed-text", show_progress=True)

    # Store embeddings and chunk contents
    all_embeddings = []
    chunk_contents = []  # List to store chunk contents

    for i, chunk in enumerate(cleaned_chunks):
        try:
            # Compute embeddings for each chunk
            embedding = embeddings.embed_documents([chunk.page_content])  # Ensure we're passing the content
            all_embeddings.append(embedding[0])  # Store the first element since embed_documents returns a list of lists
            chunk_contents.append(chunk.page_content)  # Store the chunk content
            print(f"Processed chunk {i + 1}, embedding dimension: {len(embedding[0])}\n")
        except Exception as e:
            print(f"Error processing chunk {i + 1}: {e}\n")

    # Ensure all embeddings have the same dimensionality
    first_embedding_dim = len(all_embeddings[0])
    for i, embedding in enumerate(all_embeddings):
        if len(embedding) != first_embedding_dim:
            print(f"Dimension mismatch at chunk {i + 1}. Expected {first_embedding_dim}, got {len(embedding)}")
            return None

    # Create FAISS index
    d = first_embedding_dim  # Use the dimension of the first embedding
    index = faiss.IndexFlatL2(d)  # Using L2 distance (Euclidean)

    # Convert embeddings list to numpy array
    embeddings_matrix = np.vstack(all_embeddings)
    
    # Add embeddings to FAISS index
    index.add(embeddings_matrix)
    print("All embeddings added to FAISS index.")

    # Save chunk contents to CSV without embeddings
    df = pd.DataFrame({
        'chunk_content': chunk_contents,
    })
    df.to_csv('chunks_only.csv', index=False)  # Save to a different CSV
    print("Chunk contents saved to 'chunks_only.csv'.")

    return index, cleaned_chunks  # Return cleaned chunks


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

# Example usage
if __name__ == "__main__":
    # Sample data (replace with your actual data)
    documents = [
        Document("This is the first chun\n\n\n\n\n\k. It contains some initial information."),
        Document("This is the second chunk.\n It ha\n\n\n\ns more details about the topic."),
        Document("This is \n\nn\n\n\n\n\the third chunk.n\n\n\n It covers additional points of interest."),
        Document("This is the fourt\n\n\n\n\h chunk. It summarizes the pn\n\n\nrevious chunks.")
    ]

    # Create the vector database
    index, chunks = create_vector_db(documents)

    # Example query embedding (this should be replaced with an actual query embedding)
    # Here we create a random vector for demonstration; you should compute a real embedding for your query.
    query_embedding = np.random.rand(768).astype(np.float32)  # Random vector
    k = 2  # Number of nearest neighbors to return

    # Search in FAISS
    distances, indices = search_faiss(index, query_embedding, k)
    print("Distances:", distances)
    print("Indices:", indices)

    # Save the FAISS index to a file
    save_faiss_index(index, "faiss_index.bin")

    # Load the FAISS index from the file
    loaded_index = load_faiss_index("faiss_index.bin")

    # Optionally delete the FAISS index
    delete_vector_db(loaded_index)
