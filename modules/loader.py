# modules/loader.py
from langchain_community.document_loaders import UnstructuredPDFLoader
import gc

def load_pdf(file_path):
    try:
        # Initialize the PDF loader
        loader = UnstructuredPDFLoader(file_path=file_path)
        
        # Load the PDF, which should return a list of documents/chunks
        data = loader.load()
        
        # Ensure data is a list and log the number of chunks loaded
        if isinstance(data, list):
            print(f"Successfully loaded {len(data)} chunks from the PDF.")
        else:
            print("Loaded data is not a list, please check the loader output.")
            return []

        return data
    except Exception as e:
        print(f"Error loading PDF: {e}")
        return None
    finally:
        # Optional: Collect garbage to free up memory
        gc.collect()
