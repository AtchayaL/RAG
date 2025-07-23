# main.py

from modules.loader import load_pdf
from modules.vector_db import create_vector_db, delete_vector_db
from modules.retriever import retrieve_answers
import sys

def main():
    pdf_path = "data/WEF_The_Global_Cooperation_Barometer_2024.pdf"
    
    # Load PDF
    print("Starting PDF loading process...")
    data = load_pdf(pdf_path)
    if not data:
        print("Failed to load PDF")
        sys.exit(1)
    print("PDF loading process completed.")
    
    # Create Vector Database from document chunks
    print("Starting Vector Database creation process...")
    vector_db, chunks = create_vector_db(data)
    print("Vector Database creation process completed.")
    
    while True:
        # Ask questions and retrieve answers
        question = input("Enter your question (or type 'exit' to quit): ").strip()
        if question.lower() == "exit":
            print("Exiting the program.")
            break
        
        print(f"Asking the question: '{question}'")
        
        # Retrieve relevant chunks (context)
        relevant_chunks = retrieve_answers(vector_db, chunks, question)
        
        # Optionally print relevant context before the final response
        """
        print("Relevant context (chunks):")
        for chunk in relevant_chunks:
            print(chunk, end="")
        """
        
        response = retrieve_answers(vector_db, chunks, question)
        print(f"Answer to your question: {response}")
        print("Question retrieval process completed.")
    
    """
    # Clean up the vector database
    print("Starting Vector Database cleanup process...")
    delete_vector_db(vector_db)
    print("Vector Database cleanup process completed.")
    """

if __name__ == "__main__":
    main()
