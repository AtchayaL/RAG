from modules.loader import load_pdf
from modules.vector_db import create_or_load_vector_db
from modules.retriever import retrieve_answers
import os
import sys

def main():
    # Path to the PDF file
    pdf_path = "data/WEF_The_Global_Cooperation_Barometer_2024.pdf"
    
    # Extract the PDF name (without extension) for directory management
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]

    # Load PDF
    print("Starting PDF loading process...")
    data = load_pdf(pdf_path)
    if not data:
        print("Failed to load PDF. Exiting...")
        sys.exit(1)
    print("PDF loading process completed.")

    # Create or Load Vector Database
    print("Starting Vector Database creation process...")
    try:
        vector_db, chunks = create_or_load_vector_db(pdf_name, data)
        print("Vector Database creation process completed.")
    except Exception as e:
        print(f"Failed to create or load Vector Database: {e}")
        sys.exit(1)

    # Start the question loop
    while True:
        # Ask for user input (question)
        question = input("\nEnter Prompt (or type 'exit' to quit): ").strip()
        
        # Check if the user wants to exit the loop
        if question.lower() == 'exit':
            print("Exiting the question loop...")
            break
        
        print(f"Generating prompt for : '{question}'")
        
        try:
            # Retrieve relevant context and the final answer
            response, relevant_chunks = retrieve_answers(vector_db, chunks, question)
            
            # Print relevant context (if needed)
            print("\nRelevant context (chunks):")
            #for chunk in relevant_chunks:
                #print(chunk.page_content, end="\n")  # Assuming the chunk has a 'page_content' attribute
            
            # Print the final response
            print(f"\n\nResponse:\n{response}")
            
        except Exception as e:
            print(f"Error during question retrieval: {e}")
            continue  # Skip to the next question

    print("End of Execution...")

    """
    # Uncomment this section if you want to clean up the vector database after usage
    print("Starting Vector Database cleanup process...")
    delete_vector_db(vector_db)
    print("Vector Database cleanup process completed.")
    """

if __name__ == "__main__":
    main()
