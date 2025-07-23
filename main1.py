# ground_truth_generator.py

from modules.loader import load_pdf
from modules.vector_db import create_vector_db
from modules.retriever import retrieve_answers
from modules.evaluation import Evaluator
from langchain_community.chat_models import ChatOllama
from langchain.schema import HumanMessage
import sys
import csv  # Import csv for writing to a file
import os  # Import os for directory handling

def load_mistral_model():
    # Load Mistral model for question generation
    return ChatOllama(model="mistral")

def generate_question_from_context(context, model):
    # Prepare your messages as needed
    messages = [
        HumanMessage(content=f"Generate a simple question based on this context: {context}")
    ]
    
    # Invoke the model
    response = model.invoke(messages)
    
    # Access the content correctly
    if hasattr(response, 'content'):
        return response.content
    else:
        raise ValueError("Unexpected response structure.")
  
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

    # Load Mistral model for question generation
    print("Loading Mistral model for question generation...")
    mistral_model = load_mistral_model()

    retrieved_contexts = []
    generated_answers = []
    generated_questions = []
    csv_data = []  # List to store CSV rows
    
    for chunk in chunks:
        # Generate a question based on the selected context using Mistral
        question = generate_question_from_context(chunk.page_content, mistral_model)
        print(f"Generated question: '{question}'")
        generated_questions.append(question)

        # Retrieve the answer from the context
        answer = retrieve_answers(vector_db, chunks, question)
        generated_answers.append(answer)
        retrieved_contexts.append(chunk.page_content)
        
        # Create a row for the CSV (context, question, actual answer)
        row = {
            "Context": chunk.page_content,
            "Question": question,
            "Actual Answer": answer
        }
        csv_data.append(row)

    # Ensure the output directory exists
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Write data to CSV
    csv_file_path = os.path.join(output_dir, "ground_truth_chunksize_1000.csv")
    fieldnames = ["Context", "Question", "Actual Answer"]

    with open(csv_file_path, mode="w", newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write header
        writer.writeheader()
        
        # Write rows
        for row in csv_data:
            writer.writerow(row)

    print(f"Data has been written to {csv_file_path}")

    # Evaluate the performance
    # Assuming you have your model loaded and the evaluator set up
    evaluator = Evaluator(retrieved_contexts, generated_answers)
    evaluation_results = evaluator.evaluate_all()
    print("Evaluation Results:", evaluation_results, sep='\n')
    evaluator.visualize_results(evaluation_results)
    

if __name__ == "__main__":
    main()
