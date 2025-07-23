from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaEmbeddings, ChatOllama
import numpy as np

def get_embedding(text):
    """Get the embedding for a given text."""
    # Initialize the embedding model
    embedding_model = OllamaEmbeddings(model="nomic-embed-text")

    # Get the embedding for the input text
    embedding = embedding_model.embed_query(text)
    return embedding

def retrieve_answers(vector_db, chunks, question):
    top_k = 3

    # Define the LLM
    local_model = "llama3.1"
    llm = ChatOllama(model=local_model)
    print("LLM model loaded successfully.")
    
    # Embed the question using the same embedding model
    question_embedding = get_embedding(question)

    # Perform FAISS nearest neighbor search
    D, I = vector_db.search(np.array([question_embedding]), top_k)  # D: distances, I: indices

    print("Retrieved indices (I):", I)
    print("Number of chunks available:", len(chunks))

    # Ensure all retrieved indices are valid
    valid_indices = [idx for idx in I[0] if idx < len(chunks)]
    if not valid_indices:
        print("No valid indices found. Exiting retrieval.")
        return "No relevant information found.", []  # Return empty list for chunks

    # Retrieve the top_k relevant chunks based on valid indices
    relevant_chunks = [chunks[idx] for idx in valid_indices]

    # Define the prompt template to pass the context to the LLM
    template = """
        You are an expert on AI. Based ONLY on the following context, provide a detailed and specific answer to the question. 
            Context: {context}
            Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    # Create a string with all the relevant chunks (context) to pass to the LLM
    context = "\n\n".join([chunk.page_content for chunk in relevant_chunks])

    chain = (
        prompt
        | llm
        | StrOutputParser()
    )

    # Get the answer from the LLM
    print(f"Asking question: {question}")
    try:
        result = chain.invoke({"context": context, "question": question})
    except Exception as e:
        print(f"Error during retrieval: {e}")
        return "Error occurred while generating the answer.", relevant_chunks

    return result, relevant_chunks  # Return both the answer and relevant chunks
