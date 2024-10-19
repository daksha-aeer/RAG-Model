from backend.document_processing import extract_text_from_file, split_text_into_chunks
from backend.embedding import generate_embeddings
from backend.vector_db import upsert_to_pinecone, search_in_pinecone
from backend.pinecone import get_index
from backend.cohere import generate_answer_from_cohere
from backend.cohere_embedder import CohereEmbedder  # Ensure the correct import

# Initialize the CohereEmbedder
cohere_embedder = CohereEmbedder()  # Create an instance of the class

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

COHERE_API_KEY = os.getenv('COHERE_API_KEY')

def process_query(query, uploaded_files):
    # Extract text from the uploaded files
    combined_text = ""
    for file in uploaded_files:
        text = extract_text_from_file(file)
        combined_text += text

    # Split the text into chunks
    chunks = split_text_into_chunks(combined_text)

    # Generate embeddings for the chunks
    chunk_embeddings = generate_embeddings(chunks)

    index = get_index()

    # Create metadata for each text chunk
    chunk_metadatas = [{"text": chunk} for chunk in chunks]

    # Upsert the embeddings to Pinecone
    upsert_to_pinecone(chunk_embeddings, chunk_metadatas)

    # Generate embedding for the query
    query_embedding = cohere_embedder.embed_query(query)
    print("Query Embedding:", query_embedding)

    # Inside your process_query function after generating the embeddings
    if len(chunk_embeddings) > 0 and len(chunk_metadatas) > 0:
        # Upsert just the first chunk for testing
        upsert_to_pinecone([chunk_embeddings[0]], [chunk_metadatas[0]])
        
        # Now query back the same vector
        query_embedding = chunk_embeddings[0]
        test_results = search_in_pinecone(query_embedding, top_k=1)
        
        # Print the test results to check for metadata
        print("Test Search Results:", test_results)


    # Search the relevant chunks in Pinecone
    results = search_in_pinecone(query_embedding, top_k=3)  

    # Debug: Print the results structure
    print("Search results:", results)

    # Check the structure of the matches
    if 'matches' in results:
        print("Matches:", results['matches'])
    else:
        print("No matches found.")
        return None, None  # Handle case where no matches are found

    # Extract the context from results
    context = "\n".join(
    result["metadata"]["text"] if "metadata" in result and "text" in result["metadata"] else "No metadata available"
    for result in results["matches"]
    )
    # print(results["matches"])  # Debugging line to see the structure of matches

    answer = generate_answer_from_cohere(query, context)

    return answer, context
