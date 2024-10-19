import cohere
import os
from dotenv import load_dotenv

load_dotenv()
# Load environment variables
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# Initialize Cohere client
co = cohere.Client(COHERE_API_KEY)

def generate_embeddings(chunks):
    if not chunks:
        raise ValueError("No text chunks provided for embedding.")

    print(f"Generating embeddings for {len(chunks)} chunks...")  # Debugging line

    # Call the Cohere embed API
    response = co.embed(
        texts=chunks,
        model="embed-english-v3.0",
        input_type="search_document",
        embedding_types=["float"]
    )

    embeddings = response.embeddings.float

    print(f"Generated Embeddings: {len(embeddings)} embeddings.")

    if not embeddings:
        raise ValueError("No embeddings received.")


    if hasattr(response.embeddings, '__len__'):
        print(f"Received {len(embeddings)} embeddings.")  # Debugging line
    else:
        print("Received embeddings but the structure is not what was expected.")

    # Process embeddings
    return embeddings