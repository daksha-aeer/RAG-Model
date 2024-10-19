import cohere
import os
from dotenv import load_dotenv

load_dotenv()
# Load environment variables
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# Initialize Cohere client
co = cohere.Client(COHERE_API_KEY)

# Assuming cohere is already initialized in your embedding.py file
def generate_answer_from_cohere(query, context):
    # Create a prompt for Cohere's language model with the context and query
    prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"

    # Generate an answer based on the context using Cohere
    response = co.generate(
        model='command-xlarge-nightly',  # Choose an appropriate model
        prompt=prompt,
        max_tokens=150,  # Adjust based on the required length of the answer
        temperature=0.7,  # Control creativity
        stop_sequences=["\n"]  # Stop at the end of a complete answer
    )
    print(f"Cohere response: {response.generations[0].text.strip()}")
    return response.generations[0].text.strip()
