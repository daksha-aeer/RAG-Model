import cohere
import os
from dotenv import load_dotenv

load_dotenv()
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# Initialize Cohere client
co = cohere.Client(COHERE_API_KEY)

class CohereEmbedder:  # Use CamelCase for class names
    def embed_query(self, query):
        embeds = co.embed(
            texts=[query],  # Cohere expects a list of texts
            model="embed-english-v3.0",
            input_type="search_query",
            embedding_types=['float']
        )
        return embeds.embeddings.float[0]  # Access the float embedding at index 0
