import os
from uuid import uuid4
from backend.pinecone import get_index

def upsert_to_pinecone(embeddings, metadatas):
    index = get_index()  # Get the Pinecone index
    ids = [str(uuid4()) for _ in range(len(embeddings))]

    # Debug: Print the ids, embeddings, and metadatas
    print("IDs:", ids)
    print("Embeddings:", embeddings)
    print("Metadatas:", metadatas)

    # Ensure metadatas is a list of dicts
    index.upsert(vectors=zip(ids, embeddings, metadatas))
    print(f"Upserted {len(embeddings)} vectors to Pinecone.")



def search_in_pinecone(query_embedding, top_k=3):
    index = get_index()  # Get the Pinecone index
    results = index.query(vector=query_embedding, top_k=top_k)
    
    # Debug: Print the full structure of the results
    print(f"Search results: {results}")

    return results
