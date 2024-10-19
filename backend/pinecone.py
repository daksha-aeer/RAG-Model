import pinecone
from pinecone import ServerlessSpec
from pinecone import Pinecone

import time
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

spec = ServerlessSpec(
    cloud="aws", region="us-east-1"
)

index_name = '2-sample'
existing_indexes = [
    index_info["name"] for index_info in pc.list_indexes()
]

# check if index already exists
if index_name not in existing_indexes:
    # if does not exist, create index
    pc.create_index(
        index_name,
        dimension=1024,   # dimensionality of embed-english-v3.0
        metric='dotproduct', # can use dot product, cosine similarity, and Euclidean distance as the similarity metric
        spec=spec
    )
    # wait for index to be initialized
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)

# connect to index
index = pc.Index(index_name)
time.sleep(1)

# view index stats
index.describe_index_stats()

# Return the index for use in other modules
def get_index():
    return index
