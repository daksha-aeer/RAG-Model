## Colab Notebook

Overview:
To make the Question-Answer RAG Model, I made use of Cohere AI and Pinecone. First, the data is split into smaller parts or chunks. Cohere is used to create embeddings for
these chunks. Pinecone is set up with index specifications and the dimensions are set to match the model used for Cohere - "embed-english-v3.0". The created embeddings are then
added to the index. We create an embedding for our queries as well and it is passed to the Pinecone vector store from Langchain. To generate an answer from the extracted chunks
I used Cohere's Generate.


To run the notebook, add your API keys in this form:

```python
with open('.env', 'w') as f:
    f.write('COHERE_API_KEY=key\n')
    f.write('PINECONE_API_KEY=key\n')
```

Before this code:

```python
from dotenv import load_dotenv
import os


load_dotenv()  # Load environment variables from .env file
COHERE_API_KEY = os.getenv('COHERE_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
```



## Interactive QA Interface

Overview:
The code is split into two with the Streamlit portion in the frontend folder's app.py file and the rest of the logical code in the backend file. So to run the code you need to 
run the command:

```
streamlit run frontend/app.py
```

The app.py file passes the query and the document(s) to qa_system.py where we first split the data to allow smooth processing of large files. Once the data is split we generate embeddings 
for the chunks and their corresponding embeddings. The embeddings are upserted into the Pinecone database. The query is also embedded and then a match is searched. This part
still needs some work. Even though it seems to be running, the answers being provided are not accurate. The model also hallucinates and produces its own answers. 

![image](https://github.com/user-attachments/assets/eaf752d2-b6d4-4466-89d0-5f4855431d85)

![image](https://github.com/user-attachments/assets/6e831356-d3b9-4e57-b8dc-41af44da51e6)
