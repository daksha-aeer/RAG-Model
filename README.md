## Colab Notebook

Overview:
To make the Question-Answer RAG Model, I used Cohere AI and Pinecone. First, I scraped Toyota's Wikipedia page to create a database. I was initially working with a csv file but it didn't work well with the model and produced inaccurate results. So I switched to this approach. Then, I split the data into smaller parts or chunks for better processing. To create embeddings for these chunks I used cohere. Pinecone is then set up with the index specifications and dimensions to match the model used for the Cohere embeddings - "embed-english-v3.0". The created embeddings are then added to this index. After setting up the embeddings and adding them to the Pinecone index, I created a custom embedding class for Cohere to handle query embeddings. I then instantiated the CohereEmbedder and initialized the Pinecone vector store, linking it to the custom embedding method. To retrieve the most relevant documents for a query, I performed a similarity search on the vector store. Finally, I created a function, generate_answer_from_cohere, which takes a query, retrieves the top k relevant documents from Pinecone, and generates a coherent answer using Cohere's language model

For more detailed code and examples, refer to the Colab notebook.

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
The code is split into two with the Streamlit portion in the frontend folder's app.py file and the rest of the logical code in the backend folder. To run the code you need to first download the dependencies by running the command:

```
pip install -r requirements.txt
```

Setup your API keys by creating a .env file and passing them in the form:

```
COHERE_API_KEY=key
PINECONE_API_KEY=key
```

Once the initial setup is done, run the command:

```
streamlit run frontend/app.py
```

Working:

The app.py file passes the query and the document(s) to qa_system.py where we first split the data into chunks to allow smooth processing of large files. Once the data is split we generate embeddings for the chunks. The embeddings are upserted in the Pinecone database which is set up in the pinecone.py file. The query is then embedded and we run the retrieval model to search for a match. This part still needs some work. The model generates an answer but the extraction is not accurate. The model might also hallucinate and produce its own answers. 

![image](https://github.com/user-attachments/assets/eaf752d2-b6d4-4466-89d0-5f4855431d85)

![image](https://github.com/user-attachments/assets/6e831356-d3b9-4e57-b8dc-41af44da51e6)
