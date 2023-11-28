# Overview
The project encompasses the processing of BigBasket data, where the data is formatted into documents by aggregating rows for each column. Following this, the Sentence Transformer embedding model is applied to convert these documents into embeddings. These embeddings are then stored in a Qdrant database for efficient retrieval. Furthermore, the Falcon-7B-Instruct model is harnessed for Language Model (LM) queries

# Logical Flow
* The initial processing of the BigBasket dataset involved consolidating rows in each column to construct documents.
* To transform the documents into embeddings for streamlined storage and retrieval, the Sentence Transformer embedding model was employed.
* For effective storage and querying of the generated embeddings, Qdrant DB was integrated into the system.
* Language model queries were conducted using the Falcon-7B-Instruct model, enhancing the system's capability for diverse question answering.
* To address BigBasket-specific inquiries with precision, Langchain was implemented, focusing on accurate data retrieval.
* The technologies utilized in the process include Sentence Transformer for document embedding, Qdrant DB for efficient storage and retrieval, Falcon-7B-Instruct Model for language model queries, and Langchain for specific queries related to the BigBasket dataset.


# Implementation
This API receives a user prompt via a POST request to the '/query' endpoint. It uses Sentence Transformers to convert the prompt into an embedding, performs a search in a Qdrant collection which we first create for our database to get context, and then passes the prompt along with context to LLM[Falcon 7B]. The API is designed to provide insights or additional information based on the user's input.
First I have used colab to store embeddings and query llm based on context. Later I used local environment to create API. To generate results on local environment I have used T5 for text generation but Falcon 7B is recommned for good results

Preprocessing for storing row in qDrant
BigBasket’s Products List data has many column and LLMs or NLP models have good understanding of natural language text. Hence I first converted a single row of table into text as follows

```
This is information for Garlic Oil - Vegetarian Capsule 500 mg in the subcategory "Hair Care" in the main category of "Beauty & Hygiene". Brand of the product is "Sri Sri Ayurveda ". This has sale_price of "$220.0" and market price "$220.0". It's a "Hair Oil & Serum" with a "4.1" star rating. Here is the description: "This Product contains Garlic Oil that is known to help proper digestion, maintain proper cholesterol levels, support cardiovascular and also build immunity. For Beauty tips, tricks & more visit https://bigbasket.blog/".
```

Following are the steps to create and use the api


Step 0:
Install all the requirements using following
```
pip install -r requirements.txt
```

Step 1:
Install docker and 
Store vector embeddings in Qdrant using an Encoder transformer. We have used "paraphrase-MiniLM-L6-v2" from huggingface.
I have self-hosted Qdrant in Docker. Steps for the same can be found [here](https://medium.com/@fadil.parves/qdrant-self-hosted-28a30106e9dd)

To run the qdrant instance locally, run the command below in your terminal
```
docker run -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage:z qdrant/qdrant

```

First store embeddings. To store embeddings in qDrant runn store_embeddings.py file as follows.

```
python store_embeddings.py [/path/to/your/csv/file.csv]

```
Your quadrant database will be ready to use by LLM.

Step 2:
Run main.py file to host api. Inside this we query LLM. I have tested using T5 due to resources but have also written code for Falcon which gives better results.
```
python main.py
```

Step 3:
Test API using test.py file which take two arguments
api_url and user_prompt.
Run test.py to use API endpoint as follows
```
python test.py --api_url "http://192.168.251.119:8888/query" --user_prompt "what is the rate of nissin noodles?"

```

