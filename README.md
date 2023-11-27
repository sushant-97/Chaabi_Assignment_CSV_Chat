# Chaabi_Assignment_CSV_Chat
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
Run main.py file to host api.
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

