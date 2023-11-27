# Chaabi_Assignment_CSV_Chat

Step 1:
Install docker and 
Store vector embeddings in Qdrant using an Encoder transformer. We have used "paraphrase-MiniLM-L6-v2" from huggingface.
I have self-hosted Qdrant in Docker. Steps for the same can be found [here](https://medium.com/@fadil.parves/qdrant-self-hosted-28a30106e9dd)

To run the qdrant instance locally, run the command below in your terminal
```
docker run -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage:z qdrant/qdrant

```


Step 2:
