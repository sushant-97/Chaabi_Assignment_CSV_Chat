import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

from llm_inference import query_llm, load_falcon

from qdrant_client import qdrant_client
from store_embeddings import qdrant_client

# #if Docker
# qdrant_client = QdrantClient(
#     url="localhost",
#     port=6333,
# )

#if colab
# available in memory

def get_vector(text):
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    embeddings = model.encode(text, convert_to_tensor=False)
    embeddings = torch.tensor(embeddings, dtype=torch.float32)
    
    return embeddings.numpy().tolist()

if __name__ == "__main__":
    #user_prompt = "What are the main benefits of enrolling in a DataCamp career track?"
    user_prompt = input("Enter your query: ")
    database_query_vector = get_vector()

    context_list = qdrant_client.search(
    collection_name="test_collection", query_vector=database_query_vector, limit=3
    )

    llm_res = query_llm(user_prompt, context_list)
