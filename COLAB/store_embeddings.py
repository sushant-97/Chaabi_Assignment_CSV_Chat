# api_functions.py
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
from dim_reduction import Autoencoder
import torch.optim as optim

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from qdrant_client import QdrantClient, models
from qdrant_client.http.models import PointStruct

csv_file_path = 'bigBasketProducts.csv'
df = pd.read_csv(csv_file_path)
df = df[:10]

# Placeholder for Qdrant client, replace with actual Qdrant client
#if running on colab
qdrant_client = QdrantClient(":memory:")
#if running using Docker
# qdrant_client = QdrantClient("localhost", port=6333)

def generate_embeddings(df):
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    embeddings = model.encode(df['description'].astype(str), convert_to_tensor=False)
    embeddings = torch.tensor(embeddings, dtype=torch.float32)
    return embeddings

def dimensionality_reduction(embeddings):
    input_dim = len(embeddings[0])
    encoding_dim = 200
    autoencoder = Autoencoder(input_dim, encoding_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)
    num_epochs = 50
    for epoch in range(num_epochs):
        outputs = autoencoder(embeddings)
        loss = criterion(outputs, embeddings)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Get the reduced embeddings
    with torch.no_grad():
        reduced_embeddings = autoencoder.encoder(embeddings).numpy()

    # Convert the reduced embeddings to a list of lists
    # embedding_list = reduced_embeddings.tolist()
    
    return reduced_embeddings

def query_qdrant(df, embedding_list):

    #Generate embeddings for each row in the DataFrame
    embedding_list = generate_embeddings(df).tolist()
    # embedding_list_ = dimensionality_reduction(embedding_list)

    vectors_ = models.VectorParams(
        size = len(embedding_list[0]),  # Vector size is defined by used model
        distance = models.Distance.COSINE,
    )

    qdrant_client.recreate_collection(
        collection_name="test_collection",
        vectors_config=vectors_,
    )
    # print(type(embedding_list[0][0]))

    vec = []
    for idx, x in enumerate(embedding_list):
        # print(idx)
        vec.append(PointStruct(id=idx, vector=x, payload=df.iloc[idx].to_dict()))

    operation_info = qdrant_client.upsert(
        collection_name="test_collection",
        wait=True,
        points=vec,
    )

    print(operation_info)

if __name__ == "__main__":
    #Generate embeddings
    embeddings = generate_embeddings(df)

    #Dimensionality reduction (if needed)
    reduced_embeddings = dimensionality_reduction(embeddings).tolist()

    #Query Qdrant
    query_qdrant(df, reduced_embeddings)