import sys
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
# import torch.nn as nn
# from dim_reduction import Autoencoder
# import torch.optim as optim

from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams, PointStruct

if len(sys.argv) != 2:
    print("Usage: python store_embeddings.py <csv_file_path>")
    sys.exit(1)

# csv_file_path = '/Volumes/Passport/CHAABI/bigBasketProducts.csv'
csv_file_path = sys.argv[1]
df = pd.read_csv(csv_file_path)
#df = df[:500]

# Create sentences using DataFrame columns
# index, product, category, sub_category, brand, sale_price, market_price, type, rating, description
sentences = []
for _, row in df.iterrows():
    sentence = f"""This is information for {row['product']} in the subcategory "{row['sub_category']}" in the main category of "{row['category']}". """
    sentence += f"""Brand of the product is "{row['brand']}". This has sale_price of "${row['sale_price']}" and market price "${row['market_price']}". """
    sentence += """ It's a "{row['type']}" with a "{row['rating']}" star rating. Here is the description: "{row['description']}"."""
    sentences.append(sentence)
print(len(sentences))

# Placeholder for Qdrant client, replace with actual Qdrant client
#running using Docker
qdrant_client = QdrantClient("localhost", port=6333)
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

#moving the model to GPU
if torch.cuda.is_available():
   dev = torch.device("cuda:0")
   print("Running on the GPU")
else:
   dev = torch.device("cpu")
   print("Running on the CPU, GPU Recommended otherwise time required will be more")
model.to(dev)

def generate_embeddings(df):
    embeddings = model.encode(sentences, convert_to_tensor=False)
    embeddings = torch.tensor(embeddings, dtype=torch.float32)
    return embeddings

# def dimensionality_reduction(embeddings):
#     input_dim = len(embeddings[0])
#     encoding_dim = 200
#     autoencoder = Autoencoder(input_dim, encoding_dim)
#     criterion = nn.MSELoss()
#     optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)
#     num_epochs = 50
#     for epoch in range(num_epochs):
#         outputs = autoencoder(embeddings)
#         loss = criterion(outputs, embeddings)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
    
#     # Get the reduced embeddings
#     with torch.no_grad():
#         reduced_embeddings = autoencoder.encoder(embeddings).numpy()

#     # Convert the reduced embeddings to a list of lists
#     # embedding_list = reduced_embeddings.tolist()
    
#     return reduced_embeddings

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
        # vec.append(PointStruct(id=idx, vector=x, payload=df.iloc[idx].to_dict()))
        vec.append(PointStruct(id=idx, vector=x, payload={"product" : sentences[idx]}))

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
    # reduced_embeddings = dimensionality_reduction(embeddings).tolist()

    #Query Qdrant
    query_qdrant(df, embeddings.tolist())

    # delete embeddings model after storing embeddings
    del model
