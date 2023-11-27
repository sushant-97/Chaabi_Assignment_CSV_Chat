import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
from qdrant_client import QdrantClient

from llm_inference import query_llm
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# #if Docker
qdrant_client = QdrantClient(
    url="localhost",
    port=6333,
)

# Load embedding model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def get_query_vector(text):
    embeddings = model.encode(text, convert_to_tensor=False)
    embeddings = torch.tensor(embeddings, dtype=torch.float32)
    
    return embeddings.numpy().tolist()

# @app.route('/')
# def index():
#     return render_template('index.html')

@app.route('/query', methods=['POST'])
def query_endpoint():
    if request.method == 'POST':
        try:
            # data = request.get_json()
            # user_prompt = data['user_prompt']

            user_prompt = request.form['user_prompt']

            # Get query vector and perform the search
            database_query_vector = get_query_vector(user_prompt)
            context_list = qdrant_client.search(
                collection_name="test_collection", query_vector=database_query_vector, limit=3
            )
            
            # Perform additional processing using llm_res if needed
            llm_res = query_llm(user_prompt, context_list)
            print("***********HEEHEHRR")
            # llm_res = "Print LLM Res"

            # Return the results to the HTML page
            # return render_template('index.html', user_prompt=user_prompt, result=llm_res)
            return llm_res

        except Exception as e:
            # return render_template('index.html', user_prompt=user_prompt, error=str(e))
            return "error:{}".format(e)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8888, debug=True)
