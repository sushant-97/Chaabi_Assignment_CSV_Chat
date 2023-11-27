from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import transformers
import torch

#moving the model to GPU
if torch.cuda.is_available():
   dev = torch.device("cuda:0")
   print("Running on the GPU")
else:
   dev = torch.device("cpu")
   print("Running on the CPU, GPU Recommended otherwise time required will be more")


# compute_dtype = getattr(torch, "float16")

# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=compute_dtype,
#     bnb_4bit_use_double_quant=False,
# )

def load_falcon():
    pass
#     compute_dtype = getattr(torch, "float16")

#     bnb_config = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_quant_type="nf4",
#         bnb_4bit_compute_dtype=compute_dtype,
#         bnb_4bit_use_double_quant=False,
#     )

#     model_name = "tiiuae/falcon-7b"

#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModelForCausalLM.from_pretrained(
#         model_name,
#         torch_dtype=torch.bfloat16,
#         trust_remote_code=True,
#         quantization_config=bnb_config,
#         device_map="auto",
#     )

#     pipeline = transformers.pipeline(
#         "text-generation",
#         model=model,
#         tokenizer=tokenizer,
#     )

#     return pipeline, tokenizer

#Loading the model
def load_T5():
    from transformers import T5Tokenizer, T5ForConditionalGeneration, Adafactor
    
    model_id = "MaRiOrOsSi/t5-base-finetuned-question-answering"    # "t5-small"   # "t5-base"
    tokenizer = T5Tokenizer.from_pretrained(model_id)
    model = T5ForConditionalGeneration.from_pretrained(model_id,
                                             return_dict=True)
    model.to(dev)
    
    # Output Example from Huggingface
    # input_ids = tokenizer("translate English to German: The house is wonderful.", return_tensors="pt").input_ids
    # outputs = model.generate(input_ids)
    # print(tokenizer.decode(outputs[0], skip_special_tokens=True))

    return model, tokenizer

# def falcon_output():
#     pipeline, tokenizer = load_falcon()
#     # prompt = "What are the main benefits of enrolling in a DataCamp career track?"
#     prompt = prompt + " " + context['description']
#     sequences = pipeline(
#         prompt ,
#         max_length=100,
#         do_sample=True,
#         top_k=20,
#         num_return_sequences=1,
#         eos_token_id=tokenizer.eos_token_id,
#     )
    
#     for seq in sequences:
#         print(f"Result: {seq['generated_text']}")

#     return sequences[0]['generated_text']



def query_llm(prompt, context):
    print("Loading T5")
    model, tokenizer = load_T5()
    print("T5 Loaded now converting promt to input ids")
    # prompt = prompt + " " + context['description']
    # prompt = "what is the rate of nissin noodles?"
    # context = "Nissin Mazedaar Masala Instant Cup Noodles, 70 g, Price at ruppes 50 with 10%\ discount"

    input = f"question: {prompt} context: {context}"
    encoded_input = tokenizer([input],
                                return_tensors='pt',
                                max_length=512,
                                truncation=True)
    output = model.generate(input_ids = encoded_input.input_ids,
                                attention_mask = encoded_input.attention_mask)
    output = tokenizer.decode(output[0], skip_special_tokens=True)
    print(output)

    return output

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
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

user_prompt = "List out brands that are operating in hygine?"
database_query_vector = get_query_vector(user_prompt)

context_list = qdrant_client.search(
    collection_name="test_collection", query_vector=database_query_vector, limit=10
)
print(type(context_list))
print("****")
print(type(context_list[0]))

query_llm(user_prompt,context_list)