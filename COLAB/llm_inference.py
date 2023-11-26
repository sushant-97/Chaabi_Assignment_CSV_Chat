from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import transformers
import torch


compute_dtype = getattr(torch, "float16")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)

def load_falcon():

    compute_dtype = getattr(torch, "float16")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=False,
    )

    model_name = "tiiuae/falcon-7b"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map="auto",
    )

    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )

    return pipeline, tokenizer

# #Loading the model
# def load_llm_2():
#     # Load the locally downloaded model here
#     llm = CTransformers(
#         model = "llama-2-7b-chat.ggmlv3.q8_0.bin",
#         model_type="llama",
#         max_new_tokens = 512,
#         temperature = 0.5
#     )

#     return llm
 

def query_llm(prompt, context, model):
    if model == "falcon":
        pipeline, tokenizer = load_falcon()
    
    # prompt = "What are the main benefits of enrolling in a DataCamp career track?"
    prompt = prompt + " " + context['description']
    sequences = pipeline(
        prompt ,
        max_length=100,
        do_sample=True,
        top_k=20,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
    )

    for seq in sequences:
        print(f"Result: {seq['generated_text']}")

    return sequences[0]['generated_text']