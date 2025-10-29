# model_loader.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# 1. Define Quantization Configuration
# This uses the bitsandbytes library, our "Memory Saver"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# 2. Specify the Model ID
# We'll use a model from Hugging Face, like Mistral-7B
# This line MUST be:
model_id = "mistralai/Mistral-7B-Instruct-v0.1"
# 3. Load the Model and Tokenizer
print("Loading model... This might take a moment.")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map={"": 0}, # Automatically uses GPU if available
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(model_id)

# Set padding token if it's not set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("--- Phase 1 Complete: Model Loaded Successfully ---")

# You can add a quick test. Uncomment the lines below to run it.
# print("Running a quick test...")
# prompt = "What is the capital of France?"
# inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
# outputs = model.generate(**inputs, max_new_tokens=20)
# print(f"Test Response: {tokenizer.decode(outputs[0], skip_special_tokens=True)}")
