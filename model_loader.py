import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model(preferred="mistralai/Mistral-7B-Instruct",fallback="gpt2-medium"):
  device="cuda" if torch.cuda.is_available() else "cpu"
  print(f"Loading model: {preferred} on {device}")
  try:
    tokenizer = AutoTokenizer.from_pretrained(preferred,use_fast=True)
    
    if(tokenizer.pad_token is None):
      tokenizer.pad_token = tokenizer.eos_token
      
    model=AutoModelForCausalLM.from_pretrained(preferred, device_map="auto",torch_dtype=torch.float16 if device == "cuda" else torch.float32,low_cpu_mem_usage=True,trust_remote_code=True)
    model.eval()
    return tokenizer, model,preferred
  except Exception as e:
    print(f"[MODEL LOADER] Failed to load preferred model: {e}")
    print(f"[MODEL LOADER] Falling back to: {fallback}")

    tokenizer = AutoTokenizer.from_pretrained(fallback, use_fast=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(fallback)
    model.to(device)
    model.eval()
    return tokenizer, model, fallback