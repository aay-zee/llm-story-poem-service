import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model(preferred="gpt2-medium",fallback="gpt2"):
  device="cuda" if torch.cuda.is_available() else "cpu"
  print(f"Loading model: {preferred} on {device}")
  try:
    tokenizer = AutoTokenizer.from_pretrained(preferred,use_fast=True)
    
    if(tokenizer.pad_token is None):
      tokenizer.pad_token = tokenizer.eos_token
      
    if device == "cuda":
        model=AutoModelForCausalLM.from_pretrained(preferred, device_map="auto",torch_dtype=torch.float16,low_cpu_mem_usage=True,trust_remote_code=True)
    else:
        # On CPU, avoid device_map="auto" and low_cpu_mem_usage as they can cause overhead or OOM on small instances
        model=AutoModelForCausalLM.from_pretrained(preferred, torch_dtype=torch.float32, trust_remote_code=True)
        model.to(device)

    model.eval()
    return tokenizer, model,preferred
  except Exception as e:
    print(f"[MODEL LOADER] Failed to load preferred model: {e}")
    print(f"[MODEL LOADER] Falling back to: {fallback}")

    tokenizer = AutoTokenizer.from_pretrained(fallback, use_fast=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Fallback loading
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(fallback, device_map="auto", torch_dtype=torch.float16)
    else:
        model = AutoModelForCausalLM.from_pretrained(fallback)
        model.to(device)
        
    model.eval()
    return tokenizer, model, fallback