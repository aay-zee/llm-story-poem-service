from transformers import AutoTokenizer, AutoModelForCausalLM
import os

def download_models():
    models = ["gpt2-medium", "gpt2"]
    
    for model_name in models:
        print(f"Downloading {model_name}...")
        # Download tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        # Download model
        model = AutoModelForCausalLM.from_pretrained(model_name)
        print(f"Successfully downloaded {model_name}")

if __name__ == "__main__":
    download_models()
