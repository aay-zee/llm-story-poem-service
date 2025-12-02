from ctransformers import AutoModelForCausalLM

def download_models():
    model_repo = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
    model_file = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
    
    print(f"Downloading {model_repo} ({model_file})...")
    try:
        # ctransformers will download to the cache directory automatically
        model = AutoModelForCausalLM.from_pretrained(model_repo, model_file=model_file)
        print(f"Successfully downloaded {model_repo}")
    except Exception as e:
        print(f"Error downloading model: {e}")

if __name__ == "__main__":
    download_models()
