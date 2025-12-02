from ctransformers import AutoModelForCausalLM
import os

def load_model(preferred="TheBloke/Mistral-7B-Instruct-v0.2-GGUF", fallback=None):
    model_file = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
    print(f"Loading model: {preferred} ({model_file})...")
    
    try:
        # Load the model using ctransformers
        # gpu_layers=0 means CPU only. If Cloud Run has no GPU, this is correct.
        model = AutoModelForCausalLM.from_pretrained(
            preferred, 
            model_file=model_file, 
            model_type="mistral",
            gpu_layers=0
        )
        print(f"[STARTUP] Model Loaded Successfully: {preferred}")
        
        # ctransformers model handles tokenization internally for simple generation, 
        # but for compatibility with the main loop structure, we'll return None for tokenizer
        # and handle the generation logic in generator.py
        return None, model, preferred
        
    except Exception as e:
        print(f"[MODEL LOADER] Failed to load model: {e}")
        raise e