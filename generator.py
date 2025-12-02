from typing import Dict

def build_prompt(theme: str, mode: str) -> str:
    mode = mode.lower()
    if mode not in ["story", "poem"]:
        raise ValueError("Mode must be 'story' or 'poem'.")

    # Mistral Instruct format
    # <s>[INST] Instruction [/INST]
    
    if mode == "story":
        instruction = f"Write a short fairy tale story about {theme}. Start with 'Once upon a time'."
        return f"<s>[INST] {instruction} [/INST] Once upon a time"
    else:
        instruction = f"Write a poem about {theme}."
        return f"<s>[INST] {instruction} [/INST]"


def generate_text(
    tokenizer,
    model,
    theme: str,
    mode: str,
    temperature: float,
    top_k: int,
    top_p: float,
    max_new_tokens: int
) -> Dict[str, str]:

    prompt = build_prompt(theme, mode)
    
    # Check if we are using ctransformers (tokenizer is None)
    if tokenizer is None:
        # ctransformers generation
        # model(prompt, ...) returns the generated text (completion only usually)
        
        # Adjust parameters for ctransformers if needed
        # It supports top_k, top_p, temperature, repetition_penalty, etc.
        
        generated_completion = model(
            prompt, 
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )
        
        # If the model returns just the completion, we need to handle it.
        # For story mode, we prompted with "Once upon a time", so the completion continues from there.
        # We should prepend "Once upon a time" to the result if we want the full story.
        
        if mode == "story":
            full_text = "Once upon a time " + generated_completion
            return {"generated_text": full_text.strip()}
        else:
            return {"generated_text": generated_completion.strip()}

    else:
        # Legacy/Fallback for Transformers (if we ever switch back or for local testing with other models)
        import torch
        from transformers import GenerationConfig
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)

        gen_config = GenerationConfig(
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=True,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id
        )

        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                generation_config=gen_config
            )

        generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        # Simple cleanup for legacy path
        return {"generated_text": generated}
