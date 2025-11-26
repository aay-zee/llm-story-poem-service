import torch
from transformers import GenerationConfig
from typing import Dict

def build_prompt(theme: str, mode: str) -> str:
    mode = mode.lower()
    if mode not in ["story", "poem"]:
        raise ValueError("Mode must be 'story' or 'poem'.")

    if mode == "story":
        # Base models work better with a clear start they can continue
        return f"Topic: {theme}\nGenre: Fairy Tale\n\nOnce upon a time"
    else:
        return f"Topic: {theme}\nGenre: Poem\n\n"


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

    device = "cuda" if torch.cuda.is_available() else "cpu"

    prompt = build_prompt(theme, mode)
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
    
    # For story mode, we want to keep "Once upon a time" if it was part of the prompt
    # but the prompt variable contains the whole "Topic: ... \n\nOnce upon a time"
    # The model output will contain the full prompt + generation.
    # We want to return "Once upon a time..." + generation, but hide "Topic: ..."
    
    if mode == "story":
        # Find where the story starts
        starter = "Once upon a time"
        start_index = generated.find(starter)
        if start_index != -1:
            return {"generated_text": generated[start_index:].strip()}
            
    # Fallback or for poem: remove the prompt header if possible, or just return everything after the prompt
    # If the prompt didn't have the starter (poem), we just strip the prompt.
    
    # If we just strip the prompt, we lose "Once upon a time" in story mode because it's IN the prompt.
    # So the logic above handles story mode.
    
    # For poem mode:
    if mode == "poem":
        # Prompt ends with "\n\n"
        # We can try to find the end of the header
        header_end = generated.find("\n\n")
        if header_end != -1:
             return {"generated_text": generated[header_end+2:].strip()}
             
    # Generic fallback
    return {"generated_text": generated[len(prompt):].strip()}
