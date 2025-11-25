import torch
from transformers import GenerationConfig
from typing import Dict

def build_prompt(theme: str, mode: str) -> str:
    mode = mode.lower()
    if mode not in ["story", "poem"]:
        raise ValueError("Mode must be 'story' or 'poem'.")

    instruction = "You are an imaginative writer. "

    if mode == "story":
        instruction += f"Write a short engaging story about '{theme}'."
    else:
        instruction += f"Write a short poem about '{theme}'."

    prompt = instruction + "\n\n" + mode.capitalize() + ":\n"
    return prompt


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
        max_new_tokens=max_new_tokens
    )

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            generation_config=gen_config,
            pad_token_id=tokenizer.eos_token_id
        )

    generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return {"generated_text": generated[len(prompt):].strip()}
