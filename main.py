from fastapi import FastAPI
from pydantic import BaseModel
from model_loader import load_model
from generator import generate_text


tokenizer = None
model = None
model_name = None

@app.on_event("startup")
def startup_event():
  global tokenizer, model, model_name
  print("Loading Model...")
  tokenizer, model, model_name = load_model()
  print(f"Model Loaded: {model_name}")

app=FastAPI(title="AI Story/Poem Generator API")

class GenerateRequest(BaseModel):
  theme:str
  mode:str="story"
  temperature:float=0.8
  top_k:int=50
  top_p:float=0.95
  max_new_tokens:int=150
  
  
@app.post("/generate")
def generate(req:GenerateRequest):
  result=generate_text(
    tokenizer=tokenizer,
    model=model,
    theme=req.theme,
    mode=req.mode,
    temperature=req.temperature,
    top_k=req.top_k,
    top_p=req.top_p,
    max_new_tokens=req.max_new_tokens
  )
  return result