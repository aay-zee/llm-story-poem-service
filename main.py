from fastapi import FastAPI
from fastapi import Response, status
import threading
from pydantic import BaseModel
from model_loader import load_model
from generator import generate_text


# Create the FastAPI app before registering events or routes
app = FastAPI(title="AI Story/Poem Generator API")

tokenizer = None
model = None
model_name = None


model_loading = False
model_loaded = False


def _load_model_background():
    global tokenizer, model, model_name, model_loading, model_loaded
    try:
        tokenizer, model, model_name = load_model()
        print(f"Model Loaded: {model_name}")
        model_loaded = True
    except Exception:
        import traceback
        print("Failed to load model in background:")
        traceback.print_exc()
    finally:
        model_loading = False


@app.on_event("startup")
def startup_event():
    global model_loading
    # Start loading model in background so the server becomes responsive immediately.
    if not model_loading:
        model_loading = True
        thread = threading.Thread(target=_load_model_background, daemon=True)
        thread.start()

class GenerateRequest(BaseModel):
  theme:str
  mode:str="story"
  temperature:float=0.8
  top_k:int=50
  top_p:float=0.95
  max_new_tokens:int=150
  
  
@app.post("/generate")
def generate(req: GenerateRequest):
    if not model_loaded:
        return Response(content="Model is still loading. Please retry in a moment.", status_code=status.HTTP_503_SERVICE_UNAVAILABLE)

    result = generate_text(
        tokenizer=tokenizer,
        model=model,
        theme=req.theme,
        mode=req.mode,
        temperature=req.temperature,
        top_k=req.top_k,
        top_p=req.top_p,
        max_new_tokens=req.max_new_tokens,
    )
    # Ensure a JSON-serializable dict is returned
    return {"result": result}