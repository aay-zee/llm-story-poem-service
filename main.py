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
model_error = None


def _load_model_background():
    global tokenizer, model, model_name, model_loading, model_loaded, model_error
    try:
        print("[STARTUP] Starting model load...")
        tokenizer, model, model_name = load_model()
        print(f"[STARTUP] Model Loaded Successfully: {model_name}")
        model_loaded = True
        model_error = None
    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        print("[ERROR] Failed to load model:")
        print(error_msg)
        model_error = str(e)
    finally:
        model_loading = False
        print(f"[STARTUP] Model loading finished. Loaded: {model_loaded}, Error: {model_error}")


@app.on_event("startup")
def startup_event():
    global model_loading
    # Start loading model in background so the server becomes responsive immediately.
    if not model_loading:
        model_loading = True
        thread = threading.Thread(target=_load_model_background, daemon=True)
        thread.start()


@app.get("/")
def root():
    """Health check and status endpoint"""
    return {
        "status": "running",
        "model_loaded": model_loaded,
        "model_loading": model_loading,
        "model_name": model_name if model_loaded else None,
        "error": model_error
    }


@app.get("/health")
def health():
    """Kubernetes/Cloud Run health check"""
    if model_loaded:
        return {"status": "healthy"}
    elif model_error:
        return Response(content=f"Model failed to load: {model_error}", status_code=status.HTTP_503_SERVICE_UNAVAILABLE)
    else:
        return Response(content="Model is loading", status_code=status.HTTP_503_SERVICE_UNAVAILABLE)


class GenerateRequest(BaseModel):
  theme:str
  mode:str="story"
  temperature:float=0.8
  top_k:int=50
  top_p:float=0.95
  max_new_tokens:int=150
  
  
@app.post("/generate")
def generate(req: GenerateRequest):
    if model_error:
        return Response(content=f"Model failed to load: {model_error}", status_code=status.HTTP_503_SERVICE_UNAVAILABLE)
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