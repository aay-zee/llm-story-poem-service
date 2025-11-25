# LLM Story/Poem Generator Service

A FastAPI-based microservice that generates creative stories and poems using GPT-2 language models. Designed for deployment on Google Cloud Run with automatic scaling and background model loading.

## Features

- 🎭 Generate creative stories or poems from themes
- 🚀 Fast API with automatic interactive documentation
- 🔄 Background model loading (no startup timeout)
- ☁️ Cloud Run optimized with configurable memory/CPU
- 🎛️ Customizable generation parameters (temperature, top_k, top_p)
- 🔌 RESTful API with JSON responses

## Tech Stack

- **Framework:** FastAPI + Uvicorn
- **ML Model:** GPT-2 (Hugging Face Transformers)
- **Deep Learning:** PyTorch
- **Deployment:** Docker + Google Cloud Run

## Quick Start

### Local Development

1. **Install dependencies:**

```bash
pip install -r requirements.txt
```

2. **Run the service:**

```bash
uvicorn main:app --reload --port 8080
```

3. **Access interactive API docs:**
   - Swagger UI: http://localhost:8080/docs
   - ReDoc: http://localhost:8080/redoc

### Docker (Local)

1. **Build the image:**

```bash
docker build -t llm-story-poem-service .
```

2. **Run the container:**

```bash
docker run -p 8080:8080 llm-story-poem-service
```

## API Usage

### Generate Story/Poem

**Endpoint:** `POST /generate`

**Request Body:**

```json
{
  "theme": "a knight on a quest",
  "mode": "story",
  "temperature": 0.8,
  "top_k": 50,
  "top_p": 0.95,
  "max_new_tokens": 150
}
```

**Parameters:**

- `theme` (required): The topic/theme for generation
- `mode` (optional): `"story"` or `"poem"` (default: `"story"`)
- `temperature` (optional): Creativity level 0.0-2.0 (default: `0.8`)
- `top_k` (optional): Token sampling pool size (default: `50`)
- `top_p` (optional): Nucleus sampling threshold (default: `0.95`)
- `max_new_tokens` (optional): Maximum length (default: `150`)

**Example cURL:**

```bash
curl -X POST "http://localhost:8080/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "theme": "a magical forest",
    "mode": "poem",
    "temperature": 0.9,
    "max_new_tokens": 100
  }'
```

**Example PowerShell:**

```powershell
$body = @{
    theme = "a dragon's adventure"
    mode = "story"
    temperature = 0.8
    max_new_tokens = 200
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8080/generate" -Method Post -Body $body -ContentType "application/json"
```

**Response:**

```json
{
  "result": {
    "generated_text": "Once upon a time, in a land far away..."
  }
}
```

## Deployment to Google Cloud Run

### Prerequisites

- Google Cloud CLI installed and authenticated
- Project with Cloud Run API enabled

### Deploy Command

**Recommended configuration (gpt2-medium):**

```bash
gcloud run deploy llm-story-poem-service \
  --source . \
  --region=us-central1 \
  --platform=managed \
  --memory=4Gi \
  --cpu=2 \
  --timeout=300s \
  --allow-unauthenticated \
  --max-instances=10
```

**Budget-friendly configuration (gpt2):**

```bash
gcloud run deploy llm-story-poem-service \
  --source . \
  --region=us-central1 \
  --platform=managed \
  --memory=2Gi \
  --cpu=1 \
  --timeout=300s \
  --allow-unauthenticated \
  --max-instances=5
```

### Resource Requirements

| Model       | Memory | CPU | Estimated Load Time |
| ----------- | ------ | --- | ------------------- |
| gpt2        | 2Gi    | 1   | ~30-60s             |
| gpt2-medium | 4Gi    | 2   | ~60-120s            |
| gpt2-large  | 8Gi    | 4   | ~120-180s           |

## Configuration

### Change Model

Edit `model_loader.py` to use a different model:

```python
def load_model(preferred="gpt2", fallback="gpt2"):
    # Options: "gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"
    # or any compatible Hugging Face model
```

### Environment Variables

- `PORT`: Server port (default: `8080`, auto-set by Cloud Run)

## Architecture

- **Background Loading:** Model loads in a separate thread at startup, allowing the container to become healthy immediately
- **503 Handling:** Returns `503 Service Unavailable` while model is loading
- **Auto-scaling:** Cloud Run scales instances based on traffic
- **Zero-downtime:** Scales to zero when idle (pay-per-use)

## Project Structure

```
.
├── main.py              # FastAPI application & routes
├── generator.py         # Text generation logic
├── model_loader.py      # Model initialization & fallback
├── requirements.txt     # Python dependencies
├── Dockerfile          # Container configuration
└── README.md           # This file
```

## Troubleshooting

### Container fails to start

- Check Cloud Run logs: `gcloud logging read "resource.type=cloud_run_revision"`
- Ensure memory allocation is sufficient for your model
- Verify Dockerfile ENTRYPOINT uses `${PORT:-8080}`

### Model loading timeout

- Increase `--timeout` flag during deployment
- Consider using a smaller model
- Check network connectivity for Hugging Face downloads

### Out of Memory (OOM)

- Reduce model size (use `gpt2` instead of `gpt2-medium`)
- Increase memory allocation with `--memory=8Gi`
- Use `torch_dtype=torch.float16` for GPU inference

## License

MIT

## Contributing

Pull requests welcome! Please ensure code passes linting and includes tests.

---

**Built with ❤️ using FastAPI and Transformers**
