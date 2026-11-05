# StorySynth: Multi‑Modal Narrative Generator

## 📖 Project Overview

**StorySynth** is a production‑ready, open‑source platform that generates rich, multi‑modal narratives (text, images, and audio) from high‑level prompts. It combines large language models, diffusion image generators, and optional text‑to‑speech pipelines into a single FastAPI service, enabling developers to embed story generation capabilities into games, educational tools, and creative applications.

### Key Features
- **Unified API** – One endpoint to generate story text, accompanying illustrations, and optional narration.
- **Modular Architecture** – Swap out language models, diffusion pipelines, or TTS engines without code changes.
- **Template‑Driven Rendering** – Use Jinja2 templates to format stories into HTML, Markdown, or custom layouts.
- **Streaming Responses** – Real‑time token streaming for low‑latency UI experiences.
- **Docker‑Ready** – Official Dockerfile for reproducible deployments.
- **Extensive Tests** – 100 % unit‑test coverage for core generation logic.

## 🏗️ Tech Stack

| Layer | Library / Framework |
|-------|----------------------|
| **API** | FastAPI, Uvicorn |
| **LLM** | 🤗 Transformers (e.g., `gpt2`, `opt-1.3b`) |
| **Image** | Diffusers (Stable Diffusion v1‑5) |
| **TTS (optional)** | TTS (Coqui) |
| **Templating** | Jinja2 |
| **GPU Acceleration** | PyTorch (CUDA) |
| **Testing** | pytest, httpx |
| **Containerisation** | Docker |

## 📦 Installation

### Prerequisites
- Python ≥ 3.9
- CUDA‑enabled GPU (optional but recommended for diffusion)
- `git` (to clone the repository)

### Step‑by‑step

```bash
# 1. Clone the repo
git clone https://github.com/yourorg/StorySynth.git
cd StorySynth

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. (Optional) Install CUDA‑specific torch if you have a GPU
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 5. Verify installation
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

### Docker deployment

```bash
docker build -t storiesynth:latest .
docker run -p 8000:8000 storiesynth:latest
```

## 🚀 Quick Start

### Run the API locally

```bash
uvicorn storiesynth.main:app --host 0.0.0.0 --port 8000 --reload
```

The service will be reachable at `http://localhost:8000`.

### Generate a story (curl)

```bash
curl -X POST http://localhost:8000/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
        "prompt": "A brave rabbit discovers a hidden city of clocks.",
        "max_text_tokens": 300,
        "num_images": 2,
        "tts": true,
        "template": "default.html"
      }' \
  -o story_bundle.zip
```

The response is a ZIP archive containing:
- `story.txt` – generated narrative
- `image_0.png`, `image_1.png` – illustrations
- `narration.wav` – optional audio narration
- `story.html` – rendered using the selected Jinja2 template

### Python client example

```python
import requests, zipfile, io

payload = {
    "prompt": "A shy dragon learns to paint.",
    "max_text_tokens": 250,
    "num_images": 1,
    "tts": False,
    "template": "markdown.md"
}

resp = requests.post("http://localhost:8000/v1/generate", json=payload)
resp.raise_for_status()

with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
    print(z.read("story.txt").decode())
    # Save assets if needed
    z.extractall("output")
```

## 🛠️ API Reference

### `POST /v1/generate`

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `prompt` | `str` | High‑level story idea. | – |
| `max_text_tokens` | `int` | Max tokens for the generated narrative. | `256` |
| `num_images` | `int` | Number of illustrations to generate (0‑4). | `1` |
| `tts` | `bool` | Whether to synthesize narration audio. | `false` |
| `template` | `str` | Jinja2 template filename (located in `templates/`). | `"default.html"` |
| `seed` | `int` (optional) | Random seed for reproducibility. | Random |

**Response** – `application/zip` containing the assets described above. Errors are returned as JSON with HTTP status codes.

### Streaming endpoint (experimental)

`GET /v1/stream?prompt=...` – Returns a Server‑Sent Events (SSE) stream of generated tokens. Useful for UI typing effects.

## 🧩 Architecture Overview

```
storiesynth/
├─ api/                # FastAPI routers
│   └─ v1.py
├─ core/
│   ├─ llm.py          # Text generation wrapper
│   ├─ diffusion.py    # Image generation wrapper
│   └─ tts.py          # Optional narration wrapper
├─ templates/         # Jinja2 story templates
├─ utils/
│   └─ zip_builder.py  # Assemble response ZIP
├─ main.py             # FastAPI app entry point
└─ tests/              # pytest suite
```

Each component adheres to a **provider interface** (`generate(prompt, **kwargs)`) allowing easy swapping of models (e.g., replace Stable Diffusion with DALL·E).

## 🧪 Testing

Run the full test suite with:

```bash
pytest -vv
```

Key test categories:
- **Unit** – LLM token handling, prompt sanitisation, template rendering.
- **Integration** – End‑to‑end generation pipeline (mocked diffusion for CI speed).
- **Performance** – Simple benchmark script (`scripts/benchmark.py`).

## 🤝 Contributing

1. Fork the repository.
2. Create a feature branch (`git checkout -b feat/your-feature`).
3. Write tests for new functionality.
4. Ensure `black`, `isort`, and `flake8` pass (`make lint`).
5. Submit a Pull Request.

Please read `CONTRIBUTING.md` for detailed guidelines and the code of conduct.

## 📄 License

This project is licensed under the **Apache License 2.0**. See `LICENSE` for the full text.

---

*Happy storytelling!*