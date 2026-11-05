#!/usr/bin/env python3
"""
src/main.py
-----------

Command‑line entry point for **StorySynth**, a multi‑modal narrative generator.

Features
~~~~~~~~
* Generate a story continuation using a HuggingFace language model.
* Optionally synthesize an illustration for the story using Stable Diffusion.
* Render the result as a self‑contained HTML file via Jinja2.
* Optionally launch a minimal FastAPI server exposing the same functionality.

The script is deliberately self‑contained: all heavy‑weight components are lazily
initialised, and the CLI arguments are validated with clear error messages.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import torch
from diffusers import StableDiffusionPipeline
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from jinja2 import Environment, FileSystemLoader, select_autoescape
from pydantic import BaseModel
from transformers import pipeline, set_seed

# --------------------------------------------------------------------------- #
# Logging configuration
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("StorySynth")

# --------------------------------------------------------------------------- #
# Jinja2 environment – templates are stored in ``src/templates``
# --------------------------------------------------------------------------- #
TEMPLATE_DIR = Path(__file__).parent / "templates"
if not TEMPLATE_DIR.is_dir():
    # Fallback to an inline template if the directory does not exist.
    TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)
    (TEMPLATE_DIR / "story.html").write_text(
        """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>StorySynth Output</title>
    <style>
        body {font-family: Arial, sans-serif; margin: 2rem;}
        .image {margin-top: 1rem;}
    </style>
</head>
<body>
    <h1>Generated Story</h1>
    <pre>{{ story_text }}</pre>
    {% if image_path %}
    <div class="image">
        <h2>Illustration</h2>
        <img src="{{ image_path }}" alt="Story illustration" style="max-width:100%;">
    </div>
    {% endif %}
</body>
</html>"""
    )
env = Environment(
    loader=FileSystemLoader(str(TEMPLATE_DIR)),
    autoescape=select_autoescape(["html", "xml"]),
)

# --------------------------------------------------------------------------- #
# Model loading utilities (lazy singletons)
# --------------------------------------------------------------------------- #
_text_generator: Optional[torch.nn.Module] = None
_image_generator: Optional[StableDiffusionPipeline] = None


def get_text_generator(model_name: str = "gpt2", device: str = "cpu"):
    """Return a cached text generation pipeline."""
    global _text_generator
    if _text_generator is None:
        logger.info("Loading text generation model: %s", model_name)
        _text_generator = pipeline(
            "text-generation", model=model_name, device=0 if device == "cuda" else -1
        )
    return _text_generator


def get_image_generator(
    model_name: str = "runwayml/stable-diffusion-v1-5", device: str = "cpu"
):
    """Return a cached Stable Diffusion pipeline."""
    global _image_generator
    if _image_generator is None:
        logger.info("Loading image generation model: %s", model_name)
        _image_generator = StableDiffusionPipeline.from_pretrained(
            model_name, torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )
        _image_generator = _image_generator.to(device)
    return _image_generator


# --------------------------------------------------------------------------- #
# Core generation functions
# --------------------------------------------------------------------------- #
def generate_text(prompt: str, max_length: int = 200, seed: int = 42) -> str:
    """
    Generate a story continuation from *prompt*.

    Parameters
    ----------
    prompt: str
        The seed text for the language model.
    max_length: int
        Maximum number of tokens to generate (including the prompt).
    seed: int
        Random seed for reproducibility.

    Returns
    -------
    str
        Generated text (prompt + continuation) stripped of leading/trailing whitespace.
    """
    generator = get_text_generator()
    set_seed(seed)
    logger.debug("Generating text with max_length=%d", max_length)
    result = generator(prompt, max_new_tokens=max_length, do_sample=True, top_p=0.95)
    generated = result[0]["generated_text"]
    logger.debug("Raw generated text: %s", generated[:200])
    return generated.strip()


def generate_image(prompt: str, output_path: Path, device: str = "cpu") -> Path:
    """
    Generate an illustration for *prompt* and save it to *output_path*.

    Parameters
    ----------
    prompt: str
        Text prompt describing the desired image.
    output_path: Path
        Destination file (PNG) for the rendered image.
    device: str
        ``"cpu"`` or ``"cuda"`` – determines where the model runs.

    Returns
    -------
    Path
        The absolute path to the saved image.
    """
    pipe = get_image_generator(device=device)
    logger.debug("Generating image for prompt: %s", prompt)
    image = pipe(prompt).images[0]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG")
    logger.info("Image saved to %s", output_path)
    return output_path.resolve()


def render_story_html(story_text: str, image_path: Optional[Path] = None) -> str:
    """
    Render the story and optional image into an HTML document.

    Parameters
    ----------
    story_text: str
        The narrative to embed.
    image_path: Optional[Path]
        Path to an illustration; will be referenced relative to the HTML file.

    Returns
    -------
    str
        Rendered HTML.
    """
    template = env.get_template("story.html")
    rel_image = (
        image_path.as_posix()
        if image_path is not None and image_path.is_file()
        else None
    )
    html = template.render(story_text=story_text, image_path=rel_image)
    logger.debug("Rendered HTML length: %d characters", len(html))
    return html


# --------------------------------------------------------------------------- #
# FastAPI server (optional)
# --------------------------------------------------------------------------- #
app = FastAPI(title="StorySynth API")


class GenerationRequest(BaseModel):
    prompt: str
    max_length: int = 200
    seed: int = 42
    generate_image: bool = False


class GenerationResponse(BaseModel):
    story: str
    image_url: Optional[str] = None


@app.post("/generate", response_model=GenerationResponse)
def api_generate(req: GenerationRequest):
    """Endpoint mirroring the CLI behaviour."""
    story = generate_text(req.prompt, max_length=req.max_length, seed=req.seed)

    image_url = None
    if req.generate_image:
        # Use a temporary directory under the current working directory.
        img_path = Path("generated_images") / f"{hash(story)}.png"
        generate_image(req.prompt, img_path, device="cpu")
        image_url = f"/static/{img_path.name}"

    return GenerationResponse(story=story, image_url=image_url)


@app.get("/", response_class=HTMLResponse)
def root():
    return """
    <h1>StorySynth API</h1>
    <p>POST to <code>/generate</code> with JSON payload to receive a story.</p>
    """


# --------------------------------------------------------------------------- #
# CLI implementation
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="storiesynth",
        description="Generate a story (and optional illustration) locally.",
    )
    parser.add_argument(
        "prompt", type=str, help="Seed text for story generation."
    )
    parser.add_argument(
        "-l",
        "--max-length",
        type=int,
        default=200,
        help="Maximum number of tokens to generate (default: 200).",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "-i",
        "--image",
        action="store_true",
        help="Generate an illustration using Stable Diffusion.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("story_output.html"),
        help="Path to the resulting HTML file (default: story_output.html).",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Run a local FastAPI server instead of CLI generation.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host for the FastAPI server (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for the FastAPI server (default: 8000).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Entry point for the ``storiesynth`` command."""
    args = parse_args(argv)

    if args.serve:
        # Import uvicorn lazily to avoid unnecessary dependency when only CLI is used.
        try:
            import uvicorn  # type: ignore
        except ImportError as exc:
            logger.error(
                "Running the server requires 'uvicorn'. Install it via pip."
            )
            raise SystemExit(1) from exc

        logger.info("Starting StorySynth API server at http://%s:%d", args.host, args.port)
        uvicorn.run(app, host=args.host, port=args.port)
        return

    logger.info("Generating story for prompt: %s", args.prompt)
    story = generate_text(args.prompt, max_length=args.max_length, seed=args.seed)

    image_path: Optional[Path] = None
    if args.image:
        logger.info("Generating illustration...")
        image_path = Path("generated_images") / f"{hash(story)}.png"
        generate_image(args.prompt, image_path, device="cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Illustration saved to %s", image_path)

    html_content = render_story_html(story, image_path=image_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html_content, encoding="utf-8")
    logger.info("Story written to %s", args.output.resolve())


if __name__ == "__main__":
    main()