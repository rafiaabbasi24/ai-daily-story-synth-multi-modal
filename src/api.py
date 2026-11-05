import os
import base64
import io
import logging
from typing import List, Optional

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from transformers import AutoModelForCausalLM, AutoTokenizer
from diffusers import StableDiffusionPipeline
from jinja2 import Environment, FileSystemLoader, select_autoescape

# --------------------------------------------------------------------------- #
# Configuration & Logging
# --------------------------------------------------------------------------- #

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("StorySynthAPI")

# Model identifiers can be overridden with environment variables.
TEXT_MODEL_ID = os.getenv(
    "STORY_SYNTH_TEXT_MODEL",
    "gpt2",  # lightweight default; replace with a larger model in production
)
IMAGE_MODEL_ID = os.getenv(
    "STORY_SYNTH_IMAGE_MODEL",
    "runwayml/stable-diffusion-v1-5",
)

# --------------------------------------------------------------------------- #
# FastAPI App
# --------------------------------------------------------------------------- #

app = FastAPI(
    title="StorySynth: Multi‑Modal Narrative Generator",
    description=(
        "Generate a short story and accompanying images based on a textual prompt. "
        "The endpoint returns the story text and base64‑encoded PNG images."
    ),
    version="0.1.0",
)

# --------------------------------------------------------------------------- #
# Pydantic Schemas
# --------------------------------------------------------------------------- #


class GenerateRequest(BaseModel):
    """
    Request payload for the /generate endpoint.
    """

    prompt: str = Field(..., description="User supplied prompt that seeds the story.")
    max_length: Optional[int] = Field(
        150,
        ge=10,
        le=1024,
        description="Maximum number of tokens for the generated story.",
    )
    num_images: Optional[int] = Field(
        1,
        ge=1,
        le=4,
        description="Number of images to generate (1‑4 recommended).",
    )
    image_width: Optional[int] = Field(
        512,
        ge=128,
        le=1024,
        description="Width of generated images in pixels.",
    )
    image_height: Optional[int] = Field(
        512,
        ge=128,
        le=1024,
        description="Height of generated images in pixels.",
    )

    @validator("prompt")
    def prompt_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Prompt must not be empty.")
        return v


class GenerateResponse(BaseModel):
    """
    Response payload containing the story and generated images.
    """

    story: str = Field(..., description="Generated story text.")
    images: List[str] = Field(
        ..., description="Base64‑encoded PNG images representing the story."
    )
    html: Optional[str] = Field(
        None,
        description="Optional HTML rendering of the story using a Jinja2 template.",
    )


# --------------------------------------------------------------------------- #
# Model Loading (executed once at import)
# --------------------------------------------------------------------------- #

def _load_text_model() -> tuple[AutoTokenizer, AutoModelForCausalLM]:
    """
    Load the tokenizer and language model for story generation.
    """
    logger.info("Loading text model %s", TEXT_MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(TEXT_MODEL_ID)
    # Use GPU if available
    if torch.cuda.is_available():
        model.to("cuda")
        logger.info("Text model moved to CUDA")
    else:
        logger.info("CUDA not available; using CPU for text generation")
    model.eval()
    return tokenizer, model


def _load_image_pipeline() -> StableDiffusionPipeline:
    """
    Load the Stable Diffusion pipeline for image generation.
    """
    logger.info("Loading image pipeline %s", IMAGE_MODEL_ID)
    pipeline = StableDiffusionPipeline.from_pretrained(
        IMAGE_MODEL_ID,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    if torch.cuda.is_available():
        pipeline.to("cuda")
        logger.info("Image pipeline moved to CUDA")
    else:
        logger.info("CUDA not available; using CPU for image generation (may be slow)")
    # Reduce memory usage for CPU inference
    pipeline.enable_attention_slicing()
    return pipeline


# Global model instances
TOKENIZER, TEXT_MODEL = _load_text_model()
IMAGE_PIPELINE = _load_image_pipeline()

# Jinja2 environment for optional HTML rendering
TEMPLATE_ENV = Environment(
    loader=FileSystemLoader(searchpath=os.path.join(os.path.dirname(__file__), "templates")),
    autoescape=select_autoescape(["html", "xml"]),
)

# --------------------------------------------------------------------------- #
# Helper Functions
# --------------------------------------------------------------------------- #


def generate_story(prompt: str, max_length: int) -> str:
    """
    Generate a story continuation from a prompt using the loaded language model.

    Args:
        prompt: The seed text for the model.
        max_length: Maximum number of tokens to generate (including prompt tokens).

    Returns:
        A string containing the generated story.
    """
    logger.debug("Generating story for prompt: %s (max_length=%d)", prompt, max_length)
    input_ids = TOKENIZER.encode(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        input_ids = input_ids.to("cuda")
    # The model may generate fewer tokens than max_length if it predicts EOS.
    with torch.no_grad():
        output_ids = TEXT_MODEL.generate(
            input_ids,
            max_length=max_length,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=TOKENIZER.eos_token_id,
        )
    story = TOKENIZER.decode(output_ids[0], skip_special_tokens=True)
    logger.debug("Generated story length: %d characters", len(story))
    return story


def generate_images(
    prompt: str,
    num_images: int,
    width: int,
    height: int,
) -> List[str]:
    """
    Generate images from a textual prompt using Stable Diffusion.

    Args:
        prompt: Text prompt describing the desired image.
        num_images: Number of images to generate.
        width: Desired image width in pixels.
        height: Desired image height in pixels.

    Returns:
        A list of base64‑encoded PNG images.
    """
    logger.debug(
        "Generating %d image(s) for prompt: %s (size=%dx%d)",
        num_images,
        prompt,
        width,
        height,
    )
    images_base64: List[str] = []

    # The pipeline can generate multiple images in a single call.
    generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
    generator.manual_seed(42)  # deterministic output for reproducibility

    try:
        results = IMAGE_PIPELINE(
            prompt=prompt,
            num_inference_steps=50,
            guidance_scale=7.5,
            width=width,
            height=height,
            num_images_per_prompt=num_images,
            generator=generator,
        )
    except Exception as exc:
        logger.exception("Image generation failed")
        raise HTTPException(status_code=500, detail=f"Image generation error: {exc}")

    for img in results.images:
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_bytes = buffered.getvalue()
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")
        images_base64.append(f"data:image/png;base64,{img_b64}")

    logger.debug("Generated %d base64 images", len(images_base64))
    return images_base64


def render_story_html(story: str) -> str:
    """
    Render the story inside a simple HTML template using Jinja2.

    The template file ``story.html`` should exist under ``src/templates``.
    If the template is missing, a minimal fallback HTML is returned.

    Args:
        story: The story text to embed.

    Returns:
        Rendered HTML as a string.
    """
    template_name = "story.html"
    try:
        template = TEMPLATE_ENV.get_template(template_name)
        html = template.render(story=story)
        logger.debug("Rendered HTML using template %s", template_name)
    except Exception as exc:
        logger.warning("Failed to load template %s: %s", template_name, exc)
        # Fallback minimal HTML
        html = f"<html><body><pre>{story}</pre></body></html>"
    return html


# --------------------------------------------------------------------------- #
# API Endpoint
# --------------------------------------------------------------------------- #


@app.post("/generate", response_model=GenerateResponse, tags=["Generation"])
async def generate(request: GenerateRequest) -> GenerateResponse:
    """
    Generate a narrative and accompanying images from a user prompt.

    The endpoint performs three steps:
    1. Generate a story text using a language model.
    2. Generate one or more images that illustrate the story.
    3. Optionally render the story into an HTML snippet.

    Returns:
        JSON containing the story, base64‑encoded images, and optional HTML.
    """
    logger.info(
        "Received generation request: prompt='%s', max_length=%d, num_images=%d",
        request.prompt,
        request.max_length,
        request.num_images,
    )

    # 1️⃣ Story generation
    try:
        story = generate_story(request.prompt, request.max_length)
    except Exception as exc:
        logger.exception("Story generation failed")
        raise HTTPException(status_code=500, detail=f"Story generation error: {exc}")

    # 2️⃣ Image generation
    images = generate_images(
        prompt=request.prompt,
        num_images=request.num_images,
        width=request.image_width,
        height=request.image_height,
    )

    # 3️⃣ Optional HTML rendering
    html = render_story_html(story)

    response = GenerateResponse(story=story, images=images, html=html)
    logger.info("Generation successful: story length=%d, images=%d", len(story), len(images))
    return response


# --------------------------------------------------------------------------- #
# Development server entry point
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )