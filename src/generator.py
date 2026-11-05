import os
import io
import base64
import logging
from typing import List, Dict, Any, Optional

import torch
from torch import cuda
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from diffusers.utils import export_to_gif
from PIL import Image
from jinja2 import Environment, FileSystemLoader, select_autoescape, TemplateNotFound
from fastapi import HTTPException

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class TextGenerator:
    """
    Wrapper around a causal language model from 🤗 Transformers for story generation.
    The model and tokenizer are loaded once per process and reused for subsequent calls.
    """

    def __init__(
        self,
        model_name: str = "gpt2-medium",
        device: Optional[str] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> None:
        """
        Initialise the generator.

        Args:
            model_name: HuggingFace model identifier.
            device: ``"cpu"``, ``"cuda"``, or ``None`` (auto‑detect).
            max_new_tokens: Upper bound on tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling probability.
        """
        self.device = device or ("cuda" if cuda.is_available() else "cpu")
        logger.info("Loading text model %s on %s", model_name, self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)

        self.generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )

    def generate(self, prompt: str) -> str:
        """
        Generate a continuation for ``prompt``.

        Args:
            prompt: Text prompt to condition the model.

        Returns:
            Generated text (prompt + continuation) stripped of leading/trailing whitespace.
        """
        if not prompt:
            raise ValueError("Prompt must be a non‑empty string.")

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                generation_config=self.generation_config,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return generated.strip()


class ImageGenerator:
    """
    Wrapper around a diffusion model (Stable Diffusion) from 🤗 Diffusers.
    Supports both text‑to‑image and img2img generation.
    """

    def __init__(
        self,
        model_name: str = "runwayml/stable-diffusion-v1-5",
        device: Optional[str] = None,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 50,
    ) -> None:
        """
        Initialise the diffusion pipeline.

        Args:
            model_name: HuggingFace model identifier.
            device: ``"cpu"``, ``"cuda"``, or ``None`` (auto‑detect).
            guidance_scale: Classifier‑free guidance strength.
            num_inference_steps: Number of denoising steps.
        """
        self.device = device or ("cuda" if cuda.is_available() else "cpu")
        logger.info("Loading image model %s on %s", model_name, self.device)

        self.pipeline = StableDiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            use_auth_token=os.getenv("HF_TOKEN", None),
        ).to(self.device)

        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps

    def generate(
        self,
        prompt: str,
        height: int = 512,
        width: int = 512,
        seed: Optional[int] = None,
    ) -> Image.Image:
        """
        Generate a single image conditioned on ``prompt``.

        Args:
            prompt: Text prompt for the diffusion model.
            height: Height of the generated image.
            width: Width of the generated image.
            seed: Optional random seed for reproducibility.

        Returns:
            A ``PIL.Image`` instance.
        """
        generator = torch.Generator(device=self.device)
        if seed is not None:
            generator.manual_seed(seed)

        result = self.pipeline(
            prompt=prompt,
            height=height,
            width=width,
            guidance_scale=self.guidance_scale,
            num_inference_steps=self.num_inference_steps,
            generator=generator,
        )
        return result.images[0]


def image_to_base64(img: Image.Image, fmt: str = "PNG") -> str:
    """
    Encode a ``PIL.Image`` to a base64 string suitable for JSON transport.

    Args:
        img: Image to encode.
        fmt: Output format (default ``PNG``).

    Returns:
        Base64‑encoded string (data URI without the ``data:image/...`` prefix).
    """
    buffered = io.BytesIO()
    img.save(buffered, format=fmt)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


class StorySynthesizer:
    """
    High‑level orchestrator that creates a narrative (text) and accompanying images.
    Utilises Jinja2 for templating so that the final story can be rendered as HTML or markdown.
    """

    def __init__(
        self,
        template_dir: str = "templates",
        template_name: str = "story.html",
        text_model_name: str = "gpt2-medium",
        image_model_name: str = "runwayml/stable-diffusion-v1-5",
    ) -> None:
        """
        Initialise the synthesizer.

        Args:
            template_dir: Directory containing Jinja2 templates.
            template_name: Template file used for final rendering.
            text_model_name: HuggingFace identifier for the language model.
            image_model_name: HuggingFace identifier for the diffusion model.
        """
        self.env = Environment(
            loader=FileSystemLoader(template_dir),
            autoescape=select_autoescape(["html", "xml"]),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        try:
            self.template = self.env.get_template(template_name)
        except TemplateNotFound as exc:
            raise HTTPException(status_code=500, detail=f"Template not found: {template_name}") from exc

        self.text_generator = TextGenerator(model_name=text_model_name)
        self.image_generator = ImageGenerator(model_name=image_model_name)

    def synthesize(
        self,
        prompt: str,
        image_prompts: Optional[List[str]] = None,
        image_count: int = 1,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Generate a story and a set of images.

        Args:
            prompt: Seed prompt for the story generation.
            image_prompts: Optional list of prompts for each image. If ``None``,
                the generated story text is used as a fallback for each image.
            image_count: Number of images to generate (ignored if ``image_prompts`` is provided).
            seed: Optional seed for reproducibility (applied to both text and image models).

        Returns:
            Dictionary with keys ``story`` (str) and ``images`` (list of base64 strings).
        """
        # 1️⃣ Generate story text
        story_text = self.text_generator.generate(prompt)

        # 2️⃣ Determine image prompts
        if image_prompts is None:
            image_prompts = [story_text] * image_count
        elif len(image_prompts) != image_count:
            raise ValueError("Length of image_prompts must equal image_count.")

        # 3️⃣ Generate images
        images_b64: List[str] = []
        for idx, img_prompt in enumerate(image_prompts):
            img_seed = (seed + idx) if seed is not None else None
            img = self.image_generator.generate(prompt=img_prompt, seed=img_seed)
            images_b64.append(image_to_base64(img))

        # 4️⃣ Render final HTML (or markdown) using Jinja2
        rendered = self.template.render(story=story_text, images=images_b64)

        return {"story": story_text, "images": images_b64, "rendered": rendered}
