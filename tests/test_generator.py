"""tests/test_generator.py
Unit tests for the StorySynth multi‑modal narrative generator.

The tests cover:
* Text generation – verifies that a prompt produces a non‑empty string.
* Image generation – verifies that a description produces a Pillow Image.
* FastAPI endpoint – verifies that the ``/generate`` endpoint returns the
  expected JSON payload with both text and a base64‑encoded image.

The tests are written with ``pytest`` and use FastAPI's ``TestClient``.
All imports reference real, pip‑installable packages.
"""

from __future__ import annotations

import base64
import io
import json
from typing import Any

import pytest
from fastapi.testclient import TestClient
from PIL import Image

# Import the public objects from the StorySynth package.
# The package structure is assumed to follow the conventional layout:
#   storysynth/
#       __init__.py
#       generator.py   – contains ``generate_text`` and ``generate_image``
#       main.py        – creates the FastAPI ``app`` instance
from storysynth.generator import generate_text, generate_image
from storysynth.main import app


@pytest.fixture(scope="module")
def client() -> TestClient:
    """FastAPI test client shared across tests."""
    return TestClient(app)


def test_generate_text_returns_string() -> None:
    """A simple prompt must yield a non‑empty string."""
    prompt = "A mysterious forest at dusk"
    result = generate_text(prompt)

    assert isinstance(result, str), "Generated text should be a string"
    assert result.strip(), "Generated text should not be empty"


def test_generate_image_returns_pillow_image() -> None:
    """A description must yield a Pillow Image instance with non‑zero dimensions."""
    description = "A dragon perched on a snowy mountain"
    img = generate_image(description)

    assert isinstance(img, Image.Image), "Generated image should be a Pillow Image"
    width, height = img.size
    assert width > 0 and height > 0, "Image dimensions must be greater than zero"


def test_api_generate_endpoint_success(client: TestClient) -> None:
    """
    The ``/generate`` endpoint must return a JSON payload containing:
    * ``text`` – a non‑empty string
    * ``image`` – a base64‑encoded PNG image
    """
    payload = {"prompt": "A futuristic city skyline at sunrise"}
    response = client.post("/generate", json=payload)

    # Basic HTTP contract
    assert response.status_code == 200, f"Unexpected status: {response.status_code}"
    assert response.headers["content-type"] == "application/json"

    data: dict[str, Any] = response.json()
    assert "text" in data, "Response JSON must contain a 'text' field"
    assert "image" in data, "Response JSON must contain an 'image' field"

    # Validate the text field
    text = data["text"]
    assert isinstance(text, str), "'text' should be a string"
    assert text.strip(), "'text' should not be empty"

    # Validate the image field – it should be a base64‑encoded PNG
    img_b64 = data["image"]
    assert isinstance(img_b64, str), "'image' should be a base64 string"

    # Decode and verify that the bytes represent a valid PNG image
    try:
        img_bytes = base64.b64decode(img_b64, validate=True)
        img = Image.open(io.BytesIO(img_bytes))
        img.verify()  # Will raise if the image is corrupted
    except Exception as exc:
        pytest.fail(f"Failed to decode/validate image from API response: {exc}")


def test_generate_image_is_png_encoded(client: TestClient) -> None:
    """
    Ensure that the image returned by the API is encoded as PNG.
    The test decodes the base64 payload and checks the first eight bytes
    (the PNG signature).
    """
    payload = {"prompt": "A serene lake surrounded by mountains"}
    response = client.post("/generate", json=payload)
    data = response.json()
    img_b64 = data["image"]

    img_bytes = base64.b64decode(img_b64, validate=True)
    png_signature = b"\x89PNG\r\n\x1a\n"
    assert img_bytes.startswith(png_signature), "Image data should start with PNG signature"


def test_generate_text_consistency(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verify that ``generate_text`` uses the underlying model consistently.
    By monkey‑patching the internal model call we can assert that the function
    forwards the prompt correctly and returns the mocked output.
    """
    mock_output = "Once upon a time, in a faraway land..."

    # Assume the generator module internally calls ``_model_generate``.
    # The exact name may differ; adjust according to the actual implementation.
    def fake_model_generate(prompt: str) -> str:
        assert prompt == "A lonely lighthouse", "Prompt was not forwarded correctly"
        return mock_output

    monkeypatch.setattr("storysynth.generator._model_generate", fake_model_generate)

    result = generate_text("A lonely lighthouse")
    assert result == mock_output, "generate_text should return the model's output unchanged"


def test_generate_image_consistency(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verify that ``generate_image`` forwards the description to the diffusion model.
    The test replaces the heavy diffusion pipeline with a lightweight stub that
    returns a deterministic 1×1 red pixel image.
    """
    # Create a deterministic 1x1 red image.
    red_pixel = Image.new("RGB", (1, 1), color=(255, 0, 0))

    def fake_diffusion_generate(description: str) -> Image.Image:
        assert description == "A red pixel", "Description was not forwarded correctly"
        return red_pixel

    # Patch the internal helper used by ``generate_image``.
    monkeypatch.setattr("storysynth.generator._diffusion_generate", fake_diffusion_generate)

    img = generate_image("A red pixel")
    assert isinstance(img, Image.Image), "Result should be a Pillow Image"
    assert img.size == (1, 1), "Image size should match the stubbed output"
    assert img.getpixel((0, 0)) == (255, 0, 0), "Pixel color should be red as defined in the stub"
