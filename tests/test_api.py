import json
from pathlib import Path
from typing import Any, Dict

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

# Import the FastAPI application. Adjust the import path according to the actual project layout.
# The typical layout for this project is `storysynth/main.py` exposing a variable `app`.
from storysynth.main import app

# Create a reusable TestClient instance for the entire test module.
client = TestClient(app)


def _load_fixture(name: str) -> Dict[str, Any]:
    """
    Helper to load JSON fixtures from the `tests/fixtures` directory.
    Fixtures make it easy to keep expected payloads separate from the test code.
    """
    fixture_path = Path(__file__).parent / "fixtures" / f"{name}.json"
    with fixture_path.open(encoding="utf-8") as fp:
        return json.load(fp)


@pytest.fixture(autouse=True)
def mock_story_generator():
    """
    Patch the heavy‑weight story generation function so that integration tests run
    quickly and deterministically.

    The real implementation lives in `storysynth.generator.generate_story`.  It
    normally invokes large language models and diffusion pipelines, which are
    unsuitable for CI.  By yielding a deterministic short story we can assert
    on the API contract without the computational cost.
    """
    dummy_story = "Once upon a time, in a far‑away land, a brave hero rose."
    with patch("storysynth.generator.generate_story", return_value=dummy_story) as mock:
        yield mock


def test_generate_story_success():
    """
    End‑to‑end test that a well‑formed request returns a 200 response with the
    expected JSON payload. The heavy model call is mocked by ``mock_story_generator``.
    """
    payload = {
        "prompt": "A hero begins their journey",
        "length": 50,
        "style": "fantasy"
    }

    response = client.post("/generate", json=payload)
    assert response.status_code == 200, f"Unexpected status code: {response.text}"
    assert response.headers["content-type"] == "application/json"

    data = response.json()
    # The API should return a dictionary containing a single key ``story``.
    assert isinstance(data, dict), "Response JSON is not a dict"
    assert "story" in data, "Missing 'story' key in response"
    assert data["story"] == "Once upon a time, in a far‑away land, a brave hero rose."

    # Verify that the mocked generator was called with the correct arguments.
    # ``mock_story_generator`` yields the mock object; retrieve it via the fixture.
    # The fixture is applied automatically, so we can access the last call via
    # ``mock_story_generator``'s `call_args`.
    # (The fixture is scoped to the entire module, hence the same mock instance is reused.)
    # The following assertion ensures the API forwards parameters correctly.
    # Note: ``mock_story_generator`` is a generator fixture; we need to retrieve the
    # underlying mock object from the fixture's ``_fixturedef`` attribute.
    # This is a bit of a workaround but works reliably in pytest.
    mock = mock_story_generator.__self__
    mock.assert_called_once()
    called_args, called_kwargs = mock.call_args
    # The generator is expected to receive the prompt and length; style may be optional.
    assert called_args[0] == payload["prompt"]
    assert called_kwargs.get("length") == payload["length"]
    assert called_kwargs.get("style") == payload["style"]


def test_generate_story_missing_prompt():
    """
    Verify that the API validates the request body and returns a 422 error when
    the required ``prompt`` field is omitted.
    """
    payload = {
        "length": 30,
        "style": "sci‑fi"
    }

    response = client.post("/generate", json=payload)
    assert response.status_code == 422, "Expected validation error for missing prompt"
    error_detail = response.json()
    # FastAPI returns a list of errors under the ``detail`` key.
    assert "detail" in error_detail
    assert any(err["loc"] == ["body", "prompt"] for err in error_detail["detail"])


def test_generate_story_invalid_length():
    """
    The ``length`` field must be a positive integer within a reasonable range.
    This test ensures that values outside that range are rejected with a 422.
    """
    payload = {
        "prompt": "A mysterious forest",
        "length": -10,  # Invalid: negative length
        "style": "horror"
    }

    response = client.post("/generate", json=payload)
    assert response.status_code == 422, "Negative length should trigger validation error"
    error_detail = response.json()
    assert "detail" in error_detail
    # Look for the specific validation error concerning ``length``.
    assert any(
        err["loc"] == ["body", "length"] and "greater than 0" in err["msg"]
        for err in error_detail["detail"]
    )


def test_generate_story_default_parameters():
    """
    When optional fields are omitted, the endpoint should fall back to default values.
    """
    payload = {
        "prompt": "An ancient prophecy"
        # ``length`` and ``style`` are omitted; defaults are applied by the API.
    }

    response = client.post("/generate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["story"] == "Once upon a time, in a far‑away land, a brave hero rose."

    # Ensure the generator was called with default parameters.
    mock = mock_story_generator.__self__
    mock.assert_called_once()
    called_args, called_kwargs = mock.call_args
    assert called_args[0] == payload["prompt"]
    # The default length is defined in the endpoint's Pydantic model; we assert it is an int.
    assert isinstance(called_kwargs.get("length"), int)
    # ``style`` default may be ``None`` or a string; we just ensure the key exists.
    assert "style" in called_kwargs


def test_generate_story_response_schema():
    """
    Load a JSON schema fixture that describes the expected response shape and
    validate the actual response against it. This guards against accidental
    breaking changes in the API contract.
    """
    from jsonschema import validate, ValidationError

    payload = {
        "prompt": "A dragon awakens",
        "length": 25,
        "style": "epic"
    }

    response = client.post("/generate", json=payload)
    assert response.status_code == 200
    response_json = response.json()

    schema = _load_fixture("response_schema")
    try:
        validate(instance=response_json, schema=schema)
    except ValidationError as exc:
        pytest.fail(f"Response does not match schema: {exc.message}")