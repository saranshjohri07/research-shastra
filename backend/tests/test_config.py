import pytest
from pydantic import ValidationError

from backend.app.config import Settings


def test_settings_load_with_valid_values():
    settings = Settings(
        groq_api_key="test-key",
        cors_origins="http://localhost:3000,http://localhost:8000",
    )

    assert settings.groq_api_key == "test-key"
    assert settings.groq_model == "openai/gpt-oss-120b"


def test_cors_origins_are_parsed_into_list():
    settings = Settings(
        groq_api_key="test-key",
        cors_origins="http://localhost:3000,http://localhost:8000",
    )

    assert settings.cors_origins == [
        "http://localhost:3000",
        "http://localhost:8000",
    ]


def test_missing_groq_api_key_fails_validation():
    with pytest.raises(ValidationError):
        Settings(_env_file=None)
