import pytest

from src.model.gemini import GeminiAPI


class TestGemini:
    @pytest.mark.skip(reason="Skip test to avoid consuming Google API credits")
    def test_gemini(self):
        gemini = GeminiAPI()
        response = gemini("What is the capital of France?")
        assert isinstance(response, str)
