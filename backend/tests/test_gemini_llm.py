import pytest
from backend.app.ai.llm import GeminiProvider, OpenAICompatibleProvider, create_llm_provider

def test_gemini_provider_unconfigured():
    provider = GeminiProvider(api_key=None)
    assert not provider.is_configured()

def test_gemini_provider_configured():
    provider = GeminiProvider(api_key="AIzaSyDummyKeyForTestingPurposes123456")
    assert provider.is_configured()
    assert provider.model == "gemini-2.0-flash"

def test_create_llm_provider_factory():
    p1 = create_llm_provider("gemini", api_key="AIzaSyDummyKey123")
    assert isinstance(p1, GeminiProvider)
    
    p2 = create_llm_provider("openai", api_key="sk-dummy123456789")
    assert isinstance(p2, OpenAICompatibleProvider)
    
    p3 = create_llm_provider("auto", api_key="AIzaSyDummyKey123")
    assert isinstance(p3, GeminiProvider)
