import pytest


@pytest.fixture(autouse=True)
def _no_llm_retry_sleep(monkeypatch):
    """Keep LLM retry backoff from slowing down tests."""
    from app.services import llm

    monkeypatch.setattr(llm, "_sleep", lambda seconds: None)
