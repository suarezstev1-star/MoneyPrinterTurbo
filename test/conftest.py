import pytest

from app.controllers import rate_limit


@pytest.fixture(autouse=True)
def _webui_allow_unauthenticated(monkeypatch):
    """Run WebUI tests in the unrestricted mode.

    Without a password, a WebUI whose listen address is not loopback disables
    settings and key export/import. Streamlit's AppTest has no listen address,
    so tests opt in explicitly; tests of the gate itself override this.
    """
    monkeypatch.setenv("MPT_WEBUI_ALLOW_UNAUTHENTICATED", "1")
    monkeypatch.delenv("MPT_WEBUI_PASSWORD", raising=False)


@pytest.fixture(autouse=True)
def _reset_api_rate_limits():
    """In-process limiter state must not leak between tests."""
    rate_limit.reset_all()
    yield
    rate_limit.reset_all()
