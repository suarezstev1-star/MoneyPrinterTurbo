import pytest

from app.controllers import rate_limit


@pytest.fixture(autouse=True)
def _reset_api_rate_limits():
    """In-process limiter state must not leak between tests."""
    rate_limit.reset_all()
    yield
    rate_limit.reset_all()
