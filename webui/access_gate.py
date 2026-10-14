"""Optional access gate for the Streamlit WebUI.

The WebUI has no user accounts. Anyone who can reach the port can export API
keys, edit settings and sign in to provider accounts. This module decides how
the UI should behave:

* ``gated``: a password is configured (``MPT_WEBUI_PASSWORD`` or
  ``app.webui_password``); the UI shows only a password prompt until it is
  entered correctly.
* ``open``: no password, but the server only listens on loopback (or the
  operator set ``MPT_WEBUI_ALLOW_UNAUTHENTICATED=1``); everything is available.
* ``restricted``: no password and the server is reachable from other hosts;
  key export/import, the settings editor and provider sign-in are disabled.
"""

from __future__ import annotations

import hmac
import ipaddress
import os

PASSWORD_ENV = "MPT_WEBUI_PASSWORD"
HOST_ENV = "MPT_WEBUI_HOST"
ALLOW_UNAUTHENTICATED_ENV = "MPT_WEBUI_ALLOW_UNAUTHENTICATED"
AUTH_SESSION_KEY = "webui_access_granted"

MODE_GATED = "gated"
MODE_OPEN = "open"
MODE_RESTRICTED = "restricted"


def configured_password(app_config=None) -> str:
    """Return the configured access password, or an empty string."""
    password = os.environ.get(PASSWORD_ENV)
    if not password and app_config is not None:
        password = app_config.get("webui_password", "")
    return str(password or "")


def allow_unauthenticated() -> bool:
    return os.environ.get(ALLOW_UNAUTHENTICATED_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def is_loopback_host(host) -> bool:
    """True only when ``host`` is explicitly a loopback address.

    An empty host means the server listens on every interface (Streamlit's
    default), so it is treated as exposed.
    """
    host = str(host or "").strip().strip("[]").lower()
    if not host:
        return False
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def server_host(streamlit_option=None) -> str:
    """Address the server listens on: Streamlit option first, then env."""
    host = ""
    if callable(streamlit_option):
        try:
            host = str(streamlit_option("server.address") or "")
        except Exception:
            host = ""
    return host.strip() or os.environ.get(HOST_ENV, "").strip()


def access_mode(app_config=None, streamlit_option=None) -> str:
    if configured_password(app_config):
        return MODE_GATED
    if allow_unauthenticated() or is_loopback_host(server_host(streamlit_option)):
        return MODE_OPEN
    return MODE_RESTRICTED


def verify_password(candidate, expected) -> bool:
    """Constant-time password comparison."""
    if not expected:
        return False
    return hmac.compare_digest(
        str(candidate or "").encode("utf-8"), str(expected).encode("utf-8")
    )


def sensitive_features_enabled(mode: str) -> bool:
    """Key export/import, settings editing and provider sign-in."""
    return mode in (MODE_GATED, MODE_OPEN)
