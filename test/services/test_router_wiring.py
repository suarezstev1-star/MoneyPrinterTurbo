"""Route table and auth wiring smoke tests for the real ASGI application."""

import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from app import asgi
from app.config import config
from app.controllers import rate_limit
from app.router import root_api_router

EXPECTED_ROUTES = {
    ("GET", "/ping"),
    ("POST", "/api/v1/videos"),
    ("POST", "/api/v1/subtitle"),
    ("POST", "/api/v1/audio"),
    ("GET", "/api/v1/tasks"),
    ("POST", "/api/v1/scripts"),
}


def _route_table(routes):
    table = set()
    for route in routes:
        if isinstance(route, APIRoute):
            for method in route.methods - {"HEAD", "OPTIONS"}:
                table.add((method, route.path))
    return table


@pytest.fixture
def client():
    original = dict(config.app)
    try:
        yield TestClient(asgi.app)
    finally:
        config.app.clear()
        config.app.update(original)


def test_root_router_exposes_expected_routes_without_duplicates():
    table = _route_table(root_api_router.routes)
    assert EXPECTED_ROUTES <= table
    all_pairs = [
        (method, route.path)
        for route in root_api_router.routes
        if isinstance(route, APIRoute)
        for method in route.methods - {"HEAD", "OPTIONS"}
    ]
    assert len(all_pairs) == len(set(all_pairs)), "duplicate method/path registered"
    # The application serves exactly the routes the router declares.
    assert table <= _route_table(asgi.app.routes)


def test_ping_is_public_regardless_of_api_key(client):
    config.app["api_key"] = ""
    assert client.get("/ping").json() == "pong"
    config.app["api_key"] = "secret"
    assert client.get("/ping").json() == "pong"


def test_every_v1_route_is_guarded_when_key_is_configured(client):
    """No /api/v1 route may be reachable without credentials once a key is set."""
    config.app["api_key"] = "secret"
    v1_routes = [
        route
        for route in asgi.app.routes
        if isinstance(route, APIRoute) and route.path.startswith("/api/v1")
    ]
    assert v1_routes
    for route in v1_routes:
        # Path parameters get a placeholder; auth must reject before any lookup.
        path = (
            route.path.replace("{file_path:path}", "x")
            .replace("{task_id}", "00000000-0000-0000-0000-000000000000")
            .replace("{", "")
            .replace("}", "")
        )
        for method in route.methods - {"HEAD", "OPTIONS"}:
            # Every request here is deliberately unauthenticated; reset the
            # failed-authentication throttle so it cannot answer 429 first.
            rate_limit.reset_all()
            response = client.request(method, path)
            assert response.status_code == 401, (method, route.path)


def test_wrong_key_rejected_and_right_key_reaches_handler(client):
    config.app["api_key"] = "secret"
    assert client.get("/api/v1/tasks", headers={"x-api-key": "nope"}).status_code == 401
    assert client.get("/api/v1/tasks", headers={"x-api-key": "secret"}).status_code == 200


def test_unset_key_leaves_v1_routes_open(client):
    config.app["api_key"] = ""
    assert client.get("/api/v1/tasks").status_code == 200


def test_unknown_route_is_404_when_open(client):
    config.app["api_key"] = ""
    assert client.get("/api/v1/definitely-missing").status_code == 404
