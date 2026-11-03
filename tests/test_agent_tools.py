"""Isolated tests for inventory tool schemas and HTTP dispatch."""

import json
from http.client import HTTPException
from io import BytesIO
from urllib.error import HTTPError, URLError

import pytest

import agent
from agent import INVENTORY_TOOLS, execute_inventory_tool


class FakeResponse:
    """Minimal context-managed response compatible with ``urlopen``."""

    def __init__(self, status: int = 200, body: object = None) -> None:
        self.status = status
        self._body = json.dumps(body).encode("utf-8")

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self._body


def test_exactly_four_tools_have_complete_openai_function_schemas() -> None:
    functions = [tool["function"] for tool in INVENTORY_TOOLS]

    assert [function["name"] for function in functions] == [
        "list_inventory",
        "create_product",
        "update_stock",
        "get_low_stock_alerts",
    ]
    assert all(tool["type"] == "function" for tool in INVENTORY_TOOLS)
    assert all(function["description"] for function in functions)
    assert all(function["parameters"]["type"] == "object" for function in functions)
    assert functions[0]["parameters"]["properties"] == {}
    assert functions[1]["parameters"]["required"] == ["name", "quantity", "unit"]
    assert functions[2]["parameters"]["required"] == ["product_id", "delta"]
    assert functions[3]["parameters"]["properties"]["threshold"]["type"] == "integer"
    assert all(function["parameters"]["additionalProperties"] is False for function in functions)


@pytest.mark.parametrize(
    ("name", "arguments", "method", "path", "payload"),
    [
        ("list_inventory", {}, "GET", "/inventory", None),
        (
            "create_product",
            {"name": "Apples", "quantity": 5, "unit": "kg"},
            "POST",
            "/inventory",
            {"name": "Apples", "quantity": 5, "unit": "kg"},
        ),
        (
            "update_stock",
            {"product_id": "item/42", "delta": -2},
            "PATCH",
            "/inventory/item%2F42",
            {"delta": -2},
        ),
    ],
)
def test_tool_dispatch_uses_fixed_method_path_and_payload(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    arguments: dict[str, object],
    method: str,
    path: str,
    payload: dict[str, object] | None,
) -> None:
    captured: dict[str, object] = {}

    def fake_urlopen(request, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return FakeResponse(body={"ok": True})

    monkeypatch.setattr(agent, "urlopen", fake_urlopen)

    result = execute_inventory_tool(name, arguments, api_base_url="http://inventory.test/")

    request = captured["request"]
    assert request.get_method() == method
    assert request.full_url == f"http://inventory.test{path}"
    assert captured["timeout"] > 0
    if payload is None:
        assert request.data is None
    else:
        assert json.loads(request.data) == payload
    assert result == {"ok": True, "status_code": 200, "data": {"ok": True}}


def test_low_stock_tool_encodes_threshold_as_query_parameter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    def fake_urlopen(request, timeout):
        captured["request"] = request
        return FakeResponse(body=[])

    monkeypatch.setattr(agent, "urlopen", fake_urlopen)

    result = execute_inventory_tool("get_low_stock_alerts", {"threshold": 4})

    assert captured["request"].get_method() == "GET"
    assert captured["request"].full_url.endswith("/inventory/alerts?threshold=4")
    assert result["data"] == []


def test_default_threshold_omits_query_parameter(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    def fake_urlopen(request, timeout):
        captured["request"] = request
        return FakeResponse(body=[])

    monkeypatch.setattr(agent, "urlopen", fake_urlopen)

    execute_inventory_tool("get_low_stock_alerts", {})

    assert captured["request"].full_url == "http://127.0.0.1:8000/inventory/alerts"


def test_unknown_tool_is_rejected_without_http_request(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(agent, "urlopen", lambda *args, **kwargs: pytest.fail("unexpected request"))

    result = execute_inventory_tool("delete_everything", {"url": "https://example.test"})

    assert result["ok"] is False
    assert result["error"]["type"] == "unknown_tool"


@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        ("create_product", {"name": "Apple", "unit": "kg"}),
        ("create_product", {"name": " ", "quantity": 1, "unit": "kg"}),
        ("create_product", {"name": "Apple", "quantity": True, "unit": "kg"}),
        ("update_stock", {"product_id": "p1"}),
        ("update_stock", {"product_id": "p1", "delta": 1.5}),
        ("get_low_stock_alerts", {"threshold": -1}),
        ("list_inventory", {"path": "/arbitrary"}),
    ],
)
def test_missing_invalid_or_extra_arguments_are_rejected_before_http(
    monkeypatch: pytest.MonkeyPatch, name: str, arguments: dict[str, object]
) -> None:
    monkeypatch.setattr(agent, "urlopen", lambda *args, **kwargs: pytest.fail("unexpected request"))

    result = execute_inventory_tool(name, arguments)

    assert result["ok"] is False
    assert result["error"]["type"] == "invalid_arguments"


@pytest.mark.parametrize("arguments", ["{not json", "[]", 42])
def test_malformed_tool_arguments_are_returned_as_validation_errors(
    arguments: object,
) -> None:
    result = execute_inventory_tool("list_inventory", arguments)

    assert result["ok"] is False
    assert result["error"]["type"] == "invalid_arguments"


@pytest.mark.parametrize("status_code", [404, 409, 422])
def test_http_errors_preserve_status_and_fastapi_detail(
    monkeypatch: pytest.MonkeyPatch, status_code: int
) -> None:
    error_body = {"detail": "meaningful API validation error"}

    def raise_http_error(request, timeout):
        raise HTTPError(
            request.full_url,
            status_code,
            "API error",
            hdrs=None,
            fp=BytesIO(json.dumps(error_body).encode()),
        )

    monkeypatch.setattr(agent, "urlopen", raise_http_error)

    result = execute_inventory_tool("list_inventory", {})

    assert result == {
        "ok": False,
        "error": {
            "type": "http_error",
            "message": f"Inventory API returned HTTP {status_code}",
            "status_code": status_code,
            "detail": error_body,
        },
    }


@pytest.mark.parametrize(
    ("exception", "expected_type"),
    [
        (URLError("connection refused"), "connection_error"),
        (URLError(TimeoutError("request expired")), "timeout"),
        (TimeoutError("request expired"), "timeout"),
    ],
)
def test_connection_failures_and_timeouts_are_structured(
    monkeypatch: pytest.MonkeyPatch, exception: Exception, expected_type: str
) -> None:
    def raise_error(request, timeout):
        raise exception

    monkeypatch.setattr(agent, "urlopen", raise_error)

    result = execute_inventory_tool("list_inventory", {})

    assert result["ok"] is False
    assert result["error"]["type"] == expected_type
    json.dumps(result)


def test_results_are_json_serializable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(agent, "urlopen", lambda request, timeout: FakeResponse(body=[{"id": "p1"}]))

    result = execute_inventory_tool("list_inventory", "{}")

    assert json.loads(json.dumps(result)) == result


def test_api_base_url_can_be_configured_through_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        return FakeResponse(body=[])

    monkeypatch.setattr(agent, "urlopen", fake_urlopen)
    monkeypatch.setenv("INVENTORY_API_BASE_URL", "http://localhost:9999")

    execute_inventory_tool("list_inventory", {})

    assert captured["url"] == "http://localhost:9999/inventory"


@pytest.mark.parametrize(
    "base_url",
    [
        "https://example.test/?target=elsewhere",
        "file:///tmp/inventory",
        "http://user:password@example.test",
        "http://invalid host",
    ],
)
def test_api_base_url_cannot_override_fixed_http_request_target(
    monkeypatch: pytest.MonkeyPatch, base_url: str
) -> None:
    monkeypatch.setattr(agent, "urlopen", lambda *args, **kwargs: pytest.fail("unexpected request"))

    result = execute_inventory_tool("list_inventory", {}, api_base_url=base_url)

    assert result["ok"] is False
    assert result["error"]["type"] == "invalid_configuration"


def test_transport_level_http_protocol_failure_is_structured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        agent,
        "urlopen",
        lambda *args, **kwargs: (_ for _ in ()).throw(HTTPException("protocol error")),
    )

    result = execute_inventory_tool("list_inventory", {})

    assert result["ok"] is False
    assert result["error"]["type"] == "connection_error"
