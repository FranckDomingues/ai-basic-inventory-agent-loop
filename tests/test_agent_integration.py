"""End-to-end tests for agent tool calls through the real inventory API."""

import csv
import json
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.parse import urlsplit
from typing import Any

import pytest
from fastapi.testclient import TestClient

import agent
from api.app import CsvProductStore, app, get_product_store
from conversation_logger import log_interaction


@pytest.fixture(autouse=True)
def isolate_conversation_log(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirect agent integration logging away from the operational CSV."""
    # These tests exercise the real logger, but its destination is test-local.
    monkeypatch.setattr(
        agent,
        "log_interaction",
        lambda actor, message, tool_call="": log_interaction(
            actor, message, tool_call, file_path=tmp_path / "conversation_log.csv"
        ),
    )


class InProcessResponse:
    """Adapt a serialized FastAPI response to urllib's response interface."""

    def __init__(self, status: int, body: bytes) -> None:
        self.status = status
        self._body = body

    def __enter__(self) -> "InProcessResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self._body


class MockLLM:
    """Return deterministic chat-completion responses and capture request history."""

    def __init__(self, *responses: Any) -> None:
        self.responses = list(responses)
        self.requests: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **request: Any) -> Any:
        """Capture a request snapshot and return its next scripted completion."""
        self.requests.append(json.loads(json.dumps(request)))
        if not self.responses:
            raise AssertionError("The agent requested an unexpected extra completion")
        return self.responses.pop(0)


def completion(content: str | None = None, *calls: Any) -> Any:
    """Build the response shape consumed by ``run_agent_turn``."""
    message = SimpleNamespace(content=content, tool_calls=list(calls))
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def tool_call(call_id: str, name: str, arguments: str) -> Any:
    """Build a function call with the same fields exposed by the OpenAI SDK."""
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


@pytest.fixture

def inventory_http(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """Route urllib requests through FastAPI's ASGI test client and isolate CSV.

    Each test receives a fresh temporary product store through the application's
    normal dependency override. The adapter preserves HTTP status and body
    semantics, including urllib's ``HTTPError`` behavior, without opening a
    socket or mocking the inventory dispatcher.
    """
    real_csv = Path(agent.__file__).resolve().parent / "products.csv"
    real_file_before = real_csv.read_bytes() if real_csv.exists() else None
    store = CsvProductStore(tmp_path / "products.csv")
    previous_override = app.dependency_overrides.get(get_product_store)
    app.dependency_overrides[get_product_store] = lambda: store

    with TestClient(app) as client:
        def in_process_urlopen(request, timeout):
            parsed = urlsplit(request.full_url)
            path = parsed.path + (f"?{parsed.query}" if parsed.query else "")
            response = client.request(
                request.get_method(),
                path,
                headers=dict(request.header_items()),
                content=request.data,
            )
            if response.status_code >= 400:
                raise HTTPError(
                    request.full_url,
                    response.status_code,
                    "FastAPI response error",
                    hdrs=response.headers,
                    fp=BytesIO(response.content),
                )
            return InProcessResponse(response.status_code, response.content)

        monkeypatch.setattr(agent, "urlopen", in_process_urlopen)
        try:
            yield store
        finally:
            if previous_override is None:
                app.dependency_overrides.pop(get_product_store, None)
            else:
                app.dependency_overrides[get_product_store] = previous_override
            real_file_after = real_csv.read_bytes() if real_csv.exists() else None
            assert real_file_after == real_file_before


def tool_results(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Decode dispatcher results from the agent's correlated tool messages."""
    return [
        json.loads(message["content"])
        for message in messages
        if message["role"] == "tool"
    ]


def test_agent_lists_empty_inventory_through_real_http_dispatcher(inventory_http) -> None:
    client = MockLLM(
        completion(None, tool_call("empty-list", "list_inventory", "{}")),
        completion("The inventory is empty."),
    )
    messages: list[dict[str, Any]] = []

    answer = agent.run_agent_turn(client, messages, "What is in stock?")

    assert answer == "The inventory is empty."
    assert tool_results(messages) == [{"ok": True, "status_code": 200, "data": []}]
    assert messages[-1] == {"role": "assistant", "content": answer}


def test_agent_creates_product_and_persists_it_in_isolated_csv(inventory_http) -> None:
    client = MockLLM(
        completion(
            None,
            tool_call(
                "create-1",
                "create_product",
                '{"name":"Apples","quantity":5,"unit":"kg"}',
            ),
        ),
        completion("Apples were added."),
    )
    messages: list[dict[str, Any]] = []

    answer = agent.run_agent_turn(client, messages, "Add 5 kg of apples")

    result = tool_results(messages)[0]
    product = result["data"]
    assert answer == "Apples were added."
    assert result["status_code"] == 201
    assert CsvProductStore(inventory_http.file_path).get(product["id"]).name == "Apples"
    with inventory_http.file_path.open(newline="", encoding="utf-8") as csv_file:
        assert list(csv.DictReader(csv_file)) == [
            {"id": product["id"], "name": "Apples", "quantity": "5", "unit": "kg"}
        ]


def test_agent_can_create_then_list_with_two_sequential_calls_in_one_turn(
    inventory_http,
) -> None:
    client = MockLLM(
        completion(
            None,
            tool_call("create", "create_product", '{"name":"Rice","quantity":7,"unit":"bags"}'),
        ),
        completion(None, tool_call("list", "list_inventory", "{}")),
        completion("Rice is in inventory with 7 bags."),
    )
    messages: list[dict[str, Any]] = []

    answer = agent.run_agent_turn(client, messages, "Add rice and check stock")

    assert answer == "Rice is in inventory with 7 bags."
    assert [result["status_code"] for result in tool_results(messages)] == [201, 200]
    listed = tool_results(messages)[1]["data"]
    assert [(product["name"], product["quantity"]) for product in listed] == [("Rice", 7)]
    assert len(client.requests) == 3
    assert client.requests[1]["messages"][-1]["tool_call_id"] == "create"


def test_agent_increases_and_decreases_stock_through_patch_requests(inventory_http) -> None:
    product = inventory_http.create("Flour", 5, "kg")
    client = MockLLM(
        completion(
            None,
            tool_call("increase", "update_stock", f'{{"product_id":"{product.id}","delta":4}}'),
        ),
        completion(
            None,
            tool_call("decrease", "update_stock", f'{{"product_id":"{product.id}","delta":-3}}'),
        ),
        completion("Stock is now 6 kg."),
    )
    messages: list[dict[str, Any]] = []

    answer = agent.run_agent_turn(client, messages, "Add four then use three")

    assert answer == "Stock is now 6 kg."
    results = tool_results(messages)
    assert [result["data"]["quantity"] for result in results] == [9, 6]
    assert inventory_http.get(product.id).quantity == 6


def test_agent_preserves_http_409_for_negative_resulting_stock(inventory_http) -> None:
    product = inventory_http.create("Sugar", 2, "kg")
    client = MockLLM(
        completion(
            None,
            tool_call("overdraw", "update_stock", f'{{"product_id":"{product.id}","delta":-3}}'),
        ),
        completion("That reduction would make stock negative."),
    )
    messages: list[dict[str, Any]] = []

    answer = agent.run_agent_turn(client, messages, "Use three kg of sugar")

    result = tool_results(messages)[0]
    assert answer == "That reduction would make stock negative."
    assert result["ok"] is False
    assert result["error"]["type"] == "http_error"
    assert result["error"]["status_code"] == 409
    assert "cannot be negative" in result["error"]["detail"]["detail"]
    assert inventory_http.get(product.id).quantity == 2


@pytest.mark.parametrize(
    ("arguments", "expected_names"),
    [
        ("{}", {"Apples", "Rice"}),
        ('{"threshold":8}', {"Apples", "Rice"}),
    ],
)
def test_agent_returns_default_and_custom_low_stock_alerts(
    inventory_http, arguments: str, expected_names: set[str]
) -> None:
    inventory_http.create("Apples", 4, "kg")
    inventory_http.create("Rice", 7, "bags")
    inventory_http.create("Coffee", 10, "bags")
    client = MockLLM(
        completion(
            None,
            tool_call("alerts", "get_low_stock_alerts", arguments),
        ),
        completion("Here are the low-stock items."),
    )
    messages: list[dict[str, Any]] = []

    agent.run_agent_turn(client, messages, "Check for low stock")

    result = tool_results(messages)[0]
    assert result["status_code"] == 200
    assert {product["name"] for product in result["data"]} == expected_names


def test_agent_preserves_http_404_for_unknown_product(inventory_http) -> None:
    client = MockLLM(
        completion(
            None,
            tool_call("missing", "update_stock", '{"product_id":"missing-id","delta":1}'),
        ),
        completion("I could not find that product."),
    )
    messages: list[dict[str, Any]] = []

    agent.run_agent_turn(client, messages, "Increase the missing product")

    error = tool_results(messages)[0]["error"]
    assert error["type"] == "http_error"
    assert error["status_code"] == 404
    assert error["detail"] == {"detail": "Product not found"}


@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        ("create_product", '{"name":"","quantity":2,"unit":"kg"}'),
        ("update_stock", '{"product_id":"p1","delta":true}'),
    ],
)
def test_agent_reports_invalid_tool_arguments_without_mutating_csv(
    inventory_http, name: str, arguments: str
) -> None:
    original = inventory_http.file_path.read_bytes() if inventory_http.file_path.exists() else None
    client = MockLLM(
        completion(None, tool_call("invalid", name, arguments)),
        completion("The tool arguments were invalid."),
    )
    messages: list[dict[str, Any]] = []

    answer = agent.run_agent_turn(client, messages, "Perform an invalid inventory update")

    result = tool_results(messages)[0]
    assert answer == "The tool arguments were invalid."
    assert result["ok"] is False
    assert result["error"]["type"] == "invalid_arguments"
    assert (inventory_http.file_path.read_bytes() if inventory_http.file_path.exists() else None) == original


def test_agent_retains_history_between_turns_and_finishes_after_tool_call(
    inventory_http,
) -> None:
    inventory_http.create("Beans", 3, "bags")
    client = MockLLM(
        completion(None, tool_call("turn-one", "list_inventory", "{}")),
        completion("There are 3 bags of beans."),
        completion("Beans are still on the list."),
    )
    messages: list[dict[str, Any]] = []

    first_answer = agent.run_agent_turn(client, messages, "List inventory")
    second_answer = agent.run_agent_turn(client, messages, "Remind me what is listed")

    assert first_answer == "There are 3 bags of beans."
    assert second_answer == "Beans are still on the list."
    assert messages[0] == {"role": "user", "content": "List inventory"}
    assert messages[-2:] == [
        {"role": "user", "content": "Remind me what is listed"},
        {"role": "assistant", "content": second_answer},
    ]
    assert client.requests[1]["messages"][-1]["role"] == "tool"
    assert client.requests[2]["messages"][-2]["content"] == first_answer


def test_cli_executes_real_inventory_tool_and_prints_final_response(
    inventory_http, monkeypatch: pytest.MonkeyPatch
) -> None:
    inventory_http.create("Tea", 2, "boxes")
    client = MockLLM(
        completion(None, tool_call("cli-list", "list_inventory", "{}")),
        completion("There are 2 boxes of tea."),
    )
    monkeypatch.setattr(agent, "create_llm_client", lambda: client)
    inputs = iter(["How much tea is there?", "quit"])
    output: list[str] = []

    agent.run_cli(input_fn=lambda _prompt: next(inputs), output_fn=output.append)

    assert "Assistant: There are 2 boxes of tea." in output
    assert [message["role"] for message in client.requests[0]["messages"]] == ["user"]


def test_integration_uses_temporary_csv_not_project_inventory_file(inventory_http) -> None:
    product = inventory_http.create("Temporary", 1, "item")

    assert inventory_http.file_path.exists()
    assert CsvProductStore(inventory_http.file_path).get(product.id).name == "Temporary"
    assert inventory_http.file_path.name == "products.csv"
    assert inventory_http.file_path.resolve().parent != Path(agent.__file__).resolve().parent
