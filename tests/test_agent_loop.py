"""Isolated tests for the manual LLM tool-calling conversation loop."""

import json
from copy import deepcopy
from types import SimpleNamespace
from typing import Any

import pytest

import agent
from agent import AgentLoopError, run_agent_turn


def response(content: str | None = None, tool_calls: list[Any] | None = None):
    """Create the small response shape read by the agent loop."""
    message = SimpleNamespace(content=content, tool_calls=tool_calls or [])
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def tool_call(call_id: str, name: str, arguments: str):
    """Create the small tool-call shape returned by chat completions."""
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


class MockCompletions:
    """Return queued LLM responses and retain each request for assertions."""

    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.requests: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.requests.append(deepcopy(kwargs))
        if not self.responses:
            raise AssertionError("Unexpected additional LLM request")
        next_response = self.responses.pop(0)
        if isinstance(next_response, Exception):
            raise next_response
        return next_response


class MockClient:
    """Expose a minimal OpenAI-compatible chat completions interface."""

    def __init__(self, responses: list[Any]) -> None:
        self.chat = SimpleNamespace(completions=MockCompletions(responses))


def test_final_answer_without_tool_calls_is_returned_and_appended(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_MODEL", "test-model")
    client = MockClient([response("There are 4 apples.")])
    messages = [{"role": "system", "content": "Inventory assistant."}]

    answer = run_agent_turn(client, messages, "How many apples are there?")

    assert answer == "There are 4 apples."
    assert messages[-2:] == [
        {"role": "user", "content": "How many apples are there?"},
        {"role": "assistant", "content": "There are 4 apples."},
    ]
    assert client.chat.completions.requests[0]["model"] == "test-model"
    assert len(client.chat.completions.requests[0]["tools"]) == 4


def test_single_tool_call_is_followed_by_final_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        agent,
        "execute_inventory_tool",
        lambda name, args, **kwargs: {"ok": True, "data": [{"id": "p1"}]},
    )
    client = MockClient(
        [
            response(tool_calls=[tool_call("call-1", "list_inventory", "{}")]),
            response("There is one product."),
        ]
    )
    messages: list[dict[str, Any]] = []

    answer = run_agent_turn(client, messages, "List stock")

    assert answer == "There is one product."
    assert [message["role"] for message in messages] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert messages[1]["tool_calls"][0]["id"] == "call-1"
    assert messages[2]["tool_call_id"] == "call-1"
    assert json.loads(messages[2]["content"]) == {"ok": True, "data": [{"id": "p1"}]}
    assert client.chat.completions.requests[1]["messages"] == messages[:-1]


def test_multiple_tool_calls_are_all_executed_and_correlated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executed: list[tuple[str, str]] = []

    def execute(name: str, arguments: str, **kwargs: Any) -> dict[str, Any]:
        executed.append((name, arguments))
        return {"ok": True, "data": name}

    monkeypatch.setattr(agent, "execute_inventory_tool", execute)
    client = MockClient(
        [
            response(
                tool_calls=[
                    tool_call("call-a", "list_inventory", "{}"),
                    tool_call("call-b", "get_low_stock_alerts", '{"threshold":3}'),
                ]
            ),
            response("Inventory and alerts retrieved."),
        ]
    )
    messages: list[dict[str, Any]] = []

    answer = run_agent_turn(client, messages, "Check inventory and alerts")

    assert answer == "Inventory and alerts retrieved."
    assert executed == [
        ("list_inventory", "{}"),
        ("get_low_stock_alerts", '{"threshold":3}'),
    ]
    assert [message["tool_call_id"] for message in messages if message["role"] == "tool"] == [
        "call-a",
        "call-b",
    ]
    assert [call["id"] for call in messages[1]["tool_calls"]] == ["call-a", "call-b"]


def test_multiple_sequential_tool_iterations_update_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool_results: list[str] = []

    def execute(name: str, arguments: str, **kwargs: Any) -> dict[str, Any]:
        result = {"step": len(tool_results) + 1}
        tool_results.append(json.dumps(result))
        return result

    monkeypatch.setattr(agent, "execute_inventory_tool", execute)
    client = MockClient(
        [
            response(tool_calls=[tool_call("first", "list_inventory", "{}")]),
            response(tool_calls=[tool_call("second", "get_low_stock_alerts", "{}")]),
            response("Done after two checks."),
        ]
    )
    messages: list[dict[str, Any]] = []

    answer = run_agent_turn(client, messages, "Check twice", max_iterations=3)

    assert answer == "Done after two checks."
    assert len(client.chat.completions.requests) == 3
    assert [message["tool_call_id"] for message in messages if message["role"] == "tool"] == [
        "first",
        "second",
    ]
    assert [message["role"] for message in messages] == [
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
        "assistant",
    ]
    assert client.chat.completions.requests[2]["messages"] == messages[:-1]


def test_malformed_arguments_are_returned_to_model_as_tool_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[tuple[str, str]] = []

    def execute(name: str, arguments: str, **kwargs: Any) -> dict[str, Any]:
        observed.append((name, arguments))
        return {"ok": False, "error": {"type": "invalid_arguments", "message": "bad JSON"}}

    monkeypatch.setattr(agent, "execute_inventory_tool", execute)
    client = MockClient(
        [
            response(tool_calls=[tool_call("bad-args", "create_product", "{not json")]),
            response("The product request was malformed."),
        ]
    )
    messages: list[dict[str, Any]] = []

    answer = run_agent_turn(client, messages, "Create a product")

    assert answer == "The product request was malformed."
    assert observed == [("create_product", "{not json")]
    result = json.loads(messages[2]["content"])
    assert result["error"]["type"] == "invalid_arguments"
    assert messages[2]["tool_call_id"] == "bad-args"


def test_unknown_tool_is_dispatched_and_error_is_not_skipped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def execute(name: str, arguments: str, **kwargs: Any) -> dict[str, Any]:
        return {"ok": False, "error": {"type": "unknown_tool", "message": f"Unknown: {name}"}}

    monkeypatch.setattr(agent, "execute_inventory_tool", execute)
    client = MockClient(
        [
            response(tool_calls=[tool_call("unknown-id", "delete_all", "{}")]),
            response("That action is not available."),
        ]
    )
    messages: list[dict[str, Any]] = []

    answer = run_agent_turn(client, messages, "Delete all")

    assert answer == "That action is not available."
    assert messages[2]["tool_call_id"] == "unknown-id"
    assert json.loads(messages[2]["content"])["error"]["type"] == "unknown_tool"


def test_maximum_iteration_limit_stops_before_an_extra_llm_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        agent,
        "execute_inventory_tool",
        lambda *args, **kwargs: {"ok": True, "data": []},
    )
    client = MockClient(
        [
            response(tool_calls=[tool_call("call-1", "list_inventory", "{}")]),
            response(tool_calls=[tool_call("call-2", "list_inventory", "{}")]),
        ]
    )
    messages: list[dict[str, Any]] = []

    with pytest.raises(AgentLoopError, match=r"Maximum LLM iterations \(2\)"):
        run_agent_turn(client, messages, "Keep checking", max_iterations=2)

    assert len(client.chat.completions.requests) == 2
    assert messages[-1]["tool_call_id"] == "call-2"


def test_llm_api_failure_raises_redacted_error_without_false_success() -> None:
    client = MockClient([RuntimeError("authorization failed with secret-token")])
    messages: list[dict[str, Any]] = []

    with pytest.raises(AgentLoopError) as error:
        run_agent_turn(client, messages, "Check stock")

    assert "secret-token" not in str(error.value)
    assert "final answer was produced" in str(error.value)
    assert messages == [{"role": "user", "content": "Check stock"}]


def test_conversation_history_is_retained_across_turns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_MODEL", "test-model")
    client = MockClient([response("First answer."), response("Second answer.")])
    messages: list[dict[str, Any]] = [{"role": "system", "content": "Use inventory tools."}]

    assert run_agent_turn(client, messages, "First question") == "First answer."
    assert run_agent_turn(client, messages, "Second question") == "Second answer."

    assert [message["role"] for message in messages] == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert client.chat.completions.requests[1]["messages"] == messages[:-1]
    assert client.chat.completions.requests[1]["messages"][-2]["content"] == "First answer."


def test_invalid_iteration_limit_is_rejected_without_llm_request() -> None:
    client = MockClient([])

    with pytest.raises(ValueError, match="max_iterations"):
        run_agent_turn(client, [], "Check stock", max_iterations=0)


def test_tool_execution_exception_is_converted_to_correlated_tool_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        agent,
        "execute_inventory_tool",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("execution failed")),
    )
    client = MockClient(
        [
            response(tool_calls=[tool_call("call-fail", "list_inventory", "{}")]),
            response("The inventory service could not be reached."),
        ]
    )
    messages: list[dict[str, Any]] = []

    answer = run_agent_turn(client, messages, "Check stock")

    assert answer == "The inventory service could not be reached."
    assert messages[1]["tool_calls"][0]["id"] == "call-fail"
    assert messages[2]["tool_call_id"] == "call-fail"
    assert json.loads(messages[2]["content"])["error"]["type"] == "tool_execution_error"
