"""Integration tests for agent-loop and CLI conversation logging."""

import csv
import json
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import agent
from agent import AgentLoopError, run_agent_turn, run_cli
from conversation_logger import CSV_COLUMNS, log_interaction


def make_response(content: str | None = None, tool_calls: list[Any] | None = None):
    """Build the minimal completion shape consumed by the agent loop."""
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content, tool_calls=tool_calls or [])
            )
        ]
    )


def make_tool_call(call_id: str, name: str, arguments: str = "{}"):
    """Build the minimal tool-call shape returned by the LLM."""
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


class MockClient:
    """Serve queued completion results through the OpenAI-compatible interface."""

    def __init__(self, responses: list[Any]) -> None:
        self.responses = iter(responses)
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self.create)
        )

    def create(self, **_kwargs: Any) -> Any:
        result = next(self.responses)
        if isinstance(result, BaseException):
            raise result
        return result


def make_logger(file_path: Path):
    """Bind the existing logger API to an isolated test destination."""
    return partial(log_interaction, file_path=file_path)


def read_rows(file_path: Path) -> list[dict[str, str]]:
    """Read CSV rows as dictionaries, preserving multiline values."""
    with file_path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        assert reader.fieldnames == list(CSV_COLUMNS)
        return list(reader)


def test_user_and_final_answer_are_each_logged_once(tmp_path: Path) -> None:
    file_path = tmp_path / "conversation.csv"
    messages = [{"role": "system", "content": "Keep context."}]

    answer = run_agent_turn(
        MockClient([make_response("There are 4 apples.")]),
        messages,
        "How many apples?",
        log_callback=make_logger(file_path),
    )

    rows = read_rows(file_path)
    assert answer == "There are 4 apples."
    assert [(row["actor"], row["message"]) for row in rows] == [
        ("user", "How many apples?"),
        ("agent", "There are 4 apples."),
    ]
    assert rows[0]["tool_call"] == rows[1]["tool_call"] == ""
    assert messages[0] == {"role": "system", "content": "Keep context."}


def test_tool_request_and_result_are_logged_with_correlation_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        agent, "execute_inventory_tool", lambda *_args, **_kwargs: {"ok": True, "data": []}
    )
    file_path = tmp_path / "conversation.csv"

    run_agent_turn(
        MockClient(
            [
                make_response(tool_calls=[make_tool_call("call-17", "list_inventory")]),
                make_response("Inventory is empty."),
            ]
        ),
        [],
        "List stock",
        log_callback=make_logger(file_path),
    )

    rows = read_rows(file_path)
    assert [row["actor"] for row in rows] == ["user", "agent", "tool", "agent"]
    request = json.loads(rows[1]["tool_call"])
    result = json.loads(rows[2]["tool_call"])
    assert request == result == {"name": "list_inventory", "id": "call-17"}
    assert json.loads(rows[2]["message"]) == {"ok": True, "data": []}
    assert rows[3]["message"] == "Inventory is empty."


def test_multiple_tool_calls_have_distinct_correlated_rows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        agent,
        "execute_inventory_tool",
        lambda name, *_args, **_kwargs: {"ok": True, "data": name},
    )
    file_path = tmp_path / "conversation.csv"

    run_agent_turn(
        MockClient(
            [
                make_response(
                    "Checking both.",
                    [
                        make_tool_call("id-a", "list_inventory"),
                        make_tool_call("id-b", "get_low_stock_alerts"),
                    ],
                ),
                make_response("Done."),
            ]
        ),
        [],
        "Check stock and alerts",
        log_callback=make_logger(file_path),
    )

    rows = read_rows(file_path)
    assert [row["actor"] for row in rows] == ["user", "agent", "agent", "tool", "tool", "agent"]
    request_metadata = [json.loads(row["tool_call"]) for row in rows[1:3]]
    result_metadata = [json.loads(row["tool_call"]) for row in rows[3:5]]
    assert request_metadata == result_metadata == [
        {"name": "list_inventory", "id": "id-a"},
        {"name": "get_low_stock_alerts", "id": "id-b"},
    ]


def test_sequential_tool_iterations_keep_event_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        agent, "execute_inventory_tool", lambda name, *_args, **_kwargs: {"ok": True, "data": name}
    )
    file_path = tmp_path / "conversation.csv"

    run_agent_turn(
        MockClient(
            [
                make_response(tool_calls=[make_tool_call("first", "list_inventory")]),
                make_response(tool_calls=[make_tool_call("second", "get_low_stock_alerts")]),
                make_response("Finished."),
            ]
        ),
        [],
        "Check twice",
        max_iterations=3,
        log_callback=make_logger(file_path),
    )

    rows = read_rows(file_path)
    assert [row["actor"] for row in rows] == ["user", "agent", "tool", "agent", "tool", "agent"]
    assert [json.loads(row["tool_call"])["id"] for row in rows[1:5]] == [
        "first",
        "first",
        "second",
        "second",
    ]


def test_cli_multi_turn_session_appends_to_same_log(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(agent, "create_llm_client", lambda: None)
    file_path = tmp_path / "conversation.csv"
    client = MockClient([make_response("First answer."), make_response("Second answer.")])
    inputs = iter(["first question", "second question", "exit"])

    run_cli(
        client,
        input_fn=lambda _prompt: next(inputs),
        output_fn=lambda _message: None,
        log_callback=make_logger(file_path),
    )

    rows = read_rows(file_path)
    assert [(row["actor"], row["message"]) for row in rows] == [
        ("user", "first question"),
        ("agent", "First answer."),
        ("user", "second question"),
        ("agent", "Second answer."),
    ]


def test_cli_does_not_log_blank_inputs_or_exit_commands(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(agent, "create_llm_client", lambda: None)
    file_path = tmp_path / "conversation.csv"
    inputs = iter(["", "  ", "quit"])

    run_cli(
        object(),
        input_fn=lambda _prompt: next(inputs),
        output_fn=lambda _message: None,
        log_callback=make_logger(file_path),
    )

    assert not file_path.exists()


@pytest.mark.parametrize(
    "result",
    [
        {"ok": False, "error": {"type": "unknown_tool", "message": "private detail"}},
        {"ok": False, "error": {"type": "invalid_arguments", "message": "token=secret"}},
        RuntimeError("authorization bearer-secret"),
    ],
)
def test_tool_errors_are_logged_without_sensitive_details(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, result: Any
) -> None:
    def execute(*_args: Any, **_kwargs: Any) -> Any:
        if isinstance(result, BaseException):
            raise result
        return result

    monkeypatch.setattr(agent, "execute_inventory_tool", execute)
    file_path = tmp_path / "conversation.csv"

    run_agent_turn(
        MockClient(
            [
                make_response(tool_calls=[make_tool_call("error-id", "list_inventory")]),
                make_response("The tool failed safely."),
            ]
        ),
        [],
        "Check inventory",
        log_callback=make_logger(file_path),
    )

    rows = read_rows(file_path)
    tool_row = next(row for row in rows if row["actor"] == "tool")
    assert "private detail" not in tool_row["message"]
    assert "secret" not in tool_row["message"]
    assert "bearer-secret" not in tool_row["message"]
    assert json.loads(tool_row["message"])["error"]["type"] in {
        "unknown_tool",
        "invalid_arguments",
        "tool_execution_error",
    }


def test_llm_failure_logs_failure_without_false_answer(tmp_path: Path) -> None:
    file_path = tmp_path / "conversation.csv"

    with pytest.raises(AgentLoopError):
        run_agent_turn(
            MockClient([RuntimeError("authorization bearer-secret")]),
            [],
            "Check inventory",
            log_callback=make_logger(file_path),
        )

    rows = read_rows(file_path)
    assert [row["actor"] for row in rows] == ["user", "agent"]
    assert "failed" in rows[-1]["message"]
    assert "bearer-secret" not in " ".join(row["message"] for row in rows)
    assert not any(row["message"] == "" for row in rows[1:])


def test_logger_write_failure_is_not_swallowed(tmp_path: Path) -> None:
    file_path = tmp_path / "missing-parent" / "conversation.csv"

    with pytest.raises(OSError):
        run_agent_turn(
            MockClient([make_response("Must not be logged as successful.")]),
            [],
            "Check inventory",
            log_callback=make_logger(file_path),
        )


def test_cli_surfaces_logger_write_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(agent, "create_llm_client", lambda: None)
    file_path = tmp_path / "missing-parent" / "conversation.csv"

    with pytest.raises(OSError):
        run_cli(
            MockClient([make_response("Answer")]),
            input_fn=lambda _prompt: "question",
            output_fn=lambda _message: None,
            log_callback=make_logger(file_path),
        )


def test_cli_forwards_logger_once_and_preserves_in_memory_history(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "conversation.csv"
    messages = [{"role": "system", "content": "Persistent session context."}]
    client = MockClient([make_response("First."), make_response("Second.")])
    inputs = iter(["one", "two"])
    callback = make_logger(file_path)

    # Exercise turn ownership directly while using the same caller-owned list.
    run_agent_turn(client, messages, next(inputs), log_callback=callback)
    first_snapshot = list(messages)
    run_agent_turn(client, messages, next(inputs), log_callback=callback)

    assert messages[: len(first_snapshot)] == first_snapshot
    assert messages[0] == {"role": "system", "content": "Persistent session context."}
    assert [row["actor"] for row in read_rows(file_path)] == [
        "user",
        "agent",
        "user",
        "agent",
    ]
