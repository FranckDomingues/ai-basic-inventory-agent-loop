"""Isolated tests for the interactive command-line session."""

import subprocess
import sys
from copy import deepcopy
from typing import Any
from unittest.mock import Mock

import pytest

import agent


class InputSequence:
    """Provide scripted terminal input and raise scripted control-flow events."""

    def __init__(self, values: list[str | BaseException]) -> None:
        self.values = iter(values)
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        value = next(self.values)
        if isinstance(value, BaseException):
            raise value
        return value


def test_cli_initializes_client_once_and_shows_instructions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = object()
    create_client = Mock(return_value=client)
    monkeypatch.setattr(agent, "create_llm_client", create_client)
    reader = InputSequence(["exit"])
    output: list[str] = []

    agent.run_cli(input_fn=reader, output_fn=output.append)

    create_client.assert_called_once_with()
    assert output[0].startswith("Welcome")
    assert "exit" in output[1] and "quit" in output[1]
    assert output[-1] == "Goodbye."
    assert reader.prompts == ["You: "]


def test_cli_processes_single_request_and_prints_final_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = object()
    monkeypatch.setattr(agent, "create_llm_client", lambda: client)
    turn = Mock(return_value="There are 5 apples.")
    monkeypatch.setattr(agent, "run_agent_turn", turn)
    reader = InputSequence(["How many apples?", "quit"])
    output: list[str] = []

    agent.run_cli(input_fn=reader, output_fn=output.append)

    turn.assert_called_once()
    assert turn.call_args.args[:2] == (client, [])
    assert turn.call_args.args[2] == "How many apples?"
    assert "Assistant: There are 5 apples." in output


def test_cli_retains_one_history_list_across_turns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = object()
    monkeypatch.setattr(agent, "create_llm_client", lambda: client)
    histories: list[list[dict[str, Any]]] = []
    history_snapshots: list[list[dict[str, Any]]] = []

    def respond(_client: Any, messages: list[dict[str, Any]], user_input: str) -> str:
        histories.append(messages)
        history_snapshots.append(deepcopy(messages))
        messages.extend(
            [
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": f"Answer: {user_input}"},
            ]
        )
        return f"Answer: {user_input}"

    monkeypatch.setattr(agent, "run_agent_turn", respond)
    reader = InputSequence(["first", "second", "exit"])

    agent.run_cli(input_fn=reader, output_fn=lambda _text: None)

    assert histories[0] is histories[1]
    assert history_snapshots[1] == [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "Answer: first"},
    ]


def test_cli_skips_empty_input_without_calling_agent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(agent, "create_llm_client", lambda: object())
    turn = Mock(return_value="unused")
    monkeypatch.setattr(agent, "run_agent_turn", turn)

    agent.run_cli(
        input_fn=InputSequence(["", "   ", "exit"]),
        output_fn=lambda _text: None,
    )

    turn.assert_not_called()


@pytest.mark.parametrize("command", ["exit", "quit"])
def test_cli_exit_commands_leave_session(command: str) -> None:
    reader = InputSequence([command])
    output: list[str] = []

    agent.run_cli(
        client=object(), input_fn=reader, output_fn=output.append
    )

    assert output[-1] == "Goodbye."
    assert len(reader.prompts) == 1


@pytest.mark.parametrize("command", ["EXIT", "Quit", "qUiT", " ExIt "])
def test_cli_exit_commands_are_case_insensitive(command: str) -> None:
    output: list[str] = []

    agent.run_cli(
        client=object(),
        input_fn=InputSequence([command]),
        output_fn=output.append,
    )

    assert output[-1] == "Goodbye."


def test_cli_handles_ctrl_c_during_prompt_gracefully() -> None:
    output: list[str] = []

    agent.run_cli(
        client=object(),
        input_fn=InputSequence([KeyboardInterrupt()]),
        output_fn=output.append,
    )

    assert output[-1] == "\nGoodbye."


def test_cli_handles_eof_gracefully() -> None:
    output: list[str] = []

    agent.run_cli(
        client=object(), input_fn=InputSequence([EOFError()]), output_fn=output.append
    )

    assert output[-1] == "\nGoodbye."


def test_cli_reports_recoverable_agent_error_and_continues(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(agent, "create_llm_client", lambda: object())
    calls = 0

    def respond(*args: Any) -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise agent.AgentLoopError("LLM request failed; no final answer was produced")
        return "Recovered answer."

    monkeypatch.setattr(agent, "run_agent_turn", respond)
    output: list[str] = []

    agent.run_cli(
        input_fn=InputSequence(["first attempt", "try again", "quit"]),
        output_fn=output.append,
    )

    assert calls == 2
    assert any("Agent error:" in line for line in output)
    assert "Assistant: Recovered answer." in output


def test_cli_client_initialization_failure_is_safe_and_returns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        agent,
        "create_llm_client",
        lambda: (_ for _ in ()).throw(RuntimeError("secret credential")),
    )
    output: list[str] = []
    reader = InputSequence([])

    agent.run_cli(input_fn=reader, output_fn=output.append)

    assert "Unable to initialize" in output[-1]
    assert "secret credential" not in " ".join(output)
    assert reader.prompts == []


def test_importing_module_does_not_start_cli_or_create_client() -> None:
    script = """
import builtins
from unittest.mock import Mock
import openai

builtins.input = Mock(side_effect=AssertionError('unexpected prompt'))
openai.OpenAI = Mock(side_effect=AssertionError('unexpected client'))
import agent
assert callable(agent.run_cli)
openai.OpenAI.assert_not_called()
builtins.input.assert_not_called()
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
