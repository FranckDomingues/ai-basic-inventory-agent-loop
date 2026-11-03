"""LLM configuration and inventory HTTP tools for the future AI agent."""

import json
import math
import os
from dataclasses import dataclass
from http.client import HTTPException
from typing import Any, Callable, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlsplit
from urllib.request import Request, urlopen

from openai import OpenAI

from conversation_logger import log_interaction


_GROQ_BASE_URL = "https://api.groq.com/openai/v1"
_OLLAMA_BASE_URL = "http://localhost:11434/v1"
_DEFAULT_API_BASE_URL = "http://127.0.0.1:8000"
_DEFAULT_TIMEOUT = 10.0
_DEFAULT_MAX_ITERATIONS = 8

INVENTORY_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "list_inventory",
            "description": "List all products currently in the inventory.",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_product",
            "description": "Create a product with its name, initial quantity, and unit.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "minLength": 1},
                    "quantity": {"type": "integer", "minimum": 0},
                    "unit": {"type": "string", "minLength": 1},
                },
                "required": ["name", "quantity", "unit"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_stock",
            "description": "Apply a signed integer adjustment to an existing product's stock.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {"type": "string", "minLength": 1},
                    "delta": {"type": "integer"},
                },
                "required": ["product_id", "delta"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_low_stock_alerts",
            "description": "List products with stock strictly below a non-negative threshold (default 10).",
            "parameters": {
                "type": "object",
                "properties": {"threshold": {"type": "integer", "minimum": 0}},
                "additionalProperties": False,
            },
        },
    },
]

_TOOL_NAMES = {tool["function"]["name"] for tool in INVENTORY_TOOLS}


@dataclass(frozen=True)
class LLMConfig:
    """Resolved settings for an OpenAI-compatible language model service.

    Attributes:
        provider: Selected service, either ``groq`` or ``ollama``.
        model: Model identifier sent to the service.
        base_url: OpenAI-compatible API root.
        api_key: Configured provider credential; Ollama uses a local placeholder.
    """

    provider: str
    model: str
    base_url: str
    api_key: str | None

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "LLMConfig":
        """Build configuration from environment-style values.

        Args:
            environ: Optional mapping for tests or explicit configuration. When
                omitted, the current process environment is used.

        Returns:
            Resolved provider, model, endpoint, and available API key.

        Raises:
            ValueError: If ``LLM_PROVIDER`` selects an unsupported provider.
        """
        values = os.environ if environ is None else environ
        provider = values.get("LLM_PROVIDER", "groq").strip().lower()

        if provider == "groq":
            default_model = "llama-3.3-70b-versatile"
            default_base_url = _GROQ_BASE_URL
            api_key = values.get("GROQ_API_KEY")
        elif provider == "ollama":
            default_model = "qwen2.5:7b"
            default_base_url = _OLLAMA_BASE_URL
            # Ollama ignores the SDK key, but the official client requires one.
            api_key = values.get("OLLAMA_API_KEY")
        else:
            raise ValueError("LLM_PROVIDER must be either 'groq' or 'ollama'")

        return cls(
            provider=provider,
            model=values.get("LLM_MODEL", default_model),
            base_url=values.get("LLM_BASE_URL", default_base_url),
            api_key=api_key,
        )


def create_llm_client(config: LLMConfig | None = None) -> OpenAI:
    """Initialize the official OpenAI SDK for the configured compatible API.

    Args:
        config: Optional resolved settings; environment configuration is used
            when omitted.

    Returns:
        An ``OpenAI`` client configured with the selected model service.

    Raises:
        ValueError: If the selected provider has no configured SDK API key.
    """
    settings = config or LLMConfig.from_env()
    if not settings.api_key:
        key_variable = "GROQ_API_KEY" if settings.provider == "groq" else "OLLAMA_API_KEY"
        raise ValueError(f"{key_variable} must be configured for the selected provider")

    return OpenAI(
        api_key=settings.api_key,
        base_url=settings.base_url,
    )


def _error_result(error_type: str, message: str, **details: Any) -> dict[str, Any]:
    """Build a JSON-serializable error response for a tool invocation."""
    return {
        "ok": False,
        "error": {"type": error_type, "message": message, **details},
    }


def _parse_arguments(arguments: Mapping[str, Any] | str) -> dict[str, Any]:
    """Decode tool arguments and require a JSON object."""
    if isinstance(arguments, str):
        arguments = json.loads(arguments)
    if not isinstance(arguments, Mapping):
        raise ValueError("Tool arguments must be a JSON object")
    return dict(arguments)


def _validate_arguments(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Validate the exact argument set and value types for an approved tool."""
    required_by_tool = {
        "list_inventory": set(),
        "create_product": {"name", "quantity", "unit"},
        "update_stock": {"product_id", "delta"},
        "get_low_stock_alerts": set(),
    }
    optional_by_tool = {"get_low_stock_alerts": {"threshold"}}
    required = required_by_tool[name]
    allowed = required | optional_by_tool.get(name, set())
    missing = required - arguments.keys()
    unexpected = arguments.keys() - allowed
    if missing or unexpected:
        problems = []
        if missing:
            problems.append(f"missing arguments: {', '.join(sorted(missing))}")
        if unexpected:
            problems.append(f"unexpected arguments: {', '.join(sorted(unexpected))}")
        raise ValueError("; ".join(problems))

    if name == "create_product":
        for key in ("name", "unit"):
            if not isinstance(arguments[key], str) or not arguments[key].strip():
                raise ValueError(f"{key} must be a non-empty string")
        quantity = arguments["quantity"]
        if isinstance(quantity, bool) or not isinstance(quantity, int) or quantity < 0:
            raise ValueError("quantity must be a non-negative integer")
    elif name == "update_stock":
        product_id, delta = arguments["product_id"], arguments["delta"]
        if not isinstance(product_id, str) or not product_id.strip():
            raise ValueError("product_id must be a non-empty string")
        if isinstance(delta, bool) or not isinstance(delta, int):
            raise ValueError("delta must be an integer")
    elif name == "get_low_stock_alerts" and "threshold" in arguments:
        threshold = arguments["threshold"]
        if isinstance(threshold, bool) or not isinstance(threshold, int) or threshold < 0:
            raise ValueError("threshold must be a non-negative integer")
    return arguments


def execute_inventory_tool(
    tool_name: str,
    arguments: Mapping[str, Any] | str,
    *,
    api_base_url: str | None = None,
    timeout: float = _DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    """Validate and execute one approved inventory HTTP operation.

    Args:
        tool_name: One of the four names in ``INVENTORY_TOOLS``.
        arguments: JSON object or its encoded string from an LLM tool call.
        api_base_url: Inventory API root; defaults to ``INVENTORY_API_BASE_URL``
            or ``http://127.0.0.1:8000``.
        timeout: Maximum time in seconds to wait for the HTTP response.

    Returns:
        A JSON-serializable success object with ``status_code`` and ``data``, or
        an error object preserving validation, transport, or API error details.

    Raises:
        No expected tool or HTTP errors escape this function; they are returned
        as structured error objects for a future agent to handle.
    """
    if not isinstance(tool_name, str) or tool_name not in _TOOL_NAMES:
        return _error_result("unknown_tool", f"Unknown inventory tool: {tool_name}")
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(timeout)
        or timeout <= 0
    ):
        return _error_result("invalid_arguments", "timeout must be a positive number")

    try:
        parsed = _parse_arguments(arguments)
        values = _validate_arguments(tool_name, parsed)
    except (ValueError, TypeError, json.JSONDecodeError) as error:
        return _error_result("invalid_arguments", str(error))

    configured_base_url = (
        api_base_url or os.environ.get("INVENTORY_API_BASE_URL") or _DEFAULT_API_BASE_URL
    )
    if not isinstance(configured_base_url, str):
        return _error_result("invalid_configuration", "Inventory API base URL must be a string")
    base_url = configured_base_url.rstrip("/")
    if not base_url:
        return _error_result("invalid_configuration", "Inventory API base URL cannot be empty")
    try:
        parsed_base_url = urlsplit(base_url)
    except ValueError as error:
        return _error_result("invalid_configuration", f"Invalid inventory API base URL: {error}")
    if (
        parsed_base_url.scheme not in {"http", "https"}
        or not parsed_base_url.netloc
        or any(character.isspace() for character in parsed_base_url.netloc)
        or parsed_base_url.username is not None
        or parsed_base_url.password is not None
        or parsed_base_url.query
        or parsed_base_url.fragment
    ):
        return _error_result(
            "invalid_configuration",
            "Inventory API base URL must be an HTTP(S) origin or path without credentials, query, or fragment",
        )
    method, path, query, body = "GET", "/inventory", None, None
    if tool_name == "create_product":
        method, body = "POST", values
    elif tool_name == "update_stock":
        product_id = quote(values["product_id"], safe="")
        method, path, body = "PATCH", f"/inventory/{product_id}", {"delta": values["delta"]}
    elif tool_name == "get_low_stock_alerts":
        path = "/inventory/alerts"
        if "threshold" in values:
            query = urlencode({"threshold": values["threshold"]})

    url = f"{base_url}{path}"
    if query:
        url = f"{url}?{query}"
    request_headers = {"Accept": "application/json"}
    request_data = None
    if body is not None:
        request_headers["Content-Type"] = "application/json"
        request_data = json.dumps(body).encode("utf-8")
    try:
        request = Request(url, data=request_data, headers=request_headers, method=method)
    except ValueError as error:
        return _error_result("invalid_configuration", f"Invalid inventory API base URL: {error}")

    try:
        with urlopen(request, timeout=float(timeout)) as response:
            status_code = response.status
            response_body = response.read()
    except HTTPError as error:
        status_code = error.code
        response_body = error.read()
        return _http_error_result(status_code, response_body)
    except (URLError, TimeoutError, OSError, HTTPException) as error:
        reason = getattr(error, "reason", error)
        category = "timeout" if isinstance(reason, TimeoutError) else "connection_error"
        return _error_result(category, str(reason))

    try:
        data = json.loads(response_body.decode("utf-8")) if response_body else None
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        return _error_result(
            "invalid_response", f"Inventory API returned invalid JSON: {error}", status_code=status_code
        )
    return {"ok": True, "status_code": status_code, "data": data}


def _http_error_result(status_code: int, response_body: bytes) -> dict[str, Any]:
    """Preserve FastAPI error details while safely handling non-JSON bodies."""
    try:
        detail: Any = json.loads(response_body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        detail = response_body.decode("utf-8", errors="replace") or None
    return _error_result(
        "http_error",
        f"Inventory API returned HTTP {status_code}",
        status_code=status_code,
        detail=detail,
    )


class AgentLoopError(RuntimeError):
    """Raised when the LLM conversation cannot safely continue."""


def run_agent_turn(
    client: Any,
    messages: list[dict[str, Any]],
    user_input: str,
    *,
    model: str | None = None,
    max_iterations: int = _DEFAULT_MAX_ITERATIONS,
    api_base_url: str | None = None,
    log_callback: Callable[[str, str, str], None] | None = None,
) -> str:
    """Run one user turn through the manual observe-think-act-update loop.

    The caller owns ``messages`` and should reuse the same list on later turns
    to retain the in-memory conversation history for a session. This loop owns
    logging each processed event; the CLI delegates here rather than logging
    the same user or assistant messages a second time. Supply ``log_callback``
    to inject a logger (for example, a partially applied ``log_interaction``
    targeting a temporary CSV file); otherwise the project CSV logger is used.

    Args:
        client: An initialized OpenAI-compatible client with chat completions.
        messages: Mutable conversation history, updated in place.
        user_input: New user message to process.
        model: Optional model override. When omitted, ``LLM_MODEL`` or the
            configured provider's default model is used.
        max_iterations: Maximum number of LLM responses, including the final
            answer, allowed for this turn.
        api_base_url: Optional inventory API root passed to tool dispatch.
        log_callback: Optional callback accepting actor, message, and tool-call
            metadata. CSV write errors from the callback are allowed to surface.

    Returns:
        The final assistant text, also appended to ``messages``.

    Raises:
        ValueError: If input, history, or iteration limit is invalid.
        AgentLoopError: If the LLM request fails, its response is malformed, or
            the iteration limit is reached before a final answer.
        OSError: If the configured conversation logger cannot write an event.
    """
    if not isinstance(user_input, str) or not user_input.strip():
        raise ValueError("user_input must be a non-empty string")
    if not isinstance(messages, list) or any(not isinstance(message, dict) for message in messages):
        raise ValueError("messages must be a list of message dictionaries")
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")

    config = LLMConfig.from_env()
    selected_model = model or os.environ.get("LLM_MODEL") or config.model
    write_log = log_interaction if log_callback is None else log_callback
    messages.append({"role": "user", "content": user_input})
    # Logging happens at the accepted-turn boundary, once, before any LLM call.
    write_log("user", user_input, "")

    for iteration in range(max_iterations):
        try:
            completion = client.chat.completions.create(
                model=selected_model,
                messages=messages,
                tools=INVENTORY_TOOLS,
                tool_choice="auto",
            )
        except Exception as error:
            # SDK exceptions may contain request headers or secrets; expose only
            # the exception class and a stable, non-sensitive failure message.
            write_log("agent", "LLM request failed; no final answer was produced", "")
            raise AgentLoopError(
                f"LLM request failed ({type(error).__name__}); no final answer was produced"
            ) from None

        try:
            choice = completion.choices[0]
            assistant_message = choice.message
            content = assistant_message.content
            tool_calls = assistant_message.tool_calls or []
        except (AttributeError, IndexError, TypeError) as error:
            write_log("agent", "LLM returned a malformed response", "")
            raise AgentLoopError("LLM returned a malformed response") from error

        if not tool_calls:
            if not isinstance(content, str):
                write_log("agent", "LLM returned a final response without text", "")
                raise AgentLoopError("LLM returned a final response without text")
            messages.append({"role": "assistant", "content": content})
            write_log("agent", content, "")
            return content

        assistant_entry: dict[str, Any] = {
            "role": "assistant",
            "content": content,
            "tool_calls": [],
        }
        pending_calls: list[tuple[str, Any, Any]] = []
        for call in tool_calls:
            try:
                call_id = call.id
                function = call.function
                tool_name = function.name
                arguments = function.arguments
            except AttributeError as error:
                write_log("agent", "LLM returned a malformed tool request", "")
                raise AgentLoopError("LLM returned a malformed tool call") from error
            if not isinstance(call_id, str) or not call_id:
                write_log("agent", "LLM returned a tool request without a valid ID", "")
                raise AgentLoopError("LLM returned a tool call without a valid ID")

            assistant_entry["tool_calls"].append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": tool_name, "arguments": arguments},
                }
            )
            pending_calls.append((call_id, tool_name, arguments))

        # The assistant's tool-call record must precede its results in the
        # history so the next LLM request can match every result to its call ID.
        messages.append(assistant_entry)
        # Each requested call gets its own agent row so identifiers remain
        # independently searchable and map one-to-one to subsequent tool rows.
        for call_id, tool_name, _arguments in pending_calls:
            request_metadata = json.dumps(
                {
                    "name": tool_name if isinstance(tool_name, str) else "unknown",
                    "id": call_id,
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
            write_log(
                "agent", content if isinstance(content, str) else "", request_metadata
            )
        for call_id, tool_name, arguments in pending_calls:
            try:
                result = execute_inventory_tool(
                    tool_name, arguments, api_base_url=api_base_url
                )
            except Exception:
                # The dispatcher normally returns structured errors itself;
                # protect the conversation if an unexpected failure escapes.
                result = _error_result(
                    "tool_execution_error", "Inventory tool execution failed"
                )
            result_content = json.dumps(result, ensure_ascii=False)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": result_content,
                }
            )
            # Error details can contain server or transport text; log only the
            # stable error category while retaining full results in memory.
            logged_result = result
            if result.get("ok") is False:
                error = result.get("error")
                error_type = error.get("type") if isinstance(error, dict) else "tool_error"
                if not isinstance(error_type, str):
                    error_type = "tool_error"
                logged_result = {"ok": False, "error": {"type": error_type}}
            tool_metadata = json.dumps(
                {
                    "name": tool_name if isinstance(tool_name, str) else "unknown",
                    "id": call_id,
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
            write_log("tool", json.dumps(logged_result, ensure_ascii=False), tool_metadata)

        if iteration == max_iterations - 1:
            write_log(
                "agent",
                f"Maximum LLM iterations ({max_iterations}) reached before a final answer",
                "",
            )
            raise AgentLoopError(
                f"Maximum LLM iterations ({max_iterations}) reached before a final answer"
            )

    raise AgentLoopError("Agent loop ended without a final answer")


def run_cli(
    client: Any | None = None,
    *,
    input_fn: Callable[[str], str] | None = None,
    output_fn: Callable[[str], Any] | None = None,
    log_callback: Callable[[str, str, str], None] | None = None,
) -> None:
    """Run an interactive terminal conversation with the inventory agent.

    Args:
        client: Optional initialized OpenAI-compatible client. If omitted, the
            existing provider configuration is used to create one session client.
        input_fn: Optional prompt function, injectable for isolated tests.
            Defaults to :func:`input`.
        output_fn: Optional output function, injectable for isolated tests.
            Defaults to :func:`print`.
        log_callback: Optional logger callback forwarded to ``run_agent_turn``.
            The loop owns all event records, preventing CLI duplicates.

    Returns:
        ``None`` when the user exits, the input stream ends, or the client
        cannot be initialized.

    Raises:
        OSError: If the conversation logger cannot write an event. Logging
            failures are surfaced rather than allowing an unlogged session.
        No expected terminal, initialization, or agent-loop errors other than
            logger write failures escape; they are handled with safe messages.
    """
    read_input = input if input_fn is None else input_fn
    write_output = print if output_fn is None else output_fn
    write_output("Welcome to the AI inventory assistant.")
    write_output("Ask about inventory, or type 'exit' or 'quit' to leave.")

    if client is None:
        try:
            client = create_llm_client()
        except Exception:
            # Initialization exceptions can contain provider details or secrets.
            write_output("Unable to initialize the LLM client. Check your provider configuration.")
            return

    # This one list belongs to the whole CLI session; each turn appends to it.
    messages: list[dict[str, Any]] = []
    while True:
        try:
            user_input = read_input("You: ")
        except (EOFError, KeyboardInterrupt):
            write_output("\nGoodbye.")
            return

        if not isinstance(user_input, str) or not user_input.strip():
            continue
        if user_input.strip().casefold() in {"exit", "quit"}:
            write_output("Goodbye.")
            return

        try:
            if log_callback is None:
                # Do not pass a new keyword on the default path; this preserves
                # compatibility with callers and tests replacing run_agent_turn.
                answer = run_agent_turn(client, messages, user_input)
            else:
                answer = run_agent_turn(
                    client, messages, user_input, log_callback=log_callback
                )
        except KeyboardInterrupt:
            write_output("\nRequest interrupted. Goodbye.")
            return
        except (AgentLoopError, ValueError) as error:
            write_output(f"Agent error: {error}")
            continue
        except OSError:
            # CSV destination failures are operational errors, not recoverable
            # agent failures; surface them rather than silently dropping logs.
            raise
        except Exception:
            # Keep the session usable without displaying arbitrary exception
            # text, which could include request headers or credentials.
            write_output("Agent request failed unexpectedly. You can try again or exit.")
            continue

        write_output(f"Assistant: {answer}")


if __name__ == "__main__":
    run_cli()
