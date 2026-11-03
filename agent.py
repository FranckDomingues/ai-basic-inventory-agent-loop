"""LLM configuration and inventory HTTP tools for the future AI agent."""

import json
import math
import os
from dataclasses import dataclass
from http.client import HTTPException
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlsplit
from urllib.request import Request, urlopen

from openai import OpenAI


_GROQ_BASE_URL = "https://api.groq.com/openai/v1"
_OLLAMA_BASE_URL = "http://localhost:11434/v1"
_DEFAULT_API_BASE_URL = "http://127.0.0.1:8000"
_DEFAULT_TIMEOUT = 10.0

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
