# AI Basic Inventory Agent Loop

4Geeks Academy — AI Engineering.

## Project overview

This educational project demonstrates a manually implemented AI agent that
uses an OpenAI-compatible large language model (LLM), function/tool calling, a
FastAPI inventory service, CSV persistence, and an interactive command-line
interface. The agent—not an agent framework—decides when to request one of the
available inventory tools and processes the result.

## Architecture

```text
User → CLI → Manual Agent Loop → LLM → Inventory Tools → FastAPI → products.csv
```

The CLI keeps conversation history in memory for the duration of one session.
The agent sends approved tool operations to the FastAPI service over HTTP. The
API reads and writes `products.csv` in the project root by default. Separately,
conversation events are appended to `conversation_log.csv` in the project root
by default.

## Requirements

- Python **3.12 or newer**.
- [`uv`](https://docs.astral.sh/uv/) for environment and dependency management.
- One supported LLM provider:
	- A locally running Ollama server with an installed model that supports the
		OpenAI-compatible chat completions and tool-calling interface used here.
		Tool-calling support varies by model; verify the specific model.
	- A Groq API account and API key. Groq configuration is implemented, but this
		README does not claim that a live Groq request has been validated.

## Installation

From the repository root, sync the project and development dependencies:

```sh
uv sync --all-groups
```

This uses the committed `pyproject.toml` and `uv.lock`, including the FastAPI,
Uvicorn, OpenAI-compatible SDK, and test dependencies.

## Environment configuration

The application reads these environment variables directly; it does **not**
automatically load `.env` files. Set them in the shell before starting the
agent. A `.env.example` file is provided as a reference, not loaded by the
application. If you choose to copy it to `.env`, export its values in your
shell, for example:

```sh
set -a
source .env
set +a
```

Never commit real credentials. Keep local `.env` files private.

### Local Ollama

Start Ollama separately and use a model already available locally that supports
tool calling. Set the model name to the exact installed model identifier:

```sh
export LLM_PROVIDER=ollama
export LLM_BASE_URL=http://localhost:11434/v1
export LLM_MODEL=qwen2.5:7b
export OLLAMA_API_KEY=local-placeholder
```

`OLLAMA_API_KEY` is a non-secret placeholder required by the OpenAI SDK; the
local Ollama server ignores it. The defaults in the source are provider
`groq`, model `qwen2.5:7b` for Ollama, and base URL
`http://localhost:11434/v1`. The example model must be installed locally and
must support the tool-calling behavior the agent needs; not every Ollama model
does.

### Groq

Set a real Groq API key in your shell and select the Groq-compatible endpoint:

```sh
export LLM_PROVIDER=groq
export GROQ_API_KEY='your-key-from-Groq'
export LLM_MODEL=llama-3.3-70b-versatile
export LLM_BASE_URL=https://api.groq.com/openai/v1
```

The source defaults to provider `groq`, model `llama-3.3-70b-versatile`, and
base URL `https://api.groq.com/openai/v1`. `GROQ_API_KEY` must be set for the
SDK client to initialize. Do not put a real key in this README or commit it.

### Inventory service URL

The agent sends tool requests to `INVENTORY_API_BASE_URL`. Its default is
`http://127.0.0.1:8000`; override it only when the API server uses another
address or port:

```sh
export INVENTORY_API_BASE_URL=http://127.0.0.1:8000
```

The recognized configuration variables are `LLM_PROVIDER`, `LLM_BASE_URL`,
`LLM_MODEL`, `GROQ_API_KEY`, `OLLAMA_API_KEY`, and
`INVENTORY_API_BASE_URL`.

## Running the project

Run the API and agent in two terminals from the repository root. Configure the
LLM environment in the second terminal before starting the agent.

**Terminal 1 — FastAPI inventory server:**

```sh
uv run uvicorn api.app:app --reload
```

**Terminal 2 — interactive agent:**

```sh
uv run python agent.py
```

Keep the API server running while using the agent: inventory tools make HTTP
requests to it. Type a request at the prompt; enter `exit` or `quit` to end the
agent session. The CLI creates one LLM client and retains the conversation
history in memory until the session ends.

## Inventory API

The API defaults to `http://127.0.0.1:8000`. Product responses contain `id`,
`name`, `quantity`, and `unit`.

### List inventory — `GET /inventory`

```sh
curl http://127.0.0.1:8000/inventory
```

Returns `200 OK` and a JSON array (empty when there are no products), for
example:

```json
[{"id":"<product-id>","name":"Rice","quantity":15,"unit":"bags"}]
```

### Create a product — `POST /inventory`

```sh
curl -X POST http://127.0.0.1:8000/inventory \
	-H 'Content-Type: application/json' \
	-d '{"name":"Rice","quantity":15,"unit":"bags"}'
```

Returns `201 Created` with the created product and its generated ID. Names and
units cannot be blank, and quantity must be non-negative; invalid input returns
`422 Unprocessable Entity`.

### Adjust stock — `PATCH /inventory/{product_id}`

```sh
curl -X PATCH http://127.0.0.1:8000/inventory/<product-id> \
	-H 'Content-Type: application/json' \
	-d '{"delta":-2}'
```

Returns `200 OK` with the updated product. A missing product returns `404 Not
Found`; an adjustment that would make stock negative returns `409 Conflict`.
The request body requires an integer `delta`.

### Low-stock alerts — `GET /inventory/alerts`

```sh
curl 'http://127.0.0.1:8000/inventory/alerts?threshold=10'
```

Returns `200 OK` with products whose quantity is **strictly less** than the
threshold. The default threshold is `10`; a negative threshold returns `422`.

## Agent tools

The agent exposes these four function tools to the LLM:

- `list_inventory` — list all products; no arguments.
- `create_product` — create a product with `name`, non-negative integer
	`quantity`, and `unit`.
- `update_stock` — adjust a product using `product_id` and signed integer
	`delta`; the resulting quantity cannot be negative.
- `get_low_stock_alerts` — list products below `threshold`, which is optional
	and defaults to 10 in the API.

The dispatcher validates tool names and arguments before sending HTTP requests.
The LLM may choose whether to call tools; a natural-language response alone
does not prove that an operation was performed.

## Manual agent loop

The agent implements the **Observe → Think → Act → Update → Repeat** cycle:

1. **Observe:** send the current conversation and tool definitions to the LLM.
2. **Think:** inspect the LLM response for a final answer or requested tool
	 calls.
3. **Act:** validate and execute each requested inventory operation over HTTP.
4. **Update:** add the assistant request and matching tool results to the
	 in-memory message history.
5. **Repeat:** ask the LLM to continue until it returns a final text response
	 or the iteration limit is reached.

The interactive CLI reuses the same in-memory history across turns. It is not
persisted as model context between separate CLI sessions.

## Conversation logging

Conversation events are appended to `conversation_log.csv` by default. The CSV
columns are:

```text
timestamp, actor, message, tool_call
```

The logger writes a header when the file is new or empty, then appends each
event rather than replacing existing rows. Timestamps are timezone-aware UTC
ISO 8601 values. Actors are `user`, `agent`, and `tool`; tool request and result
rows include tool names and correlation IDs in `tool_call`. The project logger
is separate from the inventory data in `products.csv`.

## Testing

Run the automated test suite with:

```sh
uv run python -m pytest -q
```

The last verified suite contained **141 passing tests**.

## Limitations

- A real local Ollama execution completed a six-turn inventory workflow using
	`qwen3.5:9b`, including tool calls against the FastAPI service.
- The tested local `qwen3.5:9b` model showed inconsistent tool behavior on a
	repeat run. Duplicate tool requests and invented product IDs are possible;
	verify actual tool results and persisted inventory rather than trusting
	natural-language claims alone.
- Tool-calling behavior is model-dependent. Do not assume that all local Ollama
	models support it reliably.
- This is an educational MVP, not a production-grade inventory service. It has
	no production-grade authentication, concurrency controls, or operational
	guarantees.
- Never commit real API credentials or other secrets.
