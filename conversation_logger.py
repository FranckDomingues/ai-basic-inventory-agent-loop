"""Append-only CSV logging for user, agent, and tool conversation messages."""

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal


DEFAULT_LOG_PATH = Path(__file__).resolve().parent / "conversation_log.csv"
CSV_COLUMNS = ("timestamp", "actor", "message", "tool_call")
Actor = Literal["user", "agent", "tool"]
_VALID_ACTORS = frozenset({"user", "agent", "tool"})


def log_interaction(
    actor: Actor,
    message: str,
    tool_call: str = "",
    *,
    file_path: str | Path = DEFAULT_LOG_PATH,
) -> None:
    """Append one conversation record to a UTF-8 CSV file.

    The file is opened in append mode, so existing records are never replaced.
    A header is written only when the destination is new or empty. Timestamps
    are generated as timezone-aware UTC ISO 8601 strings.

    Args:
        actor: Record source; must be ``user``, ``agent``, or ``tool``.
        message: Conversation message content, including any embedded newlines.
        tool_call: Optional tool-call description or serialized call data.
        file_path: Destination CSV path. Defaults to the project-root
            ``conversation_log.csv``; tests and callers may provide another path.

    Returns:
        ``None`` after the complete row has been written.

    Raises:
        ValueError: If ``actor`` is not supported or a record field is not text.
        OSError: If the destination cannot be created or written.
    """
    if not isinstance(actor, str) or actor not in _VALID_ACTORS:
        raise ValueError("actor must be one of: user, agent, tool")
    if not isinstance(message, str):
        raise ValueError("message must be a string")
    if not isinstance(tool_call, str):
        raise ValueError("tool_call must be a string")

    destination = Path(file_path)
    # newline="" lets csv.writer manage embedded line breaks and CSV quoting.
    with destination.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        if csv_file.tell() == 0:
            writer.writerow(CSV_COLUMNS)
        writer.writerow(
            (
                datetime.now(timezone.utc).isoformat(),
                actor,
                message,
                tool_call,
            )
        )
