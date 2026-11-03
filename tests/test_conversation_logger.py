"""Tests for the append-only conversation CSV logger."""

import csv
from datetime import datetime
from pathlib import Path

import pytest

from conversation_logger import CSV_COLUMNS, log_interaction


def read_log(file_path: Path) -> list[list[str]]:
    """Read CSV records while correctly decoding quoted and multiline fields."""
    with file_path.open("r", newline="", encoding="utf-8") as csv_file:
        return list(csv.reader(csv_file))


def test_creates_new_file_with_exact_column_order(tmp_path: Path) -> None:
    file_path = tmp_path / "conversation.csv"

    log_interaction("user", "How many products?", file_path=file_path)

    records = read_log(file_path)
    assert records[0] == ["timestamp", "actor", "message", "tool_call"]
    assert tuple(records[0]) == CSV_COLUMNS
    assert len(records) == 2


def test_writes_header_exactly_once_and_appends_records(tmp_path: Path) -> None:
    file_path = tmp_path / "conversation.csv"

    log_interaction("user", "First question", file_path=file_path)
    log_interaction("agent", "First answer", file_path=file_path)
    log_interaction("tool", "Inventory listed", "list_inventory", file_path=file_path)

    records = read_log(file_path)
    assert records[0] == list(CSV_COLUMNS)
    assert sum(record == list(CSV_COLUMNS) for record in records) == 1
    assert [record[1:] for record in records[1:]] == [
        ["user", "First question", ""],
        ["agent", "First answer", ""],
        ["tool", "Inventory listed", "list_inventory"],
    ]


def test_preserves_existing_rows_when_appending(tmp_path: Path) -> None:
    file_path = tmp_path / "conversation.csv"
    file_path.write_text(
        "timestamp,actor,message,tool_call\n"
        '2026-01-01T00:00:00+00:00,user,"Existing record",\n',
        encoding="utf-8",
    )

    log_interaction("agent", "New record", file_path=file_path)

    records = read_log(file_path)
    assert len(records) == 3
    assert records[1][1:] == ["user", "Existing record", ""]
    assert records[2][1:] == ["agent", "New record", ""]


@pytest.mark.parametrize("actor", ["user", "agent", "tool"])
def test_accepts_each_supported_actor(tmp_path: Path, actor: str) -> None:
    file_path = tmp_path / f"{actor}.csv"

    log_interaction(actor, "message", file_path=file_path)

    assert read_log(file_path)[1][1] == actor


@pytest.mark.parametrize("actor", ["system", "assistant", "", None])
def test_rejects_unsupported_actor_without_creating_file(
    tmp_path: Path, actor: object
) -> None:
    file_path = tmp_path / "conversation.csv"

    with pytest.raises(ValueError, match="actor"):
        log_interaction(actor, "message", file_path=file_path)  # type: ignore[arg-type]

    assert not file_path.exists()


def test_timestamp_is_timezone_aware_iso_8601(tmp_path: Path) -> None:
    file_path = tmp_path / "conversation.csv"

    log_interaction("user", "message", file_path=file_path)

    timestamp = datetime.fromisoformat(read_log(file_path)[1][0])
    assert timestamp.tzinfo is not None
    assert timestamp.utcoffset() is not None


def test_preserves_commas_quotes_multiline_and_unicode(tmp_path: Path) -> None:
    file_path = tmp_path / "conversation.csv"
    message = 'Please check "apples, pears".\nThen check café 🍎.'
    tool_call = '{"name":"list_inventory","note":"says, \'hello\'"}'

    log_interaction("tool", message, tool_call, file_path=file_path)

    row = read_log(file_path)[1]
    assert row[1:] == ["tool", message, tool_call]


def test_explicit_path_creates_nested_temporary_destination(tmp_path: Path) -> None:
    file_path = tmp_path / "nested" / "custom.csv"
    file_path.parent.mkdir()

    log_interaction("user", "Isolated", file_path=file_path)

    assert file_path.exists()
    assert read_log(file_path)[1][2] == "Isolated"


def test_empty_existing_file_receives_one_header(tmp_path: Path) -> None:
    file_path = tmp_path / "empty.csv"
    file_path.touch()

    log_interaction("agent", "First row", file_path=file_path)

    assert read_log(file_path)[0] == list(CSV_COLUMNS)
    assert len(read_log(file_path)) == 2


@pytest.mark.parametrize("destination_kind", ["directory", "missing-parent"])
def test_write_failure_is_propagated(
    tmp_path: Path, destination_kind: str
) -> None:
    if destination_kind == "directory":
        file_path = tmp_path / "directory.csv"
        file_path.mkdir()
    else:
        file_path = tmp_path / "missing" / "conversation.csv"

    with pytest.raises(OSError):
        log_interaction("user", "message", file_path=file_path)
