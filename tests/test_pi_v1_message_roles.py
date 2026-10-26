"""Pi >= 1.x message roles (system, bashExecution) migrate without sanitization."""

import json
import tempfile
from pathlib import Path

import pytest

from session_migrate import conversion
from session_migrate.errors import SessionMigrateError
from session_migrate.formats import claude, pi
from session_migrate.model import AgentFormat, EventKind, Role

FIXTURES = Path(__file__).parent / "fixtures"
NEW_ROLES_FIXTURE = FIXTURES / "pi-1.4.0" / "new-roles.jsonl"
TARGET_UUID = "22222222-2222-4222-8222-222222222222"


def test_pi_v1_fixture_parses_system_and_bash_execution_roles() -> None:
    parsed = pi.parse(NEW_ROLES_FIXTURE)

    assert parsed.session_id == "33333333-3333-4333-8333-333333333333"
    assert parsed.raw_record_count == 9
    assert [(event.kind, event.role) for event in parsed.events] == [
        (EventKind.OPAQUE, None),  # model_change
        (EventKind.OPAQUE, None),  # thinking_level_change
        (EventKind.MESSAGE, Role.SYSTEM),
        (EventKind.MESSAGE, Role.USER),
        (EventKind.MESSAGE, Role.ASSISTANT),
        (EventKind.TOOL_CALL, Role.ASSISTANT),
        (EventKind.TOOL_RESULT, Role.TOOL),
        (EventKind.MESSAGE, Role.USER),  # bashExecution projection
        (EventKind.MESSAGE, Role.USER),
    ]

    system_event = parsed.events[2]
    assert "expert coding assistant" in (system_event.text or "")
    assert "Be concise." in (system_event.text or "")

    bash_event = parsed.events[7]
    assert bash_event.text == "[bash] code ./ (exit 0)"


def test_pi_v1_bash_execution_with_output_and_nonzero_exit() -> None:
    path = NEW_ROLES_FIXTURE.read_text().splitlines()
    records = [json.loads(line) for line in path]
    for record in records:
        message = record.get("message") or {}
        if message.get("role") == "bashExecution":
            message["command"] = "npm test"
            message["output"] = "1 failed, 2 passed"
            message["exitCode"] = 1
    with tempfile.TemporaryDirectory() as tmp:
        modified = Path(tmp) / "modified.jsonl"
        modified.write_text("".join(json.dumps(record) + "\n" for record in records))
        parsed = pi.parse(modified)

    bash_event = next(
        event
        for event in parsed.events
        if event.kind == EventKind.MESSAGE
        and event.role == Role.USER
        and event.text
        and "[bash]" in event.text
    )
    assert bash_event.text == "[bash] npm test (exit 1)\n1 failed, 2 passed"


def test_pi_v1_unknown_role_is_still_rejected(tmp_path: Path) -> None:
    records = [
        {
            "type": "session",
            "version": 3,
            "id": "33333333-3333-4333-8333-333333333333",
            "timestamp": "2026-10-08T12:06:49.789Z",
            "cwd": "/tmp/session-migrate-fixture",
        },
        {
            "type": "message",
            "id": "00000001",
            "parentId": None,
            "timestamp": "2026-10-08T12:06:50.000Z",
            "message": {"role": "mystery", "content": "boom"},
        },
    ]
    path = tmp_path / "unknown-role.jsonl"
    path.write_text("".join(json.dumps(record) + "\n" for record in records))

    with pytest.raises(SessionMigrateError, match="unsupported role"):
        pi.parse(path)


def test_pi_v1_fixture_converts_to_claude_with_bash_history_preserved(
    tmp_path: Path,
) -> None:
    artifact = conversion.convert_session(
        pi.parse_session(NEW_ROLES_FIXTURE),
        conversion.ConversionOptions(
            target_format=AgentFormat.CLAUDE,
            session_id=TARGET_UUID,
            cwd=tmp_path,
        ),
    )

    # The privileged system preamble is deliberately dropped; the user-run
    # bash command must survive as ordinary user history.
    assert artifact.dropped.get("message:privileged_role") == 1
    transcript = artifact.native_bytes.decode("utf-8")
    assert "[bash] code ./ (exit 0)" in transcript
    assert "SYNTHETIC_USER_MARKER" in transcript

    out_path = tmp_path / "converted.jsonl"
    out_path.write_bytes(artifact.native_bytes)
    reparsed = claude.parse(out_path)
    assert reparsed.session_id == TARGET_UUID


def test_pi_v1_events_round_trip_through_pi_writer(tmp_path: Path) -> None:
    session = pi.parse_session(NEW_ROLES_FIXTURE)

    data, dropped = pi.serialize(
        session,
        session_id=TARGET_UUID,
        cwd=tmp_path,
        timestamp="2026-10-08T12:06:49.789Z",
    )
    pi.validate_native_bytes(data, TARGET_UUID)

    assert dropped.get("message:privileged_role") == 1
    out_path = tmp_path / "round-trip.jsonl"
    out_path.write_bytes(data)
    reparsed = pi.parse(out_path)
    assert reparsed.session_id == TARGET_UUID
    assert "SYNTHETIC_FINAL_MARKER" in "\n".join(event.text or "" for event in reparsed.events)
