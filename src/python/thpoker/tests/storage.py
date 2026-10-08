import json
import pytest
from   thpoker.storage          import (SCHEMA_VERSION, SessionLog,
                                        read_records, read_session_log)


def test_records_round_trip_in_order_with_schema_version(tmp_path):
    log = SessionLog(tmp_path / "nested" / "session.jsonl")
    log.append("event", {"hand_id": "1-1", "event": {"kind": "HandStarted", "data": {}}})
    log.append("user_action", {"hand_id": "1-1", "action": {"type": "FOLD", "amount": None}})
    records = read_session_log(log.path)
    assert [r["type"] for r in records] == ["event", "user_action"]
    assert all(r["schema_version"] == SCHEMA_VERSION for r in records)
    assert records[1]["action"] == {"type": "FOLD", "amount": None}
    # Read on from where a read stopped: only complete new lines, half a line waits.
    _, offset = read_records(log.path, 0)
    log.append("hint", {"hand_id": "1-1", "index": 0})
    with log.path.open("a", encoding="utf-8") as handle:
        handle.write('{"type": "ev')
    more, after = read_records(log.path, offset)
    assert [r["type"] for r in more] == ["hint"]
    assert read_records(log.path, after) == ([], after)
    # A deleted log stays deleted, even when a late review appends to it.
    log.delete()
    log.append("rating", {"hand_id": "1-1"})
    assert not log.path.exists()


def test_record_from_a_newer_schema_is_rejected(tmp_path):
    path = tmp_path / "future.jsonl"
    path.write_text(json.dumps({"schema_version": SCHEMA_VERSION + 1, "type": "event"}) + "\n")
    with pytest.raises(ValueError, match="newer than supported"):
        read_session_log(path)
