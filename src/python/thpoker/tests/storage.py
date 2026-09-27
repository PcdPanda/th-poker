import json

import pytest

from   thpoker.storage          import (SCHEMA_VERSION, SessionLog,
                                        read_session_log)


def test_records_round_trip_in_order_with_schema_version(tmp_path):
    log = SessionLog(tmp_path / "nested" / "session.jsonl")
    log.append("event", {"hand_id": "1-1", "event": {"kind": "HandStarted", "data": {}}})
    log.append("user_action", {"hand_id": "1-1", "action": {"type": "FOLD", "amount": None}})
    records = read_session_log(log.path)
    assert [r["type"] for r in records] == ["event", "user_action"]
    assert all(r["schema_version"] == SCHEMA_VERSION for r in records)
    assert records[1]["action"] == {"type": "FOLD", "amount": None}


def test_record_from_a_newer_schema_is_rejected(tmp_path):
    path = tmp_path / "future.jsonl"
    path.write_text(json.dumps({"schema_version": SCHEMA_VERSION + 1, "type": "event"}) + "\n")
    with pytest.raises(ValueError, match="newer than supported"):
        read_session_log(path)
