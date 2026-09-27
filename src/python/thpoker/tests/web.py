import http.client
import json
import threading

import pytest

from   thpoker.storage          import read_session_log
from   thpoker.web              import App, MAX_SESSIONS, WebError, serve

SETUP = {"mode": "cash", "seats": 3, "tier": 1, "seed": 8}


def play_out(app: App, key: str) -> dict:
    """Check or call until the hand is over, checking that no hidden card is ever shown."""
    state = app.handle("GET", ["sessions", key], {})
    while not state["hand_over"]:
        for seat in state["seats"][1:]:
            assert seat.get("cards") is None or state["hand_over"]  # opponents' cards stay hidden
        legal = state["legal"]
        state = app.handle("POST", ["sessions", key, "action"], {"kind": "check" if legal["check"] else "call"})
    return state


def test_a_session_is_played_hand_after_hand_and_reviewed():
    app = App(None)
    created = app.handle("POST", ["sessions"], SETUP)
    key, state = created["id"], created["state"]
    assert [s["name"] for s in state["seats"]][0] == "You" and len(state["seats"]) == 3
    assert state["seats"][0]["cards"] and all(s.get("cards") is None for s in state["seats"][1:])
    for _ in range(3):
        state = play_out(app, key)
        assert state["hand_over"] and any(line.startswith("  Pot") for line in state["log"])
        app.handle("POST", ["sessions", key, "next"], {})
    state = play_out(app, key)
    review = app.handle("POST", ["sessions", key, "review"], {})["lines"]
    assert review and (review[0].startswith("-- Decision 1") or "no decision" in review[0])


def test_bad_requests_are_refused():
    app = App(None)
    with pytest.raises(WebError, match="wrong type"):
        app.handle("POST", ["sessions"], {**SETUP, "seats": "three"})
    with pytest.raises(WebError, match="invalid setup"):
        app.handle("POST", ["sessions"], {**SETUP, "seats": 9})
    with pytest.raises(KeyError):
        app.handle("GET", ["sessions", "nope"], {})
    key = app.handle("POST", ["sessions"], SETUP)["id"]
    play_out(app, key)
    with pytest.raises(WebError, match="not your turn"):
        app.handle("POST", ["sessions", key, "action"], {"kind": "call"})
    with pytest.raises(WebError, match="wrong type"):
        app.handle("POST", ["sessions"], {**SETUP, "stack": float("inf")})
    your_turn(app, key)
    with pytest.raises(WebError, match="finished hand"):
        app.handle("POST", ["sessions", key, "review"], {})
    with pytest.raises(WebError, match="needs an amount"):
        app.handle("POST", ["sessions", key, "action"], {"kind": "raise", "amount": float("nan")})
    with pytest.raises(WebError, match="needs a kind"):
        app.handle("POST", ["sessions", key, "action"], {"kind": ["call"]})


def your_turn(app: App, key: str) -> str:
    """Deal hands until the user is to act (a hand the user is not in ends before then)."""
    state = app.handle("GET", ["sessions", key], {})
    while not state["your_turn"]:
        state = app.handle("POST", ["sessions", key, "next"], {})
    return key


def shove_until(app: App, key: str, done: str) -> dict:
    """Go all-in every hand until the state flag `done` is set."""
    state = app.handle("GET", ["sessions", key], {})
    for _ in range(300):
        while not state["hand_over"]:
            state = app.handle("POST", ["sessions", key, "action"], {"kind": "allin"})
            assert state["hand_over"] or state["seats"][0]["all_in"]
        if state[done]:
            return state
        state = app.handle("POST", ["sessions", key, "next"], {})
    raise AssertionError(f"{done} never happened")


def test_a_busted_cash_player_rebuys_explicitly_and_each_hand_is_logged_once(tmp_path):
    app = App(tmp_path)
    key = app.handle("POST", ["sessions"], SETUP)["id"]
    shove_until(app, key, "awaiting_rebuy")
    with pytest.raises(WebError, match="rebuy"):
        app.handle("POST", ["sessions", key, "next"], {})
    state = app.handle("POST", ["sessions", key, "rebuy"], {})
    assert not state["awaiting_rebuy"] and state["seats"][0]["stack"] == 10_000
    state = app.handle("POST", ["sessions", key, "next"], {})
    (log,) = tmp_path.glob("session-*.jsonl")
    hands = [r["hand"]["hand_id"] for r in read_session_log(log) if r["type"] == "hand"]
    dealt = app.tables[key].runner.session.hand_number
    assert len(set(hands)) == len(hands) == dealt - (0 if state["hand_over"] else 1)


def test_a_knocked_out_tournament_player_fast_forwards_to_the_result():
    app = App(None)
    key = app.handle("POST", ["sessions"], {**SETUP, "mode": "tournament"})["id"]
    state = shove_until(app, key, "knocked_out")
    assert not state["session_over"]
    before = len(app.tables[key].lines)
    state = app.handle("POST", ["sessions", key, "next"], {})
    assert state["session_over"] and not state["knocked_out"]
    added = app.tables[key].lines[before:]
    # Only how the others finished is told, not each bot-only hand.
    assert "*** Tournament over" in added
    assert not any(line.startswith(("===", "  Pot")) for line in added)
    # The review is of the hand the user was knocked out in, not the last bot-only hand.
    review = app.handle("POST", ["sessions", key, "review"], {})["lines"]
    assert review[0].startswith("-- Decision 1")


def test_only_the_newest_sessions_are_kept():
    app = App(None)
    keys = [app.handle("POST", ["sessions"], SETUP)["id"] for _ in range(MAX_SESSIONS + 1)]
    with pytest.raises(KeyError):
        app.handle("GET", ["sessions", keys[0]], {})
    assert app.handle("GET", ["sessions", keys[-1]], {})["seats"]


def test_the_state_gives_the_smallest_chip_for_bet_sizes():
    state = App(None).handle("POST", ["sessions"], {**SETUP, "blinds": "0.5/1"})["state"]
    assert state["step"] == 0.01


def request(port: int, method: str, path: str, body: dict | str | None = None, **headers: str) -> tuple[int, bytes]:
    """One HTTP request to the local server (http.client only speaks HTTP to that host). A
    string body goes as is; headers override the defaults, underscores standing for dashes."""
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    try:
        payload = body if body is None or isinstance(body, str) else json.dumps(body)
        sent = {"Content-Type": "application/json"} | {name.replace("_", "-"): value for name, value in headers.items()}
        connection.putrequest(method, path, skip_host="Host" in sent)
        for name, value in sent.items():
            connection.putheader(name, value)
        if "Content-Length" not in sent:
            connection.putheader("Content-Length", str(len(payload or "")))
        connection.endheaders(payload.encode() if payload else None)
        response = connection.getresponse()
        return response.status, response.read()
    finally:
        connection.close()


def test_the_server_serves_the_page_and_the_api_on_this_machine_only():
    server = serve(0)
    host, port = server.server_address[:2]
    assert host == "127.0.0.1"
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        status, page = request(port, "GET", "/")
        assert status == 200 and b"Hold'em trainer" in page
        status, created = request(port, "POST", "/api/sessions", SETUP)
        key = json.loads(created)["id"]
        status, state = request(port, "GET", f"/api/sessions/{key}")
        assert status == 200 and "seats" in json.loads(state)
        assert request(port, "GET", "/../thpoker/web.py")[0] == 404
        assert request(port, "POST", "/api/sessions", {"seats": "three"})[0] == 400
    finally:
        server.shutdown()
        server.server_close()


def test_the_server_answers_malformed_and_foreign_requests_without_dropping_them():
    server = serve(0)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        status, created = request(port, "POST", "/api/sessions", SETUP)
        key = your_turn(server.RequestHandlerClass.app, json.loads(created)["id"])
        action = f"/api/sessions/{key}/action"
        assert request(port, "POST", action, '{"kind": "raise", "amount": NaN}')[0] == 400
        assert request(port, "POST", action, {"kind": ["call"]})[0] == 400
        assert request(port, "POST", "/api/sessions", {**SETUP, "seed": -5})[0] == 400
        assert request(port, "POST", "/api/sessions", "{}", Content_Length="-1")[0] == 400
        assert request(port, "POST", "/api/sessions", "{}", Content_Length="many")[0] == 400
        # Another site's page: a name that resolves here, or a form post without JSON.
        assert request(port, "GET", "/", Host=f"evil.example:{port}")[0] == 403
        assert request(port, "POST", "/api/sessions", "{}", Content_Type="text/plain")[0] == 415
        assert request(port, "GET", "/")[0] == 200  # still serving
    finally:
        server.shutdown()
        server.server_close()
