import http.client
import json
import pytest
from   thpoker.analysis.ev      import PROFILES
from   thpoker.cli              import build_config, parse_args
from   thpoker.game.cards       import card_str
from   thpoker.game.state       import Action, ActionType
from   thpoker.storage          import read_session_log
import thpoker.table
from   thpoker.tests.text       import fold_to_a_bet
from   thpoker.text             import pretty_cards
from   thpoker.web              import (App, Download, MAX_SESSIONS, WebError,
                                        WebTable, _decision_json, serve)
import threading
from   typing                   import Any

SETUP = {"mode": "cash", "seats": 3, "tier": 1, "seed": 8}


def call(app: App, method: str, parts: list[str], payload: dict[str, Any]) -> dict[str, Any]:
    answer = app.handle(method, parts, payload)
    assert isinstance(answer, dict)
    return answer


def play_out(app: App, key: str) -> dict:
    """Check or call until the hand is over, checking that no hidden card is ever shown."""
    state = call(app, "GET", ["sessions", key], {})
    while not state["hand_over"]:
        for seat in state["seats"][1:]:
            assert seat.get("cards") is None or state["hand_over"]  # opponents' cards stay hidden
        legal = state["legal"]
        state = call(
            app,
            "POST",
            ["sessions", key, "action"],
            {"kind": "check" if legal["check"] else "call"},
        )
    return state


def test_a_session_is_played_hand_after_hand_and_reviewed():
    app = App(None)
    created = call(app, "POST", ["sessions"], SETUP)
    key, state = created["id"], created["state"]
    assert [s["name"] for s in state["seats"]][0] == "You" and len(state["seats"]) == 3
    assert state["seats"][0]["cards"] and all(s.get("cards") is None for s in state["seats"][1:])
    for _ in range(3):
        state = play_out(app, key)
        assert state["hand_over"] and any(" win" in line for line in state["log"])
        call(app, "POST", ["sessions", key, "next"], {})
    state = play_out(app, key)
    review = call(app, "POST", ["sessions", key, "review"], {})
    assert review["result"].startswith("You ")  # the hand may have needed no decision


def test_bad_requests_are_refused():
    app = App(None)
    with pytest.raises(WebError, match="wrong type"):
        call(app, "POST", ["sessions"], {**SETUP, "seats": "three"})
    with pytest.raises(WebError, match="can.t start that game"):
        call(app, "POST", ["sessions"], {**SETUP, "seats": 9})
    with pytest.raises(KeyError):
        call(app, "GET", ["sessions", "nope"], {})
    key = call(app, "POST", ["sessions"], SETUP)["id"]
    play_out(app, key)
    with pytest.raises(WebError, match="not your turn"):
        call(app, "POST", ["sessions", key, "action"], {"kind": "call"})
    with pytest.raises(WebError, match="wrong type"):
        call(app, "POST", ["sessions"], {**SETUP, "stack": float("inf")})
    your_turn(app, key)
    with pytest.raises(WebError, match="finished hand"):
        call(app, "POST", ["sessions", key, "review"], {})
    with pytest.raises(WebError, match="needs an amount"):
        call(app, "POST", ["sessions", key, "action"], {"kind": "raise", "amount": float("nan")})
    with pytest.raises(WebError, match="needs a kind"):
        call(app, "POST", ["sessions", key, "action"], {"kind": ["call"]})


def your_turn(app: App, key: str) -> str:
    """Deal hands until the user is to act (a hand the user is not in ends before then)."""
    state = call(app, "GET", ["sessions", key], {})
    while not state["your_turn"]:
        state = call(app, "POST", ["sessions", key, "next"], {})
    return key


def shove_until(app: App, key: str, done: str) -> dict:
    """Go all-in every hand until the state flag `done` is set."""
    state = call(app, "GET", ["sessions", key], {})
    for _ in range(300):
        while not state["hand_over"]:
            state = call(app, "POST", ["sessions", key, "action"], {"kind": "allin"})
            assert state["hand_over"] or state["seats"][0]["all_in"]
        if state[done]:
            return state
        state = call(app, "POST", ["sessions", key, "next"], {})
    raise AssertionError(f"{done} never happened")


def test_a_busted_cash_player_rebuys_explicitly_and_each_hand_is_logged_once(tmp_path):
    app = App(tmp_path)
    key = call(app, "POST", ["sessions"], SETUP)["id"]
    shove_until(app, key, "awaiting_rebuy")
    with pytest.raises(WebError, match="rebuy"):
        call(app, "POST", ["sessions", key, "next"], {})
    state = call(app, "POST", ["sessions", key, "rebuy"], {})
    assert not state["awaiting_rebuy"] and state["seats"][0]["stack"] == 10_000
    state = call(app, "POST", ["sessions", key, "next"], {})
    (log,) = tmp_path.glob("session-*.jsonl")
    hands = [r["hand"]["hand_id"] for r in read_session_log(log) if r["type"] == "hand"]
    dealt = app.tables[key].runner.session.hand_number
    assert len(set(hands)) == len(hands) == dealt - (0 if state["hand_over"] else 1)


def test_a_knocked_out_tournament_player_fast_forwards_to_the_result():
    app = App(None)
    key = call(app, "POST", ["sessions"], {**SETUP, "mode": "tournament"})["id"]
    state = shove_until(app, key, "knocked_out")
    assert not state["session_over"]
    before = len(app.tables[key].lines)
    state = call(app, "POST", ["sessions", key, "next"], {})
    assert state["session_over"] and not state["knocked_out"]
    added = app.tables[key].lines[before:]
    # Only how the others finished is told, not each bot-only hand.
    assert "*** Tournament over" in added
    assert not any(line.startswith("===") or " win" in line for line in added)
    # The review is of the hand the user was knocked out in, not the last bot-only hand.
    review = call(app, "POST", ["sessions", key, "review"], {})
    assert review["decisions"] and review["result"].startswith("You lost")


def test_only_the_newest_sessions_are_kept():
    app = App(None)
    keys = [call(app, "POST", ["sessions"], SETUP)["id"] for _ in range(MAX_SESSIONS + 1)]
    with pytest.raises(KeyError):
        call(app, "GET", ["sessions", keys[0]], {})
    assert call(app, "GET", ["sessions", keys[-1]], {})["seats"]


def test_the_state_gives_the_smallest_chip_for_bet_sizes():
    state = call(App(None), "POST", ["sessions"], {**SETUP, "blinds": "0.5/1"})["state"]
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
        status, table = request(port, "GET", f"/api/sessions/{key}/hands.csv")
        assert status == 200 and table.startswith(b"session,date,")
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


def hidden_cards(app: App, key: str) -> set[str]:
    """The bots' cards in the current hand that were never shown, as the page writes cards."""
    table = app.tables[key]
    hand = table.runner.hand
    assert hand is not None
    shown = set(hand.shown) | set(table.shows.get(hand.hand_id, {}))
    return {
        pretty_cards([card])
        for seat, cards in enumerate(hand.hole_cards)
        if seat in table.runner.bots and seat not in shown and cards
        for card in cards
    }


def test_the_move_check_matches_the_hand_review_and_counts_as_a_hint(tmp_path):
    app = App(tmp_path)
    key = your_turn(app, call(app, "POST", ["sessions"], SETUP)["id"])
    with pytest.raises(WebError, match="haven't acted"):
        call(app, "POST", ["sessions", key, "analyze"], {})
    hint = call(app, "POST", ["sessions", key, "hint"], {})["decisions"][0]
    assert sum(o["strong_share"] for o in hint["options"]) == pytest.approx(1, abs=0.01)
    legal = call(app, "GET", ["sessions", key], {})["legal"]
    state = call(app, "POST", ["sessions", key, "action"], {"kind": "check" if legal["check"] else "call"})
    went_on = state["your_turn"]
    checked = call(app, "POST", ["sessions", key, "analyze"], {})
    hidden = hidden_cards(app, key)
    assert hidden and not any(card in checked["copy_text"] for card in hidden)
    # Fold to any bet (checking when it is free). With seed 8 a bot folds before the user acts,
    # so its cards are never shown.
    while state["your_turn"]:
        kind = "fold" if state["legal"]["call"] is not None else "check"
        state = call(app, "POST", ["sessions", key, "action"], {"kind": kind})
    reviewed = call(app, "POST", ["sessions", key, "review"], {})
    # The first decision of the hand, checked right after it, reads as the review reads it later.
    assert checked["decisions"][0] == reviewed["decisions"][0]
    hidden = hidden_cards(app, key)
    assert hidden and not any(card in reviewed["copy_text"] for card in hidden)
    (log,) = tmp_path.glob("session-*.jsonl")
    hints = [r for r in read_session_log(log) if r["type"] == "hint"]
    assert len(hints) == 1 + went_on  # the hint, and the check if play went on


def test_the_coach_is_off_in_tournaments_unless_asked_for():
    app = App(None)
    off = call(app, "POST", ["sessions"], {**SETUP, "mode": "tournament"})
    assert not off["state"]["coach"]
    key = your_turn(app, off["id"])
    with pytest.raises(WebError, match="coach is off"):
        call(app, "POST", ["sessions", key, "hint"], {})
    call(app, "POST", ["sessions", key, "action"], {"kind": "fold"})
    with pytest.raises(WebError, match="coach is off"):
        call(app, "POST", ["sessions", key, "analyze"], {})
    on = call(app, "POST", ["sessions"], {**SETUP, "mode": "tournament", "coach": True})
    assert on["state"]["coach"]


def test_past_sessions_read_back_the_same_hands_as_the_live_table(tmp_path):
    decisions = tmp_path / "decisions.jsonl"
    app = App(tmp_path, decisions_log=decisions)
    key = call(app, "POST", ["sessions"], SETUP)["id"]
    live = play_out(app, key)
    # Damaged logs are left out rather than breaking the page; a damaged decisions log only
    # leaves the review columns blank.
    (tmp_path / "session-98.jsonl").write_text("[1]\n")
    (tmp_path / "session-99.jsonl").write_text('{"type": "table_config", "config": {}}\n')
    decisions.write_text('{"schema_version": 1}\n')
    (saved,) = call(app, "GET", ["history"], {})["sessions"]
    assert saved["hands"] == 1 and saved["net"] == live["session_net"]
    undated = [{k: v for k, v in row.items() if k != "date"} for row in live["hands"]]
    rows = call(app, "GET", ["history", saved["name"]], {})["rows"]
    assert [{k: v for k, v in row.items() if k != "date"} for row in rows] == undated
    review = call(app, "POST", ["history", saved["name"], "review"], {"hand": 1})
    assert review["copy_text"].startswith("No-limit Texas Hold'em cash game")
    assert review["copy_text"].count(review["result"]) == 1 and review["result"] not in review["history"]
    everything = app.handle("GET", ["history.csv"], {})
    assert isinstance(everything, Download) and len(everything.text.splitlines()) == 2
    download = app.handle("GET", ["history", saved["name"], "hands.csv"], {})
    assert isinstance(download, Download) and download.text.splitlines()[1].startswith(saved["name"])
    for name in ("..", "session-0"):  # only listed logs can be read
        with pytest.raises(KeyError):
            app.handle("GET", ["history", name], {})


def test_the_state_counts_the_chips_in_the_pot_the_pot_odds_and_the_hand_result():
    # A 10-chip big-blind ante: dead money, in the pot but in nobody's bet.
    argv = ["--seats", "3", "--tier", "1", "--seed", "8", "--ante", "10", "--ante-type", "bb-ante"]
    table = WebTable(*build_config(parse_args(argv)), None, PROFILES["pc"], True)
    table.next_hand()
    for _ in range(10):  # deal until the user faces a bet
        state = table.state()
        if state["your_turn"] and state["legal"]["call"]:
            break
        while table.runner.user_to_act():
            table.act({"kind": "check" if table.state()["legal"]["check"] else "fold"})
        table.next_hand()
    else:
        pytest.fail("the user never faced a bet")
    for seat in state["seats"]:
        assert seat["in_pot"] == seat["committed"] + (10 if seat["position"] == "BB" else 0)
    you, to_call, pot = state["seats"][0], state["legal"]["call"], state["pot"]
    assert state["call_needs"] == pytest.approx(to_call / (pot + to_call))
    table.act({"kind": "fold"})
    folded = table.state()
    assert folded["hand_over"] and folded["hand_result"] == -you["in_pot"]
    assert folded["seats"][0]["last_action"] == "folded"


def test_a_bot_that_shows_has_its_cards_on_its_seat(monkeypatch):
    monkeypatch.setattr(thpoker.table, "SHOW_CHANCE", {1: 1.0, 2: 1.0, 3: 1.0})
    app = App(None)
    key = call(app, "POST", ["sessions"], SETUP)["id"]
    for _ in range(20):  # fold until a bot wins without a showdown
        state = call(app, "GET", ["sessions", key], {})
        while not state["hand_over"]:
            kind = "fold" if state["legal"]["call"] is not None else "check"
            state = call(app, "POST", ["sessions", key, "action"], {"kind": kind})
        hand = app.tables[key].runner.hand
        assert hand is not None
        (award, *more) = hand.awards
        if not more and len(award.eligible) == 1 and award.eligible[0] != 0:
            break
        call(app, "POST", ["sessions", key, "next"], {})
    else:
        pytest.fail("no bot won without a showdown")
    winner = award.eligible[0]
    cards = hand.hole_cards[winner]
    assert cards is not None
    shown = [seat.get("cards") for seat in state["seats"]]
    assert shown[winner] == [card_str(c) for c in cards]
    assert all(c is None for seat, c in enumerate(shown) if seat not in (0, winner))


def test_the_decision_json_counts_chips_on_the_table_or_shares_of_the_prize_pool():
    mix = {Action(ActionType.CALL): 0.75}
    cash = _decision_json(fold_to_a_bet(False), {}, 1, 100, mix)["options"]
    assert [(o["vs_bots"], o["vs_bots_noise"], o["strong_share"]) for o in cash] == [
        (0, 0, 0),
        (150, 10, 0.75),
        (25, 10, 0),
    ]
    small = _decision_json(fold_to_a_bet(False), {}, 100, 100, mix)["options"]  # blinds 0.5/1
    assert [o["vs_strong"] for o in small] == [0, 1.5, 0.25]
    tournament = _decision_json(fold_to_a_bet(True), {}, 1, 100, mix)
    assert tournament["unit"] == "% of the prize pool"
    assert [o["vs_bots"] for o in tournament["options"]] == [30, 33, 31]
