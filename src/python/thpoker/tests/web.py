import csv
import http.client
import io
import json
import pytest
from   thpoker.analysis.ev      import PROFILES
import thpoker.analysis.review
from   thpoker.bots.equity_bot  import public_seed
from   thpoker.bots.range_bot   import NEUTRAL
from   thpoker.cli              import (REVIEWS, build_config, load_session,
                                        parse_args)
from   thpoker.game.cards       import card_str, parse_cards
from   thpoker.game.engine      import observation
from   thpoker.game.rng         import Rng
from   thpoker.game.state       import Action, ActionType
from   thpoker.odds             import FULL_RANGE, hand_equity
from   thpoker.storage          import read_session_log
from   thpoker.table            import STYLE_WORDS
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
    # The hand as played so far with the move rated, for the copy without the coach; the log's
    # copy has every move rated so far.
    played = "\n".join(checked["history"]) + "\n"
    rated = "rated " + checked["decisions"][0]["summary"][0].split()[1]  # "Rated 0.97 of 1 ..."
    assert checked["hand_text"].startswith(played) and rated in checked["hand_text"]
    number = call(app, "GET", ["sessions", key], {})["hand_number"]
    copied = call(app, "GET", ["sessions", key, "hands", str(number), "copy"], {})["text"]
    assert copied.startswith(played) and rated in copied
    assert not any(card in copied for card in hidden)
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
    row = call(app, "GET", ["sessions", key], {})["hands"][-1]
    assert f"your cards {pretty_cards(parse_cards(row['cards']))}" in row["history"]
    assert not any(card in row["history"] for card in hidden)
    lines = reviewed["hand_text"].splitlines()
    assert lines[: len(reviewed["history"])] == reviewed["history"]
    assert lines[-2] == reviewed["summary"] and reviewed["result"].startswith(lines[-1])
    assert "With the cards" not in lines[-1]  # the all-in average is the coach's, not the hand's
    # The player's view: no option values, ranges or styles, only the ratings.
    words = ("Options", "likely hold", *STYLE_WORDS.values())
    assert not any(word in reviewed["hand_text"] for word in words)
    assert reviewed["summary"] in reviewed["copy_text"]
    (log,) = tmp_path.glob("session-*.jsonl")
    hints = [r for r in read_session_log(log) if r["type"] == "hint"]
    assert len(hints) == 1 + went_on  # the hint, and the check if play went on


def test_no_move_is_reviewed_twice(monkeypatch):
    app = App(None)
    key = call(app, "POST", ["sessions"], SETUP)["id"]
    play_out(app, key)
    REVIEWS.submit(int).result()  # reviews of the first hand are not counted
    computed: list[tuple[str, int]] = []
    original = thpoker.analysis.review._review_decision

    def counted(*args: Any) -> Any:
        computed.append((args[1].state.hand_id, args[0]))
        return original(*args)

    monkeypatch.setattr(thpoker.analysis.review, "_review_decision", counted)
    for _ in range(20):  # a hand with two or more decisions: the first hinted, each checked
        state = call(app, "POST", ["sessions", key, "next"], {})
        checked: list[dict] = []
        while state["your_turn"]:
            if not checked:
                call(app, "POST", ["sessions", key, "hint"], {})
            kind = "check" if state["legal"]["check"] else "call"
            state = call(app, "POST", ["sessions", key, "action"], {"kind": kind})
            checked.append(call(app, "POST", ["sessions", key, "analyze"], {})["decisions"][0])
        state = play_out(app, key)
        if len(checked) >= 2:
            break
    else:
        pytest.fail("no hand with two decisions")
    reviewed = call(app, "POST", ["sessions", key, "review"], {})["decisions"]
    REVIEWS.submit(int).result()
    assert reviewed == checked
    hand = app.tables[key].finished
    assert hand is not None and len(computed) == len(set(computed))
    mine = [i for i, e in enumerate(hand.history) if e.seat == 0]
    assert sorted(i for hand_id, i in computed if hand_id == hand.hand_id) == mine


def test_a_request_waiting_for_a_review_holds_up_no_one_and_ends_with_its_table(monkeypatch, tmp_path):
    app = App(tmp_path)
    key = call(app, "POST", ["sessions"], SETUP)["id"]
    play_out(app, key)
    REVIEWS.submit(int).result()
    entered, release = threading.Event(), threading.Event()
    original = thpoker.analysis.review._review_decision

    def held(*args: Any) -> Any:
        entered.set()
        release.wait(60)
        return original(*args)

    monkeypatch.setattr(thpoker.analysis.review, "_review_decision", held)
    outcome: list[object] = []

    def ask():
        try:
            outcome.append(call(app, "POST", ["sessions", key, "hint"], {}))
        except KeyError as error:
            outcome.append(error)

    asking = threading.Thread(target=ask)
    # With the review thread kept busy, the hint request reviews the decision itself.
    REVIEWS.submit(release.wait, 60)
    try:
        your_turn(app, key)
        asking.start()
        assert entered.wait(60)
        state = call(app, "GET", ["sessions", key], {})
        assert state["your_turn"] and asking.is_alive() and not outcome
        (saved,) = call(app, "GET", ["history"], {})["sessions"]
        call(app, "POST", ["history", "delete"], {"names": [saved["name"]]})  # while the hint waits
    finally:
        release.set()
        if asking.ident is not None:
            asking.join(60)
    assert isinstance(outcome[0], KeyError)


def test_a_training_session_holds_the_seat_and_deals_the_chosen_hands(tmp_path):
    app = App(tmp_path)
    setup = {**SETUP, "mode": "training", "seats": 4, "position": "BB", "hands": "pairs"}
    created = call(app, "POST", ["sessions"], setup)
    key, state = created["id"], created["state"]
    assert state["coach"] and state["training"] == {"position": "BB", "hands": "pairs"}
    for _ in range(5):
        me = state["seats"][0]
        assert me["position"] == "BB" and me["cards"][0][0] == me["cards"][1][0]
        play_out(app, key)
        state = call(app, "POST", ["sessions", key, "next"], {})
    assert {row["mode"] for row in state["hands"]} == {"training"}
    (saved,) = call(app, "GET", ["history"], {})["sessions"]
    assert saved["mode"] == "training" and saved["training"] == state["training"]
    preview = call(app, "GET", ["hands", "0-5"], {})
    assert preview["label"] == "best 5%" and preview["classes"][0] == 1.0  # AA
    with pytest.raises(ValueError, match="unknown hands"):
        call(app, "GET", ["hands", "top"], {})
    with pytest.raises(WebError, match="seat"):
        call(app, "POST", ["sessions"], {**setup, "position": "--seed"})


@pytest.mark.parametrize(
    "spec, label",
    [
        ("25-50", "top 25-50%"),
        ("50-100", "worst 50%"),
        ("0-1", "AA, KK only"),
        ("0.5-5", "top 0.5-5%"),  # top 0.5-5%; note the best AA is left out
        ("0-100", "any hand"),
        ("small-aces", "small aces"),
    ],
)
def test_training_hands_are_named_from_the_classes_dealt(spec, label):
    assert call(App(None), "GET", ["hands", spec], {})["label"] == label


def test_training_shows_the_chance_to_win_against_the_likely_hands():
    app = App(None)
    setup = {**SETUP, "mode": "training", "position": "BTN", "hands": "0-0.1"}
    three = call(app, "POST", ["sessions"], {**setup, "seats": 3})
    state = three["state"]
    assert state["your_turn"] and [c[0] for c in state["seats"][0]["cards"]] == ["A", "A"]
    # First to act, so against any two cards (published): aces win 73.4% against two players,
    # 85.2% against one.
    first = call(app, "GET", ["sessions", three["id"], "chance"], {})
    assert first["chance"] == pytest.approx(0.734, abs=0.02)
    seats = [(o["seat"], o["chance"], o["acted"]) for o in first["opponents"]]
    assert seats == [(1, 0.852, False), (2, 0.852, False)]
    key = call(app, "POST", ["sessions"], {**setup, "seats": 2})["id"]
    state = call(app, "POST", ["sessions", key, "action"], {"kind": "call"})
    # With seed 8 the big blind checks before and after the flop: weaker than any two cards.
    hand = app.tables[key].runner.hand
    assert state["your_turn"] and len(hand.board) == 3 and hand.hole_cards[0] is not None
    seed = public_seed(observation(hand, 0))
    any_two = hand_equity(hand.hole_cards[0], hand.board, [FULL_RANGE], Rng(seed))
    answer = call(app, "GET", ["sessions", key, "chance"], {})
    assert answer["chance"] > any_two.value + 0.02 and answer["opponents"][0]["acted"]
    play_out(app, key)
    with pytest.raises(WebError, match="hand being played"):
        call(app, "GET", ["sessions", key, "chance"], {})
    cash = call(app, "POST", ["sessions"], SETUP)["id"]
    with pytest.raises(WebError, match="training only"):
        call(app, "GET", ["sessions", cash, "chance"], {})


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
    hold = threading.Event()
    REVIEWS.submit(hold.wait)  # the hand's reviews wait until it is over
    try:
        live = play_out(app, your_turn(app, key))
        row = live["hands"][-1]
        assert row["acted"] and live["pending"] == 1 and row["pending"] and row["review"] is None
        (saved,) = call(app, "GET", ["history"], {})["sessions"]
        assert call(app, "GET", ["history", saved["name"]], {})["pending"] == 1
    finally:
        hold.set()
    REVIEWS.submit(int).result()
    # The rating landed after the hand: the next read of the table shows it, as the log has it.
    live = call(app, "GET", ["sessions", key], {})
    row = live["hands"][-1]
    (log,) = tmp_path.glob("session-*.jsonl")
    (logged,) = [r for r in read_session_log(log) if r["type"] == "rating"]
    moves = logged["moves"]
    weighted = sum(m["rating"] * m["stake"] for m in moves) / sum(m["stake"] for m in moves)
    assert live["pending"] == 0 and not row["pending"]
    assert (row["win_chance"], row["rating"]) == (round(logged["chance"], 3), round(weighted, 2))
    assert row["review"][-1].startswith("Your cards") and f"{row['rating']:.2f}" in row["review"][-1]
    # Damaged logs are left out rather than breaking the page; a damaged decisions log only
    # leaves the review columns blank.
    (tmp_path / "session-98.jsonl").write_text("[1]\n")
    (tmp_path / "session-99.jsonl").write_text('{"type": "table_config", "config": {}}\n')
    decisions.write_text('{"schema_version": 1}\n')
    (saved,) = call(app, "GET", ["history"], {})["sessions"]
    assert saved["hands"] == len(live["hands"]) and saved["net"] == live["session_net"]
    undated = [{k: v for k, v in row.items() if k != "date"} for row in live["hands"]]
    rows = call(app, "GET", ["history", saved["name"]], {})["rows"]
    assert [{k: v for k, v in row.items() if k != "date"} for row in rows] == undated
    review = call(app, "POST", ["history", saved["name"], "review"], {"hand": row["hand"]})
    # The review highlights the hand's rating as its row gives it, in the band of its verdicts.
    shown = round(weighted, 2)
    band = "best" if shown == 1 else "close" if shown >= 0.75 else "mistake"
    assert (round(review["rating"], 2), review["band"], row["band"]) == (row["rating"], band, band)
    assert review["copy_text"].startswith("No-limit Texas Hold'em cash game")
    assert review["copy_text"].count(review["result"]) == 1 and review["result"] not in review["history"]
    everything = app.handle("GET", ["history.csv"], {})
    assert isinstance(everything, Download)
    *_, exported = csv.DictReader(io.StringIO(everything.text))
    assert exported["history"] == row["history"]
    assert exported["rating"] == str(row["rating"]) and exported["win_chance"] == str(row["win_chance"])
    download = app.handle("GET", ["history", saved["name"], "hands.csv"], {})
    assert isinstance(download, Download) and download.text.splitlines()[1].startswith(saved["name"])
    for name in ("..", "session-0"):  # only listed logs can be read
        with pytest.raises(KeyError):
            app.handle("GET", ["history", name], {})


def test_an_expert_session_comes_back_with_the_reads_it_was_played_with(tmp_path):
    # Heads-up against Expert, folding the first move of every third hand: a fold on the button
    # ends a hand before any bot acts, and its read must come back all the same.
    app = App(tmp_path)
    key = call(app, "POST", ["sessions"], {**SETUP, "seats": 2, "tier": 4})["id"]
    for number in range(12):
        state = call(app, "GET", ["sessions", your_turn(app, key)], {})
        if number % 3 == 2 and state["legal"]["call"] is not None:
            call(app, "POST", ["sessions", key, "action"], {"kind": "fold"})
        state = play_out(app, key)
    live = app.tables[key].runner.reads
    (log,) = tmp_path.glob("session-*.jsonl")
    records = read_session_log(log)
    acted = {
        r["event"]["data"]["hand_id"] for r in records if r["type"] == "event" and r["event"]["kind"] == "BotDecision"
    }
    assert set(live) - acted and len(live) == 12 and list(live.values())[-1] != NEUTRAL
    assert load_session(log)[2].reads == live
    (saved,) = call(app, "GET", ["history"], {})["sessions"]
    rows = call(app, "GET", ["history", saved["name"]], {})["rows"]
    assert app.saved[log].runner is not None and app.saved[log].runner.reads == live
    # The last hand was folded at once: its review values the other options by how the bots
    # answer, with the hand's read, live and saved alike.
    reviewed = call(app, "POST", ["sessions", key, "review"], {})
    again = call(app, "POST", ["history", saved["name"], "review"], {"hand": state["hand_number"]})
    assert reviewed["decisions"] == again["decisions"] and rows[-1]["hand"] == state["hand_number"]


def test_ticked_sessions_are_deleted_together_and_nothing_brings_them_back(tmp_path):
    app = App(tmp_path)
    keys = {seed: call(app, "POST", ["sessions"], {**SETUP, "seed": seed})["id"] for seed in (7, 8, 9)}
    hold = threading.Event()
    REVIEWS.submit(hold.wait)  # the hands' ratings have not run when the sessions are deleted
    try:
        for key in keys.values():
            play_out(app, key)
        with pytest.raises(KeyError):  # an unknown name deletes none of them
            app.handle("POST", ["history", "delete"], {"names": ["session-7", "session-1"]})
        assert len(call(app, "GET", ["history"], {})["sessions"]) == 3
        listed = call(app, "POST", ["history", "delete"], {"names": ["session-7", "session-9"]})
    finally:
        hold.set()
    REVIEWS.submit(int).result()
    assert [s["name"] for s in listed["sessions"]] == ["session-8"]
    assert [p.stem for p in tmp_path.glob("session-*.jsonl")] == ["session-8"]
    call(app, "GET", ["sessions", keys[8]], {})  # the table writing the kept one plays on
    for seed in (7, 9):
        with pytest.raises(KeyError):  # the tables writing them went with them
            call(app, "GET", ["sessions", keys[seed]], {})


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
