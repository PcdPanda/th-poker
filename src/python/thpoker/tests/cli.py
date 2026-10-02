import builtins
import csv
from   dataclasses              import replace
import json
from   pathlib                  import Path
import pytest
from   thpoker.analysis.review  import position_names
from   thpoker.analysis.stats   import DecisionRecord, progress
from   thpoker.charts           import seat_names
from   thpoker.cli              import build_config, main, parse_args
from   thpoker.game.cards       import preflop_class
from   thpoker.game.engine      import new_hand, observation
from   thpoker.game.session     import Mode, begin_hand, start_session
from   thpoker.game.state       import AnteType, GameState
from   thpoker.storage          import SessionLog


def answers(monkeypatch, actions=(), between=(), other="", special=None):
    """Script input(): action prompts take `actions` in turn and then "c", between-hands prompts
    take `between` and then "q", a prompt containing a key of `special` gets its value, and any
    other prompt `other`."""
    moves, breaks = iter(actions), iter(between)

    def scripted(prompt: str = "") -> str:
        if prompt.startswith("Your action"):
            return next(moves, "c")
        if "next hand" in prompt:
            return next(breaks, "q")
        return next((v for k, v in (special or {}).items() if k in prompt), other)

    monkeypatch.setattr(builtins, "input", scripted)


def test_small_user_blinds_are_scaled_to_at_least_100_units():
    config, scale = build_config(parse_args(["--blinds", "1/2", "--ante", "0.5", "--seed", "3"]))
    blinds = config.session.cash_blinds
    assert scale == 50
    assert (blinds.small_blind, blinds.big_blind, blinds.ante, blinds.ante_type) == (
        50,
        100,
        25,
        AnteType.PER_PLAYER,
    )
    assert config.session.starting_stack == 10_000


def test_tournament_preset_sets_stack_and_level_length():
    config, _ = build_config(parse_args(["--mode", "tournament", "--preset", "turbo", "--seats", "8", "--seed", "3"]))
    session = config.session
    assert session.mode == Mode.TOURNAMENT and session.num_seats == 8
    assert session.starting_stack == 5_000 and session.tournament.hands_per_level == 8


def test_quick_play_session_runs_and_quits(monkeypatch, capsys):
    answers(monkeypatch, between=[""] * 4)
    assert main(["--seed", "8", "--no-log"]) == 0
    output = capsys.readouterr().out
    assert "Session over after" in output and "=== Hand" in output


@pytest.mark.parametrize(
    "argv",
    [
        ["--seats", "9"],
        ["--blinds", "100/50"],
        ["--blinds", "0/0"],
        ["--mode", "tournament", "--ante", "10"],
        ["--mode", "tournament", "--blinds", "40/100"],
        ["--mode", "tournament", "--hands-per-level", "5", "--minutes-per-level", "3"],
        ["--preset", "deep"],
        ["--auto-rebuy", "50", "--reset-stacks"],
        ["--position", "CO"],
        ["--mode", "tournament", "--hands", "pairs"],
        ["--mode", "training", "--seats", "3", "--position", "UTG"],
        ["--mode", "training", "--hands", "top"],
    ],
)
def test_invalid_setup_exits_with_an_error(argv, capsys):
    assert main(argv + ["--no-log"]) == 2
    assert "Invalid setup" in capsys.readouterr().err


def test_training_keeps_the_user_at_the_chosen_position_every_hand():
    for seats in range(2, 9):
        for name in seat_names(seats):
            argv = ["--mode", "training", "--seats", str(seats), "--position", name.lower()]
            config, _ = build_config(parse_args(argv))
            state, _ = start_session(config.session)
            for _ in range(4):
                state, setup, _ = begin_hand(state)
                hand, _ = new_hand(setup.config, setup.seed, setup.button, setup.stacks, setup.dealt_in)
                assert position_names(observation(hand, 0))[0] == name


def test_training_decisions_are_logged_but_kept_out_of_the_measures(monkeypatch, capsys, tmp_path):
    answers(monkeypatch, between=[""])
    argv = ["--mode", "training", "--hands", "pairs", "--seed", "8", "--seats", "3"]
    assert main(argv + ["--log-dir", str(tmp_path), "--training-log", str(tmp_path / "t.jsonl")]) == 0
    assert "Training session" in capsys.readouterr().out
    log, decisions = tmp_path / "session-8.jsonl", tmp_path / "decisions.jsonl"
    records = [json.loads(line) for line in log.read_text().splitlines()]
    dealt = [GameState.from_dict(r["hand"]).hole_cards[0] for r in records if r["type"] == "hand"]
    assert len(dealt) == 2 and all(cards and len(preflop_class(*cards)) == 2 for cards in dealt)
    assert main(["review", str(log), "--decisions-log", str(decisions)]) == 0
    reviewed = DecisionRecord.read(decisions, "decision")
    assert reviewed and {r.tags["mode"] for r in reviewed} == {"training"}
    assert progress(reviewed) is None


def test_a_logged_session_can_be_reviewed_after_a_hand_and_later(monkeypatch, capsys, tmp_path):
    answers(monkeypatch, between=["r", ""] * 3)
    training = str(tmp_path / "training.jsonl")
    assert main(["--seed", "8", "--seats", "3", "--log-dir", str(tmp_path), "--training-log", training]) == 0
    played = capsys.readouterr().out
    assert "-- Decision 1" in played and " 5 Options" in played
    log = tmp_path / "session-8.jsonl"
    decisions = str(tmp_path / "decisions.jsonl")
    assert main(["review", str(log), "--top", "3", "--decisions-log", decisions]) == 0
    summary = capsys.readouterr().out
    assert "decisions:" in summary and "Net " in summary
    hands = [json.loads(line) for line in log.read_text().splitlines()]
    user_hands = [
        r["hand"]["hand_id"] for r in hands if r["type"] == "hand" and any(e["seat"] == 0 for e in r["hand"]["history"])
    ]
    assert main(["review", str(log), "--hand", user_hands[0].removeprefix("hand-")]) == 0
    assert "1 Situation" in capsys.readouterr().out
    # The export has a row per logged hand, and every reviewed decision lands on its hand's row.
    out = tmp_path / "hands.csv"
    assert main(["export", str(log), "--out", str(out), "--decisions-log", decisions]) == 0
    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    logged = [r["hand"]["hand_id"].removeprefix("hand-") for r in hands if r["type"] == "hand"]
    assert [row["hand"] for row in rows] == logged
    reviewed = DecisionRecord.read(Path(decisions), "decision")
    assert sum(int(row["decisions"] or 0) for row in rows) == len(reviewed) > 0


def test_predict_then_reveal_logs_estimates_for_calibration(monkeypatch, capsys, tmp_path):
    answers(monkeypatch, between=["p", ""] * 3, special={"(%, Enter to skip)": "50", "Best option": "1"})
    training = tmp_path / "training.jsonl"
    argv = [
        "--seed",
        "8",
        "--seats",
        "3",
        "--log-dir",
        str(tmp_path),
        "--training-log",
        str(training),
    ]
    assert main(argv) == 0
    assert "You said 50%" in capsys.readouterr().out
    kinds = {json.loads(line)["kind"] for line in training.read_text().splitlines()}
    assert {"equity", "best_option"} <= kinds
    assert main(["calibration", "--training-log", str(training)]) == 0
    assert "Equity against the range: off by" in capsys.readouterr().out


def test_drills_are_graded_logged_and_repeated_when_missed(monkeypatch, capsys, tmp_path):
    training = str(tmp_path / "training.jsonl")
    answers(monkeypatch, other="99")  # far from any threshold
    assert main(["drill", "thresholds", "--count", "2", "--seed", "1", "--training-log", training]) == 0
    first = capsys.readouterr().out
    assert first.count("Wrong.") == 2 and "It is" in first
    # Missed spots are due again at once, so the next session starts with them.
    assert main(["drill", "thresholds", "--count", "1", "--seed", "2", "--training-log", training]) == 0
    again = capsys.readouterr().out
    missed = {line[3:] for line in first.splitlines() if line[:3] in ("1. ", "2. ")}
    assert [line[3:] for line in again.splitlines() if line.startswith("1. ")][0] in missed
    records = [json.loads(line) for line in (tmp_path / "training.jsonl").read_text().splitlines()]
    assert [r["grade"] for r in records if r["type"] == "drill"] == ["wrong"] * 3


def test_reviewing_a_session_feeds_the_leak_statistics(monkeypatch, capsys, tmp_path):
    answers(monkeypatch, between=[""] * 3)
    training, decisions = str(tmp_path / "training.jsonl"), tmp_path / "decisions.jsonl"
    assert main(["--seed", "8", "--seats", "3", "--log-dir", str(tmp_path), "--training-log", training]) == 0
    log = str(tmp_path / "session-8.jsonl")
    assert main(["review", log, "--decisions-log", str(decisions)]) == 0
    first = decisions.read_text().splitlines()
    assert first and all(json.loads(line)["session"] == "session-8" for line in first)
    assert main(["review", log, "--decisions-log", str(decisions)]) == 0  # a second review adds nothing
    assert decisions.read_text().splitlines() == first
    capsys.readouterr()
    assert main(["leaks", "--decisions-log", str(decisions)]) == 0
    assert "reviewed decisions" in capsys.readouterr().out


def test_a_hint_is_shown_logged_and_kept_out_of_the_statistics(monkeypatch, capsys, tmp_path):
    asked = []

    def scripted_input(prompt: str) -> str:
        if prompt.startswith("Your action"):
            asked.append(prompt)
            return "h" if len(asked) == 1 else "c"
        # Keep dealing until the user has acted; some hands are walks.
        return "q" if "r to review" in prompt and len(asked) >= 3 else ""

    monkeypatch.setattr(builtins, "input", scripted_input)
    argv = [
        "--seed",
        "8",
        "--seats",
        "3",
        "--log-dir",
        str(tmp_path),
        "--training-log",
        str(tmp_path / "t.jsonl"),
    ]
    assert main(argv) == 0
    assert "your move" in capsys.readouterr().out
    log = tmp_path / "session-8.jsonl"
    (hinted,) = [json.loads(line) for line in log.read_text().splitlines() if json.loads(line)["type"] == "hint"]
    decisions = tmp_path / "decisions.jsonl"
    assert main(["review", str(log), "--decisions-log", str(decisions)]) == 0
    recorded = {(r["hand_id"], r["index"]) for r in map(json.loads, decisions.read_text().splitlines())}
    # The hint was asked at one of the user's decisions; every other decision is recorded.
    records = [json.loads(line) for line in log.read_text().splitlines()]
    hands = [r["hand"] for r in records if r["type"] == "hand"]
    user = {(h["hand_id"], i) for h in hands for i, e in enumerate(h["history"]) if e["seat"] == 0}
    assert (hinted["hand_id"], hinted["index"]) in user
    assert recorded == user - {(hinted["hand_id"], hinted["index"])}


def test_hints_are_off_by_default_in_tournaments(monkeypatch, capsys, tmp_path):
    asked = []

    def scripted_input(prompt: str) -> str:
        if prompt.startswith("Your action"):
            asked.append(prompt)
            return "h" if len(asked) == 1 else "f"
        if "r to review" in prompt or "next hand" in prompt:
            return "q" if asked else ""
        return "n"

    monkeypatch.setattr(builtins, "input", scripted_input)
    assert main(["--mode", "tournament", "--seed", "8", "--seats", "3", "--no-log"]) == 0
    assert "Hints are off" in capsys.readouterr().out


def test_hidden_styles_are_guessed_and_revealed(monkeypatch, capsys):
    answers(monkeypatch, actions=["f"] * 99, between=[""], other="1")  # guess nit for everyone
    assert main(["--seed", "8", "--seats", "3", "--hide-styles", "--guess-every", "2", "--no-log"]) == 0
    output = capsys.readouterr().out
    assert "Guess each opponent's style" in output and "of 2 right." in output


def test_new_drill_questions_are_new(monkeypatch, capsys, tmp_path):
    training = tmp_path / "training.jsonl"
    answers(monkeypatch, other="30")
    assert main(["drill", "thresholds", "--count", "10", "--seed", "4", "--training-log", str(training)]) == 0
    keys = [json.loads(line)["key"] for line in training.read_text().splitlines()]
    assert len(keys) == 10 and len(set(keys)) == 10


def test_chart_drills_reveal_the_chart_and_are_scheduled(monkeypatch, capsys, tmp_path):
    training = tmp_path / "training.jsonl"
    answers(monkeypatch, other="1")
    for kind in ("preflop", "pushfold"):
        drill = ["drill", kind, "--count", "2", "--seed", "4", "--training-log", str(training)]
        assert main(drill) == 0
    assert capsys.readouterr().out.count("The chart:") == 4
    keys = [json.loads(line)["key"] for line in training.read_text().splitlines()]
    assert [k.split("/")[0] for k in keys] == ["preflop"] * 2 + ["pushfold"] * 2


def test_leaks_reports_patterns_from_recorded_decisions(capsys, tmp_path):
    decisions = tmp_path / "decisions.jsonl"
    log = SessionLog(decisions)
    tags = {
        "mode": "cash",
        "street": "river",
        "facing": "large",
        "hand": "marginal",
        "position": "BB",
    }
    for index in range(10):
        action = "fold" if index < 8 else "call"
        log.append(
            "decision",
            DecisionRecord("s", f"hand-{index}", 3, tags, 1.0, 0.0, "close", action, 0.5, None).to_dict(),
        )
    assert main(["leaks", "--decisions-log", str(decisions)]) == 0
    output = capsys.readouterr().out
    assert "you continue 20% of the time against a minimum defense of 50%" in output
    assert "30 points less than the theory" in output and "Loss per decision by spot" in output


def test_reviewed_mistakes_are_drilled_from_their_session_logs(monkeypatch, capsys, tmp_path):
    decisions, training = tmp_path / "decisions.jsonl", str(tmp_path / "training.jsonl")
    drill = ["drill", "mistakes", "--count", "1", "--device", "pc", "--log-dir", str(tmp_path)]
    drill += ["--decisions-log", str(decisions), "--training-log", training]
    assert main(drill) == 0
    assert "No mistakes to drill" in capsys.readouterr().out
    answers(monkeypatch, actions=["b 333"], between=[""])  # a size the abstraction lacks
    main(["--seed", "8", "--seats", "3", "--log-dir", str(tmp_path), "--training-log", training])
    capsys.readouterr()
    records = [json.loads(line) for line in (tmp_path / "session-8.jsonl").read_text().splitlines()]
    hand, index = next(
        (r["hand"], i)
        for r in records
        if r["type"] == "hand"
        for i, e in enumerate(r["hand"]["history"])
        if e["seat"] == 0 and e["action"]["amount"] == 333
    )
    mistake = DecisionRecord(
        "session-8", hand["hand_id"], index, {"mode": "cash"}, 3.0, 0.1, "mistake", "c", None, None
    )
    SessionLog(decisions).append("decision", mistake.to_dict())
    # A costlier record whose index no longer points at a user decision is skipped.
    stale = next(i for i, e in enumerate(hand["history"]) if e["seat"] != 0)
    SessionLog(decisions).append("decision", replace(mistake, index=stale, loss=5.0).to_dict())
    answers(monkeypatch, other="1")
    assert main(drill) == 0
    captured = capsys.readouterr()
    output = captured.out
    assert f"Skipping mistake/session-8/{hand['hand_id']}/{stale}" in captured.err
    assert f"From session-8, {hand['hand_id']}" in output and "1 Situation" in output
    drilled = [json.loads(line) for line in Path(training).read_text().splitlines()]
    drilled = [r for r in drilled if r["type"] == "drill"]
    assert [r["key"] for r in drilled] == [f"mistake/session-8/{hand['hand_id']}/{index}"]
    # Folding here costs about 0.02bb: no mistake, but not the best option. The question shows
    # neither the move nor its odd size.
    question, reveal = output.split("Close: no mistake")
    assert drilled[0]["grade"] == "mixed" and "333" not in question and "you chose raise to 333" in reveal


def test_icm_drills_show_both_verdicts_and_are_scheduled(monkeypatch, capsys, tmp_path):
    training = tmp_path / "training.jsonl"
    # A due key the drill can no longer rebuild (the big blind first in) is skipped.
    stale = "icm/no_ante/10-10-10/2/-/AKs"
    SessionLog(training).append("drill", {"key": stale, "grade": "wrong", "day": 0})
    answers(monkeypatch, other="1")  # always fold
    drill = ["drill", "icm", "--count", "3", "--seed", "4", "--training-log", str(training)]
    assert main(drill) == 0
    captured = capsys.readouterr()
    assert f"Skipping {stale}" in captured.err
    assert captured.out.count("Chip EV:") == 2
    assert captured.out.count("of the prize pool against folding") == 2
    drilled = [json.loads(line) for line in training.read_text().splitlines()][1:]
    assert [r["key"].split("/")[0] for r in drilled] == ["icm", "icm"]
