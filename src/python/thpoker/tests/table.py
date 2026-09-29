from   dataclasses              import asdict
import hashlib
import json
import os
from   pathlib                  import Path
import pytest
import subprocess
import sys
from   thpoker.bots.bot         import PRESETS
from   thpoker.bots.equity_bot  import EquityBot
from   thpoker.cli              import build_config, parse_args
from   thpoker.game.engine      import apply_action, new_hand
from   thpoker.game.session     import (Mode, SessionConfig, SessionError,
                                        TournamentConfig, preset_schedule)
from   thpoker.game.state       import (Action, ActionType, ConfigError, Event,
                                        GameConfig)
import thpoker.table
from   thpoker.table            import (EXPLOITS, Hud, SeatStats, TableConfig,
                                        TableRunner, pointed_style)


def play(hud: Hud, actions: tuple[tuple[str, int | None], ...]):
    state, _ = new_hand(GameConfig(3), 1, 0, (10_000,) * 3)  # button 0, blinds 1 and 2
    for kind, amount in actions:
        state, _ = apply_action(state, Action(ActionType(kind), amount))
    hud.record(state)


def test_hud_counts_voluntary_play_raises_three_bets_and_aggression():
    hud = Hud(3)
    # Seat 0 opens, the small blind 3-bets, the big blind folds, seat 0 calls; on the flop the
    # small blind bets, seat 0 raises, the small blind calls; both check the turn and river.
    play(
        hud,
        (
            ("RAISE", 300),
            ("RAISE", 1_000),
            ("FOLD", None),
            ("CALL", None),
            ("BET", 1_000),
            ("RAISE", 3_000),
            ("CALL", None),
        )
        + (("CHECK", None),) * 4,
    )
    # Seat 0 opens and both blinds face that single raise: the small blind calls, the big blind
    # folds; seat 0 bets the flop and the small blind folds.
    play(
        hud,
        (
            ("RAISE", 300),
            ("CALL", None),
            ("FOLD", None),
            ("CHECK", None),
            ("BET", 400),
            ("FOLD", None),
        ),
    )
    # A limped pot: the big blind's check is not voluntary.
    play(hud, (("CALL", None), ("CALL", None), ("CHECK", None)) + (("CHECK", None),) * 9)
    assert hud.seats[0] == SeatStats(3, 3, 2, 0, 0, 2, 0)
    assert hud.seats[1] == SeatStats(3, 3, 1, 2, 1, 1, 1)
    assert hud.seats[2] == SeatStats(3, 0, 0, 1, 0, 0, 0)
    assert hud.seats[0].summary() == "VPIP 100 PFR 67 3B - AF - (3h)"
    assert hud.seats[1].summary() == "VPIP 100 PFR 33 3B 50 AF 1.0 (3h)"
    assert SeatStats().summary() == "no hands yet"


def bot_tournament(num_seats: int, seed: int) -> TableConfig:
    schedule = TournamentConfig(preset_schedule(100), hands_per_level=8)
    session = SessionConfig(Mode.TOURNAMENT, seed, num_seats, 5_000, user_seat=None, tournament=schedule)
    return TableConfig(session, (None,) * num_seats)


@pytest.mark.parametrize("num_seats", [2, 5, 8])
def test_bot_only_tournament_runs_to_a_winner(num_seats):
    runner = TableRunner(bot_tournament(num_seats, seed=num_seats))
    events = runner.fast_forward()
    session = runner.session
    assert session.finished
    assert None not in session.places and session.places.count(1) == 1
    assert abs(sum(session.prizes) - 1.0) < 1e-9
    assert sum(session.stacks) == 5_000 * num_seats
    assert sum(1 for e in events if e.kind == "HandStarted") == session.hand_number


def session_digest() -> str:
    events = TableRunner(bot_tournament(4, seed=21)).fast_forward()
    return hashlib.sha256(json.dumps([e.to_dict() for e in events]).encode()).hexdigest()


def test_same_seed_replays_the_same_session_in_another_process():
    # A separate interpreter with a different hash seed catches any dependence on set or dict
    # ordering of strings, which an in-process rerun cannot.
    source_root = Path(__file__).resolve().parents[2]
    environment = {**os.environ, "PYTHONHASHSEED": "12345", "PYTHONPATH": str(source_root)}
    script = "from thpoker.tests.table import session_digest; print(session_digest())"
    result = subprocess.run([sys.executable, "-c", script], env=environment, capture_output=True, text=True, check=True)
    assert result.stdout.strip() == session_digest()


def test_fast_forward_keeps_the_pace_of_a_minute_based_tournament():
    schedule = TournamentConfig(preset_schedule(100), hands_per_level=None, minutes_per_level=10)
    session = SessionConfig(Mode.TOURNAMENT, 3, 4, 5_000, user_seat=None, tournament=schedule)
    runner = TableRunner(TableConfig(session, (None,) * 4))
    runner.start_hand(elapsed_minutes=45)  # level 5 after one 45-minute hand
    events = runner.fast_forward(elapsed_minutes=45)
    levels = [e.data["level"] for e in events if e.kind == "BlindLevelChanged"]
    assert levels and levels == sorted(levels) and levels[0] > 5


def test_bot_names_are_distinct_and_hidden_styles_do_not_change_them():
    session = SessionConfig(Mode.CASH, 5, 6, 10_000, cash_blinds=GameConfig(6))

    def labels(styles: tuple[str, ...], hide: bool) -> list[str]:
        runner = TableRunner(TableConfig(session, styles, hide_styles=hide))
        return [runner.seat_label(seat) for seat in range(1, 6)]

    hidden = labels(("tag",) * 6, hide=True)
    assert len(set(hidden)) == 5
    assert hidden == labels(("maniac",) * 6, hide=True)
    assert all(label.endswith(" (tag)") for label in labels(("tag",) * 6, hide=False))


def test_user_acts_only_on_their_turn_and_bots_play_around_them():
    session = SessionConfig(Mode.CASH, 5, 4, 10_000, user_seat=0, cash_blinds=GameConfig(4))
    runner = TableRunner(TableConfig(session, ("tag",) * 4))
    for _ in range(20):
        runner.start_hand()
        while runner.user_to_act():
            legal = runner.user_legal_actions()
            runner.act(Action(ActionType.CHECK if legal.can_check else ActionType.FOLD))
        with pytest.raises(SessionError, match="not the user's turn"):
            runner.act(Action(ActionType.FOLD))
    assert sum(runner.session.stacks) == sum(runner.session.buy_ins)


def test_fast_forward_refuses_while_the_user_is_still_playing():
    session = SessionConfig(Mode.TOURNAMENT, 5, 3, 5_000, tournament=TournamentConfig(preset_schedule(100)))
    runner = TableRunner(TableConfig(session, (None,) * 3))
    with pytest.raises(SessionError, match="out of the tournament"):
        runner.fast_forward()


@pytest.mark.parametrize(
    "styles, panel, message",
    [
        ((None,), "online_micro", "bot style entries"),
        ((None, "shark"), "online_micro", "unknown styles"),
        ((None, None), "casino", "unknown panel"),
    ],
)
def test_invalid_table_config_rejected(styles, panel, message):
    session = SessionConfig(Mode.CASH, 5, 2, 10_000, cash_blinds=GameConfig(2))
    with pytest.raises(ConfigError, match=message):
        TableConfig(session, styles, panel)


def test_tier_two_bots_finish_a_tournament_and_unknown_tiers_are_rejected():
    schedule = TournamentConfig(preset_schedule(100), hands_per_level=4)
    session = SessionConfig(Mode.TOURNAMENT, 9, 3, 2_000, user_seat=None, tournament=schedule)
    runner = TableRunner(TableConfig(session, (None,) * 3, tier=2))
    runner.fast_forward()
    assert runner.session.finished and sum(runner.session.stacks) == 6_000
    assert all(isinstance(bot, EquityBot) for bot in runner.bots.values())
    with pytest.raises(ConfigError, match="tier"):
        TableConfig(session, (None,) * 3, tier=4)


@pytest.mark.parametrize(
    "argv",
    [["--seats", "4", "--seed", "5"], ["--mode", "tournament", "--seats", "7", "--seed", "5"]],
)
def test_a_logged_table_config_rebuilds_the_same_config(argv):
    config, _ = build_config(parse_args(argv))
    logged = json.loads(json.dumps(asdict(config)))
    assert TableConfig.from_dict(logged) == config


def test_the_numbers_point_to_the_nearest_style():
    # 100 hands at VPIP 45 / PFR 7 read as a calling station; VPIP 12 / PFR 9 as a nit.
    assert pointed_style(SeatStats(hands=100, voluntary=45, preflop_raises=7)) == "calling_station"
    assert pointed_style(SeatStats(hands=100, voluntary=12, preflop_raises=9)) == "nit"
    assert pointed_style(SeatStats(hands=100, voluntary=60, preflop_raises=45)) == "maniac"


def test_too_few_hands_point_nowhere_and_every_style_has_advice():
    assert pointed_style(SeatStats(hands=10, voluntary=5, preflop_raises=4)) is None
    assert set(EXPLOITS) == set(PRESETS)


def fold_every_hand(tier: int, hands: int) -> tuple[int, int, list[Event]]:
    """The user folds or checks every hand: how many pots a bot won uncontested, how many of
    those it showed, and every bot decision."""
    argv = ["--seats", "4", "--tier", str(tier), "--seed", "21", "--no-log"]
    runner = TableRunner(build_config(parse_args(argv))[0])
    uncontested = shown = 0
    decisions: list[Event] = []
    for _ in range(hands):
        events = runner.start_hand()
        while runner.user_to_act():
            check = runner.user_legal_actions().can_check
            events += runner.act(Action(ActionType.CHECK if check else ActionType.FOLD))
        hand = runner.hand
        assert hand is not None
        (award, *more) = hand.awards
        alone = not more and len(award.eligible) == 1 and award.eligible[0] != runner.user_seat
        shows = [e for e in events if e.kind == "CardsShown"]
        assert not shows or alone  # only a bot that won without a showdown shows
        uncontested += alone
        shown += len(shows)
        decisions += [e for e in events if e.kind == "BotDecision"]
    return uncontested, shown, decisions


def test_bots_that_win_uncontested_show_by_difficulty_without_changing_play(monkeypatch):
    uncontested, shown, decisions = fold_every_hand(1, 80)
    assert uncontested > 40 and 0.3 < shown / uncontested < 0.7  # SHOW_CHANCE[1] is 0.5
    medium, medium_shown, _ = fold_every_hand(2, 80)
    assert medium > 40 and 0.1 < medium_shown / medium < 0.4  # SHOW_CHANCE[2] is 0.25
    monkeypatch.setattr(thpoker.table, "SHOW_CHANCE", {1: 0.0, 2: 0.0, 3: 0.0})
    assert fold_every_hand(1, 80) == (uncontested, 0, decisions)
