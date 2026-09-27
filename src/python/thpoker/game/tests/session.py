from   dataclasses              import replace

import pytest

from   thpoker.game.engine      import apply_action, new_hand
from   thpoker.game.session     import (BlindLevel, Mode, SessionConfig,
                                        SessionError, TournamentConfig,
                                        begin_hand, blind_level,
                                        default_payouts, end_hand, end_session,
                                        preset_schedule, rebuy, start_session)
from   thpoker.game.state       import (Action, ActionType, ConfigError,
                                        GameConfig, GameState)


def tournament(num_seats: int, hands_per_level: int = 15) -> SessionConfig:
    schedule = TournamentConfig(preset_schedule(100), hands_per_level=hands_per_level)
    return SessionConfig(Mode.TOURNAMENT, 11, num_seats, 10_000, tournament=schedule)


def cash(num_seats: int, **options) -> SessionConfig:
    return SessionConfig(Mode.CASH, 11, num_seats, 10_000, cash_blinds=GameConfig(num_seats), **options)


def finished_hand(starting: tuple[int, ...], ending: tuple[int, ...], big_blind_seat: int) -> GameState:
    """A completed hand with chosen stacks; end_hand reads only the fields set here."""
    n = len(starting)
    state, _ = new_hand(GameConfig(n), 1, 0, (1_000,) * n)
    while state.to_act is not None:
        state, _ = apply_action(state, Action(ActionType.FOLD))
    return replace(
        state,
        starting_stacks=starting,
        stacks=ending,
        big_blind_seat=big_blind_seat,
        dealt_in=tuple(s > 0 for s in starting),
    )


def test_tournament_places_prizes_and_finish():
    state, _ = start_session(tournament(4))
    # Seats 1 and 2 bust on the same hand; the smaller starting stack finishes lower.
    state, events = end_hand(state, finished_hand((1_000, 300, 500, 1_200), (2_000, 0, 0, 1_000), 2))
    assert state.places == (None, 4, 3, None) and not state.finished
    assert [e.kind for e in events] == ["PlayerEliminated", "PlayerEliminated"]
    state, events = end_hand(state, finished_hand((2_000, 0, 0, 1_000), (3_000, 0, 0, 0), 3))
    assert state.finished and state.places == (1, 4, 3, 2)
    assert state.prizes == (0.65, 0.0, 0.0, 0.35)  # 4 entrants: 65/35
    assert events[-1].kind == "TournamentFinished"


def test_equal_stacks_busting_together_share_the_better_place_and_average_prize():
    state, _ = start_session(tournament(7))
    state = replace(state, active=(True, True, True, False, False, False, False))
    starting = (500, 500, 1_000, 0, 0, 0, 0)
    state, _ = end_hand(state, finished_hand(starting, (0, 0, 2_000, 0, 0, 0, 0), 1))
    # Places 2 and 3 of a 50/30/20 payout: both take place 2 and (30% + 20%) / 2.
    assert state.places[:3] == (2, 2, 1)
    assert state.prizes[:3] == (0.25, 0.25, 0.5)


def test_going_heads_up_gives_the_last_big_blind_the_button():
    state, _ = start_session(tournament(3))
    # Last hand: button 0, big blind 2; seat 0 busted. Plain rotation would make seat 1 the
    # button and seat 2 the big blind again.
    state = replace(
        state,
        button=0,
        last_big_blind_seat=2,
        last_dealt_count=3,
        active=(False, True, True),
        stacks=(0, 15_000, 15_000),
    )
    _, setup, _ = begin_hand(state)
    assert setup.button == 2 and setup.dealt_in == (False, True, True)


def test_button_moves_to_the_next_seat_still_in():
    state, _ = start_session(tournament(4))
    state = replace(state, button=1, active=(True, True, False, True), stacks=(1, 1, 0, 1))
    _, setup, _ = begin_hand(state)
    assert setup.button == 3


def test_blind_level_advances_after_the_configured_number_of_hands():
    state, _ = start_session(tournament(3, hands_per_level=2))
    configs, level_events = [], []
    for _ in range(3):
        state, setup, events = begin_hand(state)
        configs.append((setup.config.small_blind, setup.config.big_blind))
        level_events += [e.data for e in events if e.kind == "BlindLevelChanged"]
    assert configs == [(50, 100), (50, 100), (75, 150)]
    assert level_events == [{"level": 2, "small_blind": 75, "big_blind": 150, "ante": 0}]


def test_preset_schedule_and_doubling_past_the_end():
    schedule = preset_schedule(100)
    assert [level.big_blind for level in schedule[:6]] == [100, 150, 200, 300, 400, 600]
    assert schedule[3].ante == 0 and schedule[4] == BlindLevel(200, 400, 400)
    assert blind_level(schedule, len(schedule)) == BlindLevel(10_000, 20_000, 20_000)


@pytest.mark.parametrize(
    "players, payouts",
    [
        (2, (1.0,)),
        (3, (1.0,)),
        (4, (0.65, 0.35)),
        (6, (0.65, 0.35)),
        (7, (0.5, 0.3, 0.2)),
        (8, (0.5, 0.3, 0.2)),
    ],
)
def test_default_payouts(players, payouts):
    assert default_payouts(players) == payouts


def test_cash_reloads_busted_bots_and_auto_rebuys_below_threshold():
    state, _ = start_session(cash(3, auto_rebuy=True, rebuy_threshold=5_000))
    state = replace(state, stacks=(4_000, 0, 12_000))
    state, setup, events = begin_hand(state)
    assert setup.stacks == (10_000, 10_000, 12_000)
    assert state.buy_ins == (16_000, 20_000, 10_000)
    assert [e.data for e in events] == [{"seat": 0, "amount": 6_000}, {"seat": 1, "amount": 10_000}]


def test_busted_user_must_rebuy_before_the_next_hand():
    state, _ = start_session(cash(2))
    state, events = end_hand(state, finished_hand((10_000, 10_000), (0, 20_000), 1))
    assert state.awaiting_rebuy and events[-1].kind == "UserBusted"
    with pytest.raises(SessionError, match="rebuy"):
        begin_hand(state)
    state, _ = rebuy(state, 0)
    _, setup, _ = begin_hand(state)
    assert setup.stacks == (10_000, 20_000)


def test_reset_stacks_keep_the_session_net_equal_to_the_sum_of_hands():
    state, _ = start_session(cash(2, reset_stacks_each_hand=True))
    for _ in range(2):
        state, setup, events = begin_hand(state)
        assert setup.stacks == (10_000, 10_000) and events == []
        state, _ = end_hand(state, finished_hand(setup.stacks, (9_950, 10_050), 1))
    _, events = end_session(state)
    assert events[0].data["net"] == [-100, 100]


def test_end_session_reports_stack_minus_buy_ins():
    state, _ = start_session(cash(2))
    state = replace(state, stacks=(4_000, 26_000), buy_ins=(20_000, 10_000))
    ended, events = end_session(state)
    assert ended.finished and events[0].data["net"] == [-16_000, 16_000]


def test_hand_ids_do_not_reveal_the_seed():
    config = SessionConfig(Mode.CASH, 987_654_321, 2, 10_000, cash_blinds=GameConfig(2))
    _, setup, _ = begin_hand(start_session(config)[0])
    assert "987654321" not in setup.hand_id


def test_minute_levels_follow_elapsed_time_and_never_go_back():
    schedule = TournamentConfig(preset_schedule(100), hands_per_level=None, minutes_per_level=10)
    state, _ = start_session(SessionConfig(Mode.TOURNAMENT, 11, 3, 10_000, tournament=schedule))
    state, setup, _ = begin_hand(state, elapsed_minutes=45)
    assert setup.config.big_blind == 400  # level 5 after 45 minutes of 10-minute levels
    state, setup, events = begin_hand(state, elapsed_minutes=0)
    assert setup.config.big_blind == 400 and events == []


@pytest.mark.parametrize(
    "build, message",
    [
        (lambda: SessionConfig(Mode.CASH, 1, 9, 100, cash_blinds=GameConfig(8)), "num_seats"),
        (lambda: SessionConfig(Mode.CASH, 1, 3, 100), "needs cash_blinds"),
        (lambda: SessionConfig(Mode.CASH, 1, 3, 100, cash_blinds=GameConfig(2)), "must equal"),
        (
            lambda: SessionConfig(Mode.CASH, 1, 3, 100, user_seat=3, cash_blinds=GameConfig(3)),
            "user seat",
        ),
        (
            lambda: SessionConfig(Mode.CASH, 1, 3, 100, cash_blinds=GameConfig(3), auto_rebuy=True),
            "rebuy threshold",
        ),
        (lambda: SessionConfig(Mode.TOURNAMENT, 1, 3, 100), "needs a tournament"),
        (
            lambda: SessionConfig(
                Mode.TOURNAMENT,
                1,
                2,
                100,
                tournament=TournamentConfig(preset_schedule(100), payouts=(0.5, 0.3, 0.2)),
            ),
            "paid places",
        ),
        (lambda: TournamentConfig(preset_schedule(100), hands_per_level=None), "exactly one"),
        (lambda: TournamentConfig(preset_schedule(100), payouts=(0.5, 0.4)), "sum to 1"),
    ],
)
def test_invalid_configs_rejected(build, message):
    with pytest.raises(ConfigError, match=message):
        build()
