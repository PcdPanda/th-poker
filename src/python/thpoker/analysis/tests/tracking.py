from   dataclasses              import replace
from   typing                   import Any

import numpy as np
import pytest

from   thpoker.analysis.tracking \
                                import REFERENCE_FLOOR, _likelihoods, track
from   thpoker.bots.abstraction import (AbstractAction, legal_abstract_actions,
                                        to_action)
from   thpoker.bots.bot         import Bot, PRESETS
from   thpoker.bots.range_bot   import RangeBot
from   thpoker.game.cards       import COMBOS, combo_index, parse_cards
from   thpoker.game.engine      import observation
from   thpoker.game.rng         import Rng
from   thpoker.game.session     import Mode, SessionConfig
from   thpoker.game.state       import (Action, ActionType, AnteType,
                                        GameConfig, Observation)
from   thpoker.game.tests.decks import CHECK, heads_up, play
from   thpoker.table            import TableConfig, TableRunner

REFERENCE = RangeBot("reference", PRESETS["balanced"])


def weights(*hands: str) -> np.ndarray:
    result = np.zeros(len(COMBOS))
    for hand in hands:
        result[combo_index(*parse_cards(hand))] = 1.0
    return result


class Calls(Bot):
    """Calls or checks with every hand, so its actions say nothing about its cards."""

    name, style = "calls", PRESETS["balanced"]

    def policy(self, view: Observation) -> tuple[dict[AbstractAction, float], dict[str, Any]]:
        legal = legal_abstract_actions(view)
        return {AbstractAction.CALL if AbstractAction.CALL in legal else AbstractAction.CHECK: 1.0}, {}


class RaisesPairs(Bot):
    """Opens with pocket pairs only and folds everything else."""

    name, style = "pairs", PRESETS["balanced"]

    def policy(self, view: Observation) -> tuple[dict[AbstractAction, float], dict[str, Any]]:
        a, b = view.my_cards
        return ({AbstractAction.OPEN: 1.0} if a // 4 == b // 4 else {AbstractAction.FOLD: 1.0}), {}


def test_a_player_whose_actions_carry_no_information_keeps_a_uniform_range():
    # The user raises the button and the calling bot calls down.
    state = play(
        heads_up("AsKd", "7c2h"),
        Action(ActionType.RAISE, 300),
        Action(ActionType.CALL),
        *[CHECK] * 6,
    )
    final = track(state, 0, {1: Calls()}, REFERENCE)[-1].ranges[1]
    dead = set(state.board) | set(parse_cards("AsKd"))
    possible = np.array([not set(c) & dead for c in COMBOS])
    assert np.allclose(final[possible], 1 / possible.sum()) and not final[~possible].any()


def test_a_raise_that_only_pairs_make_leaves_only_pairs():
    # Seat 1 is the user; the bot on the button raises and the user folds.
    state = play(heads_up("7h7c", "AsKd"), Action(ActionType.RAISE, 250), Action(ActionType.FOLD))
    after_raise = track(state, 1, {0: RaisesPairs()}, REFERENCE)[-1].ranges[0]
    pairs = np.array([a // 4 == b // 4 for a, b in COMBOS])
    assert not after_raise[~pairs].any() and after_raise[pairs].sum() == pytest.approx(1.0)


def test_the_users_range_is_read_with_the_reference_policy_and_a_floor():
    # Heads-up at 100bb the reference opens AA from the button and never opens 72o.
    opened = play(heads_up("QsJs", "8c8d"), Action(ActionType.RAISE, 250), Action(ActionType.CALL))
    after_open = track(opened, 0, {1: Calls()}, REFERENCE)[1].ranges[0]
    aces, seven_deuce = combo_index(*parse_cards("AcAd")), combo_index(*parse_cards("7c2d"))
    assert after_open[seven_deuce] / after_open[aces] == pytest.approx(REFERENCE_FLOOR, rel=0.01)
    # It never limps either, so a limp leaves every hand at the floor and the range stays whole.
    limped = play(heads_up("7h2c", "8c8d"), Action(ActionType.CALL), CHECK)
    after_limp = track(limped, 0, {1: Calls()}, REFERENCE)[1].ranges[0]
    assert np.allclose(after_limp, after_limp.max())


def test_a_hand_its_bots_could_not_have_played_is_rejected():
    # On the flop the big blind acts first, and this bot never bets.
    limped = heads_up("7h2c", "AsKd")
    state = play(limped, Action(ActionType.CALL), CHECK, Action(ActionType.BET, 100), Action(ActionType.FOLD))
    with pytest.raises(ValueError, match="impossible"):
        track(state, 0, {1: Calls()}, REFERENCE)


def play_path_likelihood(bot: Bot, view: Observation, hole: int, taken: Action) -> float:
    """Chance of `taken` from the bot's own `policy` holding `hole`: how it really plays."""
    cards = list(view.hole_cards)
    cards[view.seat] = COMBOS[hole]
    view = replace(view, hole_cards=tuple(cards))
    return sum(p for a, p in bot.action_probabilities(view).items() if to_action(a, view) == taken)


def user_action(runner: TableRunner, rng: Rng) -> Action:
    """Mostly checks and calls, sometimes the smallest raise, rarely all in."""
    legal = runner.user_legal_actions()
    draw = rng.random()
    aggressive = ActionType.BET if legal.can_bet else ActionType.RAISE
    if (legal.can_bet or legal.can_raise) and draw < 0.05:
        return Action(aggressive, legal.max_raise_to)
    if (legal.can_bet or legal.can_raise) and draw < 0.25:
        return Action(aggressive, legal.min_raise_to)
    return CHECK if legal.can_check else Action(ActionType.CALL)


@pytest.mark.parametrize("tier", [1, 2, 3])
@pytest.mark.parametrize(
    "num_seats, stack, ante",
    [(4, 10_000, AnteType.NONE), (6, 1_200, AnteType.BIG_BLIND_ANTE)],
)
def test_tracking_reads_every_bot_action_the_way_the_bot_plays(tier, num_seats, stack, ante):
    blinds = GameConfig(num_seats, ante=100 if ante == AnteType.BIG_BLIND_ANTE else 0, ante_type=ante)
    session = SessionConfig(Mode.CASH, 12, num_seats, stack, 0, blinds, reset_stacks_each_hand=True)
    runner = TableRunner(TableConfig(session, (None,) * num_seats, tier=tier))
    rng = Rng(tier)
    checked = 0
    for _ in range(5):
        runner.start_hand()
        while runner.user_to_act():
            runner.act(user_action(runner, rng))
        hand = runner.hand
        assert hand is not None
        snapshots = track(hand, 0, runner.bots, REFERENCE)
        for snapshot, entry in zip(snapshots, hand.history):
            if entry.seat == 0:
                continue
            real = hand.hole_cards[entry.seat]
            assert real is not None
            prior = snapshot.ranges[entry.seat]
            holes = [combo_index(*real)] + [int(i) for i in np.flatnonzero(prior)[::97]]
            bot = runner.bots[entry.seat]
            tracked = _likelihoods(bot, snapshot.state, entry.seat, holes, entry.action, exact=True)
            view = observation(snapshot.state, entry.seat)
            for hole in holes:
                assert tracked[hole] == pytest.approx(play_path_likelihood(bot, view, hole, entry.action))
            checked += 1
        for snapshot in snapshots:
            for seat, weights in snapshot.ranges.items():
                real = hand.hole_cards[seat]
                assert seat == 0 or (real is not None and weights[combo_index(*real)] > 0)
    assert checked >= 20
