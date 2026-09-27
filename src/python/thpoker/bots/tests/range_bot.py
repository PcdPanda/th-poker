import pytest

from   thpoker.bots             import range_bot
from   thpoker.bots.abstraction import AbstractAction, legal_abstract_actions
from   thpoker.bots.bot         import PRESETS
from   thpoker.bots.equity_bot  import EquityBot, preflop_range
from   thpoker.bots.range_bot   import RangeBot, chart_range, seat_range
from   thpoker.charts           import push_chart, strategy
from   thpoker.game.cards       import COMBOS, COMBO_CLASS, combo_index
from   thpoker.game.engine      import (apply_action, is_terminal, new_hand,
                                        observation)
from   thpoker.game.rng         import Rng
from   thpoker.game.state       import (Action, ActionType, GameConfig,
                                        Observation)
from   thpoker.game.tests.decks import (CALL, CHECK, FOLD, heads_up, play,
                                        stacked_deck)

TAG = RangeBot("probe", PRESETS["tag"])
# The stacked deck needs cards for every dealt seat; the tests override the seats they study.
SIX_SEATS = {0: "KdQd", 1: "9s9c", 2: "Th6h", 3: "5d4d", 4: "Jc3s", 5: "8h8s"}


def test_policy_is_a_legal_distribution_that_never_folds_when_checking_is_free():
    rng = Rng(21)
    views: list[Observation] = []
    while len(views) < 150:
        seats = 2 + rng.randbelow(7)
        stacks = tuple(500 + rng.randbelow(15_000) for _ in range(seats))
        state, _ = new_hand(GameConfig(seats), rng.randbelow(1 << 30), 0, stacks)
        while not is_terminal(state):
            view = observation(state, state.to_act)
            views.append(view)
            state, _ = apply_action(state, TAG.decide(view, rng.derive(len(views))).action)
    for view in views:
        distribution, _ = TAG.policy(view)
        assert abs(sum(distribution.values()) - 1.0) < 1e-9
        assert set(distribution) <= set(legal_abstract_actions(view))
        if AbstractAction.CHECK in legal_abstract_actions(view):
            assert AbstractAction.FOLD not in distribution


def test_first_in_follows_the_preflop_table():
    deck = stacked_deck(0, 6, SIX_SEATS | {3: "AsAh", 4: "7c2d"}, "")
    state, _ = new_hand(GameConfig(6), 1, 0, (10_000,) * 6, deck=deck)
    assert TAG.action_probabilities(observation(state, 3)) == {AbstractAction.OPEN: 1.0}
    state = play(state, FOLD)
    assert TAG.action_probabilities(observation(state, 4)) == {AbstractAction.FOLD: 1.0}


@pytest.mark.parametrize("hole, action", [("Jh8c", AbstractAction.ALL_IN), ("7c2d", AbstractAction.FOLD)])
def test_short_stack_first_in_follows_the_push_fold_chart(hole, action):
    # Heads-up at 8bb: J8o shoves up to 13bb on the chart, 72o only up to 1.5bb.
    deck = stacked_deck(0, 2, {0: hole, 1: "AsKs"}, "")
    state, _ = new_hand(GameConfig(2), 1, 0, (800, 800), deck=deck)
    assert TAG.action_probabilities(observation(state, 0)) == {action: 1.0}


@pytest.mark.parametrize("stack, action", [(500, AbstractAction.CALL), (900, AbstractAction.FOLD)])
def test_short_stack_facing_a_shove_follows_the_call_chart(stack, action):
    # The big blind calls a heads-up shove with J8o up to 7bb on the chart.
    deck = stacked_deck(0, 2, {0: "AsKs", 1: "Jh8c"}, "")
    state, _ = new_hand(GameConfig(2), 1, 0, (stack, stack), deck=deck)
    state = play(state, Action(ActionType.RAISE, stack))
    assert TAG.action_probabilities(observation(state, 1)) == {action: 1.0}


def _big_blind_facing_a_pot_bet() -> Observation:
    state = heads_up("QsJs", "8c8d", "Kh7d2c3s9h")
    state = play(state, Action(ActionType.RAISE, 250), CALL, CHECK, Action(ActionType.BET, 500))
    return observation(state, 1)


def test_facing_a_pot_bet_it_defends_at_least_the_minimum_defense_frequency():
    view = _big_blind_facing_a_pot_bet()
    own = seat_range(view, 1)
    holes = [i for i, (a, b) in enumerate(COMBOS) if own[i] > 0 and a not in view.board and b not in view.board]
    decisions = TAG.policies(view, holes)
    kept = sum(own[h] * (1 - decisions[h].get(AbstractAction.FOLD, 0.0)) for h in holes) / sum(own[h] for h in holes)
    _, rationale = TAG.policy(view)
    # A pot-size bet needs 50% defense (1/(1+b)); the style shades it slightly.
    assert rationale["defense_share"] == pytest.approx(0.5, abs=0.05)
    assert kept >= rationale["defense_share"] - 0.03


def test_policy_and_policies_agree_on_the_hand_held():
    view = _big_blind_facing_a_pot_bet()
    hero = combo_index(*view.my_cards)
    assert TAG.policies(view, [hero])[hero] == TAG.action_probabilities(view)


def test_a_limp_leaves_the_charts_and_falls_back_to_the_tier_2_bands():
    deck = stacked_deck(0, 6, SIX_SEATS, "")
    state, _ = new_hand(GameConfig(6), 1, 0, (10_000,) * 6, deck=deck)
    state = play(state, CALL)  # first to act limps
    view = observation(state, 4)
    assert chart_range(view, 3) is None
    assert seat_range(view, 3) == preflop_range(view, 3)


def test_a_hand_without_an_equity_falls_back_the_same_way_when_played_and_tracked(monkeypatch):
    # Card removal can leave an opponent range empty for a hand; both paths then use Tier 2.
    view = _big_blind_facing_a_pot_bet()
    hero = combo_index(*view.my_cards)
    computed = range_bot.range_equities

    def without_hero(*args, **kwargs):
        return {h: e for h, e in computed(*args, **kwargs).items() if h != hero}

    monkeypatch.setattr(range_bot, "range_equities", without_hero)
    tier_2 = EquityBot("probe", PRESETS["tag"]).action_probabilities(view)
    assert TAG.action_probabilities(view) == tier_2
    assert TAG.policies(view, [hero])[hero] == tier_2


def test_an_oversized_deep_open_leaves_the_charts():
    # The charts were solved for a 2.5bb open; a 100bb open-shove goes to the size-aware Tier 2.
    state, _ = new_hand(GameConfig(2), 1, 0, (10_000, 10_000))
    shoved = play(state, Action(ActionType.RAISE, 10_000))
    assert chart_range(observation(shoved, 1), 0) is None
    assert seat_range(observation(shoved, 1), 0) == preflop_range(observation(shoved, 1), 0)
    opened = play(state, Action(ActionType.RAISE, 250))
    assert chart_range(observation(opened, 1), 0) is not None
    # The charts' own all-in 4-bet after a 3-bet stays on them.
    four_bet = play(
        state,
        Action(ActionType.RAISE, 250),
        Action(ActionType.RAISE, 1_000),
        Action(ActionType.RAISE, 10_000),
    )
    assert chart_range(observation(four_bet, 1), 0) is not None


def test_a_short_shove_reads_and_is_answered_from_the_push_fold_charts_at_its_depth():
    # Three-handed, a 12bb button shove against 100bb blinds: the shover's depth governs, so its
    # range is the 12bb push chart and the small blind answers from the 12bb call chart.
    state, _ = new_hand(GameConfig(3), 1, 0, (1_200, 10_000, 10_000))
    view = observation(play(state, Action(ActionType.RAISE, 1_200)), 1)
    pushes = [range_bot._soft_threshold(t, 12.0) for t in push_chart(3, 0, False)]
    assert chart_range(view, 0) == tuple(pushes[c] for c in COMBO_CLASS)
    _, rationale = TAG.policy(view)
    assert rationale["rule_triggered"] == "push/fold chart: facing a shove"
    assert rationale["stack_bb"] == 12.0


def test_a_deep_shove_leaves_the_charts_even_for_a_short_viewer():
    # A 100bb shove into a 12bb small blind and a 100bb big blind is a genuine 100bb shove.
    state, _ = new_hand(GameConfig(3), 1, 0, (10_000, 1_200, 10_000))
    view = observation(play(state, Action(ActionType.RAISE, 10_000)), 1)
    assert chart_range(view, 0) is None
    assert seat_range(view, 0) == preflop_range(view, 0)


def test_a_short_stack_calling_an_open_leaves_the_charts():
    # The push/fold call chart is for calling a shove, not a 2.5bb open.
    state, _ = new_hand(GameConfig(3), 1, 0, (10_000, 1_200, 10_000))
    view = observation(play(state, Action(ActionType.RAISE, 250), CALL), 2)
    assert chart_range(view, 1) is None


def test_an_opener_facing_a_short_three_bet_plays_at_the_three_bettors_depth():
    # The 100bb button opens and the 20bb small blind 3-bets all in (its usual 4x size doubled,
    # so still on the charts): the button answers from the 20bb chart, not the 100bb one.
    state, _ = new_hand(GameConfig(3), 1, 0, (10_000, 2_000, 10_000))
    state = play(state, Action(ActionType.RAISE, 250), Action(ActionType.RAISE, 2_000), FOLD)
    _, rationale = TAG.policy(observation(state, 0))
    assert rationale["stack_bb"] == 20.0


def test_an_open_is_read_at_the_depth_it_was_made_before_a_short_three_bet():
    # Six-handed, 100bb except a 20bb big blind: UTG opens, the big blind 3-bets and UTG calls.
    # The open comes from the 100bb chart and the call from the 20bb one.
    state, _ = new_hand(GameConfig(6), 1, 0, (10_000, 10_000, 2_000, 10_000, 10_000, 10_000))
    state = play(state, Action(ActionType.RAISE, 250), FOLD, FOLD, FOLD, FOLD)
    state = play(state, Action(ActionType.RAISE, 1_000), CALL)
    opens = strategy("open", 6, "0", 100.0, False)
    calls = strategy("versus_3bet", 6, "0-5", 20.0, False)
    expected = tuple(opens[c][1] * calls[c][1] for c in COMBO_CLASS)
    assert chart_range(observation(state, 2), 3) == pytest.approx(expected)
