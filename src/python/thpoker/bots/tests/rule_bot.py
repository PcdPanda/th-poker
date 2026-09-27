from   dataclasses              import replace

import pytest

from   thpoker.bots.abstraction import AbstractAction, legal_abstract_actions
from   thpoker.bots.bot         import PRESETS
from   thpoker.bots.rule_bot    import (DRAW, MEDIUM, MONSTER, RuleBot, STRONG,
                                        WEAK, classify)
from   thpoker.bots.tests.bot   import random_views
from   thpoker.game.cards       import combo_index, parse_cards
from   thpoker.game.engine      import apply_action, new_hand, observation
from   thpoker.game.state       import Action, ActionType, GameConfig
from   thpoker.game.tests.decks import CALL, CHECK, heads_up, stacked_deck


@pytest.mark.parametrize(
    "hole, board, bucket",
    [
        ("7c7d", "7h2s9d", MONSTER),  # set
        ("KcKd", "7h2s9d", STRONG),  # overpair
        ("As5d", "Ah8s3d", MEDIUM),  # top pair, weak kicker
        ("AsKd", "Ah8s3d", STRONG),  # top pair, good kicker
        ("9s8s", "As7s2d", DRAW),  # flush draw
        ("9h8d", "7s6c2d", DRAW),  # open-ended straight draw
        ("Kc4d", "Ah8s3d", WEAK),  # air
        ("2c3d", "TsJdQcKhAh", WEAK),  # the board's straight plays
        ("2c3d", "7c7d7h7s", WEAK),  # quads on a four-card board play for everyone
    ],
)
def test_classify(hole, board, bucket):
    assert classify(tuple(parse_cards(hole)), tuple(parse_cards(board))) == bucket


@pytest.mark.parametrize("style", sorted(PRESETS))
def test_policy_is_a_legal_distribution_that_never_folds_when_checking_is_free(style):
    bot = RuleBot("probe", PRESETS[style])
    for view in random_views(RuleBot("probe", PRESETS["tag"]), 7, 300):
        distribution, rationale = bot.policy(view)
        assert abs(sum(distribution.values()) - 1.0) < 1e-9
        assert set(distribution) <= set(legal_abstract_actions(view))
        assert all(p > 0 for p in distribution.values())
        if AbstractAction.CHECK in legal_abstract_actions(view):
            assert AbstractAction.FOLD not in distribution
        assert bot.policy(view) == (distribution, rationale)


@pytest.mark.parametrize("style", sorted(PRESETS))
def test_aces_never_fold_to_a_reraise(style):
    deck = stacked_deck(0, 3, {0: "AsAd", 1: "KcKd", 2: "7h2c"}, "8h5s3dTc2h")
    state, _ = new_hand(GameConfig(3), 1, 0, (10_000,) * 3, deck=deck)
    for action in (
        Action(ActionType.RAISE, 300),
        Action(ActionType.RAISE, 1_200),
        Action(ActionType.FOLD),
    ):
        state, _ = apply_action(state, action)
    assert AbstractAction.FOLD not in RuleBot("probe", PRESETS[style]).action_probabilities(observation(state, 0))


@pytest.mark.parametrize("style", sorted(PRESETS))
def test_a_set_never_folds_to_a_bet(style):
    state = heads_up("7c7d", "AsKd", "7h2s9dTc3h")
    for action in (Action(ActionType.CALL), Action(ActionType.CHECK), Action(ActionType.BET, 150)):
        state, _ = apply_action(state, action)
    assert AbstractAction.FOLD not in RuleBot("probe", PRESETS[style]).action_probabilities(observation(state, 0))


def test_first_to_act_at_a_full_table_folds_seven_deuce_and_raises_aces():
    bot = RuleBot("probe", PRESETS["tag"])
    for hole, action, minimum in (
        ("7c2d", AbstractAction.FOLD, 0.95),
        ("AsAd", AbstractAction.OPEN, 0.95),
    ):
        deck = stacked_deck(
            0,
            8,
            {s: h for s, h in enumerate(["Kc4d", "Qh5c", "Jd6s", hole, "9c3h", "8d4s", "Tc2h", "6d5h"])},
            "",
        )
        state, _ = new_hand(GameConfig(8), 1, 0, (10_000,) * 8, deck=deck)
        assert state.to_act == 3
        assert bot.action_probabilities(observation(state, 3))[action] > minimum


def test_ten_big_blinds_only_shoves_or_folds_preflop():
    deck = stacked_deck(0, 3, {0: "QsQd", 1: "7h2c", 2: "8d3s"}, "")
    state, _ = new_hand(GameConfig(3), 1, 0, (1_000, 10_000, 10_000), deck=deck)
    probabilities = RuleBot("probe", PRESETS["tag"]).action_probabilities(observation(state, 0))
    assert set(probabilities) <= {AbstractAction.FOLD, AbstractAction.CALL, AbstractAction.ALL_IN}
    assert probabilities[AbstractAction.ALL_IN] > 0.9


def test_the_order_of_the_hole_cards_does_not_matter():
    # Aces up with a ten on the turn facing a pot bet, dealt in either order.
    state = heads_up("AsTc", "8c8d", "AdQcQhTh2s")
    for action in (CALL, CHECK, CHECK, CHECK, Action(ActionType.BET, 200)):
        state, _ = apply_action(state, action)
    view = observation(state, 0)
    swapped = replace(view, hole_cards=((view.hole_cards[0][1], view.hole_cards[0][0]), view.hole_cards[1]))
    bot = RuleBot("probe", PRESETS["tag"])
    hole = combo_index(*view.my_cards)
    assert bot.action_probabilities(view) == bot.action_probabilities(swapped) == bot.policies(view, [hole])[hole]
