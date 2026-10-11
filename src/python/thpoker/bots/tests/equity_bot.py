from   dataclasses              import replace

import pytest

from   thpoker.bots.abstraction import AbstractAction, legal_abstract_actions
from   thpoker.bots.bot         import PRESETS
from   thpoker.bots.equity_bot  import (EquityBot, assumed_ranges, bet_shifted,
                                        preflop_range)
from   thpoker.bots.tests.bot   import random_views
from   thpoker.game.cards       import COMBOS, combo_index, parse_cards
from   thpoker.game.engine      import apply_action, new_hand, observation
from   thpoker.game.state       import (Action, ActionType, GameConfig,
                                        GameState)
from   thpoker.game.tests.decks import (CALL, CHECK, FOLD, heads_up, play,
                                        stacked_deck)


def test_policy_is_a_legal_distribution_that_never_folds_when_checking_is_free():
    for style in ("nit", "maniac"):
        bot = EquityBot("probe", PRESETS[style])
        for view in random_views(EquityBot("probe", PRESETS["tag"]), 11, 120):
            distribution, rationale = bot.policy(view)
            assert abs(sum(distribution.values()) - 1.0) < 1e-9
            assert set(distribution) <= set(legal_abstract_actions(view))
            if AbstractAction.CHECK in legal_abstract_actions(view):
                assert AbstractAction.FOLD not in distribution
            assert bot.policy(view) == (distribution, rationale)


def test_hands_clearly_priced_in_never_fold_heads_up_after_the_flop():
    bot = EquityBot("probe", PRESETS["tag"])
    checked = 0
    for view in random_views(EquityBot("probe", PRESETS["tag"]), 13, 400):
        distribution, rationale = bot.policy(view)
        heads_up_bet = view.board and rationale["opponents"] == 1 and rationale["required_equity"]
        if heads_up_bet and rationale["equity_estimate"] - rationale["required_equity"] >= 0.3:
            assert AbstractAction.FOLD not in distribution
            checked += 1
    assert checked  # the sample must contain such spots for the check to mean anything


def test_a_short_big_blind_calls_a_shove_it_is_priced_into():
    # The big blind has 3bb and holds Qh7c; the small blind shoves. Only 2bb more buys a pot of
    # 6bb (the rest of the shove is returned), so the price is 1/3, not about 1/2.
    deck = stacked_deck(0, 2, {0: "AsKd", 1: "Qh7c"}, "")
    state = play(new_hand(GameConfig(2), 1, 0, (10_000, 300), deck=deck)[0], Action(ActionType.RAISE, 10_000))
    distribution, rationale = EquityBot("probe", PRESETS["tag"]).policy(observation(state, 1))
    assert rationale["required_equity"] == pytest.approx(1 / 3, abs=1e-4)
    assert distribution[AbstractAction.CALL] > 0.8


@pytest.mark.parametrize("style", sorted(PRESETS))
def test_aces_never_fold_preflop_and_the_nuts_never_fold_on_the_river(style):
    bot = EquityBot("probe", PRESETS[style])
    deck = stacked_deck(0, 3, {0: "AsAd", 1: "KcKd", 2: "7h2c"}, "Ah5s3dTc2h")
    state = play(
        new_hand(GameConfig(3), 1, 0, (10_000,) * 3, deck=deck)[0],
        Action(ActionType.RAISE, 300),
        Action(ActionType.RAISE, 1_200),
        FOLD,
    )
    assert AbstractAction.FOLD not in bot.action_probabilities(observation(state, 0))
    # Heads-up to the river with the nut straight flush facing a half-pot bet.
    state = play(
        heads_up("6h7h", "AsKd", "3h4h5hKcAc"),
        CALL,
        CHECK,
        CHECK,
        CHECK,
        CHECK,
        CHECK,
        Action(ActionType.BET, 100),
    )
    assert AbstractAction.FOLD not in bot.action_probabilities(observation(state, 0))


def test_assumed_ranges_follow_public_actions():
    # Button 0, blinds 1 and 2, first to act seat 3 of 6.
    utg_open = play(new_hand(GameConfig(6), 1, 0, (10_000,) * 6)[0], Action(ActionType.RAISE, 250))
    button_open = play(
        new_hand(GameConfig(6), 1, 0, (10_000,) * 6)[0],
        FOLD,
        FOLD,
        FOLD,
        Action(ActionType.RAISE, 250),
    )
    # Ranges come in seat order of the opponents still in: seat 3 is fourth from seat 4's view,
    # and the button is first from the small blind's view once seats 3 to 5 have folded.
    early = assumed_ranges(observation(utg_open, utg_open.to_act))[3]
    late = assumed_ranges(observation(button_open, button_open.to_act))[0]
    assert sum(early) < sum(late)  # an early open is narrower than a button open
    aces = combo_index(*parse_cards("AsAh"))
    called = play(utg_open, CALL)  # seat 4 flats
    caller_range = assumed_ranges(observation(called, called.to_act))[4]  # seat 4
    assert caller_range[aces] == 0  # a flat call rules out the very best hands


def test_a_postflop_bet_shifts_weight_toward_strong_hands():
    state = play(heads_up("QcJd", "8c8d", "8s5h2dKcAc"), CALL, CHECK)
    before = assumed_ranges(observation(state, 0))[0]
    state = play(state, Action(ActionType.BET, 150))  # big blind bets the flop
    after = assumed_ranges(observation(state, 0))[0]
    set_of_fives, air = combo_index(*parse_cards("5c5d")), combo_index(*parse_cards("7c6d"))
    assert before[set_of_fives] == before[air]
    assert after[set_of_fives] > after[air]


def test_the_big_blind_defends_a_button_open_far_wider_than_the_button_calls_an_early_open():
    bot = EquityBot("probe", PRESETS["tag"])

    def continue_rate(state: GameState) -> float:
        view = observation(state, state.to_act)
        total = 0.0
        for a, b in COMBOS:
            cards = list(view.hole_cards)
            cards[view.seat] = (a, b)
            total += 1 - bot.action_probabilities(replace(view, hole_cards=tuple(cards))).get(AbstractAction.FOLD, 0)
        return total / len(COMBOS)

    start = new_hand(GameConfig(6), 1, 0, (10_000,) * 6)[0]
    big_blind = play(start, FOLD, FOLD, FOLD, Action(ActionType.RAISE, 250), FOLD)
    button = play(start, Action(ActionType.RAISE, 250), FOLD, FOLD)
    # Position and price: the big blind gets 3-to-1 against a wide range.
    assert continue_rate(big_blind) > 2 * continue_rate(button)


def test_policy_and_policies_agree_on_the_hand_held():
    bot = EquityBot("probe", PRESETS["tag"])
    for view in random_views(EquityBot("probe", PRESETS["tag"]), 17, 40):
        hero = combo_index(*view.my_cards)
        assert bot.policies(view, [hero])[hero] == bot.action_probabilities(view)


def test_an_oversized_deep_open_reads_as_a_much_stronger_range():
    # Heads-up at 100bb a 2.5bb open is a wide range; a 100bb open-shove reads as the top 15%.
    state, _ = new_hand(GameConfig(2), 1, 0, (10_000, 10_000))
    opened, _ = apply_action(state, Action(ActionType.RAISE, 250))
    shoved, _ = apply_action(state, Action(ActionType.RAISE, 10_000))
    width = [sum(preflop_range(observation(s, 1), 0)) / len(COMBOS) for s in (opened, shoved)]
    assert width[0] > 0.3 and width[1] == pytest.approx(0.15, abs=0.03)  # whole classes
    # At 20bb shoving is routine, so the shove reads as an ordinary open.
    short, _ = new_hand(GameConfig(2), 1, 0, (2_000, 2_000))
    short_shove, _ = apply_action(short, Action(ActionType.RAISE, 2_000))
    assert sum(preflop_range(observation(short_shove, 1), 0)) == pytest.approx(
        sum(preflop_range(observation(opened, 1), 0))
    )


def test_a_re_raise_never_reads_wider_than_its_band():
    # Heads-up at 100bb: a 4x 3-bet out of position and an all-in 4-bet read as the usual 3-bet
    # and 4-bet bands; the narrowest oversized read (the top 15%) is wider than both.
    state, _ = new_hand(GameConfig(2), 1, 0, (10_000, 10_000))
    opened = play(state, Action(ActionType.RAISE, 250))

    def width(hand: GameState, seat: int) -> float:
        return sum(preflop_range(observation(hand, 1 - seat), seat))

    small_3bet = play(opened, Action(ActionType.RAISE, 750))
    four_x_3bet = play(opened, Action(ActionType.RAISE, 1_000))
    assert width(four_x_3bet, 1) == width(small_3bet, 1)
    small_4bet = play(four_x_3bet, Action(ActionType.RAISE, 2_500))
    all_in_4bet = play(four_x_3bet, Action(ActionType.RAISE, 10_000))
    assert width(all_in_4bet, 0) == width(small_4bet, 0)


def test_an_oversized_raise_is_read_by_the_raisers_depth_not_the_viewers():
    # Three-handed, the button (seat 0) opens first. A 20bb shove into 100bb stacks is routine
    # and reads as an open; a 100bb shove reads narrow even to a 20bb blind.
    short_button = new_hand(GameConfig(3), 1, 0, (2_000, 10_000, 10_000))[0]
    shove = play(short_button, Action(ActionType.RAISE, 2_000))
    opened = play(short_button, Action(ActionType.RAISE, 250))
    assert preflop_range(observation(shove, 1), 0) == preflop_range(observation(opened, 1), 0)
    deep_button = new_hand(GameConfig(3), 1, 0, (10_000, 2_000, 10_000))[0]
    deep_shove = play(deep_button, Action(ActionType.RAISE, 10_000))
    width = sum(preflop_range(observation(deep_shove, 1), 0)) / len(COMBOS)
    assert width == pytest.approx(0.15, abs=0.03)


def test_the_oversized_read_phases_in_with_depth():
    # No jump just past 25bb: a 26bb shove reads almost like a 25bb one.
    def shove_width(stack: int) -> float:
        state, _ = new_hand(GameConfig(2), 1, 0, (stack, stack))
        shoved = play(state, Action(ActionType.RAISE, stack))
        return sum(preflop_range(observation(shoved, 1), 0)) / len(COMBOS)

    assert shove_width(2_600) == pytest.approx(shove_width(2_500), abs=0.03)
    assert shove_width(2_500) > 0.3


def test_a_bet_over_the_pot_shrinks_the_floor_but_keeps_a_small_one():
    # The big blind bets two pots (400 into 200): Hard's floor of 0.25 shrinks by the square
    # of the size, and Expert's lower floor for a user who rarely bets stops at 0.05.
    state = play(heads_up("QcJd", "8c8d", "8s5h2dKcAc"), CALL, CHECK, Action(ActionType.BET, 400))
    view = observation(state, 0)
    full = tuple(1.0 for _ in COMBOS)
    for floor, weakest in ((0.25, 0.0625), (0.1, 0.05)):
        shifted = bet_shifted(view, 1, full, (floor, floor), sized=True)
        live = [w for w, (a, b) in zip(shifted, COMBOS) if not {a, b} & set(view.board)]
        assert min(live) == pytest.approx(weakest, abs=1e-3)
    # Unsized, as Medium reads it, the floor stays (the weakest tied hands rank a little above 0).
    assert 0.25 <= min(bet_shifted(view, 1, full)) < 0.26
