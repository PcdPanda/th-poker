from   dataclasses              import replace
from   typing                   import Any

import pytest

import thpoker.analysis.ev
from   thpoker.analysis.ev      import (PHONE_SECONDS, PROFILES, icm_equities,
                                        option_values, pick_profile)
from   thpoker.analysis.tests.tracking \
                                import Calls, REFERENCE, weights
from   thpoker.bots.abstraction import AbstractAction, legal_abstract_actions
from   thpoker.bots.bot         import Bot, PRESETS
from   thpoker.bots.equity_bot  import public_seed
from   thpoker.game.cards       import parse_cards
from   thpoker.game.engine      import new_hand, observation
from   thpoker.game.evaluator   import evaluate
from   thpoker.game.rng         import Rng
from   thpoker.game.state       import (Action, ActionType, GameConfig,
                                        GameState, Observation)
from   thpoker.game.tests.decks import (CALL, CHECK, heads_up, play,
                                        stacked_deck)
from   thpoker.odds             import settled_equity

BOARD = "Qh9d4c3s2d"
USER_HOLE = "KsKc"
# Against KK on this board AA wins and the rest lose.
VILLAIN_HANDS = ["AhAc", "JhJc", "TsTc", "8h8c"]


class CallsWithAces(Bot):
    """Checks when it can; facing a bet it calls with AA and folds everything else."""

    name, style = "aces", PRESETS["balanced"]

    def policy(self, view: Observation) -> tuple[dict[AbstractAction, float], dict[str, Any]]:
        if AbstractAction.CHECK in legal_abstract_actions(view):
            return {AbstractAction.CHECK: 1.0}, {}
        aces = min(view.my_cards) // 4 == 12
        return {AbstractAction.CALL if aces else AbstractAction.FOLD: 1.0}, {}


class CallsWithAcesAndHalfTheJacks(CallsWithAces):
    """Like `CallsWithAces`, but also calls half the time with JJ."""

    def policy(self, view: Observation) -> tuple[dict[AbstractAction, float], dict[str, Any]]:
        if AbstractAction.CHECK not in legal_abstract_actions(view) and min(view.my_cards) // 4 == 9:
            return {AbstractAction.CALL: 0.5, AbstractAction.FOLD: 0.5}, {}
        return super().policy(view)


def spot(villain_hole: str = "AhAc", street_checks: int = 2) -> GameState:
    """Heads-up limped pot of 200 checked through `street_checks` streets, so the user (big
    blind, seat 1) acts first on the turn (1) or the river (2)."""
    state = heads_up(villain_hole, USER_HOLE, BOARD)
    return play(state, CALL, CHECK, *[CHECK] * (2 * street_checks))


def test_against_a_caller_every_bet_is_worth_its_showdown_share():
    # KK wins 3 of 4 villain hands: checking wins 0.75 * 200; a bet B called by everything
    # adds 0.75 * 2B - B = 0.5B.
    values = option_values(spot(), {0: Calls()}, REFERENCE, {0: weights(*VILLAIN_HANDS)})
    for value in values:
        amount = value.action.amount or 0
        assert value.exact and value.stderr == 0
        assert value.ev == pytest.approx((150 + 0.5 * amount) / 100)
    assert {v.action.type for v in values} == {ActionType.CHECK, ActionType.BET}
    shove = next(v for v in values if v.abstract == AbstractAction.ALL_IN)
    assert all(v.ev <= shove.ev for v in values)


def test_the_villain_range_splits_by_its_answer_to_a_bet():
    # A bet B: TT and 88 fold, JJ folds half the time (the user wins the 200 pot), AA calls
    # and wins, JJ calls the other half and loses: 0.625 * 200 - 0.25B + 0.125 * (200 + B).
    bots = {0: CallsWithAcesAndHalfTheJacks()}
    for value in option_values(spot(), bots, REFERENCE, {0: weights(*VILLAIN_HANDS)}):
        amount = value.action.amount or 0
        expected = 150 if value.action.type == ActionType.CHECK else 150 - 0.125 * amount
        assert value.exact and value.ev == pytest.approx(expected / 100)


def test_an_all_in_before_the_river_is_valued_on_every_river_card():
    # Turn all in against a caller: equity over the rivers, not the villain's real (winning) AA.
    turn = spot(street_checks=1)
    board, hero = turn.board, parse_cards(USER_HOLE)
    total = 0.0
    for hand in VILLAIN_HANDS:
        villain = parse_cards(hand)
        rivers = [c for c in range(52) if c not in (*board, *hero, *villain)]
        mine = [evaluate([*hero, *board, r]) for r in rivers]
        theirs = [evaluate([*villain, *board, r]) for r in rivers]
        total += sum((m > t) + 0.5 * (m == t) for m, t in zip(mine, theirs)) / len(rivers)
    equity = total / len(VILLAIN_HANDS)
    shove = next(
        v
        for v in option_values(turn, {0: Calls()}, REFERENCE, {0: weights(*VILLAIN_HANDS)})
        if v.abstract == AbstractAction.ALL_IN
    )
    expected = (equity * 20_000 - 9_900) / 100
    assert shove.exact and shove.ev == pytest.approx(expected)


def test_values_use_only_public_information_and_the_users_cards():
    # On the turn the leaves sample runouts; neither the villain's real hand nor the hand's
    # seed may change what they see.
    ranges = {0: weights(*VILLAIN_HANDS)}
    with_aces = option_values(spot("AhAc", street_checks=1), {0: CallsWithAces()}, REFERENCE, ranges)
    with_eights = option_values(spot("8h8c", street_checks=1), {0: CallsWithAces()}, REFERENCE, ranges)
    reseeded = option_values(replace(spot("AhAc", street_checks=1), seed=99), {0: CallsWithAces()}, REFERENCE, ranges)
    assert with_aces == with_eights == reseeded
    assert any(v.stderr > 0 for v in with_aces)


def test_a_three_way_river_walks_each_opponent_in_turn():
    # Seat 1 (the user, small blind) acts first; seat 2 always calls, seat 0 folds to a bet.
    deck = stacked_deck(0, 3, {0: "JhJc", 1: USER_HOLE, 2: "AhAc"}, BOARD)
    state, _ = new_hand(GameConfig(3), 1, 0, (10_000,) * 3, deck=deck)
    river = play(state, CALL, CALL, CHECK, *[CHECK] * 6)
    ranges = {0: weights("JhJc", "TsTc"), 2: weights(*VILLAIN_HANDS)}
    values = option_values(river, {0: CallsWithAces(), 2: Calls()}, REFERENCE, ranges)
    for value in values:
        if value.action.type == ActionType.BET:
            # Heads-up showdown against seat 2 for the 300 pot plus both bets.
            assert value.exact and value.ev == pytest.approx(
                (0.75 * (300 + 2 * value.action.amount) - value.action.amount) / 100
            )
        else:
            assert not value.exact  # a three-way showdown multiplies per-opponent shares


def test_a_size_outside_the_abstraction_gets_its_own_value():
    taken = Action(ActionType.BET, 130)
    values = option_values(spot(), {0: Calls()}, REFERENCE, {0: weights(*VILLAIN_HANDS)}, taken)
    custom = [v for v in values if v.abstract is None]
    assert [v.action for v in custom] == [taken] and custom[0].ev == pytest.approx((150 + 65) / 100)


def test_with_one_prize_tournament_equity_is_the_chip_share():
    values = option_values(spot(), {0: Calls()}, REFERENCE, {0: weights(*VILLAIN_HANDS)}, payouts=[1.0])
    for value in values:
        amount = value.action.amount or 0
        # KK wins 3 of 4: expected chips after the hand out of 20,000 in play.
        assert value.icm == pytest.approx((9_900 - amount + 0.75 * (200 + 2 * amount)) / 20_000)


def test_tournament_equity_weights_each_finish_by_its_chance():
    # Three-handed, 65/35: the button folds and the user (small blind) shoves the river into a
    # caller. Winning busts the big blind, leaving the user 20,000 against 10,000 (first with
    # 2/3); losing busts the user in 3rd.
    deck = stacked_deck(0, 3, {0: "JhJc", 1: USER_HOLE, 2: "AhAc"}, BOARD)
    state, _ = new_hand(GameConfig(3), 1, 0, (10_000,) * 3, deck=deck)
    river = play(state, Action(ActionType.FOLD), CALL, CHECK, *[CHECK] * 4)
    values = option_values(river, {2: Calls()}, REFERENCE, {2: weights(*VILLAIN_HANDS)}, payouts=[0.65, 0.35])
    shove = next(v for v in values if v.abstract == AbstractAction.ALL_IN)
    assert shove.icm == pytest.approx(0.75 * (0.65 * 2 / 3 + 0.35 / 3))
    check = next(v for v in values if v.action.type == ActionType.CHECK)
    won = icm_equities([10_000, 10_100, 9_900], [10_000] * 3, [0.65, 0.35])[1]
    lost = icm_equities([10_000, 9_900, 10_100], [10_000] * 3, [0.65, 0.35])[1]
    assert check.icm == pytest.approx(0.75 * won + 0.25 * lost)


class RarelyShoves(Bot):
    """Checks when it can; facing a bet it calls, but shoves 0.5% of the time."""

    name, style = "rare", PRESETS["balanced"]

    def policy(self, view: Observation) -> tuple[dict[AbstractAction, float], dict[str, Any]]:
        legal = legal_abstract_actions(view)
        if AbstractAction.CHECK in legal:
            return {AbstractAction.CHECK: 1.0}, {}
        if AbstractAction.ALL_IN not in legal:  # facing an all-in
            return {AbstractAction.CALL: 1.0}, {}
        return {AbstractAction.CALL: 0.995, AbstractAction.ALL_IN: 0.005}, {}


def test_pruned_branches_make_a_value_inexact_with_an_error():
    ranges = {0: weights(*VILLAIN_HANDS)}
    full = option_values(spot(), {0: RarelyShoves()}, REFERENCE, ranges)
    rough = option_values(spot(), {0: RarelyShoves()}, REFERENCE, ranges, min_branch=1e-2)
    small_bet = Action(ActionType.BET, 100)  # the smallest bet; a third of the pot rounds up to it
    kept = next(v for v in full if v.action == small_bet)
    pruned = next(v for v in rough if v.action == small_bet)
    assert kept.exact and not pruned.exact and pruned.stderr > 0
    assert abs(pruned.ev - kept.ev) <= pruned.stderr


def test_a_user_all_in_takes_its_equity_while_others_can_still_bet():
    # The short small blind shoves the flop and both deep players call: nobody can push the
    # user off its share of the 3,000 pot, so no realization discount applies.
    deck = stacked_deck(0, 3, {0: "JhJc", 1: USER_HOLE, 2: "AhAc"}, BOARD)
    state, _ = new_hand(GameConfig(3), 1, 0, (10_000, 1_000, 10_000), deck=deck)
    flop = play(state, CALL, CALL, CHECK)
    ranges = {0: weights("JhJc", "TsTc"), 2: weights(*VILLAIN_HANDS)}
    values = option_values(flop, {0: Calls(), 2: Calls()}, REFERENCE, ranges)
    shove = next(v for v in values if v.abstract == AbstractAction.ALL_IN)
    rng = Rng(public_seed(observation(flop, 1)))
    equity = settled_equity(tuple(parse_cards(USER_HOLE)), flop.board, [ranges[0], ranges[2]], rng)
    assert shove.ev == pytest.approx((equity.value * 3_000 - 900) / 100)


def test_tournament_side_pots_are_won_in_order_with_the_main_pot():
    # Seat 2 is short, the user is in the middle, seat 0 is deep. The user shoves the river and
    # both call: a 9,000 main pot and a 6,000 side pot against seat 0 alone. KK beats seat 2's
    # jacks and wins only if seat 0 holds eights, and then it wins both pots.
    deck = stacked_deck(0, 3, {0: "AhAc", 1: USER_HOLE, 2: "JhJc"}, BOARD)
    state, _ = new_hand(GameConfig(3), 1, 0, (10_000, 6_000, 3_000), deck=deck)
    river = play(state, CALL, CALL, CHECK, *[CHECK] * 6)
    ranges = {0: weights("AhAc", "8h8c"), 2: weights("JhJc")}
    values = option_values(river, {0: Calls(), 2: Calls()}, REFERENCE, ranges, payouts=[0.65, 0.35])
    shove = next(v for v in values if v.abstract == AbstractAction.ALL_IN)
    both = icm_equities([4_000, 15_000, 0], [10_000, 6_000, 3_000], [0.65, 0.35])[1]
    assert shove.icm == pytest.approx(0.5 * both)  # losing busts the user in third, unpaid


def test_with_one_prize_equity_is_the_chip_share():
    assert icm_equities([6_000, 3_000, 1_000], [5_000] * 3, [1.0]) == pytest.approx([0.6, 0.3, 0.1])


def test_three_players_match_the_malmuth_harville_hand_calculation():
    # First: 0.5/0.3/0.2. The big stack is second with 0.3 * 5/7 + 0.2 * 5/8 and third with the
    # rest, so 0.5 * 0.5 + 0.3 * 0.33929 + 0.2 * 0.16071 = 0.38393.
    equities = icm_equities([5_000, 3_000, 2_000], [1] * 3, [0.5, 0.3, 0.2])
    second = 0.3 * 5 / 7 + 0.2 * 5 / 8
    assert equities[0] == pytest.approx(0.5 * 0.5 + 0.3 * second + 0.2 * (1 - 0.5 - second))
    assert sum(equities) == pytest.approx(1.0)
    # Chips are worth more to a short stack: 20% of the chips is worth more than 20%.
    assert equities[2] > 0.2


def test_players_busting_in_the_hand_finish_below_the_rest_by_start_stack():
    # Seats 1 and 2 bust in the same hand; seat 1 started it with more chips, so it takes 2nd.
    equities = icm_equities([10_000, 0, 0, 0], [4_000, 3_000, 2_000, 0], [0.5, 0.3, 0.2])
    assert equities == pytest.approx([0.5, 0.3, 0.2, 0.0])


def test_players_busting_with_equal_start_stacks_share_their_places():
    equities = icm_equities([8_000, 0, 0], [4_000, 2_000, 2_000], [0.65, 0.35])
    assert equities == pytest.approx([0.65, 0.175, 0.175])


def test_seats_already_out_of_the_tournament_decide_nothing():
    # Four entrants remain at the start of the hand, so they decide places 1 to 4.
    equities = icm_equities([5_000, 5_000, 0, 5_000, 5_000], [5_000, 5_000, 0, 5_000, 5_000], [0.5, 0.3, 0.2])
    assert equities == pytest.approx([0.25, 0.25, 0.0, 0.25, 0.25])


def test_nobody_with_chips_is_an_error():
    with pytest.raises(ValueError, match="nobody"):
        icm_equities([0, 0], [100, 100], [1.0])


def test_profiles_are_picked_by_name_or_by_the_benchmark(monkeypatch):
    assert pick_profile("phone") is PROFILES["phone"]
    monkeypatch.setattr(thpoker.analysis.ev, "benchmark", lambda: PHONE_SECONDS * 2)
    assert pick_profile().name == "phone"
    monkeypatch.setattr(thpoker.analysis.ev, "benchmark", lambda: PHONE_SECONDS / 4)
    assert pick_profile().name == "pc"
    with pytest.raises(ValueError, match="unknown device profile"):
        pick_profile("toaster")


def test_a_phone_prunes_more_and_solves_longer():
    pc, phone = PROFILES["pc"], PROFILES["phone"]
    assert phone.min_branch > pc.min_branch and phone.triage_branch > pc.triage_branch
    assert phone.solver_seconds > pc.solver_seconds
