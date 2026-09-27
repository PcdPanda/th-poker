from   dataclasses              import replace
from   typing                   import Any

import numpy as np
import pytest

from   thpoker.analysis.ev      import PROFILES
from   thpoker.analysis.review  import (TRIAGE_BRANCH, all_in_net, hint,
                                        needs_full_review, position_names,
                                        range_view, review_decision,
                                        review_hand, situation, summarize,
                                        thresholds)
from   thpoker.analysis.tests.tracking \
                                import Calls, REFERENCE
from   thpoker.bots.abstraction import AbstractAction, legal_abstract_actions
from   thpoker.bots.bot         import Bot, PRESETS
from   thpoker.game.cards       import (COMBOS, PREFLOP_CLASSES, combo_index,
                                        parse_cards)
from   thpoker.game.engine      import new_hand, observation
from   thpoker.game.evaluator   import evaluate
from   thpoker.game.rng         import Rng
from   thpoker.game.state       import (Action, ActionType, GameConfig,
                                        Observation, Street)
from   thpoker.game.tests.decks import (CALL, CHECK, FOLD, heads_up, play,
                                        stacked_deck)
from   thpoker.text             import render_decision


class BetsTheRiver(Bot):
    """Calls or checks until the river, where it bets the pot when checked to."""

    name, style = "river", PRESETS["balanced"]

    def policy(self, view: Observation) -> tuple[dict[AbstractAction, float], dict[str, Any]]:
        legal = legal_abstract_actions(view)
        if view.street == Street.RIVER and AbstractAction.BET_100 in legal:
            return {AbstractAction.BET_100: 1.0}, {}
        return {AbstractAction.CALL if AbstractAction.CALL in legal else AbstractAction.CHECK: 1.0}, {}


def test_seats_are_named_by_the_players_dealt_in():
    six, _ = new_hand(GameConfig(6), 1, 0, (10_000,) * 6)
    assert position_names(observation(six, 0)) == {
        0: "BTN",
        1: "SB",
        2: "BB",
        3: "UTG",
        4: "HJ",
        5: "CO",
    }
    eight, _ = new_hand(GameConfig(8), 1, 0, (10_000,) * 8)
    assert [position_names(observation(eight, 0))[s] for s in range(3, 8)] == [
        "UTG",
        "UTG+1",
        "UTG+2",
        "HJ",
        "CO",
    ]
    heads_up, _ = new_hand(GameConfig(2), 1, 0, (10_000,) * 2)
    assert position_names(observation(heads_up, 0)) == {0: "BTN", 1: "BB"}


def test_situation_describes_a_three_bet_pot():
    deck = stacked_deck(0, 6, {s: h for s, h in enumerate(["KdQd", "9s9c", "Th6h", "5d4d", "Jc3s", "8h8s"])}, "")
    state, _ = new_hand(GameConfig(6), 1, 0, (10_000, 10_000, 4_000, 10_000, 10_000, 10_000), deck=deck)
    # UTG opens to 250, HJ 3-bets to 750, the rest fold to the big blind.
    state = play(state, Action(ActionType.RAISE, 250), Action(ActionType.RAISE, 750), FOLD, FOLD, FOLD)
    spot = situation(observation(state, 2))
    assert (spot.position, spot.pot_type, spot.players) == ("BB", "3-bet", 3)
    assert spot.pot_bb == pytest.approx(11.5) and spot.to_call_bb == pytest.approx(6.5)
    # The big blind's 4,000 is the shortest stack, and 100 of it is already in.
    assert spot.effective_bb == pytest.approx(39.0) and spot.spr is None
    # The 3-bet adds 750 to the 400 before it; the stack counts from the start of the hand.
    assert (spot.role, spot.facing, spot.stack_bb) == (
        "none",
        pytest.approx(750 / 400),
        pytest.approx(40.0),
    )
    # On the flop the big blind called the 3-bet and the 3-bettor made the last raise.
    flop = play(state, CALL, CALL)
    assert situation(observation(flop, 2)).role == "caller"
    assert situation(observation(play(flop, CHECK, CHECK), 4)).role == "aggressor"


def test_a_uniform_range_is_all_there_and_a_set_is_two_pair_or_better():
    board = tuple(parse_cards("Ad7c2s"))
    uniform = range_view(np.ones(len(COMBOS)), board)
    assert uniform.width == pytest.approx(1.0) and min(uniform.classes) == pytest.approx(1.0)
    sets = np.zeros(len(COMBOS))
    sets[combo_index(*parse_cards("AhAc"))] = sets[combo_index(*parse_cards("7h7d"))] = 1.0
    view = range_view(sets, board)
    assert view.groups == {"two pair+": 1.0, "one pair": 0.0, "draw": 0.0, "air": 0.0}
    assert view.classes[PREFLOP_CLASSES.index("AA")] == pytest.approx(1 / 3)  # 1 of the 3 combos left


@pytest.mark.parametrize(
    "hand, board, group",
    [
        ("KsKc", "Qh9d4c", "one pair"),  # an overpair
        ("AsKc", "7c7d2s", "air"),  # the board's pair is not the hand's
        ("AhKh", "Qh7h2c", "draw"),  # four hearts
        ("9s8c", "7h6d2c", "draw"),  # four in a row
        ("Qs9c", "Qh9d4c", "two pair+"),
    ],
)
def test_hands_group_by_what_the_hole_cards_add(hand, board, group):
    weights = np.zeros(len(COMBOS))
    weights[combo_index(*parse_cards(hand))] = 1.0
    view = range_view(weights, tuple(parse_cards(board)))
    assert view.groups is not None and view.groups[group] == 1.0


def test_thresholds_facing_a_pot_bet_heads_up_and_three_way():
    state = play(heads_up("QsJs", "8c8d", "Kh7d2c3s9h"), CALL, CHECK, CHECK, Action(ActionType.BET, 200))
    two = thresholds(observation(state, 1))
    assert two.required_equity == pytest.approx(1 / 3) and two.defense_share == pytest.approx(0.5)
    three, _ = new_hand(GameConfig(3), 1, 0, (10_000,) * 3)
    three = play(three, CALL, CALL, CHECK, Action(ActionType.BET, 300))
    # A pot bet into two players: each defends 1 - 0.5 ** (1/2).
    assert thresholds(observation(three, 2)).defense_share == pytest.approx(1 - 0.5**0.5)


def test_an_all_in_is_judged_by_its_equity_not_the_river_card():
    # Turn all in with KK against AA's shown hand; the river is dealt from the rest of the deck.
    state = heads_up("AhAc", "KsKc")
    state = play(state, CALL, CHECK, CHECK, CHECK, Action(ActionType.BET, 9_900), CALL)
    board, kings, aces = state.board[:4], parse_cards("KsKc"), parse_cards("AhAc")
    rivers = [c for c in range(52) if c not in (*board, *kings, *aces)]
    wins = sum(evaluate([*kings, *board, r]) > evaluate([*aces, *board, r]) for r in rivers)
    assert all_in_net(state, 1) == pytest.approx(wins / len(rivers) * 20_000 - 10_000)
    assert state.stacks[1] == 0  # the real river lost


def test_folding_the_nuts_to_a_river_bet_is_a_mistake_and_calling_is_not():
    state = heads_up("8c8d", "JhTh", "AhKhQh2c3d")
    river_bet = play(state, CALL, CHECK, *[CHECK] * 4, CHECK, Action(ActionType.BET, 200))
    folded = review_hand(play(river_bet, FOLD), 1, {0: BetsTheRiver()}, REFERENCE)
    called = review_hand(play(river_bet, CALL), 1, {0: BetsTheRiver()}, REFERENCE)
    fold_decision, call_decision = folded.decisions[-1], called.decisions[-1]
    # Calling wins the 200 pot plus the 200 bet whatever the bot holds.
    for options in (call_decision.exploitative, call_decision.reference):
        call = next(o for o in options if o.action == CALL)
        assert call.ev == pytest.approx(4.0)
    # Against the bot the walk is exact; the heads-up river reference is the solver.
    assert next(o for o in call_decision.exploitative if o.action == CALL).exact
    assert call_decision.reference_note is not None and call_decision.reference_note.startswith("river solver")
    assert fold_decision.verdict() == "mistake" and fold_decision.loss(fold_decision.reference)[0] >= 4.0
    # Checking the nuts earlier can cost even more, so the fold is one mistake among several.
    summary = summarize([folded, called], top=20)
    losses = [d.loss(d.reference)[0] for _, d in summary.mistakes]
    assert (folded.hand.hand_id, fold_decision) in summary.mistakes and losses == sorted(losses, reverse=True)
    assert summary.net_bb == pytest.approx(-1.0 + 3.0)


def test_the_river_solver_reviews_a_user_who_covers_the_opponent():
    # The user's all-in is more than the 30bb opponent can call, so it plays as a bet of
    # everything the opponent has; the solver still prices the decision.
    deck = stacked_deck(0, 2, {0: "8c8d", 1: "JhTh"}, "AhKhQh2c3d")
    state, _ = new_hand(GameConfig(2), 1, 0, (3_000, 10_000), deck=deck)
    hand = play(state, CALL, CHECK, *[CHECK] * 4, CHECK, Action(ActionType.BET, 200), CALL)
    review = review_hand(hand, 1, {0: BetsTheRiver()}, REFERENCE)
    for decision in review.decisions[-2:]:  # the river check, then the call
        assert any(o.action.amount == 9_900 for o in decision.reference)
        assert decision.reference_note is not None
        assert decision.reference_note.startswith("river solver")
    # With the nuts, calling wins the 200 pot and the 200 bet; the all-in can only add to that.
    options = {o.action: o.ev for o in review.decisions[-1].reference}
    assert options[CALL] == pytest.approx(4.0)
    assert options[Action(ActionType.RAISE, 9_900)] >= options[CALL]


class FoldsToARaise(Bot):
    """Calls or checks until the river, where it bets the pot and folds to a raise."""

    name, style = "folds", PRESETS["balanced"]

    def policy(self, view: Observation) -> tuple[dict[AbstractAction, float], dict[str, Any]]:
        legal = legal_abstract_actions(view)
        if view.street == Street.RIVER:
            return {AbstractAction.FOLD if AbstractAction.CALL in legal else AbstractAction.BET_100: 1.0}, {}
        return {AbstractAction.CALL if AbstractAction.CALL in legal else AbstractAction.CHECK: 1.0}, {}


class RaisesTheRiver(Bot):
    """Calls or checks until the river, where it raises a bet by the pot when it can."""

    name, style = "raises", PRESETS["balanced"]

    def policy(self, view: Observation) -> tuple[dict[AbstractAction, float], dict[str, Any]]:
        legal = legal_abstract_actions(view)
        facing = AbstractAction.CALL in legal
        if view.street == Street.RIVER and facing and AbstractAction.BET_100 in legal:
            return {AbstractAction.BET_100: 1.0}, {}
        return {AbstractAction.CALL if AbstractAction.CALL in legal else AbstractAction.CHECK: 1.0}, {}


def test_the_river_solver_counts_a_folded_players_river_bet_in_the_pot():
    # Three-handed: seat 1 bets the river, the user calls, seat 0 raises and seat 1 folds. With
    # the nuts, calling wins the whole pot, seat 1's dead bet included.
    deck = stacked_deck(0, 3, {0: "8c8d", 1: "7c7d", 2: "JhTh"}, "AhKhQh2c3d")
    state, _ = new_hand(GameConfig(3), 1, 0, (10_000,) * 3, deck=deck)
    bots: dict[int, Bot] = {0: RaisesTheRiver(), 1: FoldsToARaise()}
    state = play(state, CALL, CALL, CHECK, *[CHECK] * 6)
    state = play(state, bots[1].decide(observation(state, 1), Rng(0)).action, CALL)
    facing = play(state, bots[0].decide(observation(state, 0), Rng(0)).action, FOLD)
    decision = review_hand(play(facing, CALL), 2, bots, REFERENCE).decisions[-1]
    assert facing.folded[1] and facing.committed_this_street[1] > 0
    assert decision.reference_note is not None
    assert decision.reference_note.startswith("river solver")
    call = next(o for o in decision.reference if o.action == CALL)
    assert call.ev == pytest.approx(facing.pot / facing.config.big_blind)


def test_one_decision_is_reviewed_as_in_the_whole_hand():
    state = heads_up("8c8d", "JhTh", "AhKhQh2c3d")
    hand = play(state, CALL, CHECK, *[CHECK] * 4, CHECK, Action(ActionType.BET, 200), CALL)
    bots: dict[int, Bot] = {0: BetsTheRiver()}
    whole = review_hand(hand, 1, bots, REFERENCE, None, TRIAGE_BRANCH, None).decisions[-1]
    one = review_decision(hand, 1, bots, REFERENCE, whole.index, None, TRIAGE_BRANCH, None)
    assert (one.reference, one.exploitative, one.chosen) == (
        whole.reference,
        whole.exploitative,
        whole.chosen,
    )
    with pytest.raises(ValueError, match="not a decision"):
        review_decision(hand, 1, bots, REFERENCE, 0)  # the button's limp


def test_the_phone_profile_still_solves_rivers_and_the_triage_pass_does_not():
    state = heads_up("8c8d", "JhTh", "AhKhQh2c3d")
    hand = play(state, CALL, CHECK, *[CHECK] * 4, CHECK, Action(ActionType.BET, 200), CALL)
    bots: dict[int, Bot] = {0: BetsTheRiver()}
    index = len(hand.history) - 1
    phone = PROFILES["phone"]
    solved = review_decision(hand, 1, bots, REFERENCE, index, None, phone.min_branch, phone.solver_seconds)
    assert solved.reference_note is not None and solved.reference_note.startswith("river solver")
    walked = review_decision(hand, 1, bots, REFERENCE, index, None, phone.triage_branch, None)
    assert walked.reference_note is None


def test_tournament_decisions_are_judged_in_prize_equity():
    state = heads_up("8c8d", "JhTh", "AhKhQh2c3d")
    folded = play(state, CALL, CHECK, *[CHECK] * 4, CHECK, Action(ActionType.BET, 200), FOLD)
    decision = review_hand(folded, 1, {0: BetsTheRiver()}, REFERENCE, payouts=[1.0]).decisions[-1]
    # With one prize equity is the chip share, so half a big blind is 50 of 20,000 chips.
    assert decision.icm_threshold == pytest.approx(50 / 20_000)
    assert decision.verdict() == "mistake"


def test_an_exploit_spot_is_a_different_best_option_by_a_clear_margin():
    state = heads_up("8c8d", "JhTh", "AhKhQh2c3d")
    decision = review_hand(play(state, CALL, CHECK, *[CHECK] * 4, CHECK, CHECK), 1, {0: Calls()}, REFERENCE).decisions[
        -1
    ]
    check, small, shove = (
        next(o for o in decision.exploitative if o.abstract == a)
        for a in (AbstractAction.CHECK, AbstractAction.BET_33, AbstractAction.ALL_IN)
    )
    against_bot = [replace(check, ev=2.0), replace(small, ev=4.0), replace(shove, ev=50.0)]
    reference = [replace(check, ev=2.0), replace(small, ev=4.0), replace(shove, ev=3.0)]
    assert replace(decision, exploitative=against_bot, reference=reference).exploit_spot()
    agreeing = [replace(check, ev=2.0), replace(small, ev=4.0), replace(shove, ev=4.5)]
    assert not replace(decision, exploitative=agreeing, reference=reference).exploit_spot()


def test_triage_sends_only_possible_mistakes_to_a_full_review():
    # Heads-up the reference never opens 72o from the button, so folding it loses nothing.
    trash = heads_up("7c2d", "8c8d", "AhKhQh2c3d")
    folded_trash = review_hand(play(trash, FOLD), 0, {1: Calls()}, REFERENCE, min_branch=TRIAGE_BRANCH)
    assert not needs_full_review(folded_trash)
    nuts = heads_up("8c8d", "JhTh", "AhKhQh2c3d")
    folded_nuts = play(nuts, CALL, CHECK, *[CHECK] * 4, CHECK, Action(ActionType.BET, 200), FOLD)
    assert needs_full_review(review_hand(folded_nuts, 1, {0: BetsTheRiver()}, REFERENCE, min_branch=TRIAGE_BRANCH))


def test_thresholds_facing_a_raise_and_a_shove_the_user_cannot_cover():
    deck = stacked_deck(0, 2, {0: "QsJs", 1: "8c8d"}, "Kh7d2c3s9h")
    state, _ = new_hand(GameConfig(2), 1, 0, (10_000, 1_100), deck=deck)
    # The big blind bets 100 into 200 and the button raises to 400: the raise risks 400 to win
    # the 300 before it, so the user must continue with 300 / 700 of its range.
    raised = play(state, CALL, CHECK, Action(ActionType.BET, 100), Action(ActionType.RAISE, 400))
    assert thresholds(observation(raised, 1)).defense_share == pytest.approx(300 / 700)
    # The user has 1,000 behind facing a 9,900 shove: calling 900 wins only chips it covers.
    shoved = play(state, CALL, CHECK, CHECK, Action(ActionType.BET, 9_900))
    assert thresholds(observation(shoved, 1)).required_equity == pytest.approx(1_000 / (1_100 + 1_100))


def test_all_in_results_are_not_adjusted_when_others_bet_on_later_streets():
    # The short big blind is all in preflop; the other two bet the flop and get all in on the
    # turn, so the big blind's result also rode on the flop and turn.
    deck = stacked_deck(0, 3, {0: "AhAc", 1: "KsKc", 2: "QdQh"}, "Js9d4c3s2d")
    state, _ = new_hand(GameConfig(3), 1, 0, (10_000, 3_000, 500), deck=deck)
    flop_bet, turn_shove = Action(ActionType.BET, 1_000), Action(ActionType.BET, 1_500)
    hand = play(state, Action(ActionType.RAISE, 500), CALL, CALL, flop_bet, CALL, turn_shove, CALL)
    assert hand.all_in[2] and all_in_net(hand, 2) is None
    assert all_in_net(hand, 1) is not None  # the small blind's own all-in on the turn counts


@pytest.mark.parametrize(
    "hand, board, group",
    [
        ("AhKd", "7c7d7h7s", "air"),  # quads on the board are everyone's
        ("JdTc", "9h8s7c6d5h", "two pair+"),  # a higher straight than the board's
        ("2c3c", "9h8s7c6d5h", "air"),  # the board's straight plays
    ],
)
def test_board_made_hands_count_only_when_the_hole_cards_beat_them(hand, board, group):
    weights = np.zeros(len(COMBOS))
    weights[combo_index(*parse_cards(hand))] = 1.0
    view = range_view(weights, tuple(parse_cards(board)))
    assert view.groups is not None and view.groups[group] == 1.0


def test_a_hint_shows_the_decision_ahead_without_judging_it():
    state = heads_up("8c8d", "JhTh", "AhKhQh2c3d")
    facing_bet = play(state, CALL, CHECK, *[CHECK] * 4, CHECK, Action(ActionType.BET, 200))
    decision = hint(facing_bet, 1, {0: BetsTheRiver()}, REFERENCE)
    assert decision.chosen is None and decision.index == len(facing_bet.history)
    assert {o.action for o in decision.reference} >= {FOLD, CALL}
    lines = render_decision(decision, {0: "Alex"}, 1)
    assert lines[0].endswith("your move") and not any("Mistake" in line or "Best option" in line for line in lines)
