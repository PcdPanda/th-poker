from   dataclasses              import replace
import pytest
from   thpoker.analysis.drills  import MIXED, RIGHT, WRONG
from   thpoker.analysis.ev      import OptionValue
from   thpoker.analysis.review  import review_hand
from   thpoker.analysis.stats   import DecisionRecord
from   thpoker.analysis.tests.review \
                                import BetsTheRiver
from   thpoker.analysis.tests.tracking \
                                import REFERENCE
from   thpoker.analysis.training \
                                import (BEST_OPTION, EQUITY, Estimate,
                                        REQUIRED_EQUITY, best_action,
                                        calibration, grade_option,
                                        mistake_keys, parse_mistake_key,
                                        questions, score)
from   thpoker.game.state       import Action, ActionType
from   thpoker.game.tests.decks import CALL, CHECK, FOLD, heads_up, play


def nuts_facing_a_pot_bet():
    """A royal flush on the river facing a pot-size bet, and the check before it."""
    state = heads_up("8c8d", "JhTh", "AhKhQh2c3d")
    hand = play(state, CALL, CHECK, *[CHECK] * 4, CHECK, Action(ActionType.BET, 200), CALL)
    decisions = review_hand(hand, 1, {0: BetsTheRiver()}, REFERENCE).decisions
    return decisions[-2], decisions[-1]


def test_estimates_are_scored_against_the_review():
    checked, facing = nuts_facing_a_pot_bet()
    assert questions(checked) == [EQUITY, BEST_OPTION]
    assert questions(facing) == [EQUITY, REQUIRED_EQUITY, BEST_OPTION]
    # A royal flush has all the equity; calling 200 to win 600 needs a third.
    assert score(facing, EQUITY, 0.9, "hand-1") == Estimate(EQUITY, 0.9, 1.0, "hand-1")
    assert score(facing, REQUIRED_EQUITY, 0.25, "hand-1").actual == pytest.approx(1 / 3)
    assert score(facing, BEST_OPTION, FOLD, "hand-1").actual == 0.0
    assert score(facing, BEST_OPTION, best_action(facing), "hand-1").actual == 1.0


def test_calibration_reports_error_bias_and_the_recent_window():
    shares = [Estimate(EQUITY, 0.9, 0.4, "h")] * 5 + [Estimate(EQUITY, 0.4, 0.4, "h")] * 20
    picks = [Estimate(BEST_OPTION, 1.0, hit, "h") for hit in (1.0, 0.0, 1.0)]
    equity, option = calibration(shares + picks)
    assert (equity.count, equity.error, equity.recent_error) == (25, pytest.approx(0.1), 0.0)
    assert equity.bias == pytest.approx(0.1)  # every miss was too high
    assert (option.count, option.error, option.bias) == (3, pytest.approx(2 / 3), None)


def test_estimates_round_trip_through_their_log_records():
    estimate = Estimate(REQUIRED_EQUITY, 0.25, 1 / 3, "hand-4")
    assert Estimate.from_dict(estimate.to_dict()) == estimate


def test_mistakes_become_drill_keys_that_parse_back():
    records = [
        DecisionRecord("session-8", "hand-3", 4, {}, 2.0, 0.1, "mistake", "fold", None, None),
        DecisionRecord("session-8", "hand-5", 1, {}, 0.1, 0.1, "close", "call", None, None),
    ]
    assert mistake_keys(records) == ["mistake/session-8/hand-3/4"]
    assert parse_mistake_key("mistake/session-8/hand-3/4") == ("session-8", "hand-3", 4)
    with pytest.raises(ValueError):
        parse_mistake_key("preflop/2/100/no_ante/0/-AA")


def test_a_named_option_is_graded_by_what_it_would_lose():
    _, facing = nuts_facing_a_pot_bet()
    raise_to = Action(ActionType.RAISE, 800)
    spot = replace(
        facing,
        reference=[
            OptionValue(None, FOLD, 0.0, 0.0, True),
            OptionValue(None, CALL, 6.0, 0.0, True),
            OptionValue(None, raise_to, 6.2, 0.0, True),
        ],
    )
    assert grade_option(spot, raise_to) == RIGHT
    assert grade_option(spot, CALL) == MIXED  # 0.2bb short of the best: no mistake
    assert grade_option(spot, FOLD) == WRONG
