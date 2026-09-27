from   dataclasses              import replace

import numpy as np
import pytest

import thpoker.analysis.drills
from   thpoker.analysis.drills  import (DrillResult, MIXED, PAYOUTS, PREFLOP,
                                        PUSHFOLD, RIGHT, WRONG, _final, _posts,
                                        boxes, chart_question, due_keys,
                                        grade_chart, grade_icm, grade_share,
                                        icm_question, new_chart_question,
                                        new_icm_question,
                                        new_threshold_question, option_values,
                                        threshold_question)
from   thpoker.analysis.ev      import icm_equities
from   thpoker.charts           import call_chart, push_chart
from   thpoker.game.rng         import Rng
from   thpoker.odds             import Classes


@pytest.mark.parametrize("kind", [PREFLOP, PUSHFOLD])
def test_a_question_is_rebuilt_from_its_key(kind):
    for n in range(20):
        question = new_chart_question(kind, Rng(n))
        assert chart_question(question.key) == question
        assert sum(question.frequencies) == pytest.approx(1.0)


def test_chart_answers_follow_the_published_heads_up_spots():
    # Heads-up at 8bb the button shoves J8o (up to 13bb) and folds 72o.
    shove = chart_question("pushfold/2/8/no_ante/0/-/J8o")
    assert (grade_chart(shove, 1), grade_chart(shove, 0)) == (RIGHT, WRONG)
    assert grade_chart(chart_question("pushfold/2/8/no_ante/0/-/72o"), 0) == RIGHT
    # At 100bb the button always opens AA.
    assert grade_chart(chart_question("preflop/2/100/no_ante/0/-/AA"), 1) == RIGHT
    assert chart_question("preflop/6/100/no_ante/0/-/AKs").describe().startswith("6 players, 100bb. You are UTG")


def test_a_mixed_chart_action_is_graded_mixed():
    question = replace(chart_question("preflop/6/100/no_ante/0/-/AA"), frequencies=(0.7, 0.3))
    assert (grade_chart(question, 0), grade_chart(question, 1)) == (RIGHT, MIXED)
    assert grade_chart(replace(question, frequencies=(0.9, 0.1)), 1) == WRONG
    assert grade_chart(replace(question, frequencies=(0.45, 0.55)), 0) == MIXED


def test_thresholds_match_the_section_7_4_table():
    assert threshold_question("thresholds/0.5/required_equity").answer == pytest.approx(0.25)
    assert threshold_question("thresholds/0.5/bluff_break_even").answer == pytest.approx(1 / 3)
    assert threshold_question("thresholds/1/defense").answer == pytest.approx(0.5)
    half = threshold_question("thresholds/0.5/required_equity")
    assert [grade_share(half, g) for g in (0.27, 0.30, 0.35)] == [RIGHT, MIXED, WRONG]
    question = new_threshold_question(Rng(4))
    assert threshold_question(question.key) == question


def test_spots_move_through_the_boxes_and_come_back_when_due():
    history = [
        DrillResult("preflop/a", RIGHT, 100),
        DrillResult("preflop/a", RIGHT, 101),  # box 2: due three days later
        DrillResult("preflop/b", WRONG, 101),  # box 0: due at once
        DrillResult("preflop/c", RIGHT, 100),
        DrillResult("preflop/c", MIXED, 101),  # stays in box 1: due a day later
        DrillResult("pushfold/d", WRONG, 90),
    ]
    assert boxes(history)["preflop/a"] == (2, 101)
    assert due_keys(history, 101, PREFLOP) == ["preflop/b"]
    assert due_keys(history, 102, PREFLOP) == ["preflop/b", "preflop/c"]
    assert due_keys(history, 104, PREFLOP) == ["preflop/b", "preflop/c", "preflop/a"]
    assert due_keys(history, 104, PUSHFOLD) == ["pushfold/d"]


def test_an_early_right_answer_does_not_move_a_spot_up():
    # Right on day 100 puts the spot in box 1, due on day 101; right again on day 100 is early.
    history = [DrillResult("preflop/a", RIGHT, 100), DrillResult("preflop/a", RIGHT, 100)]
    assert boxes(history)["preflop/a"] == (1, 100)
    assert boxes([*history, DrillResult("preflop/a", RIGHT, 101)])["preflop/a"] == (2, 101)


def test_hand_written_keys_rebuild_their_spots():
    # Heads-up at 7.5bb with an ante, the big blind (seat 1) facing the button's shove.
    call = chart_question("pushfold/2/7.5/bb_ante/1/0/A2o")
    assert call.options == ("fold", "call") and call.key == "pushfold/2/7.5/bb_ante/1/0/A2o"
    assert call.describe() == "2 players, 7.5bb, big-blind ante. You are BB with A2o; BTN shoves, folded to you."
    assert grade_chart(call, 1) == RIGHT  # any ace calls a 7.5bb shove heads-up


def test_chip_ev_answers_agree_with_the_solved_push_fold_charts():
    # With equal 10bb stacks the spot is the charts' own model, so the chip-EV best response to
    # the other players' charts is the chart itself, up to the hands the solver mixes.
    stacks = (10.0, 10.0, 10.0)
    fold, shove, *_ = option_values(stacks, False, 1, None)  # small blind first in
    pushes = np.array(push_chart(3, 1, False)) >= 10
    assert np.sum((shove > fold) == pushes) >= 163
    fold, call, *_ = option_values((10.0, 10.0, 10.0, 10.0), False, 3, 0)  # big blind vs UTG
    calls = np.array(call_chart(4, 0, 3, False)) >= 10
    assert np.sum((call > fold) == calls) >= 163


def test_on_the_bubble_icm_calls_a_covering_shove_with_fewer_hands_than_chip_ev():
    # Four players, three paid: the 30bb UTG covers the 10bb big blind, so busting costs the
    # big blind more prize money than doubling up gains.
    fold_chips, call_chips, fold_prizes, call_prizes = option_values((30.0, 10.0, 10.0, 10.0), False, 3, 0)
    chip_calls = call_chips > fold_chips
    icm_calls = call_prizes > fold_prizes
    assert not np.any(icm_calls & ~chip_calls)
    assert icm_calls.sum() < chip_calls.sum()


def test_a_short_small_blind_shove_is_priced_from_the_calls_at_its_depth():
    # Three-handed, the 5bb small blind shoves into a 20bb big blind (the 20bb button folded).
    # The big blind calls by the 5bb call chart; the small blind busting third is paid 20%.
    stacks = (20.0, 5.0, 20.0)
    classes = Classes.load()
    calls = np.array(call_chart(3, 1, 2, False)) >= 5
    pairs, equity = classes.pairs, classes.equity
    called = pairs @ calls
    share = called / pairs.sum(axis=1)
    won = (pairs * equity) @ calls / called
    steal = icm_equities([20.0, 6.0, 19.0], stacks, PAYOUTS)[1]
    double = icm_equities([20.0, 10.0, 15.0], stacks, PAYOUTS)[1]
    expected = (1 - share) * steal + share * (won * double + (1 - won) * 0.20)
    _, _, _, shove = option_values(stacks, False, 1, None)
    assert shove == pytest.approx(expected)


@pytest.mark.parametrize("bb_ante", [False, True])
def test_every_outcome_keeps_the_chips_on_the_table(bb_ante):
    stacks = (12.0, 4.5, 20.0, 3.0)
    live, dead = _posts(4, bb_ante)
    outcomes = [_final(stacks, live, dead, None, 3), _final(stacks, live, dead, None, 0)]
    for contest in ((0, 3), (1, 2), (0, 1)):
        outcomes += [_final(stacks, live, dead, contest, winner) for winner in contest]
    for final in outcomes:
        assert sum(final) == pytest.approx(sum(stacks))
        assert min(final) >= 0
    # With an ante the 3bb big blind has 2bb live, so each side stakes 2bb; winning takes both
    # stakes, the small blind, and the ante back.
    stake, ante = (2.0, 1.0) if bb_ante else (3.0, 0.0)
    assert _final(stacks, live, dead, (0, 3), 3)[3] == pytest.approx(2 * stake + 0.5 + ante)


def test_the_answer_with_the_larger_prize_share_is_right():
    question = new_icm_question(Rng(3))
    spot = replace(question, prizes=(0.30, 0.31), threshold=0.02)
    assert grade_icm(spot, 1) == RIGHT
    assert grade_icm(spot, 0) == MIXED  # 1% of the prize pool, under the 2% threshold
    assert grade_icm(replace(spot, threshold=0.005), 0) == WRONG


def test_questions_rebuild_from_their_keys_and_bad_keys_are_refused():
    question = new_icm_question(Rng(11))
    assert icm_question(question.key) == question
    assert "(you)" in question.describe() and "paid" in question.describe()
    with pytest.raises(ValueError, match="not an ICM drill key"):
        icm_question("icm/no_ante/10-10-10/2/-/AKs")  # the big blind cannot be first in
    with pytest.raises(ValueError, match="not an ICM drill key"):
        icm_question("icm/no_ante/10-10-10/0/1/AKs")  # the shover acts after the user
    # One spelling per spot, so a spot is never scheduled twice.
    for alias in ("icm/BBANTE/10-10-10/0/-/AKs", "icm/no_ante/10.0-10-10/0/-/AKs"):
        with pytest.raises(ValueError, match="not an ICM drill key"):
            icm_question(alias)
    for outside in ("inf-10-10", "2-10-10", "25-10-10"):  # too deep, too short, beyond the charts
        with pytest.raises(ValueError, match="stacks outside the drill"):
            icm_question(f"icm/no_ante/{outside}/0/-/AKs")


def test_a_shover_whose_chart_shoves_nothing_is_refused(monkeypatch):
    monkeypatch.setattr(thpoker.analysis.drills, "push_chart", lambda *args: (0.0,) * 169)
    with pytest.raises(ValueError, match="never shoves"):
        icm_question("icm/no_ante/10-10-10/2/0/AKs")
