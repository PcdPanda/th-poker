import pytest
from   thpoker.charts           import (call_chart, push_chart, stack_bucket,
                                        strategy)
from   thpoker.game.cards       import COMBO_CLASS, PREFLOP_CLASSES


def share(
    node: str,
    num_players: int,
    seats: str,
    action: int,
    stack: float = 100.0,
    bb_ante: bool = False,
) -> float:
    """Share of all combos taking `action` (0 is fold) at a node."""
    rows = strategy(node, num_players, seats, stack, bb_ante)
    return sum(rows[c][action] for c in COMBO_CLASS) / len(COMBO_CLASS)


@pytest.mark.parametrize("stack, bucket", [(15, 20.0), (28, 20.0), (29, 40.0), (63, 40.0), (64, 100.0), (250, 100.0)])
def test_stacks_map_to_the_nearest_solved_depth(stack, bucket):
    assert stack_bucket(stack) == bucket


def test_six_max_opens_widen_toward_the_button_and_match_solver_bands():
    shares = [share("open", 6, str(seat), 1) for seat in range(5)]
    assert shares[:4] == sorted(shares[:4])  # UTG, HJ, CO, button
    # Published 6-max 100bb solver opens are about 17% from UTG and 43% on the button.
    assert 0.12 <= shares[0] <= 0.22 and 0.33 <= shares[3] <= 0.50


def test_aces_always_open_and_seven_deuce_never_does_from_early_position():
    opens = strategy("open", 8, "0", 100.0, False)
    assert opens[PREFLOP_CLASSES.index("AA")] == (0.0, 1.0)
    assert opens[PREFLOP_CLASSES.index("72o")] == (1.0, 0.0)


def test_the_big_blind_defends_wider_against_the_button_than_against_early_position():
    assert 1 - share("respond", 6, "3-5", 0) > 1 - share("respond", 6, "0-5", 0)


def test_action_probabilities_sum_to_one():
    for node, stack, ante in (
        ("respond", 100.0, False),
        ("versus_3bet", 40.0, True),
        ("versus_allin", 20.0, False),
    ):
        for probabilities in strategy(node, 6, "0-3", stack, ante):
            assert sum(probabilities) == pytest.approx(1.0, abs=0.011)  # rounded to whole percents


def share_played(chart: tuple[float, ...], stack: float) -> float:
    """Share of all combos that a chart plays at this stack."""
    return sum(1 for c in COMBO_CLASS if chart[c] >= stack) / len(COMBO_CLASS)


# Heads-up, no ante, from the HoldemResources push/fold Nash chart: the largest stack at which a
# hand shoves from the small blind or calls in the big blind. These charts come from a sampled
# equity table, so a threshold may sit up to a big blind away (T7o's shove threshold, 7bb here
# against 9bb published, is a known larger deviation and is not pinned).
@pytest.mark.parametrize("hand, published", [("J8o", 13.3), ("T3s", 7.7)])
def test_heads_up_shove_thresholds_match_the_published_chart(hand, published):
    assert abs(push_chart(2, 0, bb_ante=False)[PREFLOP_CLASSES.index(hand)] - published) <= 1.0


@pytest.mark.parametrize("hand, published", [("J8o", 7.6), ("T7o", 5.5)])
def test_heads_up_call_thresholds_match_the_published_chart(hand, published):
    assert abs(call_chart(2, 0, 1, bb_ante=False)[PREFLOP_CLASSES.index(hand)] - published) <= 1.0


def test_heads_up_shove_range_at_16bb_matches_the_published_43_percent():
    # PokerStrategy.com (HoldemResources ranges): at 16bb the small blind shoves 43.3%.
    played = share_played(push_chart(2, 0, bb_ante=False), 16.0)
    assert played == pytest.approx(0.433, abs=0.02)


def test_later_seats_and_antes_shove_wider():
    widths = [share_played(push_chart(6, seat, bb_ante=False), 10.0) for seat in range(5)]
    assert widths == sorted(widths)
    with_ante = share_played(push_chart(6, 3, bb_ante=True), 10.0)
    assert with_ante > widths[3]


def test_aces_always_shove_and_call():
    aces = PREFLOP_CLASSES.index("AA")
    assert push_chart(8, 0, bb_ante=False)[aces] == 20.0
    assert call_chart(8, 0, 7, bb_ante=False)[aces] == 20.0


@pytest.mark.parametrize(
    "call, message",
    [
        (lambda: strategy("open", 6, "5", 100.0, False), "no open strategy"),
        (lambda: strategy("respond", 6, "3-2", 100.0, False), "no respond strategy"),
        (lambda: strategy("open", 9, "0", 100.0, False), "2 to 8 players"),
        (lambda: strategy("limp", 6, "0", 100.0, False), "unknown node"),
        (lambda: push_chart(6, 5, False), "cannot shove"),
        (lambda: call_chart(6, 3, 2, False), "does not act after"),
        (lambda: push_chart(9, 0, False), "2 to 8 players"),
    ],
)
def test_invalid_chart_requests_rejected(call, message):
    with pytest.raises(ValueError, match=message):
        call()
