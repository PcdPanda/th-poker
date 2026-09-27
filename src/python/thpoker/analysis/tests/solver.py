import time

import numpy as np
import pytest

from   thpoker.analysis.solver  import _Matchup, solve_river
from   thpoker.analysis.tests.tracking \
                                import weights
from   thpoker.game.cards       import COMBOS, combo_index, parse_cards
from   thpoker.game.evaluator   import evaluate
from   thpoker.game.rng         import Rng

BOARD = tuple(parse_cards("KsKd7h4c2s"))
TRIPS = ["KhAs", "KhAd", "KcAs", "KcAd"]  # trip kings: beat the queens
AIR = ["6c5d", "6d5c", "6h5s", "6s5h"]  # six high: lose to the queens
QUEENS = ["QhQc", "QdQs", "QcQs"]  # kings and queens: beat only air


def test_the_clairvoyance_game_bluffs_and_calls_at_the_indifference_frequencies():
    # Pot 100 and 100 behind: an all-in is a pot-size bet (b = 1). The bettor value-bets all its
    # trips and bluffs b / (1 + b) as much air, half of it; the caller calls 1 / (1 + b), half.
    solution = solve_river(BOARD, 100, (0, 0), (100, 100), 0, 0, (weights(*TRIPS, *AIR), weights(*QUEENS)), 1500)
    root = solution.root
    shove = root.actions.index(100)
    strategy = solution.strategy(root)
    rows = {int(c): i for i, c in enumerate(solution.combos[0])}
    trips = [strategy[rows[combo_index(*parse_cards(h))], shove] for h in TRIPS]
    air = [strategy[rows[combo_index(*parse_cards(h))], shove] for h in AIR]
    assert min(trips) > 0.95
    assert np.mean(air) == pytest.approx(0.5, abs=0.08)
    calls = solution.strategy(root.children[shove])[:, 1]  # fold, call
    assert np.mean(calls) == pytest.approx(0.5, abs=0.08)
    assert solution.exploitability < 0.01 * 100


def test_root_values_price_each_action_for_one_hand():
    # Facing an all-in from a range of equally many trips and bluffs, queens win half the time.
    # Zero-sum values (chips won since the street began, minus half the 100 pot): folding loses
    # the half pot, calling wins or loses 150 evenly.
    solution = solve_river(BOARD, 100, (100, 0), (0, 100), 1, 1, (weights(*TRIPS, *AIR), weights(*QUEENS)), 50)
    values = solution.root_values(combo_index(*parse_cards("QhQc")))
    assert values == {-1: pytest.approx(-50.0), 100: pytest.approx(0.0)}


def test_the_matchup_lookups_equal_a_hand_by_hand_count():
    rng = Rng(3)
    live = [i for i, (a, b) in enumerate(COMBOS) if a not in BOARD and b not in BOARD]
    mine = np.array(live[:7])
    theirs = np.array(live[::5])
    value = {i: evaluate([*COMBOS[i], *BOARD]) for i in set(mine) | set(theirs)}
    reach = np.array([rng.random() for _ in theirs])
    matchup = _Matchup(mine, np.array([value[i] for i in mine]), theirs, np.array([value[i] for i in theirs]))
    net, mass = matchup.against(reach)
    for row, i in enumerate(mine):
        met = [(r, value[j]) for j, r in zip(theirs, reach) if not set(COMBOS[i]) & set(COMBOS[j])]
        expected = sum(r * ((value[i] > v) - (value[i] < v)) for r, v in met)
        assert net[row] == pytest.approx(expected) and mass[row] == pytest.approx(sum(r for r, _ in met))


def test_root_targets_replace_the_root_sizes_within_what_can_be_called():
    # Pot 100 and 100 behind each: a bet to 150 is more than the caller has, so it is left out.
    ranges = (weights(*TRIPS, *AIR), weights(*QUEENS))
    solution = solve_river(BOARD, 100, (0, 0), (100, 100), 0, 0, ranges, 10, root_targets=[30, 100, 150])
    assert solution.root.actions == [0, 30, 100]


def test_a_bet_the_caller_cannot_cover_plays_as_an_all_in_for_less():
    # A bet to 300 into a caller with 100 left: calling puts in 100, and the rest is not at stake.
    ranges = (weights(*TRIPS, *AIR), weights(*QUEENS))
    solution = solve_river(BOARD, 100, (300, 0), (0, 100), 1, 1, ranges, 50)
    assert solution.root.actions == [-1, 100]
    values = solution.root_values(combo_index(*parse_cards("QhQc")))
    assert values == {-1: pytest.approx(-50.0), 100: pytest.approx(0.0)}


def test_the_time_budget_stops_the_solve_early():
    ranges = (weights(*TRIPS, *AIR), weights(*QUEENS))
    started = time.perf_counter()
    solve_river(BOARD, 100, (0, 0), (1_000, 1_000), 0, 0, ranges, 10**7, budget=0.2)
    assert time.perf_counter() - started < 5
