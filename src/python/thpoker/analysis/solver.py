"""Heads-up river subgame solver (DESIGN.md Section 5.3, Tier 4): CFR+ over both players' ranges
on a small betting abstraction, the reference opponent for heads-up river review.

The tree starts from any river state, so a decision can be re-solved from where it stands. Each
player bets a third, three quarters, or one and a half of the pot, or all in; facing a bet it
folds, calls, or raises the pot or all in, with at most three bets on the street. Showdowns use a
win/tie/loss matrix with card removal. Values are zero-sum: each player's chips won since the
street began, minus half the pot it began with. Exploitability (the average gain of best
responses against the solution) is reported with every solve.
"""

from __future__ import annotations

from   dataclasses              import dataclass, field
import time

import numpy as np

from   thpoker.game.cards       import COMBOS, COMBO_CARDS
from   thpoker.game.evaluator   import evaluate_combos

BET_SIZES = (0.33, 0.75, 1.5)
RAISE_SIZES = (1.0,)
MAX_BETS = 3
ITERATIONS = 400
TRIM = 1e-3  # hands lighter than this share of a range's heaviest are dropped


@dataclass
class _Node:
    """A decision (`player` 0 or 1) or a terminal: a fold by `folder`, or a showdown."""

    player: int | None
    committed: tuple[int, int]
    folder: int | None = None
    actions: list[int] = field(default_factory=list)  # chips the actor has in after each action
    children: list[_Node] = field(default_factory=list)
    regrets: np.ndarray | None = None
    strategy_sum: np.ndarray | None = None


def _build(
    to_act: int,
    committed: tuple[int, int],
    limits: tuple[int, int],
    bets: int,
    pot: int,
    targets_here: list[int] | None = None,
) -> _Node:
    """The betting tree from a state: `committed` this street, `limits` the most each player can
    have in on the street (its stack), `bets` the bets and raises made so far. `targets_here`
    replaces this node's bet or raise sizes (chips in after the action)."""
    node = _Node(to_act, committed)
    mine, theirs = committed[to_act], committed[1 - to_act]
    cap = min(limits)
    facing = theirs - mine
    targets: list[int] = []
    if facing > 0:
        node.actions.append(-1)  # fold
        node.children.append(_Node(None, committed, folder=to_act))
        call = tuple(theirs if s == to_act else committed[s] for s in range(2))
        node.actions.append(theirs)
        node.children.append(_Node(None, (call[0], call[1])))
        if bets < MAX_BETS and cap > theirs:
            in_pot = pot + committed[0] + committed[1] + facing
            targets = [min(cap, theirs + round(size * in_pot)) for size in RAISE_SIZES] + [cap]
    else:
        node.actions.append(mine)  # check
        if to_act == 1 or committed[0] != committed[1] or bets:
            node.children.append(_Node(None, committed))  # checked through
        else:
            node.children.append(_build(1, committed, limits, bets, pot))
        if bets < MAX_BETS and cap > mine:
            in_pot = pot + committed[0] + committed[1]
            targets = [min(cap, mine + round(size * in_pot)) for size in BET_SIZES] + [cap]
    if targets_here is not None:
        low = theirs if facing > 0 else mine
        targets = [t for t in targets_here if low < t <= cap]
    for target in sorted(set(targets)):
        after = tuple(target if s == to_act else committed[s] for s in range(2))
        node.actions.append(target)
        node.children.append(_build(1 - to_act, (after[0], after[1]), limits, bets + 1, pot))
    return node


_SPAN = 1 << 32  # above every hand value, so (card, value) keys sort by card first


class _Matchup:
    """One player's hands against the other's range on the board. Every hand keeps its value
    for the whole solve, so the searches are done once; then, for any reach of the other
    player, each hand's reach-weighted wins minus losses and the reach it can meet at all (with
    exact card removal) take a few vector operations instead of a dense matrix product."""

    def __init__(self, combos: np.ndarray, values: np.ndarray, others: np.ndarray, other_values: np.ndarray):
        cards, other_cards = COMBO_CARDS[combos], COMBO_CARDS[others]
        self.order = np.argsort(other_values, kind="stable")
        ordered = other_values[self.order]
        self.low = np.searchsorted(ordered, values, "left")
        self.high = np.searchsorted(ordered, values, "right")
        # The other's hands listed under both their cards, sorted by (card, value).
        keys = (other_cards * _SPAN + other_values[:, None]).ravel()
        self.key_order = np.argsort(keys, kind="stable")
        keys = keys[self.key_order]
        card_keys = cards * _SPAN
        self.start = np.searchsorted(keys, card_keys, "left")
        self.end = np.searchsorted(keys, card_keys + _SPAN, "left")
        self.card_low = np.searchsorted(keys, card_keys + values[:, None], "left")
        self.card_high = np.searchsorted(keys, card_keys + values[:, None], "right")
        position = {int(c): i for i, c in enumerate(others)}
        self.same = np.array([position.get(int(c), -1) for c in combos])

    def against(self, reach: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        running = np.concatenate(([0.0], np.cumsum(reach[self.order])))
        by_card = np.concatenate(([0.0], np.cumsum(np.repeat(reach, 2)[self.key_order])))
        start = by_card[self.start]
        below = running[self.low] - (by_card[self.card_low] - start).sum(axis=1)
        through = running[self.high] - (by_card[self.card_high] - start).sum(axis=1)
        total = running[-1] - (by_card[self.end] - start).sum(axis=1)
        # The identical hand was removed once per card, so it comes back once (it ties).
        same = np.where(self.same >= 0, reach[np.maximum(self.same, 0)], 0.0)
        through, total = through + same, total + same
        return below - (total - through), total


@dataclass
class RiverSolution:
    """The solved subgame: per player, its combos (indexes into `COMBOS`) and their weights, the
    tree, and the exploitability in chips."""

    combos: tuple[np.ndarray, np.ndarray]
    weights: tuple[np.ndarray, np.ndarray]
    root: _Node
    pot: int
    exploitability: float
    matchups: tuple[_Matchup, _Matchup]  # each player's hands against the other's range

    def root_values(self, combo: int) -> dict[int, float]:
        """For the root player holding `combo`, its value after each root action (chips in
        after the action, -1 for fold), against the other's range under the solution."""
        player = self.root.player
        assert player is not None
        index = int(np.flatnonzero(self.combos[player] == combo)[0])
        values = {}
        for action, child in zip(self.root.actions, self.root.children):
            reach = [w.copy() for w in self.weights]
            reach[player] = np.zeros_like(reach[player])
            reach[player][index] = 1.0
            own = _values(self, child, reach, average=True)[player][index]
            opponent_mass = self.matchups[player].against(self.weights[1 - player])[1][index]
            values[action] = own / opponent_mass if opponent_mass > 0 else 0.0
        return values

    def strategy(self, node: _Node) -> np.ndarray:
        """The average strategy at `node`: rows are the acting player's combos."""
        assert node.strategy_sum is not None
        total = node.strategy_sum.sum(axis=1, keepdims=True)
        uniform = np.full_like(node.strategy_sum, 1.0 / node.strategy_sum.shape[1])
        return np.where(total > 0, node.strategy_sum / np.where(total > 0, total, 1.0), uniform)


def _current(node: _Node) -> np.ndarray:
    assert node.regrets is not None
    positive = np.maximum(node.regrets, 0.0)
    total = positive.sum(axis=1, keepdims=True)
    return np.where(total > 0, positive / np.where(total > 0, total, 1.0), 1.0 / positive.shape[1])


def _values(
    solution: RiverSolution,
    node: _Node,
    reach: list[np.ndarray],
    average: bool = False,
    update: int | None = None,
    weight: float = 0.0,
    best_response: int | None = None,
) -> list[np.ndarray]:
    """Counterfactual values for both players' combos below `node`, given each player's reach.
    `update` accumulates regrets and strategy for that player; `best_response` lets that player
    pick its best action everywhere against the other's average strategy."""
    half = solution.pot / 2
    if node.player is None:
        (net0, mass0), (net1, mass1) = (solution.matchups[p].against(reach[1 - p]) for p in (0, 1))
        if node.folder is None:
            stake = half + node.committed[0]
            return [stake * net0, stake * net1]
        stake = half + node.committed[node.folder]
        sign = -1.0 if node.folder == 0 else 1.0
        return [sign * stake * mass0, -sign * stake * mass1]
    player = node.player
    strategy = solution.strategy(node) if (average or best_response is not None) else _current(node)
    child_values = []
    for action in range(len(node.children)):
        child_reach = list(reach)
        if best_response != player:
            child_reach[player] = reach[player] * strategy[:, action]
        child_values.append(
            _values(solution, node.children[action], child_reach, average, update, weight, best_response)
        )
    own = np.stack([v[player] for v in child_values], axis=1)
    other = np.sum([v[1 - player] for v in child_values], axis=0)
    if best_response == player:
        mine = own.max(axis=1)
    else:
        mine = (own * strategy).sum(axis=1)
    if update == player:
        assert node.regrets is not None and node.strategy_sum is not None
        node.regrets = np.maximum(node.regrets + own - mine[:, None], 0.0)
        node.strategy_sum += weight * reach[player][:, None] * strategy
    result = [np.zeros(0), np.zeros(0)]
    result[player], result[1 - player] = mine, other
    return result


def _nodes(node: _Node) -> list[_Node]:
    if node.player is None:
        return []
    return [node] + [n for child in node.children for n in _nodes(child)]


def solve_river(
    board: tuple[int, ...],
    pot: int,
    committed: tuple[int, int],
    behind: tuple[int, int],
    to_act: int,
    bets: int,
    ranges: tuple[np.ndarray, np.ndarray],
    iterations: int = ITERATIONS,
    keep: int | None = None,
    budget: float | None = None,
    root_targets: list[int] | None = None,
) -> RiverSolution:
    """Solve a heads-up river from its current state. Player 0 acts first on the street; `pot`
    is what the pot held when the street began, `committed` what each has put in since, and
    `behind` what each has left, and `root_targets` replaces the root's bet or raise sizes (to
    price exactly the options under review). `ranges` are weights over `COMBOS`; hands under `TRIM` of a
    range's heaviest weight are dropped, except `keep` in the range of the player to act (the
    hand being reviewed). With a `budget` in seconds it stops early when the time runs out; the
    exploitability tells how far it got. Raises `ValueError` without five board cards or with an
    empty range."""
    if len(board) != 5:
        raise ValueError(f"the river needs five board cards, got {len(board)}")
    combos, weights = [], []
    for player, weights_all in enumerate(ranges):
        array = np.asarray(weights_all, dtype=float)
        live = [i for i, (a, b) in enumerate(COMBOS) if a not in board and b not in board]
        top = max((array[i] for i in live), default=0.0)
        chosen = [i for i in live if array[i] > TRIM * top or (player == to_act and i == keep)]
        if top <= 0 or not chosen:
            raise ValueError("a range is empty on this board")
        mass = np.maximum(array[chosen], TRIM * TRIM * top)  # the kept hand may weigh nothing
        combos.append(np.array(chosen))
        weights.append(mass / mass.sum())
    values = [np.array(evaluate_combos(board, [COMBOS[i] for i in c]), dtype=np.int64) for c in combos]
    matchups = (
        _Matchup(combos[0], values[0], combos[1], values[1]),
        _Matchup(combos[1], values[1], combos[0], values[0]),
    )
    limits = (committed[0] + behind[0], committed[1] + behind[1])
    # Chips one player has in beyond what the other can ever match come back to it.
    committed = (min(committed[0], min(limits)), min(committed[1], min(limits)))
    root = _build(to_act, committed, limits, bets, pot, root_targets)
    for node in _nodes(root):
        assert node.player is not None
        shape = (len(combos[node.player]), len(node.children))
        node.regrets = np.zeros(shape)
        node.strategy_sum = np.zeros(shape)
    solution = RiverSolution((combos[0], combos[1]), (weights[0], weights[1]), root, pot, 0.0, matchups)
    started = time.perf_counter()
    for t in range(1, iterations + 1):
        _values(solution, root, list(solution.weights), update=t % 2, weight=float(t))
        if budget is not None and t % 2 == 0 and time.perf_counter() - started > budget:
            break
    best = [_values(solution, root, list(solution.weights), best_response=p)[p] @ solution.weights[p] for p in (0, 1)]
    # Each best response wins at least the game value for its side, and the two values cancel,
    # so their sum is what the solution concedes; it is zero at an equilibrium.
    joint = solution.weights[0] @ matchups[0].against(solution.weights[1])[1]
    solution.exploitability = float(best[0] + best[1]) / 2 / joint
    return solution
