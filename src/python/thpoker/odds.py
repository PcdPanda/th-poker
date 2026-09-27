"""Preflop hand ranking, 1326-combo ranges, equity against ranges, and board texture.

A range is a tuple of 1326 non-negative weights indexed like `COMBOS`.

Equity preflop uses the class-versus-class table from `thpoker/data/preflop_equity.json.gz`,
which also gives the class-level arrays of the preflop models (`Classes`); `caller_share` is
the model of how much of its equity a caller realizes after the flop.
Postflop evaluates the hand exactly against each range on every sampled complete board. Against
several opponents the per-opponent shares are multiplied, which ignores card removal between
opponents and treats ties as half wins: an approximation, as DESIGN.md Section 2 principle 9
allows.

Board texture tags (DESIGN.md Section 6.4) serve bet sizing and review tags.
"""

from __future__ import annotations

from   bisect                   import bisect_right
from   collections.abc          import Iterable, Sequence
from   dataclasses              import dataclass
from   functools                import cache, lru_cache
import gzip
import json
from   pathlib                  import Path

import numpy as np

from   thpoker.game.cards       import (COMBOS, COMBOS_OF_CLASS, COMBO_CARDS,
                                        COMBO_CLASS, PREFLOP_CLASSES,
                                        combo_index, rank_of, suit_of)
from   thpoker.game.evaluator   import evaluate_combos
from   thpoker.game.rng         import Rng

Range = tuple[float, ...]
Weights = Range | np.ndarray  # a weight per combo index; the analysis passes arrays

FULL_RANGE: Range = (1.0,) * len(COMBOS)


def chen_score(a: int, b: int) -> float:
    """Bill Chen's preflop hand score without its final round-up of half points, which keeps
    the ordering finer (20 for aces, about 0 for the weakest hands)."""
    high, low = max(rank_of(a), rank_of(b)), min(rank_of(a), rank_of(b))
    points = {12: 10.0, 11: 8.0, 10: 7.0, 9: 6.0}.get(high, (high + 2) / 2)
    if high == low:
        return max(points * 2, 5.0)
    score = points + (2 if suit_of(a) == suit_of(b) else 0)
    gap = high - low - 1
    score -= {0: 0, 1: 1, 2: 2, 3: 4}.get(gap, 5)
    if gap <= 1 and high < 10:  # both cards below a queen
        score += 1
    return score


def _percentiles() -> tuple[float, ...]:
    scores = [chen_score(a, b) for a, b in COMBOS]
    ordered = sorted(scores, reverse=True)
    first: dict[float, int] = {}
    last: dict[float, int] = {}
    for index, score in enumerate(ordered):
        first.setdefault(score, index)
        last[score] = index + 1
    return tuple((first[s] + last[s]) / 2 / len(COMBOS) for s in scores)


# By combo index: the midpoint of the hand's tie group in the Chen ordering, as a fraction of
# all combos (near 0 is the best hand). The midpoint keeps a width cut from admitting a whole
# tie group at its edge, so a width of 20% plays about 20% of hands.
PREFLOP_PERCENTILE = _percentiles()


def ranked_range(low: float, high: float) -> Range:
    """Combos whose preflop percentile is in [low, high); (0, 0.2) is roughly the best 20%."""
    return tuple(1.0 if low <= p < high else 0.0 for p in PREFLOP_PERCENTILE)


RUNOUT_SAMPLES = 24
# More runouts when nobody can bet any more: the value is then all there is to estimate.
SETTLED_FLOP_SAMPLES = 96
PREFLOP_TABLE = Path(__file__).resolve().parent / "data" / "preflop_equity.json.gz"


@dataclass(frozen=True)
class Equity:
    """`stderr` is the standard error from sampling; exact results have 0."""

    value: float
    stderr: float
    exact: bool


@cache
def preflop_table() -> tuple[tuple[tuple[float, ...], ...], float]:
    data = json.loads(gzip.decompress(PREFLOP_TABLE.read_bytes()))
    if tuple(data["classes"]) != PREFLOP_CLASSES:
        raise ValueError(f"{PREFLOP_TABLE} lists classes in a different order")
    return tuple(tuple(row) for row in data["equity"]), data["stderr"]


@cache
def _equity_vs_random_by_class() -> tuple[float, ...]:
    table, _ = preflop_table()
    sizes = [COMBO_CLASS.count(c) for c in range(len(PREFLOP_CLASSES))]
    return tuple(sum(e * n for e, n in zip(row, sizes)) / len(COMBOS) for row in table)


def equity_vs_random(hole: tuple[int, int]) -> float:
    """Preflop equity heads-up against any two cards (card removal ignored)."""
    return _equity_vs_random_by_class()[COMBO_CLASS[combo_index(*hole)]]


@cache
def _representative_conflicts() -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[float, ...], ...]]:
    """For each class's first combo in `COMBOS_OF_CLASS`: the 99 combos sharing a card with
    it, and its table equity against each of those combos' classes."""
    table, _ = preflop_table()
    conflicts, equities = [], []
    for klass, name in enumerate(PREFLOP_CLASSES):
        cards = set(COMBOS[COMBOS_OF_CLASS[name][0]])
        blocked = tuple(i for i, combo in enumerate(COMBOS) if cards & set(combo))
        conflicts.append(blocked)
        equities.append(tuple(table[klass][COMBO_CLASS[i]] for i in blocked))
    return tuple(conflicts), tuple(equities)


def representative_equities(weights: Weights, classes: Iterable[int]) -> dict[int, float]:
    """Preflop equity against one range of the first combo of each class in `classes` (indexes
    into `PREFLOP_CLASSES`), with exact card removal: what `hand_equity` gives that combo, much
    faster when many classes are needed. A class whose combo leaves the range empty gets 0."""
    every = _representative_equities(_frozen(weights))
    return {klass: every[klass] for klass in classes}


@lru_cache(maxsize=64)
def _representative_equities(weights: Range) -> tuple[float, ...]:
    table, _ = preflop_table()
    conflicts, conflict_equities = _representative_conflicts()
    mass = [0.0] * len(PREFLOP_CLASSES)
    for index, weight in enumerate(weights):
        mass[COMBO_CLASS[index]] += weight
    total_mass = sum(mass)
    result = []
    for klass in range(len(PREFLOP_CLASSES)):
        blocked = [weights[i] for i in conflicts[klass]]
        total = total_mass - sum(blocked)
        won = sum(e * m for e, m in zip(table[klass], mass))
        won -= sum(w * e for w, e in zip(blocked, conflict_equities[klass]))
        result.append(won / total if total > 0 else 0.0)
    return tuple(result)


def preflop_class_equities(ranges: Sequence[Weights]) -> list[float]:
    """Preflop equity of every class, in `PREFLOP_CLASSES` order, against all of `ranges`
    (multiplied per opponent, card removal ignored): a quick way to rank a whole range.
    Raises `ValueError` if a range is empty."""
    table, _ = preflop_table()
    result = [1.0] * len(table)
    for weights in ranges:
        mass = [0.0] * len(table)
        for index, weight in enumerate(weights):
            mass[COMBO_CLASS[index]] += weight
        total = sum(mass)
        if total <= 0:
            raise ValueError("an opponent range is empty")
        for klass, row in enumerate(table):
            result[klass] *= sum(e * m for e, m in zip(row, mass)) / total
    return result


@cache
def _equities_vs_random_descending() -> tuple[float, ...]:
    by_class = _equity_vs_random_by_class()
    return tuple(sorted((by_class[c] for c in COMBO_CLASS), reverse=True))


def equity_to_reach_top(fraction: float) -> float:
    """The heads-up equity against any two cards that the best `fraction` of hands reach."""
    ordered = _equities_vs_random_descending()
    return ordered[min(len(ordered) - 1, max(0, round(fraction * len(ordered)) - 1))]


@dataclass(frozen=True)
class Classes:
    """Class-level inputs: equity of row class against column class, card-disjoint combo pairs
    per class pair, and combos per class."""

    equity: np.ndarray
    pairs: np.ndarray
    sizes: np.ndarray

    @classmethod
    @cache
    def load(cls) -> Classes:
        """Loaded once; the arrays are shared, so they must not be changed in place."""
        disjoint, membership = combo_matrices()
        pairs = membership @ disjoint @ membership.T
        return cls(np.array(preflop_table()[0]), pairs, membership.sum(axis=1))


def combo_matrices() -> tuple[np.ndarray, np.ndarray]:
    """Whether each two combos share no card (1326 x 1326), and each class's combos as 0/1 rows
    (169 x 1326)."""
    disjoint = ~(COMBO_CARDS[:, None, :, None] == COMBO_CARDS[None, :, None, :]).any(axis=(2, 3))
    membership = np.zeros((len(PREFLOP_CLASSES), len(COMBOS)))
    membership[list(COMBO_CLASS), np.arange(len(COMBOS))] = 1.0
    return disjoint, membership


# Share of its equity a caller realizes after the flop, before the hand-strength adjustment in
# `caller_share`.
# Calibrated against published solver ranges (DESIGN.md Section 14 records the comparison).
# Raising the in-position share from 1.0 to 1.1 widened 6-max button opens toward published
# ranges without cutting big-blind defense; lowering the out-of-position share instead cut
# the defense well below published figures.
_CALLER_REALIZATION = {True: 1.1, False: 0.9}  # by caller in position
_STRENGTH_SLOPE = 1.0
# A cold call with players still to act invites squeezes and overcalls, which the heads-up model
# leaves out; each such player cuts the caller's realization by this share.
_SQUEEZE_COST = 0.06


def caller_share(equity: np.ndarray | float, in_position: bool, behind: int = 0) -> np.ndarray | float:
    """Pot share the caller (the player facing the last raise) wins after the flop: its equity
    scaled by position and by hand strength, since weak hands get pushed off the pot and strong
    ones win extra bets. The raiser wins the rest, so the split stays zero-sum."""
    realization = _CALLER_REALIZATION[in_position] * (1 - _SQUEEZE_COST * behind)
    return np.clip(equity * realization * (1 + _STRENGTH_SLOPE * (equity - 0.5)), 0.0, 1.0)


def _share(hero: int, weights: Weights, values: dict[int, int]) -> tuple[float, float]:
    """The part of a range the hero beats (ties count half), and the range weight still
    possible on this board."""
    total = won = 0.0
    for index, value in values.items():
        weight = weights[index]
        if weight:
            total += weight
            won += weight if hero > value else weight / 2 if hero == value else 0.0
    return (won / total if total else 0.0), total


def _live_combos(dead: set[int], ranges: Sequence[Weights]) -> list[int]:
    """Combos that avoid `dead` and have weight in at least one range."""
    return [
        index
        for index, (a, b) in enumerate(COMBOS)
        if a not in dead and b not in dead and any(weights[index] for weights in ranges)
    ]


def _board_equity(
    hole: tuple[int, int], board: tuple[int, ...], ranges: Sequence[Weights], live: list[int]
) -> tuple[float, float]:
    """Equity on one complete board, and that board's weight: the product of the opponents'
    range weight still possible on it. Weighting boards this way makes the average over boards
    match drawing the opponents' hands and the board together."""
    board_values = _board_values(board)
    values = {i: board_values[i] for i in live if i in board_values}
    hero = board_values[combo_index(*hole)]
    equity = weight = 1.0
    for weights in ranges:
        share, total = _share(hero, weights, values)
        equity *= share
        weight *= total
    return equity, weight


def _preflop_equity(hole: tuple[int, int], ranges: Sequence[Weights]) -> Equity:
    table, table_stderr = preflop_table()
    row = table[COMBO_CLASS[combo_index(*hole)]]
    equity = 1.0
    for weights in ranges:
        total = won = 0.0
        for index, (a, b) in enumerate(COMBOS):
            weight = weights[index]
            if weight and a not in hole and b not in hole:
                total += weight
                won += weight * row[COMBO_CLASS[index]]
        if not total:
            raise ValueError("an opponent range is empty after card removal")
        equity *= won / total
    return Equity(equity, table_stderr, exact=False)


def hand_equity(hole: tuple[int, int], board: tuple[int, ...], ranges: Sequence[Weights], rng: Rng) -> Equity:
    """Equity of `hole` against every range in `ranges` (one per opponent still in the hand).

    On the river the result is exact against one opponent. On the flop and turn, `RUNOUT_SAMPLES` completions of the
    board are drawn from `rng`; seeding `rng` from public information only makes every
    hypothetical hand see the same boards, which keeps bot play and range tracking consistent.
    Raises `ValueError` without opponents, with a board of 1, 2, or more than 5 cards, or when
    card removal leaves an opponent's range empty.
    """
    if not ranges:
        raise ValueError("equity needs at least one opponent range")
    if len(board) not in (0, 3, 4, 5):
        raise ValueError(f"board must have 0, 3, 4, or 5 cards, got {len(board)}")
    if not board:
        return _preflop_equity(hole, ranges)
    live = _live_combos(set(hole) | set(board), ranges)
    if len(board) == 5:
        equity, weight = _board_equity(hole, board, ranges, live)
        if not weight:
            raise ValueError("an opponent range is empty after card removal")
        # Against several opponents the per-opponent product is still an approximation.
        return Equity(equity, 0.0, exact=len(ranges) == 1)
    return _sampled_equity(hole, _completions(board, RUNOUT_SAMPLES, rng), ranges, live)


def _completions(board: tuple[int, ...], count: int, rng: Rng) -> list[tuple[int, ...]]:
    """`count` complete boards drawn from the deck minus the board only, so they do not
    depend on anyone's hole cards."""
    deck = [c for c in range(52) if c not in board]
    boards = []
    for _ in range(count):
        drawn: list[int] = []
        while len(drawn) < 5 - len(board):
            card = deck[rng.randbelow(len(deck))]
            if card not in drawn:
                drawn.append(card)
        boards.append(board + tuple(drawn))
    return boards


def _sampled_equity(
    hole: tuple[int, int], boards: list[tuple[int, ...]], ranges: Sequence[Weights], live: list[int]
) -> Equity:
    """The board-weighted mean over sampled complete boards; a board that uses one of this
    hand's cards is skipped for this hand."""
    samples = [
        _board_equity(hole, full, ranges, live) for full in boards if hole[0] not in full and hole[1] not in full
    ]
    total = sum(w for _, w in samples)
    if not total:
        raise ValueError("an opponent range is empty after card removal")
    mean = sum(e * w for e, w in samples) / total
    stderr = sum((w * (e - mean)) ** 2 for e, w in samples) ** 0.5 / total
    return Equity(mean, stderr, exact=False)


def settled_equity(hole: tuple[int, int], board: tuple[int, ...], ranges: Sequence[Weights], rng: Rng) -> Equity:
    """Equity once nobody can bet any more: exact over every river on the turn (the
    per-opponent product against several opponents), `SETTLED_FLOP_SAMPLES` runouts on the
    flop, and as `hand_equity` preflop and on the river. Raises like `hand_equity`."""
    if len(board) not in (3, 4):
        return hand_equity(hole, board, ranges, rng)
    if not ranges:
        raise ValueError("equity needs at least one opponent range")
    live = _live_combos(set(hole) | set(board), ranges)
    if len(board) == 3:
        return _sampled_equity(hole, _completions(board, SETTLED_FLOP_SAMPLES, rng), ranges, live)
    rivers = [board + (c,) for c in range(52) if c not in board and c not in hole]
    samples = [_board_equity(hole, full, ranges, live) for full in rivers]
    total = sum(w for _, w in samples)
    if not total:
        raise ValueError("an opponent range is empty after card removal")
    return Equity(sum(e * w for e, w in samples) / total, 0.0, exact=len(ranges) == 1)


# Fewer boards than `hand_equity`: this scores every hand of a range, and all hands share the
# boards, so their ranking is steadier than each estimate alone.
RANGE_RUNOUT_SAMPLES = 12
# Below this share of a range's weight on a board, card removal has left the range empty.
_EMPTY = 1e-9
_VALUE_SPAN = 1 << 32  # above every hand value, so (card, value) keys sort by card first


@lru_cache(maxsize=128)
def _board_values(board: tuple[int, ...]) -> dict[int, int]:
    """Hand value of every combo that avoids a complete board. Decisions on one street share
    their sampled boards, so these repeat."""
    live = [i for i, (a, b) in enumerate(COMBOS) if a not in board and b not in board]
    return dict(zip(live, evaluate_combos(board, [COMBOS[i] for i in live])))


def _add_board(
    wanted: list[int],
    values: dict[int, int],
    ranges: list[np.ndarray],
    weighted: dict[int, float],
    totals: dict[int, float],
):
    """Adds one complete board's equity and weight for every hand in `wanted`, at once: each
    hand's share of every range with exact card removal, multiplied across the ranges."""
    on_board = [i for i in wanted if i in values]
    if not on_board:
        return
    live = np.fromiter(values, dtype=np.int64, count=len(values))
    live_values = np.fromiter(values.values(), dtype=np.int64, count=len(values))
    hands = np.array(on_board)
    hand_values = np.array([values[i] for i in on_board], dtype=np.int64)
    # Card removal looks up (card, value) keys: each range combo is listed under both its cards.
    hand_keys = COMBO_CARDS[hands] * _VALUE_SPAN
    equity = np.ones(len(on_board))
    weight = np.ones(len(on_board))
    for weights in ranges:
        mass = weights[live]
        kept = mass > 0
        order = np.argsort(live_values[kept], kind="stable")
        combos, sorted_values, mass = live[kept][order], live_values[kept][order], mass[kept][order]
        running = np.concatenate(([0.0], np.cumsum(mass)))
        below = running[np.searchsorted(sorted_values, hand_values, "left")]
        through = running[np.searchsorted(sorted_values, hand_values, "right")]
        total = np.full(len(on_board), running[-1])
        keys = (COMBO_CARDS[combos] * _VALUE_SPAN + sorted_values[:, None]).ravel()
        by_key = np.argsort(keys, kind="stable")
        keys = keys[by_key]
        card_running = np.concatenate(([0.0], np.cumsum(np.repeat(mass, 2)[by_key])))
        start = card_running[np.searchsorted(keys, hand_keys, "left")]
        end = card_running[np.searchsorted(keys, hand_keys + _VALUE_SPAN, "left")]
        queries = hand_keys + hand_values[:, None]
        below -= (card_running[np.searchsorted(keys, queries, "left")] - start).sum(axis=1)
        through -= (card_running[np.searchsorted(keys, queries, "right")] - start).sum(axis=1)
        total -= (end - start).sum(axis=1)
        # The hand itself was removed once per card; one of those removals was extra.
        own = weights[hands]
        through += own
        total += own
        possible = total > _EMPTY * running[-1]
        equity *= np.where(possible, (below + (through - below) / 2) / np.where(possible, total, 1.0), 0.0)
        weight *= np.where(possible, total, 0.0)
    for combo, e, w in zip(on_board, equity.tolist(), weight.tolist()):
        weighted[combo] += e * w
        totals[combo] += w


def range_equities(board: tuple[int, ...], combos: list[int], ranges: Sequence[Weights], rng: Rng) -> dict[int, float]:
    """Equity against `ranges` of each combo in `combos` that avoids the board.

    Like `hand_equity`, but for many hands at once on the same `RANGE_RUNOUT_SAMPLES` boards,
    drawn from public cards only, so one hand gets the same number whichever other hands are
    scored with it. Card removal between a hand and each range is exact; boards are weighted by
    the range weight they leave possible; several opponents multiply per board. Raises
    `ValueError` without opponents or with a board of fewer than 3 cards.
    """
    if not ranges:
        raise ValueError("equity needs at least one opponent range")
    if len(board) not in (3, 4, 5):
        raise ValueError(f"range equities need a board of 3, 4, or 5 cards, got {len(board)}")
    boards: list[tuple[int, ...]] = [board] if len(board) == 5 else _completions(board, RANGE_RUNOUT_SAMPLES, rng)
    frozen = tuple(_frozen(weights) for weights in ranges)
    return dict(_range_equities(board, tuple(boards), tuple(combos), frozen))


def _frozen(weights: Weights) -> Range:
    """A hashable copy of a range, for the caches."""
    return weights if isinstance(weights, tuple) else tuple(float(w) for w in weights)


@lru_cache(maxsize=128)
def _range_equities(
    board: tuple[int, ...],
    boards: tuple[tuple[int, ...], ...],
    combos: tuple[int, ...],
    ranges: tuple[Range, ...],
) -> dict[int, float]:
    """`range_equities` on the given completions of `board`. Cached: in an EV walk the same
    spot recurs whenever the actions between two decisions did not change the ranges."""
    wanted = [i for i in combos if COMBOS[i][0] not in board and COMBOS[i][1] not in board]
    weighted = dict.fromkeys(wanted, 0.0)
    totals = dict.fromkeys(wanted, 0.0)
    arrays = [np.asarray(weights, dtype=float) for weights in ranges]
    for full in boards:
        _add_board(wanted, _board_values(full), arrays, weighted, totals)
    return {c: weighted[c] / totals[c] for c in wanted if totals[c] > 0}


@dataclass(frozen=True)
class Texture:
    """Tags for a 3-5 card board. `change` describes the last card against the board before it
    ("flush completed", "straight completed", "board paired", "overcard", "brick"), or is None
    on the flop."""

    high: str  # "A-high", "K-high", "broadway", "middle", "low"
    pairing: str  # "unpaired", "paired", "trips"
    suits: str  # "rainbow", "two-tone", "monotone" (flop); later "flush possible" or "no flush"
    connectivity: str  # "connected", "semi-connected", "disconnected"
    wet: bool
    change: str | None


def _straight_windows(ranks: set[int]) -> int:
    """Most board ranks inside any five-rank window (the ace also plays low)."""
    extended = ranks | ({-1} if 12 in ranks else set())
    return max(sum(1 for r in extended if low <= r < low + 5) for low in range(-1, 9))


def _flush_possible(board: tuple[int, ...]) -> bool:
    return max(sum(1 for c in board if suit_of(c) == s) for s in range(4)) >= 3


def texture(board: tuple[int, ...]) -> Texture:
    """Classify a board of 3 to 5 cards. Raises `ValueError` for other sizes."""
    if not 3 <= len(board) <= 5:
        raise ValueError(f"texture needs 3 to 5 board cards, got {len(board)}")
    ranks = [rank_of(c) for c in board]
    top = max(ranks)
    high = {12: "A-high", 11: "K-high"}.get(top, ("low", "middle", "broadway")[bisect_right((5, 8), top)])
    most = max(ranks.count(r) for r in ranks)
    pairing = "unpaired" if most == 1 else "paired" if most == 2 else "trips"
    suit_counts = sorted((sum(1 for c in board if suit_of(c) == s) for s in range(4)), reverse=True)
    if len(board) == 3:
        suits = {1: "rainbow", 2: "two-tone", 3: "monotone"}[suit_counts[0]]
    else:
        suits = "flush possible" if suit_counts[0] >= 3 else "no flush"
    window = _straight_windows(set(ranks))
    connectivity = "connected" if window >= 3 else "semi-connected" if window == 2 else "disconnected"
    draws = suit_counts[0] >= 2 and len(board) == 3 or suit_counts[0] >= 3
    wet = draws and connectivity != "disconnected" or suit_counts[0] >= 3 or connectivity == "connected"
    change = None
    if len(board) > 3:
        before, card = board[:-1], board[-1]
        if suit_counts[0] >= 3 and not _flush_possible(before):
            change = "flush completed"
        elif window >= 3 and _straight_windows({rank_of(c) for c in before}) < 3:
            change = "straight completed"
        elif rank_of(card) in {rank_of(c) for c in before}:
            change = "board paired"
        elif rank_of(card) > max(rank_of(c) for c in before):
            change = "overcard"
        else:
            change = "brick"
    return Texture(high, pairing, suits, connectivity, wet, change)
