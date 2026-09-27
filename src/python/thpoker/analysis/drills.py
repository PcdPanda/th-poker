"""Drills with spaced repetition (DESIGN.md Section 7.6): preflop charts, push/fold, the bet-size
thresholds of Section 7.4, and ICM push/fold.

Chart drills favour hands near the edge of a range, where the answer is not obvious, and grade
an answer against the chart's mixing frequencies. Every spot has a key that rebuilds it, and
spots are scheduled in Leitner boxes: a right answer moves the spot up a box (due again later),
a wrong one sends it back to the first box.

ICM drills ask shove or fold first in, or call or fold facing a first-in shove, at a
short-stacked tournament table, with the chip-EV and ICM verdicts side by side so the user
learns when the prize structure changes the answer. The spot follows the push/fold chart model
(`generators/pushfold.py`): the other players shove and call by the chip-EV charts at their
effective stacks, the first caller goes heads-up and the rest fold, and card removal is counted
between hand classes. When the user folds, the rest fold too, so the big blind (or the shover)
takes the pot. Answers are graded against the ICM verdict, the one that decides the prize money.
"""

from   collections.abc          import Sequence
from   dataclasses              import dataclass
import math
from   typing                   import Any

import numpy as np

from   thpoker.analysis.ev      import icm_equities, icm_value_of
from   thpoker.analysis.review  import MISTAKE_BB
from   thpoker.charts           import (STACK_BUCKETS, call_chart, push_chart,
                                        seat_names, strategy)
from   thpoker.game.cards       import COMBOS, COMBOS_OF_CLASS, PREFLOP_CLASSES
from   thpoker.game.rng         import Rng
from   thpoker.odds             import Classes, PREFLOP_PERCENTILE
from   thpoker.storage          import Record

PREFLOP, PUSHFOLD, THRESHOLDS, ICM = "preflop", "pushfold", "thresholds", "icm"
BOX_DAYS = (0, 1, 3, 7, 14, 30)  # days until a spot in each box is due again
RIGHT, MIXED, WRONG = "right", "mixed", "wrong"
# A chart answer is right when it is the chart's main action (or played at least half the time)
# and mixed when the chart plays it at least this often.
MIXED_SHARE = 0.2
# Threshold answers within this distance of the formula are right, within twice it close.
SHARE_TOLERANCE = 0.03
BET_FRACTIONS = (0.25, 0.33, 0.5, 0.66, 0.75, 1.0, 1.25, 1.5, 2.0)
THRESHOLD_KINDS = ("required_equity", "bluff_break_even", "defense")


@dataclass(frozen=True)
class ChartQuestion:
    """A preflop spot: seats in preflop order, `opener` the seat whose open or shove is faced
    (None when first in), and the chart's frequency for each option."""

    key: str
    kind: str
    num_players: int
    stack: float
    bb_ante: bool
    seat: int
    opener: int | None
    hand: str
    options: tuple[str, ...]
    frequencies: tuple[float, ...]

    def describe(self) -> str:
        names = seat_names(self.num_players)
        ante = ", big-blind ante" if self.bb_ante else ""
        action = action_text(names, self.seat, self.opener, "shoves" if self.kind == PUSHFOLD else "opens")
        return f"{self.num_players} players, {self.stack:g}bb{ante}. You are {names[self.seat]} with {self.hand}; {action}."


@dataclass(frozen=True)
class ThresholdQuestion:
    key: str
    bet_fraction: float  # the bet as a share of the pot before it
    kind: str  # one of THRESHOLD_KINDS
    answer: float

    def describe(self) -> str:
        prompt = {
            "required_equity": "the equity a call needs",
            "bluff_break_even": "how often a pure bluff of this size must work",
            "defense": "the share of its range the player facing it must continue with",
        }[self.kind]
        return f"A bet of {self.bet_fraction:.0%} of the pot: {prompt}?"


@dataclass(frozen=True)
class DrillResult(Record):
    key: str
    grade: str
    day: int  # date ordinal


def _edge_weighted_class(rng: Rng, main_share: list[float]) -> int:
    """A hand class drawn by its number of combos, favouring classes whose `main_share` is
    near one half (the edge of the range)."""
    weights = [len(COMBOS_OF_CLASS[name]) * (0.03 + min(p, 1 - p)) for name, p in zip(PREFLOP_CLASSES, main_share)]
    draw = rng.random() * sum(weights)
    for klass, weight in enumerate(weights):
        draw -= weight
        if draw < 0:
            return klass
    return len(weights) - 1


def draw_seats(rng: Rng, fewest: int, most: int) -> tuple[int, bool, int | None, int]:
    """A random drill table: the number of players, whether a big-blind ante is played, the seat
    of a first-in opener (None when the user is first in), and the user's seat after it."""
    players = fewest + rng.randbelow(most - fewest + 1)
    bb_ante = rng.random() < 0.5
    opener = rng.randbelow(players - 1) if rng.random() < 0.5 else None
    if opener is None:
        seat = rng.randbelow(players - 1)
    else:
        seat = opener + 1 + rng.randbelow(players - 1 - opener)
    return players, bb_ante, opener, seat


def action_text(names: Sequence[str], seat: int, opener: int | None, verb: str) -> str:
    """What the user faces: "first to act", "folded to you" or "UTG shoves, folded to you"."""
    if opener is None:
        return "folded to you" if seat else "first to act"
    return f"{names[opener]} {verb}, folded to you"


def _chart(
    kind: str, num_players: int, seat: int, opener: int | None, stack: float, bb_ante: bool
) -> tuple[tuple[Any, ...], tuple[str, ...]]:
    """Per hand class, the chart's shove or call threshold (push/fold) or its action
    frequencies (preflop tables), and the options they belong to."""
    if opener is None:
        if kind == PUSHFOLD:
            return push_chart(num_players, seat, bb_ante), ("fold", "shove")
        return strategy("open", num_players, str(seat), stack, bb_ante), ("fold", "raise")
    if kind == PUSHFOLD:
        return call_chart(num_players, opener, seat, bb_ante), ("fold", "call")
    rows = strategy("respond", num_players, f"{opener}-{seat}", stack, bb_ante)
    return rows, ("fold", "call", "3-bet")


def chart_question(key: str) -> ChartQuestion:
    """Rebuild a chart question from its key. Raises `ValueError` for a malformed key."""
    kind, players, stack, ante, seat, opener, hand = key.split("/")
    num_players, stack_bb, bb_ante = int(players), float(stack), ante == "bb_ante"
    seat_index, opener_index = int(seat), None if opener == "-" else int(opener)
    klass = PREFLOP_CLASSES.index(hand)
    if kind not in (PUSHFOLD, PREFLOP):
        raise ValueError(f"not a chart drill key: {key!r}")
    values, options = _chart(kind, num_players, seat_index, opener_index, stack_bb, bb_ante)
    frequencies = values[klass]
    if kind == PUSHFOLD:
        play = 1.0 if values[klass] >= stack_bb else 0.0
        frequencies = (1 - play, play)
    return ChartQuestion(
        key,
        kind,
        num_players,
        stack_bb,
        bb_ante,
        seat_index,
        opener_index,
        hand,
        options,
        frequencies,
    )


def new_chart_question(kind: str, rng: Rng) -> ChartQuestion:
    """A random spot of `kind` (PREFLOP or PUSHFOLD) with a hand near the edge of its range."""
    num_players, bb_ante, opener, seat = draw_seats(rng, 2, 8)
    ante = "bb_ante" if bb_ante else "no_ante"
    if kind == PUSHFOLD:
        stack = 2 + rng.randbelow(27) / 2  # 2 to 15bb
        thresholds, _ = _chart(kind, num_players, seat, opener, stack, bb_ante)
        # Pure charts: favour hands whose threshold is near this stack.
        main_share = [0.5 if abs(t - stack) <= 2 else float(t >= stack) for t in thresholds]
    else:
        stack = STACK_BUCKETS[rng.randbelow(len(STACK_BUCKETS))]
        rows, _ = _chart(kind, num_players, seat, opener, stack, bb_ante)
        played = [1 - row[0] for row in rows]
        # Charts mostly play a class all the time or never, so the edge is where the hand's
        # strength percentile meets the share of hands the chart plays.
        cutoff = sum(p * len(COMBOS_OF_CLASS[name]) for name, p in zip(PREFLOP_CLASSES, played)) / len(COMBOS)
        main_share = [
            max(
                min(p, 1 - p),
                0.5 * math.exp(-abs(PREFLOP_PERCENTILE[COMBOS_OF_CLASS[name][0]] - cutoff) / 0.04),
            )
            for name, p in zip(PREFLOP_CLASSES, played)
        ]
    hand = PREFLOP_CLASSES[_edge_weighted_class(rng, main_share)]
    key = f"{kind}/{num_players}/{stack:g}/{ante}/{seat}/{'-' if opener is None else opener}/{hand}"
    return chart_question(key)


def grade_chart(question: ChartQuestion, answer: int) -> str:
    """RIGHT for the chart's main action or one it plays at least half the time, MIXED for one
    it plays at least MIXED_SHARE of the time, else WRONG."""
    share = question.frequencies[answer]
    if share >= 0.5 or share >= max(question.frequencies) - 1e-9:
        return RIGHT
    return MIXED if share >= MIXED_SHARE else WRONG


def threshold_question(key: str) -> ThresholdQuestion:
    """Rebuild a threshold question from its key. Raises `ValueError` for a malformed key."""
    kind_name, fraction, kind = key.split("/")
    if kind_name != THRESHOLDS or kind not in THRESHOLD_KINDS:
        raise ValueError(f"not a threshold drill key: {key!r}")
    b = float(fraction)
    answer = {
        "required_equity": b / (1 + 2 * b),
        "bluff_break_even": b / (1 + b),
        "defense": 1 / (1 + b),
    }[kind]
    return ThresholdQuestion(key, b, kind, answer)


def new_threshold_question(rng: Rng) -> ThresholdQuestion:
    fraction = BET_FRACTIONS[rng.randbelow(len(BET_FRACTIONS))]
    kind = THRESHOLD_KINDS[rng.randbelow(len(THRESHOLD_KINDS))]
    return threshold_question(f"{THRESHOLDS}/{fraction:g}/{kind}")


def grade_share(question: ThresholdQuestion, guess: float) -> str:
    error = abs(guess - question.answer)
    return RIGHT if error <= SHARE_TOLERANCE else MIXED if error <= 2 * SHARE_TOLERANCE else WRONG


def boxes(results: list[DrillResult]) -> dict[str, tuple[int, int]]:
    """Each drilled key's Leitner box and the day its interval started, replaying `results` in
    order: a right answer on or after the due day moves up a box (an early one changes
    nothing), mixed restarts the same box, wrong goes back to the first."""
    state: dict[str, tuple[int, int]] = {}
    for result in results:
        if result.key not in state:
            state[result.key] = (1 if result.grade == RIGHT else 0, result.day)
            continue
        box, since = state[result.key]
        if result.grade == WRONG:
            state[result.key] = (0, result.day)
        elif result.grade == MIXED:
            state[result.key] = (box, result.day)
        elif result.day >= since + BOX_DAYS[box]:
            state[result.key] = (min(box + 1, len(BOX_DAYS) - 1), result.day)
    return state


def due_keys(results: list[DrillResult], today: int, kind: str) -> list[str]:
    """Keys of `kind` due today, the longest overdue first."""
    due = []
    for key, (box, day) in boxes(results).items():
        if key.startswith(f"{kind}/") and day + BOX_DAYS[box] <= today:
            due.append((day + BOX_DAYS[box], key))
    return [key for _, key in sorted(due)]


PAYOUTS = (0.5, 0.3, 0.2)
# New questions pick a hand where chip EV and ICM disagree this often, when the spot has one.
DISAGREE_SHARE = 0.6
EDGE_HANDS = 20  # otherwise one of the hands closest to the ICM decision
# Every stack covers the blind and ante, and the user and any shover stay within the charts.
MIN_STACK, CHART_LIMIT = 3.0, 20.0


@dataclass(frozen=True)
class IcmQuestion:
    """A spot with seats in preflop order (the last two are the blinds), stacks in big blinds at
    the start of the hand, and per option ("fold" first) the expected final stack in big blinds
    and the expected share of the prize pool."""

    key: str
    stacks: tuple[float, ...]
    bb_ante: bool
    seat: int
    shover: int | None
    hand: str
    options: tuple[str, str]
    chips: tuple[float, float]
    prizes: tuple[float, float]
    threshold: float  # prize share of MISTAKE_BB at the user's stack

    def describe(self) -> str:
        names = seat_names(len(self.stacks))
        paid = "/".join(f"{p:.0%}" for p in PAYOUTS)
        ante = ", big-blind ante" if self.bb_ante else ""
        stacks = ", ".join(
            f"{names[s]} {stack:g}bb" + (" (you)" if s == self.seat else "") for s, stack in enumerate(self.stacks)
        )
        action = action_text(names, self.seat, self.shover, "shoves")
        return (
            f"{len(self.stacks)} players, {len(PAYOUTS)} paid ({paid}), blinds 0.5/1{ante}. "
            f"Stacks: {stacks}. You are {names[self.seat]} with {self.hand}; {action}."
        )

    def verdicts(self) -> str:
        play = self.options[1]
        chips = self.chips[1] - self.chips[0]
        prizes = self.prizes[1] - self.prizes[0]
        note = "ICM changes the answer." if (chips > 0) != (prizes > 0) else "Both say the same."
        return (
            f"Chip EV: {play} {chips:+.2f}bb against folding. "
            f"ICM: {play} {prizes:+.2%} of the prize pool against folding. {note}"
        )


def _posts(players: int, bb_ante: bool) -> tuple[list[float], list[float]]:
    """Live blinds and dead antes per seat, in big blinds."""
    live, dead = [0.0] * players, [0.0] * players
    live[-2], live[-1] = 0.5, 1.0
    if bb_ante:
        dead[-1] = 1.0
    return live, dead


def _final(
    stacks: tuple[float, ...],
    live: list[float],
    dead: list[float],
    contest: tuple[int, int] | None,
    winner: int,
) -> list[float]:
    """Stacks after the hand: the two players in `contest` each put in the smaller of their live
    stacks, everyone else loses its posts, and `winner` takes the pot."""
    final = [stack - blind - ante for stack, blind, ante in zip(stacks, live, dead)]
    pot = sum(live) + sum(dead)
    if contest is not None:
        stake = min(stacks[p] - dead[p] for p in contest)
        for p in contest:
            final[p] += live[p] - stake
            pot += stake - live[p]
    final[winner] += pot
    return final


def _valued(stacks: tuple[float, ...], seat: int, final: list[float]) -> tuple[float, float]:
    """The user's final stack and prize share."""
    return final[seat], icm_equities(final, stacks, PAYOUTS)[seat]


def option_values(
    stacks: tuple[float, ...], bb_ante: bool, seat: int, shover: int | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per hand class of the user at `seat`: expected final stack (big blinds) and prize share
    after folding and after shoving (or calling `shover`). Raises `ValueError` if the shover's
    chart shoves nothing at its depth."""
    classes = Classes.load()
    pairs, equity = classes.pairs, classes.equity
    players = len(stacks)
    live, dead = _posts(players, bb_ante)
    count = len(PREFLOP_CLASSES)
    if shover is None:
        fold = _valued(stacks, seat, _final(stacks, live, dead, None, players - 1))
        chips, prizes = np.zeros(count), np.zeros(count)
        reach = np.ones(count)
        totals = pairs.sum(axis=1)
        for caller in range(seat + 1, players):
            depth = min(stacks[seat], stacks[caller])
            calls = np.array([t >= depth for t in call_chart(players, seat, caller, bb_ante)], float)
            called = pairs @ calls
            share = called / totals
            won = (pairs * equity) @ calls / np.maximum(called, 1e-12)
            win = _valued(stacks, seat, _final(stacks, live, dead, (seat, caller), seat))
            lose = _valued(stacks, seat, _final(stacks, live, dead, (seat, caller), caller))
            chips += reach * share * (won * win[0] + (1 - won) * lose[0])
            prizes += reach * share * (won * win[1] + (1 - won) * lose[1])
            reach *= 1 - share
        everyone_folds = _valued(stacks, seat, _final(stacks, live, dead, None, seat))
        chips += reach * everyone_folds[0]
        prizes += reach * everyone_folds[1]
    else:
        fold = _valued(stacks, seat, _final(stacks, live, dead, None, shover))
        depth = min(stacks[shover], max(stacks[shover + 1 :]))
        shoves = np.array([t >= depth for t in push_chart(players, shover, bb_ante)], float)
        if not shoves.any():
            raise ValueError(f"the chart never shoves from seat {shover} at {depth:g}bb")
        weights = pairs * shoves[None, :]
        mass = weights.sum(axis=1)
        won = (weights * equity).sum(axis=1) / np.maximum(mass, 1e-12)
        win = _valued(stacks, seat, _final(stacks, live, dead, (shover, seat), seat))
        lose = _valued(stacks, seat, _final(stacks, live, dead, (shover, seat), shover))
        chips = won * win[0] + (1 - won) * lose[0]
        prizes = won * win[1] + (1 - won) * lose[1]
    return np.full(count, fold[0]), chips, np.full(count, fold[1]), prizes


def _key(bb_ante: bool, stacks: tuple[float, ...], seat: int, shover: int | None, hand: str) -> str:
    ante = "bb_ante" if bb_ante else "no_ante"
    shover_text = "-" if shover is None else str(shover)
    return f"{ICM}/{ante}/{'-'.join(f'{s:g}' for s in stacks)}/{seat}/{shover_text}/{hand}"


def icm_question(key: str) -> IcmQuestion:
    """Rebuild a question from its key. Raises `ValueError` for a malformed or non-canonical key
    or a spot outside the drill (a stack under 3bb, or the user or shover deeper than the charts'
    20bb)."""
    try:
        kind, ante, stack_text, seat_text, shover_text, hand = key.split("/")
        stacks = tuple(float(s) for s in stack_text.split("-"))
        seat, shover = int(seat_text), None if shover_text == "-" else int(shover_text)
        klass = PREFLOP_CLASSES.index(hand)
    except ValueError as error:
        raise ValueError(f"not an ICM drill key: {key!r}") from error
    bb_ante = ante == "bb_ante"
    # The big blind cannot be first in, and a shover acts before the user.
    misplaced = seat == len(stacks) - 1 if shover is None else not 0 <= shover < seat
    if (
        kind != ICM
        or key != _key(bb_ante, stacks, seat, shover, hand)
        or not 3 <= len(stacks) <= 8
        or not 0 <= seat < len(stacks)
        or misplaced
    ):
        raise ValueError(f"not an ICM drill key: {key!r}")
    short = [stacks[seat]] + ([] if shover is None else [stacks[shover]])
    if not all(MIN_STACK <= s < math.inf for s in stacks) or max(short) > CHART_LIMIT:
        raise ValueError(f"stacks outside the drill in {key!r}")
    fold_chips, play_chips, fold_prizes, play_prizes = option_values(stacks, bb_ante, seat, shover)
    return IcmQuestion(
        key,
        stacks,
        bb_ante,
        seat,
        shover,
        hand,
        ("fold", "shove" if shover is None else "call"),
        (float(fold_chips[klass]), float(play_chips[klass])),
        (float(fold_prizes[klass]), float(play_prizes[klass])),
        icm_value_of(MISTAKE_BB, stacks, seat, PAYOUTS),
    )


def new_icm_question(rng: Rng) -> IcmQuestion:
    """A random spot: 3 to 6 players, the user (and any shover) 3 to 15bb deep, the others 3 to
    27bb, with a hand where chip EV and ICM disagree or one close to the ICM decision."""
    players, bb_ante, shover, seat = draw_seats(rng, 3, 6)
    stacks = [(6 + rng.randbelow(49)) / 2 for _ in range(players)]
    for short in (seat, shover):
        if short is not None:
            stacks[short] = (6 + rng.randbelow(25)) / 2
    fold_chips, play_chips, fold_prizes, play_prizes = option_values(tuple(stacks), bb_ante, seat, shover)
    chip_gain, prize_gain = play_chips - fold_chips, play_prizes - fold_prizes
    disagree = np.flatnonzero((chip_gain > 0) != (prize_gain > 0))
    if len(disagree) and rng.random() < DISAGREE_SHARE:
        klass = int(disagree[rng.randbelow(len(disagree))])
    else:
        closest = np.argsort(np.abs(prize_gain))[:EDGE_HANDS]
        klass = int(closest[rng.randbelow(len(closest))])
    return icm_question(_key(bb_ante, tuple(stacks), seat, shover, PREFLOP_CLASSES[klass]))


def grade_icm(question: IcmQuestion, answer: int) -> str:
    """RIGHT for the option with the larger prize share, MIXED for the other one when it loses
    less than the mistake threshold, else WRONG."""
    loss = max(question.prizes) - question.prizes[answer]
    if loss <= 0:
        return RIGHT
    return MIXED if loss < question.threshold else WRONG
