"""Generate `thpoker/data/pushfold.json.gz`: chip-EV push/fold charts for first-in shoves.

Run `bin/generate_data.py pushfold`.

Model: the first player to enter shoves all-in or folds; each player behind, in order, calls or
folds against that shove. A caller is assumed to go heads-up (later players fold), which ignores
overcalls. All stacks are equal, and a big-blind ante counts as part of the big blind's stake.
Strategies over the 169 classes are found by regret matching+, with card removal through the
number of card-disjoint combo pairs between two classes. Each chart stores, per hand class, the
largest stack (in big blinds) at which the hand shoves or calls, like published Nash charts; the
best-response gain measures how far the averaged strategies are from equilibrium.
"""

from   concurrent.futures       import ProcessPoolExecutor
from   dataclasses              import dataclass
from   typing                   import Any

import numpy as np

from   thpoker.game.cards       import PREFLOP_CLASSES
from   thpoker.odds             import Classes

STACKS = tuple(s / 2 for s in range(2, 41))  # 1bb to 20bb in half big blinds
_ITERATIONS = 400


@dataclass(frozen=True)
class _Spot:
    classes: Classes
    posts: np.ndarray  # chips posted per seat, in preflop order
    pusher: int
    stack: float

    @property
    def callers(self) -> range:
        return range(self.pusher + 1, len(self.posts))

    def pot(self, caller: int) -> float:
        """Pot when `caller` calls: both stacks plus everything the other seats posted."""
        return 2 * self.stack + self.posts.sum() - self.posts[self.pusher] - self.posts[caller]

    def push_values(self, calls: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """Chip result of shoving and of folding, per pusher class."""
        pairs, equity = self.classes.pairs, self.classes.equity
        totals = pairs.sum(axis=1)
        reach = np.ones(len(PREFLOP_CLASSES))
        shove = np.zeros(len(PREFLOP_CLASSES))
        for caller, call in zip(self.callers, calls):
            showdown = ((pairs * (self.pot(caller) * equity - self.stack)) @ call) / totals
            shove += reach * showdown
            reach *= 1 - (pairs @ call) / totals
        shove += reach * (self.posts.sum() - self.posts[self.pusher])
        return shove, np.full(len(PREFLOP_CLASSES), -self.posts[self.pusher])

    def call_values(self, caller: int, push: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Chip result of calling and of folding against the shove, per caller class."""
        weights = self.classes.pairs * push[None, :]  # rows: caller class, columns: pusher class
        mass = weights.sum(axis=1)
        value = (weights * (self.pot(caller) * (1 - self.classes.equity.T) - self.stack)).sum(axis=1)
        fold = np.full(len(PREFLOP_CLASSES), -self.posts[caller])
        return np.where(mass > 0, value / np.maximum(mass, 1e-12), fold), fold


def _current(regret: np.ndarray) -> np.ndarray:
    """Play probability of the first action (shove or call) from positive regrets."""
    total = regret.sum(axis=0)
    return np.where(total > 0, regret[0] / np.maximum(total, 1e-300), 0.5)


def _accumulate(regret: np.ndarray, play: np.ndarray, first: np.ndarray, second: np.ndarray):
    expected = play * first + (1 - play) * second
    regret[0] = np.maximum(0.0, regret[0] + first - expected)
    regret[1] = np.maximum(0.0, regret[1] + second - expected)


def solve(
    classes: Classes, num_players: int, pusher: int, stack: float, bb_ante: bool
) -> tuple[np.ndarray, list[np.ndarray], float]:
    """Averaged push strategy of `pusher` (seats in preflop order, blinds last), call strategies of
    each player behind, and the largest best-response gain in big blinds per hand."""
    posts = np.zeros(num_players)
    posts[-2] += 0.5
    posts[-1] += 2.0 if bb_ante else 1.0
    spot = _Spot(classes, posts, pusher, stack)
    # Regret matching+ with alternating updates; the average strategy weights iteration t by t.
    push_regret = np.zeros((2, len(PREFLOP_CLASSES)))  # rows: shove, fold
    call_regrets = [np.zeros((2, len(PREFLOP_CLASSES))) for _ in spot.callers]
    push_total = np.zeros(len(PREFLOP_CLASSES))
    call_totals = [np.zeros(len(PREFLOP_CLASSES)) for _ in spot.callers]
    for iteration in range(1, _ITERATIONS + 1):
        push = _current(push_regret)
        calls = [_current(r) for r in call_regrets]
        shove, fold = spot.push_values(calls)
        _accumulate(push_regret, push, shove, fold)
        push = _current(push_regret)
        for caller, regret, total in zip(spot.callers, call_regrets, call_totals):
            call, fold_caller = spot.call_values(caller, push)
            _accumulate(regret, _current(regret), call, fold_caller)
            total += iteration * _current(regret)
        push_total += iteration * push
    weight = _ITERATIONS * (_ITERATIONS + 1) / 2
    push = push_total / weight
    calls = [total / weight for total in call_totals]
    sizes = classes.sizes
    shove, fold = spot.push_values(calls)
    gains = [(sizes * (np.maximum(shove, fold) - (push * shove + (1 - push) * fold))).sum()]
    for caller, strategy in zip(spot.callers, calls):
        call, fold_caller = spot.call_values(caller, push)
        mixed = strategy * call + (1 - strategy) * fold_caller
        gains.append((sizes * (np.maximum(call, fold_caller) - mixed)).sum())
    return push, calls, float(max(gains) / sizes.sum())


def _thresholds(strategies: list[np.ndarray]) -> list[float]:
    """Per class, the largest stack whose strategy plays the hand at least half the time."""
    played = np.array([s >= 0.5 for s in strategies])  # rows: stacks, columns: classes
    return [float(STACKS[np.flatnonzero(c).max()]) if c.any() else 0.0 for c in played.T]


def _solve_task(task: tuple[int, int, float, bool]) -> tuple[np.ndarray, list[np.ndarray], float]:
    return solve(Classes.load(), *task)


def generate(workers: int) -> dict[str, Any]:
    """All charts: 2 to 8 players, without and with a big-blind ante, every pusher position."""
    tasks = [
        (num_players, pusher, stack, bb_ante)
        for num_players in range(2, 9)
        for bb_ante in (False, True)
        for pusher in range(num_players - 1)
        for stack in STACKS
    ]
    with ProcessPoolExecutor(workers) as pool:
        results = dict(zip(tasks, pool.map(_solve_task, tasks, chunksize=4)))
    tables: dict[str, dict[str, dict[str, dict[str, list[float]]]]] = {}
    for num_players in range(2, 9):
        by_ante: dict[str, dict[str, dict[str, list[float]]]] = {}
        for bb_ante in (False, True):
            pushes: dict[str, list[float]] = {}
            calls: dict[str, list[float]] = {}
            for pusher in range(num_players - 1):
                solved = [results[(num_players, pusher, s, bb_ante)] for s in STACKS]
                pushes[str(pusher)] = _thresholds([push for push, _, _ in solved])
                for index, caller in enumerate(range(pusher + 1, num_players)):
                    calls[f"{pusher}-{caller}"] = _thresholds([c[index] for _, c, _ in solved])
            by_ante["bb_ante" if bb_ante else "no_ante"] = {"push": pushes, "call": calls}
        tables[str(num_players)] = by_ante
    return {
        "classes": list(PREFLOP_CLASSES),
        "stacks": list(STACKS),
        "best_response_gain_bb": round(max(gain for _, _, gain in results.values()), 5),
        "tables": tables,
    }
