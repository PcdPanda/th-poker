"""Generate `thpoker/data/preflop_ranges.json.gz`: preflop strategies for 20bb-and-deeper stacks.

Run `bin/generate_data.py preflop_ranges`.

Model, per table size (2-8 players), stack (20, 40, 100 big blinds) and ante (none or a big-blind
ante): the first player in opens to 2.5bb or folds. Each player behind, in order, folds, calls,
or 3-bets (to 7.5bb in position, 10bb out of position); the first to call or 3-bet goes heads-up
with the opener. Facing the 3-bet the opener folds, calls, or moves all-in; facing that all-in
the 3-bettor calls or folds. All-ins are settled by equity. A flop splits the pot by realized
equity: the player who called the last raise realizes its equity scaled by position (1.1 in
position, 0.9 out of position), by hand strength, and by 6% less per player still to act behind
a cold call (squeeze and overcall risk the heads-up model leaves out); the raiser gets the rest. Limps, squeezes, cold 4-bets and
multiway pots are left out. Strategies over the 169 classes come from CFR+ (regret matching+
with linear averaging, reach-weighted where a decision follows the player's own earlier one).
"""

from __future__ import annotations

from   collections.abc          import Callable
from   concurrent.futures       import ProcessPoolExecutor
from   dataclasses              import dataclass
from   typing                   import Any

import numpy as np

from   thpoker.charts           import STACK_BUCKETS
from   thpoker.game.cards       import PREFLOP_CLASSES
from   thpoker.odds             import Classes, caller_share

OPEN = 2.5
_ITERATIONS = 1500
_CLASSES = len(PREFLOP_CLASSES)


def _positions(num_players: int) -> np.ndarray:
    """Order of acting after the flop per preflop seat: small blind 0, big blind 1, then the
    rest (preflop seats run from first to act to the big blind). Heads-up the small blind is the
    button and acts last."""
    if num_players == 2:
        return np.array([1, 0])
    return (np.arange(num_players) - (num_players - 2)) % num_players


@dataclass
class _Pair:
    """Opener `p` against responder `q`: sizes and chip results for every class pair (rows:
    opener class, columns: responder class)."""

    three_bet: float
    dead: float
    opener_call: np.ndarray  # responder called the open
    opener_call3: np.ndarray  # opener called the 3-bet
    opener_allin: np.ndarray  # all-in called
    responder_call: np.ndarray
    responder_call3: np.ndarray
    responder_allin: np.ndarray
    ante_q: float


def _pair(classes: Classes, blinds: np.ndarray, ante: float, p: int, q: int, stack: float) -> _Pair:
    """The big blind's ante is dead money in every pot; `ante_q` is what the responder loses
    on top of its bets when it is the big blind (an opener is never the big blind)."""
    equity = classes.equity
    opener_ip = _positions(len(blinds))[p] > _positions(len(blinds))[q]
    three_bet = min(stack, 3 * OPEN if not opener_ip else 4 * OPEN)
    dead = blinds.sum() - blinds[p] - blinds[q] + ante
    ante_q = ante if q == len(blinds) - 1 else 0.0
    pot = 2 * OPEN + dead
    # Single-raised pot: the responder calls the opener's raise. 3-bet pot: the opener calls.
    share = 1 - caller_share(1 - equity, not opener_ip, behind=len(blinds) - 1 - q)
    pot3 = 2 * three_bet + dead
    share3 = caller_share(equity, opener_ip)
    pot_allin = 2 * stack + dead
    return _Pair(
        three_bet=three_bet,
        dead=dead,
        opener_call=share * pot - OPEN,
        opener_call3=share3 * pot3 - three_bet,
        opener_allin=equity * pot_allin - stack,
        responder_call=(1 - share) * pot - OPEN - ante_q,
        responder_call3=(1 - share3) * pot3 - three_bet - ante_q,
        responder_allin=(1 - equity) * pot_allin - stack - ante_q,
        ante_q=ante_q,
    )


class _Regrets:
    """Regret matching+ over one decision with several actions, for every class at once, and
    the linear, reach-weighted average strategy."""

    def __init__(self, actions: int):
        self.regret = np.zeros((actions, _CLASSES))
        self.total = np.zeros((actions, _CLASSES))

    def current(self) -> np.ndarray:
        positive = self.regret.sum(axis=0)
        uniform = np.full_like(self.regret, 1 / len(self.regret))
        return np.where(positive > 0, self.regret / np.maximum(positive, 1e-300), uniform)

    def update(self, values: np.ndarray, own_reach: np.ndarray | float, iteration: int):
        strategy = self.current()
        expected = (strategy * values).sum(axis=0)
        self.regret = np.maximum(0.0, self.regret + values - expected)
        self.total += iteration * own_reach * strategy

    def average(self) -> np.ndarray:
        mass = self.total.sum(axis=0)
        return np.where(mass > 0, self.total / np.maximum(mass, 1e-300), 1 / len(self.total))


@dataclass(frozen=True)
class _Strategies:
    """Current or average strategies, as action-by-class arrays per decision."""

    opening: dict[int, np.ndarray]  # open probability per opener class
    responses: dict[tuple[int, int], np.ndarray]  # fold, call, 3-bet
    replies: dict[tuple[int, int], np.ndarray]  # opener facing the 3-bet: fold, call, all-in
    calls: dict[tuple[int, int], np.ndarray]  # 3-bettor facing the all-in: fold, call

    @classmethod
    def of(
        cls,
        pick: Callable[[_Regrets], np.ndarray],
        opening: dict[int, _Regrets],
        responses: dict[tuple[int, int], _Regrets],
        replies: dict[tuple[int, int], _Regrets],
        calls: dict[tuple[int, int], _Regrets],
    ) -> _Strategies:
        """`pick` (current or average strategy) of each decision's regrets."""
        return cls(
            {p: pick(r)[0] for p, r in opening.items()},
            {k: pick(r) for k, r in responses.items()},
            {k: pick(r) for k, r in replies.items()},
            {k: pick(r) for k, r in calls.items()},
        )


def _after_three_bet(m: _Pair, reply: np.ndarray, allin: np.ndarray) -> np.ndarray:
    """Opener's chip result once the responder 3-bets (rows: opener class, columns: responder)."""
    shove = allin[0][None, :] * (m.three_bet + m.dead) + allin[1][None, :] * m.opener_allin
    return reply[0][:, None] * -OPEN + reply[1][:, None] * m.opener_call3 + reply[2][:, None] * shove


def _open_values(
    classes: Classes,
    posts: np.ndarray,
    p: int,
    matchups: dict[tuple[int, int], _Pair],
    plays: _Strategies,
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    # `posts` holds everything each seat has put in before acting: blinds plus the big blind's ante.
    """Opener's chip result of opening, per class, and the chance that every responder before
    each responder folded (per opener class)."""
    pairs, totals = classes.pairs, classes.pairs.sum(axis=1)
    reach = np.ones(_CLASSES)
    reached: dict[int, np.ndarray] = {}
    value = np.zeros(_CLASSES)
    for q in range(p + 1, len(posts)):
        m = matchups[(p, q)]
        fold, call, three = plays.responses[(p, q)]
        after_three = _after_three_bet(m, plays.replies[(p, q)], plays.calls[(p, q)])
        value += reach * ((pairs * m.opener_call) @ call + (pairs * after_three) @ three) / totals
        reached[q] = reach.copy()
        reach = reach * (pairs @ fold) / totals
    return value + reach * (posts.sum() - posts[p]), reached


def solve(classes: Classes, num_players: int, stack: float, bb_ante: bool) -> tuple[dict[str, dict[str, Any]], float]:
    """Average strategies for one configuration, and the largest best-response gain (big blinds
    per hand) at any first-in decision."""
    blinds = np.zeros(num_players)
    blinds[-2:] = 0.5, 1.0
    ante = 1.0 if bb_ante else 0.0
    posts = blinds.copy()
    posts[-1] += ante
    pairs = classes.pairs
    openers = range(num_players - 1)
    matchups = {(p, q): _pair(classes, blinds, ante, p, q, stack) for p in openers for q in range(p + 1, num_players)}
    open_regret = {p: _Regrets(2) for p in openers}  # open, fold
    respond = {k: _Regrets(3) for k in matchups}
    versus_three_bet = {k: _Regrets(3) for k in matchups}
    versus_allin = {k: _Regrets(2) for k in matchups}
    for iteration in range(1, _ITERATIONS + 1):
        plays = _Strategies.of(_Regrets.current, open_regret, respond, versus_three_bet, versus_allin)
        for p in openers:
            value, reached = _open_values(classes, posts, p, matchups, plays)
            open_regret[p].update(np.array([value, np.full(_CLASSES, -posts[p])]), 1.0, iteration)
            for q, reach in reached.items():
                k, m = (p, q), matchups[(p, q)]
                three, reply, allin = plays.responses[k][2], plays.replies[k], plays.calls[k]
                # Opener facing the 3-bet; opponent reach is the 3-bet and the folds before it.
                three_bettors = pairs * three[None, :]
                shove = allin[0][None, :] * (m.three_bet + m.dead) + allin[1][None, :] * m.opener_allin
                reply_values = reach * np.array(
                    [
                        (three_bettors * -OPEN).sum(axis=1),
                        (three_bettors * m.opener_call3).sum(axis=1),
                        (three_bettors * shove).sum(axis=1),
                    ]
                )
                versus_three_bet[k].update(reply_values, plays.opening[p], iteration)
                # Responder facing the open (rows: responder class, columns: opener class, using
                # the symmetry of the pair counts); opponent reach is the open and earlier folds.
                openers_weight = pairs * (plays.opening[p] * reach)[None, :]
                facing_shove = allin[0][:, None] * -(m.three_bet + m.ante_q) + allin[1][:, None] * m.responder_allin.T
                three_bet_result = (
                    reply[0][None, :] * (OPEN + m.dead - m.ante_q)
                    + reply[1][None, :] * m.responder_call3.T
                    + reply[2][None, :] * facing_shove
                )
                respond_values = np.array(
                    [
                        openers_weight.sum(axis=1) * -posts[q],
                        (openers_weight * m.responder_call.T).sum(axis=1),
                        (openers_weight * three_bet_result).sum(axis=1),
                    ]
                )
                respond[k].update(respond_values, 1.0, iteration)
                # 3-bettor facing the all-in; opponent reach adds the opener's all-in.
                shovers = openers_weight * reply[2][None, :]
                allin_values = np.array(
                    [
                        shovers.sum(axis=1) * -(m.three_bet + m.ante_q),
                        (shovers * m.responder_allin.T).sum(axis=1),
                    ]
                )
                versus_allin[k].update(allin_values, three, iteration)
    average = _Strategies.of(_Regrets.average, open_regret, respond, versus_three_bet, versus_allin)
    worst = 0.0
    for p in openers:
        value, _ = _open_values(classes, posts, p, matchups, average)
        opened, fold_value = average.opening[p], -posts[p]
        regret = np.maximum(value, fold_value) - (opened * value + (1 - opened) * fold_value)
        worst = max(worst, float((classes.sizes * regret).sum() / classes.sizes.sum()))
    strategies: dict[str, dict[str, Any]] = {
        "open": {str(p): np.rint(100 * average.opening[p]).astype(int).tolist() for p in openers},
        "respond": {f"{p}-{q}": _percent(average.responses[(p, q)]) for p, q in matchups},
        "versus_3bet": {f"{p}-{q}": _percent(average.replies[(p, q)]) for p, q in matchups},
        "versus_allin": {f"{p}-{q}": _percent(average.calls[(p, q)]) for p, q in matchups},
    }
    return strategies, worst


def _percent(probabilities: np.ndarray) -> list[list[int]]:
    """Per class, the probability of each action after fold, as whole percentages."""
    return np.rint(100 * probabilities[1:].T).astype(int).tolist()


def _solve_task(task: tuple[int, float, bool]) -> tuple[dict[str, dict[str, Any]], float]:
    return solve(Classes.load(), *task)


def generate(workers: int) -> dict[str, Any]:
    tasks = [(n, stack, ante) for n in range(2, 9) for stack in STACK_BUCKETS for ante in (False, True)]
    with ProcessPoolExecutor(workers) as pool:
        results = list(pool.map(_solve_task, tasks))
    return {
        "classes": list(PREFLOP_CLASSES),
        "best_response_gain_bb": round(max(gain for _, gain in results), 4),
        "configs": {
            f"{n}/{stack:g}/{'bb_ante' if ante else 'no_ante'}": strategies
            for (n, stack, ante), (strategies, _) in zip(tasks, results)
        },
    }
