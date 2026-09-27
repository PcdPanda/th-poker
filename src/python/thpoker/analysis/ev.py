"""EV of every option at a decision (DESIGN.md Sections 6.5 and 6.6), ICM tournament equity, and
the device profiles that set how much work a review may do.

The rest of the current betting round is walked exactly: each opponent answers with its policy
applied to every hand in its range, so the range splits by action and later answers see the
updated range, and the player's own later actions in the round follow the reference policy.
When the round closes, the hand is valued from equity against the ranges that got there: exact
once the hand is over or nobody can bet any more, otherwise scaled by the preflop solver's
realization model, since later streets are not walked (a walk of every later street does not fit
a phone's budget). In tournaments every leaf is also valued in ICM equity; each pot then goes to
the player with probability equal to its share of that pot, and a pot it cannot win is split
evenly in chance among the players who can.

ICM equity follows the Malmuth-Harville model: each remaining player finishes first with
probability proportional to its stack; given the first place, the second is drawn the same way
from the rest, and so on. Players who bust in the hand being valued finish below everyone with
chips, ordered by their stacks at the start of the hand (DESIGN.md Section 4.6), with tied
players sharing the prizes of their places.

A device profile (DESIGN.md Section 10) sets the pruning of the full and triage passes and the
time for each river solve. A slower device prunes more and solves for longer (fewer iterations
fit), which widens the reported errors rather than silently lowering accuracy. `pick_profile`
times a small fixed workload to choose one. The PC profile is the analysis default.
"""

from   collections.abc          import Sequence
from   dataclasses              import dataclass
import time

import numpy as np

from   thpoker.bots.abstraction import (AbstractAction, acts_last,
                                        legal_abstract_actions, to_action)
from   thpoker.bots.bot         import Bot
from   thpoker.bots.equity_bot  import public_seed
from   thpoker.game.cards       import COMBOS
from   thpoker.game.engine      import (Pot, apply_action, build_pots,
                                        is_terminal, observation)
from   thpoker.game.rng         import Rng
from   thpoker.game.state       import Action, ActionType, GameState, Street
from   thpoker.odds             import (caller_share, hand_equity,
                                        settled_equity)


@dataclass(frozen=True)
class Profile:
    name: str
    min_branch: float  # pruning of the full review pass
    triage_branch: float  # pruning of the triage pass
    solver_seconds: float  # time for each heads-up river solve


PROFILES = {
    "pc": Profile("pc", 1e-3, 1e-2, 2.0),
    "phone": Profile("phone", 3e-3, 3e-2, 4.0),
}
# A PC runs the benchmark in about 40 ms; phones are 5-10 times slower (Section 10).
PHONE_SECONDS = 0.15


def benchmark() -> float:
    """Seconds for a fixed loop of plain Python arithmetic: the analysis is plain Python and
    numpy, and the hand evaluator's caches would make any poker workload read fast."""
    started = time.perf_counter()
    sum(i * i % 7 for i in range(300_000))
    return time.perf_counter() - started


def pick_profile(name: str = "auto") -> Profile:
    """The named profile, or with "auto" the one the benchmark suggests. Raises `ValueError`
    for an unknown name."""
    if name == "auto":
        return PROFILES["phone" if benchmark() > PHONE_SECONDS else "pc"]
    if name not in PROFILES:
        raise ValueError(f"unknown device profile {name!r}; choose from auto, {', '.join(PROFILES)}")
    return PROFILES[name]


# Branches reached less often than this from the option are dropped, their siblings renormalized.
MIN_BRANCH = PROFILES["pc"].min_branch


@dataclass(frozen=True)
class OptionValue:
    """EV of one option in big blinds, counted from the decision: chips already in the pot are
    sunk, so folding is worth 0. `exact` means neither the realization model nor sampled equity
    was needed, so the chip EV is exact given the policies and ranges. In tournaments `icm` is
    the expected share of the prize pool after the hand."""

    abstract: AbstractAction | None  # None for a size outside the abstraction
    action: Action
    ev: float
    stderr: float
    exact: bool
    icm: float | None = None
    icm_stderr: float | None = None


def _share(branch: tuple[Action, float, dict[int, np.ndarray]]) -> float:
    return branch[1]


class _Walk:
    """Values are arrays: chips won from the decision on, then ICM equity in tournaments."""

    def __init__(
        self,
        root: GameState,
        bots: dict[int, Bot],
        reference: Bot,
        ranges: dict[int, np.ndarray],
        payouts: Sequence[float] | None,
        min_branch: float,
    ):
        assert root.to_act is not None
        self.root = root
        self.user = root.to_act
        hole = root.hole_cards[self.user]
        assert hole is not None
        self.hole = hole
        self.bots = bots
        self.reference = reference
        self.ranges = {s: w / w.sum() for s, w in ranges.items() if s != self.user}
        self.payouts = payouts
        self.min_branch = min_branch
        self.width = 1 if payouts is None else 2
        self.icm_cache: dict[tuple[float, ...], float] = {}
        # Every leaf draws the same runouts, so the options' errors largely cancel in their gaps.
        self.seed = public_seed(observation(root, self.user))

    def valuation(self, final: tuple[float, ...]) -> np.ndarray:
        chips = final[self.user] - self.root.stacks[self.user]
        if self.payouts is None:
            return np.array([chips])
        if final not in self.icm_cache:
            equities = icm_equities(final, self.root.starting_stacks, self.payouts)
            self.icm_cache[final] = equities[self.user]
        return np.array([chips, self.icm_cache[final]])

    def value(
        self, state: GameState, reach: dict[int, np.ndarray], reached: float = 1.0
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        """Values from this state on, their standard errors, and whether the chip EV is exact.
        `reached` is the chance of getting here from the option being valued."""
        if is_terminal(state) or state.street != self.root.street:
            return self._leaf(state, reach)
        if state.folded[self.user] and self.payouts is None:  # nothing the user can still win
            added = state.committed_total[self.user] - self.root.committed_total[self.user]
            return np.array([-float(added)]), np.zeros(1), True
        seat = state.to_act
        assert seat is not None
        view = observation(state, seat)
        concrete = {a: to_action(a, view) for a in legal_abstract_actions(view)}
        branches: list[tuple[Action, float, dict[int, np.ndarray]]] = []
        if seat == self.user:
            shares: dict[Action, float] = {}
            for abstract, probability in self.reference.action_probabilities(view).items():
                action = concrete[abstract]
                shares[action] = shares.get(action, 0.0) + probability
            branches = [(a, p, reach) for a, p in shares.items()]
        else:
            weights = reach[seat]
            holes = [int(i) for i in np.flatnonzero(weights)]
            # Bots share one distribution object among hands decided alike (a class, a bucket).
            groups: dict[int, tuple[dict[AbstractAction, float], list[int]]] = {}
            for hole, distribution in self.bots[seat].policies(view, holes).items():
                groups.setdefault(id(distribution), (distribution, []))[1].append(hole)
            split: dict[Action, np.ndarray] = {}
            for distribution, members in groups.values():
                for abstract, probability in distribution.items():
                    action = concrete[abstract]
                    if action not in split:
                        split[action] = np.zeros(len(COMBOS))
                    split[action][members] += probability
            for action, likelihood in split.items():
                joint = weights * likelihood
                share = float(joint.sum())
                if share > 0:
                    branches.append((action, share, {**reach, seat: joint / share}))
        kept = [b for b in branches if reached * b[1] >= self.min_branch] or [max(branches, key=_share)]
        total_share = sum(share for _, share, _ in kept)
        value, stderr = np.zeros(self.width), np.zeros(self.width)
        exact = True
        children = []
        for action, share, child_reach in kept:
            child, _ = apply_action(state, action)
            weight = share / total_share
            child_value, child_stderr, child_exact = self.value(child, child_reach, reached * weight)
            value += weight * child_value
            stderr += weight * child_stderr  # leaves share runouts
            exact = exact and child_exact
            children.append(child_value)
        pruned = 1 - total_share / sum(share for _, share, _ in branches)
        if pruned > 1e-12:
            # A dropped branch can move the user's chips by at most the pot and twice the
            # effective stack (an all-in either way).
            opponents = [
                state.stacks[s]
                for s in range(state.config.num_seats)
                if s != self.user and state.dealt_in[s] and not state.folded[s]
            ]
            effective = min(state.stacks[self.user], max(opponents, default=0))
            spread = np.ptp(np.array(children), axis=0)
            spread[0] = max(spread[0], float(state.pot + 2 * effective))
            stderr += pruned * spread
            exact = False
        return value, stderr, exact

    def _leaf(self, state: GameState, reach: dict[int, np.ndarray]) -> tuple[np.ndarray, np.ndarray, bool]:
        n = state.config.num_seats
        live = tuple(state.dealt_in[s] and not state.folded[s] for s in range(n))
        # Chips behind before any pot was awarded: the replayed hand awards pots by real cards.
        behind = [float(c) for c in state.stacks]
        for award in state.awards:
            for seat, amount in zip(award.winners, award.shares):
                behind[seat] -= amount
        # Nobody can push the user off the pot once the river closes, the user is all in, or at
        # most one live player has chips left.
        settled = (
            self.root.street == Street.RIVER
            or state.all_in[self.user]
            or sum(live[s] and not state.all_in[s] for s in range(n)) <= 1
        )
        pots = build_pots(state.committed_total, live, state.dead_money)
        # The user's shares of the pots it contests, main pot first; nothing else is uncertain.
        contested: list[tuple[int, float, float]] = []  # pot index, share, error
        exact = True
        for index, pot in enumerate(pots):
            rivals = [s for s in pot.eligible if s != self.user]
            if self.user in pot.eligible and rivals:
                ranges = [reach[s] for s in rivals]
                rng = Rng(self.seed)
                equity = (
                    settled_equity(self.hole, self.root.board, ranges, rng)
                    if settled
                    else hand_equity(self.hole, self.root.board, ranges, rng)
                )
                share = equity.value if settled else self._realized(state, equity.value)
                contested.append((index, share, equity.stderr))
                exact = exact and equity.exact and settled
        if self.payouts is None:
            won = float(sum(pot.amount for pot in pots if pot.eligible == (self.user,)))
            won += sum(share * pots[index].amount for index, share, _ in contested)
            error = sum(error * pots[index].amount for index, _, error in contested)
            return (
                np.array([behind[self.user] + won - self.root.stacks[self.user]]),
                np.array([error]),
                exact,
            )
        return (*self._tournament_leaf(pots, contested, behind), exact)

    def _final_stacks(self, pots: list[Pot], won: set[int], behind: list[float]) -> tuple[float, ...]:
        """Stacks after the user wins the pots in `won`; every other pot is split evenly in
        expectation among the other players who can win it."""
        final = list(behind)
        for index, pot in enumerate(pots):
            if index in won:
                final[self.user] += pot.amount
                continue
            others = [s for s in pot.eligible if s != self.user] or list(pot.eligible)
            for seat in others:
                final[seat] += pot.amount / len(others)
        return tuple(final)

    def _tournament_leaf(
        self, pots: list[Pot], contested: list[tuple[int, float, float]], behind: list[float]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Expected values and their errors over the user's nested pot outcomes: the side pots
        it contests have fewer rivals, and winning a pot means holding the best hand among a
        superset of those rivals, so the user wins some last run of its contested pots (main
        pot first)."""
        sole = {index for index, pot in enumerate(pots) if pot.eligible == (self.user,)}
        shares = []
        best = 0.0
        for _, share, _ in contested:
            best = max(best, share)  # later pots have fewer rivals, so no lower share
            shares.append(best)
        value = np.zeros(self.width)
        previous = 0.0
        for first in range(len(contested) + 1):
            upper = shares[first] if first < len(contested) else 1.0
            chance, previous = upper - previous, upper
            won = sole | {index for index, _, _ in contested[first:]}
            value += chance * self.valuation(self._final_stacks(pots, won, behind))
        error = np.zeros(self.width)
        for index, _, equity_error in contested:
            swing = self.valuation(self._final_stacks(pots, sole | {index}, behind)) - self.valuation(
                self._final_stacks(pots, sole, behind)
            )
            error += equity_error * np.abs(swing)
        return value, error

    def _realized(self, state: GameState, equity: float) -> float:
        """Pot share when betting continues on later streets. The caller of the round's last
        bet, or the player out of position when nobody bet, realizes `caller_share`; the other
        side wins the rest, as in the preflop solver."""
        in_position = acts_last(state, self.user)
        raisers = [
            e.seat
            for e in state.history
            if e.street == self.root.street and e.action.type in (ActionType.BET, ActionType.RAISE)
        ]
        if raisers:
            is_caller = raisers[-1] != self.user
        else:
            is_caller = not in_position
        if is_caller:
            return float(caller_share(equity, in_position))
        return 1.0 - float(caller_share(1.0 - equity, not in_position))


def option_values(
    state: GameState,
    bots: dict[int, Bot],
    reference: Bot,
    ranges: dict[int, np.ndarray],
    taken: Action | None = None,
    payouts: Sequence[float] | None = None,
    min_branch: float = MIN_BRANCH,
) -> list[OptionValue]:
    """EV of every legal abstract option of the seat to act in `state`, plus `taken` when it
    is a size outside the abstraction. Opponents answer with `bots` holding hands weighted by
    `ranges` (by seat, as from `tracking.track`); only public information and the acting seat's
    own cards are used, so `state` may be a replayed hand with every card in it. With the
    tournament's `payouts`, each option is also valued in ICM equity. A larger `min_branch`
    prunes more of the tree: faster and rougher."""
    walk = _Walk(state, bots, reference, ranges, payouts, min_branch)
    view = observation(state, walk.user)
    options: dict[Action, AbstractAction | None] = {}
    for legal in legal_abstract_actions(view):
        options.setdefault(to_action(legal, view), legal)
    if taken is not None:
        options.setdefault(taken, None)
    big_blind = state.config.big_blind
    values = []
    for action, abstract in options.items():
        child, _ = apply_action(state, action)
        value, stderr, exact = walk.value(child, walk.ranges)
        icm = (float(value[1]), float(stderr[1])) if payouts is not None else (None, None)
        values.append(
            OptionValue(
                abstract,
                action,
                float(value[0]) / big_blind,
                float(stderr[0]) / big_blind,
                exact,
                *icm,
            )
        )
    return values


def icm_equities(stacks: Sequence[float], start_stacks: Sequence[float], payouts: Sequence[float]) -> list[float]:
    """Each seat's expected share of the prizes still to be decided, as fractions of the prize
    pool. Seats with start stacks above 0 were in the tournament at the start of the hand and
    decide the last places of `payouts` among them; `stacks` are their chips now. Raises
    `ValueError` if the lengths differ or nobody has chips left.
    """
    if len(stacks) != len(start_stacks):
        raise ValueError(f"{len(stacks)} stacks but {len(start_stacks)} start stacks")
    entrants = [s for s, start in enumerate(start_stacks) if start > 0]
    alive = [s for s in entrants if stacks[s] > 0]
    if not alive:
        raise ValueError("nobody has chips left")
    prizes = list(payouts[: len(entrants)]) + [0.0] * (len(entrants) - len(payouts))
    equities = [0.0] * len(stacks)
    _finish_order(stacks, alive, prizes, 0, 1.0, equities)
    # Busted players take the places below the living, the larger start stack placing higher.
    place = len(alive)
    busted = sorted((s for s in entrants if stacks[s] <= 0), key=start_stacks.__getitem__, reverse=True)
    while busted:
        group = [s for s in busted if start_stacks[s] == start_stacks[busted[0]]]
        shared = sum(prizes[place : place + len(group)]) / len(group)
        for seat in group:
            equities[seat] = shared
        place += len(group)
        busted = busted[len(group) :]
    return equities


def icm_value_of(chips: float, stacks: Sequence[float], user: int, payouts: Sequence[float]) -> float:
    """Prize equity of `chips` more for `user` at `stacks`, the chips coming from the other
    players in proportion to their stacks (and going back to them for the loss)."""
    others = sum(c for s, c in enumerate(stacks) if s != user)
    up = [c + chips if s == user else c - chips * c / others for s, c in enumerate(stacks)]
    down = [c - chips if s == user else c + chips * c / others for s, c in enumerate(stacks)]
    return (icm_equities(up, stacks, payouts)[user] - icm_equities(down, stacks, payouts)[user]) / 2


def _finish_order(
    stacks: Sequence[float],
    remaining: list[int],
    prizes: list[float],
    place: int,
    probability: float,
    equities: list[float],
):
    """Add each remaining player's chance of `place` and below, stopping at the last paid place."""
    if not any(prizes[place:]):
        return
    total = sum(stacks[s] for s in remaining)
    for seat in remaining:
        chance = probability * stacks[seat] / total
        equities[seat] += chance * prizes[place]
        if len(remaining) > 1:
            _finish_order(stacks, [s for s in remaining if s != seat], prizes, place + 1, chance, equities)
