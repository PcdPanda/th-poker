"""Tier 3: plays preflop from solved charts and postflop by where its hand sits in its range.

Preflop, stacks of 15bb or less follow the push/fold charts and deeper stacks follow the preflop
range tables in the spots they cover (first in, facing a single open, facing a 3-bet, facing an
all-in); other preflop spots use the Tier 2 policy. Postflop the bot ranks every hand of its own
range by equity against the opponents' ranges (built the same way: charts where they apply,
then shifted toward strong hands by each postflop bet or raise). Facing a bet it continues with
the top share of its range that the minimum defense frequency asks for, or whenever it is
clearly priced in; when checked to it bets its value hands plus a balanced share of bluffs
(b/(1+b) of the value hands on the river, more on earlier streets), sized by board texture.
"""

from   bisect                   import bisect_left, bisect_right
from   collections.abc          import Callable, Sequence
from   dataclasses              import dataclass, field
from   functools                import cache
import math
from   thpoker.bots.abstraction import (AbstractAction, NEVER_FOLD_EQUITY,
                                        POT_FRACTIONS, PUSH_FOLD_BIG_BLINDS,
                                        call_price, defense_share,
                                        effective_stack, finish, last_bet,
                                        legal_abstract_actions, sigmoid,
                                        usual_raises)
from   thpoker.bots.bot         import PRESETS, Policy, Style, TierBot
from   thpoker.bots.equity_bot  import (EquityBot, bet_shifted, preflop_range,
                                        public_seed)
from   thpoker.charts           import call_chart, push_chart, strategy
from   thpoker.game.cards       import (COMBOS, COMBOS_OF_CLASS, COMBO_CLASS,
                                        PREFLOP_CLASSES)
from   thpoker.game.engine      import observation, replay_states
from   thpoker.game.rng         import Rng
from   thpoker.game.rules       import legal_actions
from   thpoker.game.state       import (ActionType, GameState, Observation,
                                        Street)
from   thpoker.odds             import (Range, hand_range, range_equities,
                                        ranked_range, texture)
from   typing                   import Any

_BLUFF_STREET_FACTOR = {Street.FLOP: 1.5, Street.TURN: 1.2, Street.RIVER: 1.0}
# Chart thresholds sit on a half-big-blind grid; mix across about that width.
_CHART_SOFTNESS_BB = 0.3
# How much wider (above 1) or narrower a seat's continuing and re-raising chart rows are, from
# the view, the seats in preflop order, the raisers and the actor (as places in that order),
# and the chart node.
Factors = Callable[[Observation, tuple[int, ...], list[int], int, str], tuple[float, float]]


@dataclass(frozen=True)
class PreflopSpot:
    """The preflop situation in chart terms: seats in preflop order among players dealt in."""

    num_players: int
    order: tuple[int, ...]  # table seats in preflop order
    stack: float  # effective stack at the start of the hand, in big blinds
    bb_ante: bool


def preflop_order(view: Observation) -> tuple[int, ...]:
    """Table seats of the players dealt in, in preflop order (heads-up the small blind first)."""
    n = view.config.num_seats
    if sum(view.dealt_in) == 2:
        return (view.small_blind_seat, view.big_blind_seat)
    start = (view.big_blind_seat + 1) % n
    return tuple(s for s in ((start + k) % n for k in range(n)) if view.dealt_in[s])


def preflop_spot(view: Observation, seat: int | None = None, before: int | None = None) -> PreflopSpot:
    """The spot at the depth that governs `seat`'s action at history index `before` (the
    viewer's decision now by default): its own effective stack, capped by what every other
    earlier raiser started with. A 12bb shove reads from the push/fold chart, and a 100bb player
    facing it plays the 12bb call chart."""
    seat = view.seat if seat is None else seat
    order = preflop_order(view)
    depth = effective_stack(view, seat)
    raisers = [
        e.seat
        for e in view.history[:before]
        if e.street == Street.PREFLOP and e.action.type in (ActionType.BET, ActionType.RAISE)
    ]
    depth = min([depth] + [view.starting_stacks[r] for r in raisers if r != seat])
    return PreflopSpot(len(order), order, depth / view.config.big_blind, view.config.ante > 0)


def _class_weights(probabilities: list[float]) -> Range:
    return tuple(probabilities[c] for c in COMBO_CLASS)


def _soft_threshold(threshold: float, stack: float) -> float:
    """Probability of playing a hand whose chart threshold is `threshold` at this stack."""
    return 1.0 / (1.0 + math.exp(-(threshold - stack + 0.25) / _CHART_SOFTNESS_BB))


def _oversized(view: Observation) -> bool:
    """Whether the open or the 3-bet was more than twice the bots' own size from a raiser deeper
    than the push/fold charts: the preflop charts were solved for standard sizes (the all-in
    4-bet after a 3-bet is part of them), so such a line is off the charts."""
    limit = PUSH_FOLD_BIG_BLINDS * view.config.big_blind
    return any(
        amount > 2 * usual and effective_stack(view, seat) > limit for seat, amount, usual in usual_raises(view)[:2]
    )


@cache
def _strongest_first() -> tuple[int, ...]:
    return tuple(PREFLOP_CLASSES.index(name) for name in hand_range("0-100"))


_CLASS_COMBOS = tuple(len(COMBOS_OF_CLASS[name]) for name in PREFLOP_CLASSES)


def _combos(weights: Sequence[float]) -> float:
    """How many of the 1326 deals class weights hold."""
    return sum(w * n for w, n in zip(weights, _CLASS_COMBOS))


def stretched(weights: Sequence[float], factor: float) -> list[float]:
    """Class weights at `factor` times their share of deals (at most all of them): a wider
    range fills the strongest classes first, a narrower one empties the weakest first."""
    result = list(weights)
    if factor == 1:
        return result
    share = _combos(weights)
    left = min(len(COMBOS), factor * share) - share
    order = _strongest_first() if left > 0 else reversed(_strongest_first())
    for klass in order:
        if abs(left) < 1e-9:
            break
        room = 1 - result[klass] if left > 0 else -result[klass]
        change = min(room, left / _CLASS_COMBOS[klass], key=abs)
        result[klass] += change
        left -= change * _CLASS_COMBOS[klass]
    return result


def _stretched_rows(
    node: str, rows: tuple[tuple[float, ...], ...], factors: tuple[float, float]
) -> tuple[tuple[float, ...], ...]:
    """Open rows (fold, open) with the opens at `factors[0]` times their share; respond rows
    (fold, call, re-raise) with the continuing hands at `factors[0]` and the re-raises at
    `factors[1]` times theirs, a re-raise never above the class's continuing weight."""
    keep, again = factors
    if (keep, again) == (1.0, 1.0):
        return rows
    if node == "open":
        return tuple((1 - w, w) for w in stretched([row[1] for row in rows], keep))
    going = stretched([row[1] + row[2] for row in rows], keep)
    raising = [min(r, g) for r, g in zip(stretched([row[2] for row in rows], again), going)]
    return tuple((1 - g, g - r, r) for g, r in zip(going, raising))


def chart_range(view: Observation, seat: int, factors: Factors | None = None) -> Range | None:
    """The range the charts, stretched by `factors`, give `seat` for its preflop actions so far,
    or None when its line leaves the charted spots (limps, multiway pots, cold 4-bets,
    short-stack raises that are not all-in, raises of an unusual size)."""
    if _oversized(view):
        return None
    order = preflop_spot(view, seat).order
    index = order.index(seat)
    raisers: list[int] = []
    callers: list[int] = []
    weights: list[float] | None = None
    for position, entry in enumerate(view.history):
        if entry.street != Street.PREFLOP:
            break
        actor, action = order.index(entry.seat), entry.action
        aggressive = action.type in (ActionType.BET, ActionType.RAISE)
        if actor == index:
            # An open follows the chart for the depth when it was made, before any short 3-bet.
            spot = preflop_spot(view, seat, position)
            node = _chart_node(
                spot,
                raisers,
                callers,
                index,
                aggressive,
                action.type == ActionType.CALL,
                view,
                factors,
            )
            if node is None:
                return None
            weights = node if weights is None else [w * p for w, p in zip(weights, node)]
        if aggressive:
            raisers.append(actor)
        elif action.type == ActionType.CALL:
            callers.append(actor)
    return _class_weights(weights) if weights is not None else None


def _chart_node(
    spot: PreflopSpot,
    raisers: list[int],
    callers: list[int],
    index: int,
    aggressive: bool,
    called: bool,
    view: Observation,
    factors: Factors | None,
) -> list[float] | None:
    """Per-class probability that a player at `index` took this action, or None if off-chart."""
    if callers:
        return None
    short = spot.stack <= PUSH_FOLD_BIG_BLINDS
    all_in = view.all_in[spot.order[index]]
    if short:
        if not raisers and aggressive and all_in:
            return [_soft_threshold(t, spot.stack) for t in push_chart(spot.num_players, index, spot.bb_ante)]
        if len(raisers) == 1 and called and view.all_in[spot.order[raisers[0]]]:
            return [
                _soft_threshold(t, spot.stack) for t in call_chart(spot.num_players, raisers[0], index, spot.bb_ante)
            ]
        return None
    if not raisers and aggressive:
        rows = strategy("open", spot.num_players, str(index), spot.stack, spot.bb_ante)
        stretch = factors(view, spot.order, raisers, index, "open") if factors else (1.0, 1.0)
        return [row[1] for row in _stretched_rows("open", rows, stretch)]
    if len(raisers) == 1 and raisers[0] != index:
        rows = strategy("respond", spot.num_players, f"{raisers[0]}-{index}", spot.stack, spot.bb_ante)
        stretch = factors(view, spot.order, raisers, index, "respond") if factors else (1.0, 1.0)
        rows = _stretched_rows("respond", rows, stretch)
        return [row[2] if aggressive else row[1] for row in rows] if (aggressive or called) else None
    if len(raisers) == 2 and raisers[0] == index:
        rows = strategy("versus_3bet", spot.num_players, f"{index}-{raisers[1]}", spot.stack, spot.bb_ante)
        return [row[2] if aggressive else row[1] for row in rows] if (aggressive or called) else None
    if len(raisers) == 3 and raisers[1] == index and raisers[2] == raisers[0] and called:
        rows = strategy("versus_allin", spot.num_players, f"{raisers[0]}-{index}", spot.stack, spot.bb_ante)
        return [row[1] for row in rows]
    return None


def seat_range(
    view: Observation,
    seat: int,
    factors: Factors | None = None,
    limp: Range | None = None,
    floors: tuple[float, float] = (0.25, 0.25),
) -> Range:
    """A seat's range from its public actions: charts stretched by `factors` (or the Tier 2
    bands off-chart, with `limp` for a limp) preflop, then shifted toward strong hands by each
    of its postflop bets and raises, with `strength_weighted`'s `floors` on the flop and later."""
    charted = chart_range(view, seat, factors)
    # A line the charts give no weight to any hand leaves nothing to score against.
    if charted is not None and sum(charted) > 1.0:
        weights = charted
    else:
        weights = preflop_range(view, seat) if limp is None else preflop_range(view, seat, limp)
    return bet_shifted(view, seat, weights, floors)


class RangeBot(TierBot):
    def __init__(self, name: str, style: Style):
        self.name = name
        self.style = style
        self._fallback = EquityBot(name, style)

    def chart_factors(
        self, view: Observation, order: tuple[int, ...], raisers: list[int], index: int, node: str
    ) -> tuple[float, float]:
        """See `Factors`: Hard plays and reads the charts as they are."""
        return 1.0, 1.0

    def seat_range(self, view: Observation, seat: int) -> Range:
        return seat_range(view, seat)

    def exploits(self, view: Observation, opponents: list[int]) -> tuple[float, float, float]:
        """After the flop, what to add to the share of its range it continues with facing a
        bet, what to add to its value bar, and what to multiply its bluffs by."""
        return 0.0, 0.0, 1.0

    def decisions(self, view: Observation, holes: list[int]) -> dict[int, Policy]:
        """One chart lookup preflop or one equity pass postflop serves every hand. Off the
        charts, and for hands whose equity cannot be computed (card removal left an opponent
        range empty), the Tier 2 policy decides."""
        legal = legal_abstract_actions(view)
        result: dict[int, Policy] = {}
        if view.street == Street.PREFLOP:
            charted = self._preflop(view)
            if charted is None:
                for hole, (distribution, rationale) in self._fallback.decisions(view, holes).items():
                    rule = f"tier 2 fallback: {rationale['rule_triggered']}"
                    result[hole] = distribution, {**rationale, "rule_triggered": rule}
                return result
            table, shared, thresholds = charted
            by_class: dict[int, Policy] = {}
            for hole in holes:
                klass = COMBO_CLASS[hole]
                if klass not in by_class:
                    rationale = dict(shared)
                    if thresholds is not None:
                        rationale["chart_threshold_bb"] = thresholds[klass]
                    rationale["style"] = self.style.name
                    by_class[klass] = finish(table[klass], view, legal), rationale
                result[hole] = by_class[klass]
            return result
        for hole, (weights, rationale) in self._postflop(view, holes).items():
            rationale["style"] = self.style.name
            result[hole] = finish(weights, view, legal), rationale
        missing = [h for h in holes if h not in result]
        if missing:
            result.update(self._fallback.decisions(view, missing))
        return result

    def _preflop(
        self, view: Observation
    ) -> tuple[list[dict[AbstractAction, float]], dict[str, Any], list[float] | None] | None:
        """The charted action weights of every preflop class in this spot (they do not depend
        on the hand held), the rationale they share, and the push/fold chart thresholds by
        class; None when the spot is off the charts."""
        spot = preflop_spot(view)
        index = spot.order.index(view.seat)
        entries = [e for e in view.history if e.street == Street.PREFLOP]
        raisers = [spot.order.index(e.seat) for e in entries if e.action.type in (ActionType.BET, ActionType.RAISE)]
        if any(e.action.type == ActionType.CALL for e in entries) or _oversized(view):
            return None
        rationale: dict[str, Any] = {
            "stack_bb": round(spot.stack, 1),
            "seat_index": index,
            "players": spot.num_players,
        }
        fold, call = AbstractAction.FOLD, AbstractAction.CALL
        if spot.stack <= PUSH_FOLD_BIG_BLINDS:
            if not raisers:
                thresholds = list(push_chart(spot.num_players, index, spot.bb_ante))
                play, rule = AbstractAction.ALL_IN, "push/fold chart: first in"
            elif len(raisers) == 1 and view.all_in[spot.order[raisers[0]]]:
                thresholds = list(call_chart(spot.num_players, raisers[0], index, spot.bb_ante))
                play, rule = call, "push/fold chart: facing a shove"
            else:
                return None
            shares = [_soft_threshold(t, spot.stack) for t in thresholds]
            table = [{play: p, fold: 1 - p} for p in shares]
            return table, {**rationale, "rule_triggered": rule}, thresholds
        # Each chart row lists its actions' probabilities in the order of `actions`.
        actions: tuple[AbstractAction, ...]
        if not raisers:
            node, seats, actions, rule = "open", str(index), (fold, AbstractAction.OPEN), "first in"
        elif len(raisers) == 1 and index not in raisers:
            node, seats, rule = "respond", f"{raisers[0]}-{index}", "facing an open"
            actions = (fold, call, AbstractAction.RERAISE)
        elif len(raisers) == 2 and raisers[0] == index:
            node, seats, rule = "versus_3bet", f"{index}-{raisers[1]}", "facing a 3-bet"
            actions = (fold, call, AbstractAction.ALL_IN)
        elif (
            len(raisers) == 3
            and raisers[1] == index
            and raisers[2] == raisers[0]
            and view.all_in[spot.order[raisers[0]]]
        ):
            node, seats, actions, rule = (
                "versus_allin",
                f"{raisers[0]}-{index}",
                (fold, call),
                "facing an all-in",
            )
        else:
            return None
        rows = strategy(node, spot.num_players, seats, spot.stack, spot.bb_ante)
        if node in ("open", "respond"):
            stretch = self.chart_factors(view, spot.order, raisers, index, node)
            rows = _stretched_rows(node, rows, stretch)
            if stretch != (1.0, 1.0):
                rationale["chart_stretch"] = stretch
        table = [dict(zip(actions, row)) for row in rows]
        return table, {**rationale, "rule_triggered": f"preflop table: {rule}"}, None

    def _postflop(
        self, view: Observation, holes: list[int]
    ) -> dict[int, tuple[dict[AbstractAction, float], dict[str, Any]]]:
        """Action weights and rationale for each hand in `holes` held by this seat. Hands whose
        equity cannot be computed (card removal emptied a range) are left out."""
        style = self.style
        opponents = [
            s for s in range(view.config.num_seats) if s != view.seat and view.dealt_in[s] and not view.folded[s]
        ]
        opponent_ranges = [self.seat_range(view, s) for s in opponents]
        own = self.seat_range(view, view.seat)
        call_shift, value_shift, bluff_factor = self.exploits(view, opponents)
        combos = sorted({i for i, w in enumerate(own) if w > 0} | set(holes))
        equities = range_equities(view.board, combos, opponent_ranges, Rng(public_seed(view)))
        ranked = sorted((e, own[i]) for i, e in equities.items() if own[i] > 0)
        values = [e for e, _ in ranked]
        running = [0.0]
        for _, w in ranked:
            running.append(running[-1] + w)
        weight_total = running[-1] or 1.0
        softness = 0.03 + 0.1 * style.temperature
        legal = legal_actions(view)
        tags = texture(view.board)
        common: dict[str, Any] = {
            "opponents": len(opponents),
            "board": f"{tags.high}, {tags.pairing}, {tags.suits}, {tags.connectivity}",
        }
        if (call_shift, value_shift, bluff_factor) != (0.0, 0.0, 1.0):
            common["exploits"] = [call_shift, value_shift, bluff_factor]
        if legal.can_check:
            size = 0.75 if tags.wet or view.street == Street.RIVER else 0.33
            size = min(1.0, max(0.33, size + 0.3 * (style.sizing_preference - 0.5)))
            value_bar = 0.5 + 0.08 * (len(opponents) - 1)
            value_share = min(
                1.0,
                style.aggression * (weight_total - running[bisect_left(values, value_bar)]) / weight_total,
            )
            bluff_share = min(
                0.3,
                value_share
                * size
                / (1 + size)
                * _BLUFF_STREET_FACTOR[view.street]
                * style.bluff_multiplier
                * bluff_factor
                / len(opponents),
            )
            if value_shift:  # bluffs stay balanced against the value share before the shift
                value_share = min(
                    1.0,
                    style.aggression
                    * (weight_total - running[bisect_left(values, value_bar + value_shift)])
                    / weight_total,
                )
            action = _size_action(size)
            common.update(
                {
                    "rule_triggered": "checked to: value and balanced bluffs",
                    "value_share": round(value_share, 4),
                    "bluff_share": round(bluff_share, 4),
                    "bet_size_pot": size,
                }
            )
        else:
            price = call_price(view)
            defense, bet_fraction = _defense(view, len(opponents))
            continue_share = min(
                1.0,
                max(
                    0.05,
                    defense
                    + 0.15 * (style.call_down_tendency - 0.5)
                    - 0.1 * style.fold_to_pressure * max(0.0, bet_fraction - 0.75)
                    + call_shift,
                ),
            )
            raise_share = continue_share * 0.2 * style.aggression
            common.update(
                {
                    "rule_triggered": "facing bet: minimum defense and price",
                    "required_equity": round(price, 4),
                    "defense_share": round(continue_share, 4),
                }
            )
        decided = {}
        for hole in holes:
            if hole not in equities:
                continue
            strength = equities[hole]
            low, high = bisect_left(values, strength), bisect_right(values, strength)
            percentile = (running[low] + (running[high] - running[low]) / 2) / weight_total
            rationale = {
                **common,
                "equity_estimate": round(strength, 4),
                "range_percentile": round(percentile, 4),
            }
            if legal.can_check:
                bet = min(
                    1.0,
                    sigmoid((percentile - (1 - value_share)) / softness)
                    + sigmoid((bluff_share - percentile) / softness),
                )
                decided[hole] = ({AbstractAction.CHECK: 1 - bet, action: bet}, rationale)
                continue
            keep = sigmoid((percentile - (1 - continue_share)) / softness)
            keep = max(keep, sigmoid((strength - price - 0.05) / 0.02))  # clearly priced in
            if strength >= NEVER_FOLD_EQUITY:
                keep = 1.0
            raises = min(keep, sigmoid((percentile - (1 - raise_share)) / softness) * (strength >= 0.6))
            decided[hole] = (
                {
                    AbstractAction.FOLD: 1 - keep,
                    AbstractAction.CALL: keep - raises,
                    AbstractAction.BET_75: raises,
                },
                rationale,
            )
        return decided


def _defense(view: Observation, opponents: int) -> tuple[float, float]:
    """The share of its range a seat keeps facing this street's last bet by the minimum defense
    frequency (in its multiway form where `defense_share` has none), and the bet's size as a
    share of the pot before it."""
    faced = last_bet(view)
    assert faced is not None  # after the flop, a seat that cannot check faces a bet
    bet_fraction = faced[1] / faced[0]
    defense = defense_share(view)
    if defense is None:
        defense = 1 - (bet_fraction / (1 + bet_fraction)) ** (1 / opponents)
    return defense, bet_fraction


def _size_action(pot_fraction: float) -> AbstractAction:
    """The abstract bet closest to a pot fraction."""
    return min(POT_FRACTIONS, key=lambda a: abs(POT_FRACTIONS[a] - pot_fraction))


# Behind every factor of the read, this many events of solid play (DESIGN.md Section 5.3), so a
# short session reads like a solid player; factors stay within the clamp.
READ_PRIOR = 4.0
READ_CLAMP = (0.4, 2.5)
_AGGRESSIVE = (ActionType.BET, ActionType.RAISE)


@dataclass(frozen=True)
class UserRead:
    """How often the user has done each thing this session against what Hard expects of a
    solid player in the same spots (1 is as expected), and how wide its limping hands are."""

    opens: float = 1.0
    defends: float = 1.0  # calls or re-raises facing one open
    three_bets: float = 1.0
    bets_flop: float = 1.0  # bets when checked to
    bets_late: float = 1.0  # the same on the turn and river
    folds_flop: float = 1.0  # folds facing a bet
    folds_late: float = 1.0
    limp_width: float = 0.5


NEUTRAL = UserRead()


def _floor(bets: float) -> float:
    """`strength_weighted`'s floor for the bets of a user who bets `bets` times as often as
    expected: the more often, the less a bet says."""
    return min(0.7, max(0.1, 1 - 0.75 / bets))


class ExpertBot(RangeBot):
    """Hard, adjusted to the user's play this session: `reads` holds the read each hand was
    dealt with, by hand id, so a policy stays a pure function of its hand."""

    def __init__(
        self,
        name: str,
        style: Style,
        reads: dict[str, UserRead] | None = None,
        user: int | None = None,
    ):
        super().__init__(name, style)
        self.reads = {} if reads is None else reads
        self.user = user

    def chart_factors(
        self, view: Observation, order: tuple[int, ...], raisers: list[int], index: int, node: str
    ) -> tuple[float, float]:
        """The user's rows by its read; its own rows wider against a user in the blinds who
        defends too little, or facing an open from a user who opens too much."""
        if self.user not in order:
            return 1.0, 1.0
        read = self.reads.get(view.hand_id, NEUTRAL)
        user = order.index(self.user)
        if node == "open":
            if index == user:
                return read.opens, 1.0
            if user == len(order) - 1:
                return read.defends**-0.5, 1.0
            # In the small blind, the big blind behind it still defends.
            if user == len(order) - 2 and index < user:
                return read.defends**-0.25, 1.0
            return 1.0, 1.0
        if index == user:
            return read.defends, read.three_bets
        if raisers[0] == user:
            return read.opens**0.5, read.opens**0.5
        return 1.0, 1.0

    def seat_range(self, view: Observation, seat: int) -> Range:
        if seat != self.user:
            return seat_range(view, seat, self.chart_factors)
        read = self.reads.get(view.hand_id, NEUTRAL)
        limp = ranked_range(0.1, min(1.0, 0.1 + read.limp_width))
        floors = (_floor(read.bets_flop), _floor(read.bets_late))
        return seat_range(view, seat, self.chart_factors, limp, floors)

    def exploits(self, view: Observation, opponents: list[int]) -> tuple[float, float, float]:
        """Against a user who bets more than expected it calls down more; heads-up against one
        who folds less, it bluffs less and value bets thinner (and the other way round)."""
        if self.user not in opponents:
            return 0.0, 0.0, 1.0
        read = self.reads.get(view.hand_id, NEUTRAL)
        flop = view.street == Street.FLOP
        bets, folds = (read.bets_flop, read.folds_flop) if flop else (read.bets_late, read.folds_late)
        street = [e for e in view.history if e.street == view.street]
        bettor = next((e.seat for e in reversed(street) if e.action.type in _AGGRESSIVE), None)
        call_shift = 0.15 * min(1.0, max(-1.0, math.log2(bets))) if bettor == self.user else 0.0
        if opponents != [self.user]:
            return call_shift, 0.0, 1.0
        return call_shift, min(0.03, max(-0.03, 0.03 * (folds - 1))), folds


@dataclass
class UserTally:
    """What the user did this session and what a solid player would have done in the same
    spots, by factor of the read, and its first-in spots and limps."""

    observed: dict[str, float] = field(default_factory=dict)
    expected: dict[str, float] = field(default_factory=dict)
    spots: int = 0
    limps: int = 0

    def read(self) -> UserRead:
        low, high = READ_CLAMP
        factors = {
            name: min(
                high,
                max(low, (self.observed[name] + READ_PRIOR) / (self.expected[name] + READ_PRIOR)),
            )
            for name in self.expected
        }
        return UserRead(**factors, limp_width=max(0.1, (self.limps + 5) / (self.spots + 10)))

    def count(self, hand: GameState, user: int, read: UserRead):
        """Add the user's decisions in a finished hand, judged with the read it was dealt with."""
        states = replay_states(hand)
        for index, entry in enumerate(hand.history):
            if entry.seat != user:
                continue
            view = observation(states[index], user)
            kind = entry.action.type
            if view.street == Street.PREFLOP:
                self._preflop(view, kind)
                continue
            street = "flop" if view.street == Street.FLOP else "late"
            if legal_actions(view).can_check:
                self._add(f"bets_{street}", kind in _AGGRESSIVE, _bet_rate(view, read))
            else:
                opponents = sum(
                    view.dealt_in[s] and not view.folded[s] for s in range(view.config.num_seats) if s != user
                )
                self._add(f"folds_{street}", kind == ActionType.FOLD, 1 - _defense(view, opponents)[0])

    def _preflop(self, view: Observation, kind: ActionType):
        """The user's first decision before the flop, when it is first in or facing one open
        with no callers, deeper than the push/fold charts."""
        spot = preflop_spot(view)
        entries = [e for e in view.history if e.street == Street.PREFLOP]
        if (
            spot.stack <= PUSH_FOLD_BIG_BLINDS
            or _oversized(view)
            or any(e.seat == view.seat or e.action.type == ActionType.CALL for e in entries)
        ):
            return
        raisers = [spot.order.index(e.seat) for e in entries if e.action.type in _AGGRESSIVE]
        index = spot.order.index(view.seat)
        if not raisers:
            self.spots += 1
            if kind == ActionType.CALL:
                self.limps += 1
                return
            rows = strategy("open", spot.num_players, str(index), spot.stack, spot.bb_ante)
            self._add("opens", kind in _AGGRESSIVE, _combos([row[1] for row in rows]) / len(COMBOS))
        elif len(raisers) == 1:
            rows = strategy("respond", spot.num_players, f"{raisers[0]}-{index}", spot.stack, spot.bb_ante)
            self._add(
                "defends",
                kind in (ActionType.CALL, *_AGGRESSIVE),
                _combos([row[1] + row[2] for row in rows]) / len(COMBOS),
            )
            self._add("three_bets", kind in _AGGRESSIVE, _combos([row[2] for row in rows]) / len(COMBOS))

    def _add(self, name: str, happened: bool, expected: float):
        self.observed[name] = self.observed.get(name, 0.0) + happened
        self.expected[name] = self.expected.get(name, 0.0) + expected


def _bet_rate(view: Observation, read: UserRead) -> float:
    """How often a solid player would bet here, checked to, holding the user's range as Expert
    reads it: Hard's balanced play over that range, from the user's seat."""
    solid = ExpertBot("solid", PRESETS["balanced"], {view.hand_id: read}, view.seat)
    own = solid.seat_range(view, view.seat)
    board = set(view.board)
    holes = [i for i, w in enumerate(own) if w > 0 and not board.intersection(COMBOS[i])]
    policies = solid.policies(view, holes)
    total = sum(own[h] for h in policies)
    return sum(own[h] * (1 - p.get(AbstractAction.CHECK, 0.0)) for h, p in policies.items()) / total
