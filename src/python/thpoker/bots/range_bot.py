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
from   dataclasses              import dataclass
import math
from   typing                   import Any

from   thpoker.bots.abstraction import (AbstractAction, NEVER_FOLD_EQUITY,
                                        POT_FRACTIONS, PUSH_FOLD_BIG_BLINDS,
                                        call_price, defense_share,
                                        effective_stack, finish, last_bet,
                                        legal_abstract_actions, sigmoid,
                                        usual_raises)
from   thpoker.bots.bot         import Policy, Style, TierBot
from   thpoker.bots.equity_bot  import (EquityBot, bet_shifted, preflop_range,
                                        public_seed)
from   thpoker.charts           import call_chart, push_chart, strategy
from   thpoker.game.cards       import COMBO_CLASS
from   thpoker.game.rng         import Rng
from   thpoker.game.rules       import legal_actions
from   thpoker.game.state       import ActionType, Observation, Street
from   thpoker.odds             import Range, range_equities, texture

_BLUFF_STREET_FACTOR = {Street.FLOP: 1.5, Street.TURN: 1.2, Street.RIVER: 1.0}
# Chart thresholds sit on a half-big-blind grid; mix across about that width.
_CHART_SOFTNESS_BB = 0.3


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


def chart_range(view: Observation, seat: int) -> Range | None:
    """The range the charts give `seat` for its preflop actions so far, or None when its line
    leaves the charted spots (limps, multiway pots, cold 4-bets, short-stack raises that are
    not all-in, raises of an unusual size)."""
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
            node = _chart_node(spot, raisers, callers, index, aggressive, action.type == ActionType.CALL, view)
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
        return [row[1] for row in strategy("open", spot.num_players, str(index), spot.stack, spot.bb_ante)]
    if len(raisers) == 1 and raisers[0] != index:
        rows = strategy("respond", spot.num_players, f"{raisers[0]}-{index}", spot.stack, spot.bb_ante)
        return [row[2] if aggressive else row[1] for row in rows] if (aggressive or called) else None
    if len(raisers) == 2 and raisers[0] == index:
        rows = strategy("versus_3bet", spot.num_players, f"{index}-{raisers[1]}", spot.stack, spot.bb_ante)
        return [row[2] if aggressive else row[1] for row in rows] if (aggressive or called) else None
    if len(raisers) == 3 and raisers[1] == index and raisers[2] == raisers[0] and called:
        rows = strategy("versus_allin", spot.num_players, f"{raisers[0]}-{index}", spot.stack, spot.bb_ante)
        return [row[1] for row in rows]
    return None


def seat_range(view: Observation, seat: int) -> Range:
    """A seat's range from its public actions: charts (or the Tier 2 bands off-chart) preflop,
    then shifted toward strong hands by each of its postflop bets and raises."""
    charted = chart_range(view, seat)
    # A line the charts give no weight to any hand leaves nothing to score against.
    weights = charted if charted is not None and sum(charted) > 1.0 else preflop_range(view, seat)
    return bet_shifted(view, seat, weights)


class RangeBot(TierBot):
    def __init__(self, name: str, style: Style):
        self.name = name
        self.style = style
        self._fallback = EquityBot(name, style)

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
        opponent_ranges = [seat_range(view, s) for s in opponents]
        own = seat_range(view, view.seat)
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
                / len(opponents),
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
            faced = last_bet(view)
            assert faced is not None  # after the flop, a seat that cannot check faces a bet
            bet_fraction = faced[1] / faced[0]
            defense = defense_share(view)
            if defense is None:
                defense = 1 - (bet_fraction / (1 + bet_fraction)) ** (1 / len(opponents))
            continue_share = min(
                1.0,
                max(
                    0.05,
                    defense
                    + 0.15 * (style.call_down_tendency - 0.5)
                    - 0.1 * style.fold_to_pressure * max(0.0, bet_fraction - 0.75),
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


def _size_action(pot_fraction: float) -> AbstractAction:
    """The abstract bet closest to a pot fraction."""
    return min(POT_FRACTIONS, key=lambda a: abs(POT_FRACTIONS[a] - pot_fraction))
