"""Tier 2: compares its equity against assumed opponent ranges with the price it is offered.

Each opponent's range is assumed from its public actions: its strongest preflop action sets a
starting range, and every later bet or raise shifts weight toward hands that are strong on the
board at the time. Scores are equity margins (equity minus what the action needs, adjusted by
style); a softmax with the style's temperature turns them into a mixed strategy. Weak hands
also bluff at a controlled rate when checked to.
"""

from   dataclasses              import dataclass
from   functools                import lru_cache
import hashlib
import math
from   thpoker.bots.abstraction import (AbstractAction, NEVER_FOLD_EQUITY,
                                        acts_last, bet_sizes, call_price,
                                        effective_stack, finish,
                                        legal_abstract_actions, players_behind,
                                        position_width, raises_this_street,
                                        usual_raises)
from   thpoker.bots.bot         import Policy, Style, TierBot
from   thpoker.game.cards       import COMBOS, COMBO_CLASS
from   thpoker.game.evaluator   import evaluate_combos
from   thpoker.game.rng         import Rng
from   thpoker.game.rules       import can_act, legal_actions
from   thpoker.game.state       import (ActionType, LegalActions, Observation,
                                        Street)
from   thpoker.odds             import (FULL_RANGE, Range, equity_to_reach_top,
                                        equity_vs_random, range_equities,
                                        ranked_range, representative_equities)

RERAISE_TOPS = (0.06, 0.025)
# Assumed preflop ranges of a typical opponent, as preflop percentile bands.
_THREE_BET_TOP = 0.06
_FOUR_BET_TOP = 0.025
# The narrowest read of an oversized raise, reached 50bb deep: read tighter, a defender calls
# a 100bb shove so rarely that shoving any two cards profits; this width leaves a calling range
# (about 88+, AJ+) that makes a deep shove worse than a normal raise for weak and strong hands.
_TIGHTEST_TOP = 0.15
_DEEP_BIG_BLINDS = 25  # up to this deep, an oversized raise reads like a normal one
_FULL_DEPTH_BIG_BLINDS = 50  # from this deep, it reads down to the narrowest width
_CALL_RANGE = ranked_range(0.04, 0.3)
_LIMP_RANGE = ranked_range(0.1, 0.6)
_CHECKED_OPTION_RANGE = ranked_range(0.08, 1.0)
_STREET_CARDS = {Street.FLOP: 3, Street.TURN: 4, Street.RIVER: 5}


def public_seed(view: Observation) -> int:
    """A seed from public cards only, so every hypothetical hole-card pair sees the same sampled
    boards (DESIGN.md Section 5.1), and every decision on a street shares them, which lets the
    analysis reuse each board's hand values."""
    digest = hashlib.blake2b(repr((view.hand_id, view.board)).encode(), digest_size=8)
    return int.from_bytes(digest.digest(), "big")


def _open_width(view: Observation, opener: int) -> float:
    """The share of hands a typical first raise holds: wider the fewer dealt-in seats act after
    the opener preflop."""
    behind = 0
    seat = opener
    while seat != view.big_blind_seat:
        seat = (seat + 1) % view.config.num_seats
        behind += view.dealt_in[seat]
    return min(0.5, position_width(0.2, behind))


def preflop_range(
    view: Observation, seat: int, limp: Range = _LIMP_RANGE, reraise_tops: tuple[float, float] = RERAISE_TOPS
) -> Range:
    """A seat's range from its preflop actions: percentile bands by its strongest action, with
    `limp` for a limp. A raise bigger than usual from a deep stack reads as a stronger range, in
    proportion to its size (a 100bb open-shove is not a 2.5bb open), phased in between 25bb and
    50bb deep since short stacks shove as a matter of course; it never reads looser than the
    usual band."""
    big_blind = view.config.big_blind
    sizes = iter(usual_raises(view))
    raises = 0
    strongest: Range | None = None
    for entry in view.history:
        if entry.street != Street.PREFLOP:
            continue
        kind = entry.action.type
        if kind in (ActionType.BET, ActionType.RAISE):
            _, amount, usual = next(sizes)
            if entry.seat == seat:
                top = _open_width(view, seat) if raises == 0 else reraise_tops[min(raises, 2) - 1]
                depth = effective_stack(view, seat) / big_blind
                if amount > usual and depth > _DEEP_BIG_BLINDS:
                    ramp = min(
                        1.0,
                        (depth - _DEEP_BIG_BLINDS) / (_FULL_DEPTH_BIG_BLINDS - _DEEP_BIG_BLINDS),
                    )
                    floor = top + (_TIGHTEST_TOP - top) * ramp
                    top = min(top, max(floor, top * usual / amount))
                strongest = ranked_range(0.0, top)
            raises += 1
        elif entry.seat == seat and kind == ActionType.CALL and strongest is None:
            strongest = _CALL_RANGE if raises else limp
        elif entry.seat == seat and kind == ActionType.CHECK and strongest is None:
            strongest = _CHECKED_OPTION_RANGE
    return strongest if strongest is not None else FULL_RANGE


@lru_cache(maxsize=128)
def strength_weighted(weights: Range, board: tuple[int, ...], floor: float = 0.25) -> Range:
    """Shift a range toward hands that are strong on `board`: weight times floor + (1 - floor) *
    rank, where rank is the hand's made-hand percentile (0 weakest, 1 strongest) within the
    range. Cached: every bot decision on a street rebuilds the same seat ranges."""
    live = [i for i, (a, b) in enumerate(COMBOS) if weights[i] and a not in board and b not in board]
    if len(live) < 2:
        return weights
    values = evaluate_combos(board, [COMBOS[i] for i in live])
    order = sorted(range(len(live)), key=values.__getitem__)
    shifted = list(weights)
    start = 0
    while start < len(order):
        end = start
        while end + 1 < len(order) and values[order[end + 1]] == values[order[start]]:
            end += 1
        # Equal hands share their group's midpoint rank, so suits do not tilt the weights.
        rank = (start + end) / 2 / (len(live) - 1)
        for position in order[start : end + 1]:
            shifted[live[position]] *= floor + (1 - floor) * rank
        start = end + 1
    return tuple(shifted)


def bet_shifted(view: Observation, seat: int, weights: Range, floors: tuple[float, float] = (0.25, 0.25)) -> Range:
    """`weights` shifted toward strong hands by each of the seat's bets and raises after the
    flop, with `strength_weighted`'s floor for flop bets and for later ones. Only public cards
    count: the range must not depend on whose hand is being scored."""
    for entry in view.history:
        if (
            entry.seat == seat
            and entry.street != Street.PREFLOP
            and entry.action.type in (ActionType.BET, ActionType.RAISE)
        ):
            floor = floors[entry.street != Street.FLOP]
            weights = strength_weighted(weights, view.board[: _STREET_CARDS[entry.street]], floor)
    return weights


def assumed_ranges(view: Observation) -> list[Range]:
    """One range per opponent still in the hand, from that opponent's public actions."""
    return [
        bet_shifted(view, seat, preflop_range(view, seat))
        for seat in range(view.config.num_seats)
        if seat != view.seat and view.dealt_in[seat] and not view.folded[seat]
    ]


def _facing_bet_scores(
    equity: float,
    price: float,
    *,
    realization: float,
    raise_bar: float,
    call_adjustment: float,
    style_edge: float,
    aggressive: AbstractAction,
) -> dict[AbstractAction, float]:
    """Call when the equity a call realizes beats the price (out of position realizes less);
    raise instead when the equity also clears `raise_bar`. Folding wins only below the price."""
    call = equity * realization - price + call_adjustment
    return {
        AbstractAction.FOLD: 0.0,
        AbstractAction.CALL: call,
        aggressive: call + equity - raise_bar + style_edge,
    }


@dataclass(frozen=True)
class _Context:
    """Everything about a decision except the hand's own equity."""

    kind: str  # "unopened", "preflop", or "postflop"
    view: Observation
    ranges: list[Range]
    legal: LegalActions
    aggressive: AbstractAction
    price: float
    final_call: bool
    enter: float  # unopened: equity against any two cards needed to play
    raise_first: float  # unopened: equity needed to raise
    strongest: Range  # preflop: the range the hand is measured against
    abstract_legal: list[AbstractAction]  # legal abstract actions, computed once


class EquityBot(TierBot):
    def __init__(self, name: str, style: Style, reraise_tops: tuple[float, float] = RERAISE_TOPS):
        self.name = name
        self.style = style
        self.reraise_tops = reraise_tops

    def _context(self, view: Observation) -> _Context:
        ranges = assumed_ranges(view)
        legal = legal_actions(view)
        abstract_legal = legal_abstract_actions(view)
        raises = raises_this_street(view)
        preflop = view.street == Street.PREFLOP
        aggressive = (AbstractAction.RERAISE if raises else AbstractAction.OPEN) if preflop else AbstractAction.BET_75
        price = call_price(view) if legal.can_call else 0.0
        # When this call ends the betting, the hand realizes all of its equity.
        others_can_bet = any(can_act(view, s) for s in range(view.config.num_seats) if s != view.seat)
        final_call = legal.call_amount >= view.stacks[view.seat] or not others_can_bet
        kind, enter, raise_first, strongest = "postflop", 0.0, 0.0, FULL_RANGE
        if preflop and raises == 0 and legal.can_call:
            # Unopened: rank the hand heads-up against any two cards and play the style's share
            # of hands for this position (multiplying equities across players behind would
            # badly understate multiway equity preflop).
            behind = players_behind(view)
            kind = "unopened"
            enter = equity_to_reach_top(position_width(self.style.vpip_target, behind))
            raise_first = equity_to_reach_top(position_width(self.style.pfr_target, behind))
        elif preflop:
            # Facing a raise (or checked to in the big blind): equity against the strongest
            # range only, with a penalty for each other player still in.
            kind = "preflop"
            if view.last_aggressor is not None:
                strongest = preflop_range(view, view.last_aggressor, reraise_tops=self.reraise_tops)
            else:
                strongest = _LIMP_RANGE
        return _Context(
            kind,
            view,
            ranges,
            legal,
            aggressive,
            price,
            final_call,
            enter,
            raise_first,
            strongest,
            abstract_legal,
        )

    def _equities(self, context: _Context, holes: list[int]) -> dict[int, float]:
        view = context.view
        if context.kind == "unopened":
            return {h: equity_vs_random(COMBOS[h]) for h in holes}
        if context.kind == "preflop":
            # Preflop decisions go by hand class: each class is measured from one fixed combo,
            # so range tracking can score 169 classes instead of 1326 combos.
            by_class = representative_equities(context.strongest, {COMBO_CLASS[h] for h in holes})
            return {h: by_class[COMBO_CLASS[h]] for h in holes}
        # Rounded, so hands of nearly equal strength share one decision (and its computation).
        equities = range_equities(view.board, holes, context.ranges, Rng(public_seed(view)))
        return {h: round(e, 3) for h, e in equities.items()}

    def decisions(self, view: Observation, holes: list[int]) -> dict[int, Policy]:
        """One context and one equity pass serve every hand."""
        context = self._context(view)
        decided: dict[float, Policy] = {}
        result = {}
        for hole, strength in self._equities(context, holes).items():
            if strength not in decided:  # preflop, hands of a class share their strength
                decided[strength] = self._decide(context, strength)
            result[hole] = decided[strength]
        return result

    def _decide(self, context: _Context, strength: float) -> Policy:
        style, view, legal, aggressive = self.style, context.view, context.legal, context.aggressive
        opponents = len(context.ranges)
        style_edge = 0.05 * (style.aggression - 1)
        temperature = 0.03 + 0.15 * style.temperature
        if context.kind == "unopened":
            enter = strength - context.enter
            scores = {
                AbstractAction.FOLD: 0.0,
                # Limping gets less attractive the more of its entered hands the style raises.
                AbstractAction.CALL: enter - 0.02 * style.pfr_target / style.vpip_target,
                aggressive: enter + strength - context.raise_first,
            }
            # Adjacent hands differ by less than 0.01 in this equity, so sharpen the boundary.
            temperature /= 6
            rule = "unopened"
        elif context.kind == "preflop":
            crowd = 0.03 * (opponents - 1)
            if legal.can_check:
                scores = {
                    AbstractAction.CHECK: 0.0,
                    aggressive: strength - 0.55 - crowd + style_edge,
                }
                rule = "checked to"
            else:
                in_position = acts_last(view, view.seat)
                scores = _facing_bet_scores(
                    strength,
                    context.price,
                    realization=1.0 if context.final_call else 0.9 if in_position else 0.75,
                    raise_bar=0.55 + crowd,
                    call_adjustment=0.08 * (style.call_down_tendency - 0.5) - crowd,
                    style_edge=style_edge,
                    aggressive=aggressive,
                )
                rule = "facing bet"
        elif legal.can_check:
            threshold = 0.5 + 0.08 * (opponents - 1)
            scores = {AbstractAction.CHECK: 0.0, aggressive: strength - threshold + style_edge}
            rule = "checked to"
        else:
            full_realization = context.final_call or view.street == Street.RIVER or acts_last(view, view.seat)
            pressure = 0.1 * style.fold_to_pressure * max(0.0, context.price - 0.25)
            scores = _facing_bet_scores(
                strength,
                context.price,
                realization=1.0 if full_realization else 0.9,
                raise_bar=0.6 + 0.08 * (opponents - 1),
                call_adjustment=0.08 * (style.call_down_tendency - 0.5) - pressure,
                style_edge=style_edge,
                aggressive=aggressive,
            )
            rule = "facing bet"
        top = max(scores.values())
        powered = {a: math.exp((s - top) / temperature) for a, s in scores.items()}
        total = sum(powered.values())
        distribution = {a: p / total for a, p in powered.items()}
        if rule == "checked to" and context.kind == "postflop" and scores[aggressive] < 0:
            # Below the value threshold (draws included), bet a fixed share as a bluff.
            bluff = min(0.35, 0.1 * style.bluff_multiplier) / opponents
            distribution[aggressive] = max(distribution[aggressive], bluff)
            distribution[AbstractAction.CHECK] = 1.0 - distribution[aggressive]
        if rule == "facing bet" and strength >= NEVER_FOLD_EQUITY:
            distribution[AbstractAction.FOLD] = 0.0
        if context.kind == "postflop":
            share = distribution.pop(aggressive)
            large = min(1.0, style.sizing_preference + (0.3 if strength > 0.75 else 0.0))
            for size, weight in bet_sizes(large).items():
                distribution[size] = share * weight
        rationale = {
            "rule_triggered": rule,
            "equity_estimate": round(strength, 4),
            "opponents": opponents,
            "required_equity": round(context.price, 4),
        }
        return finish(distribution, view, context.abstract_legal), rationale
