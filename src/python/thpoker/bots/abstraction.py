"""The small abstract action set bots choose from, and its mapping to legal chip amounts."""

from   collections.abc          import Mapping
import enum
from   functools                import lru_cache
import math
from   thpoker.game.rules       import legal_actions
from   thpoker.game.state       import (Action, ActionType, AnteType,
                                        GameState, LegalActions, Observation,
                                        Street)
from   types                    import MappingProxyType


class AbstractAction(enum.StrEnum):
    FOLD = "FOLD"
    CHECK = "CHECK"
    CALL = "CALL"
    OPEN = "OPEN"  # preflop first raise: 2.5bb plus 1bb per limper
    RERAISE = "RERAISE"  # preflop re-raise: 3x in position, 4x out of position, plus 1x per caller
    BET_33 = "BET_33"  # postflop bet or raise sized as a fraction of the pot
    BET_50 = "BET_50"
    BET_75 = "BET_75"
    BET_100 = "BET_100"
    BET_150 = "BET_150"
    ALL_IN = "ALL_IN"


AGGRESSIVE = frozenset(AbstractAction) - {
    AbstractAction.FOLD,
    AbstractAction.CHECK,
    AbstractAction.CALL,
}
POT_FRACTIONS = {
    AbstractAction.BET_33: 0.33,
    AbstractAction.BET_50: 0.5,
    AbstractAction.BET_75: 0.75,
    AbstractAction.BET_100: 1.0,
    AbstractAction.BET_150: 1.5,
}
# Below this share of the pot after the bet, keeping chips back has no strategic value.
_ALL_IN_REMAINDER = 0.25
_WIDTH_SCALE = 6.0
# At or below this effective stack (in big blinds), preflop play is fold, call, or all-in.
PUSH_FOLD_BIG_BLINDS = 15


def raises_this_street(view: GameState | Observation) -> int:
    return sum(
        1
        for entry in view.history
        if entry.street == view.street and entry.action.type in (ActionType.BET, ActionType.RAISE)
    )


def call_price(view: Observation) -> float:
    """What calling costs as a share of the pot after the call, counting only chips this seat
    can win: an all-in call for less competes for less than the whole pot."""
    call = legal_actions(view).call_amount
    mine = view.committed_total[view.seat] + call
    winnable = sum(min(committed, mine) for seat, committed in enumerate(view.committed_total) if seat != view.seat)
    return call / (mine + winnable + view.dead_money)


def last_bet(view: Observation) -> tuple[int, int] | None:
    """The pot before this street's last bet or raise and the chips that bet added, replayed
    from the street's actions (preflop the blinds count as posted, not as bets); None when
    nobody has bet or raised on this street. Street commitments only grow, so each replayed
    one is capped at the seat's commitment now: that covers short blinds and calls for less."""
    now = view.committed_this_street
    committed = [0] * view.config.num_seats
    level = 0
    if view.street == Street.PREFLOP:
        for seat, blind in (
            (view.small_blind_seat, view.config.small_blind),
            (view.big_blind_seat, view.config.big_blind),
        ):
            committed[seat] = min(blind, now[seat])
        # A limper pays the full big blind even when it was posted short.
        level = view.config.big_blind
    carried = view.pot - sum(now)
    found = None
    for entry in view.history:
        if entry.street != view.street:
            continue
        if entry.action.type in (ActionType.BET, ActionType.RAISE):
            amount = entry.action.amount
            assert amount is not None
            found = (carried + sum(committed), amount - committed[entry.seat])
            committed[entry.seat] = level = min(amount, now[entry.seat])
        elif entry.action.type == ActionType.CALL:
            committed[entry.seat] = min(level, now[entry.seat])
    return found


def bet_fractions(view: Observation) -> dict[int, float]:
    """Each postflop bet or raise in the history, by index: the chips it added over the pot
    before it, as `last_bet` measures it. One pass forward from the antes and blinds, each call
    capped at what the seat had left, since range reads call this at every opponent node of a
    review."""
    n = view.config.num_seats
    start = view.starting_stacks
    spent = [0] * n  # chips each seat put in on earlier streets, antes included
    if view.config.ante_type == AnteType.PER_PLAYER:
        spent = [min(view.config.ante, start[s]) if view.dealt_in[s] else 0 for s in range(n)]
    spent[view.big_blind_seat] += view.dead_money
    pot = sum(spent)
    committed = {
        seat: min(blind, start[seat] - spent[seat])
        for seat, blind in (
            (view.small_blind_seat, view.config.small_blind),
            (view.big_blind_seat, view.config.big_blind),
        )
    }
    level, street = view.config.big_blind, Street.PREFLOP
    fractions = {}
    for index, entry in enumerate(view.history):
        if entry.street != street:
            for seat, chips in committed.items():
                spent[seat] += chips
                pot += chips
            committed, level, street = {}, 0, entry.street
        if entry.action.type in (ActionType.BET, ActionType.RAISE):
            amount = entry.action.amount
            assert amount is not None
            if street != Street.PREFLOP:
                added = amount - committed.get(entry.seat, 0)
                fractions[index] = added / (pot + sum(committed.values()))
            committed[entry.seat] = level = amount
        elif entry.action.type == ActionType.CALL:
            committed[entry.seat] = min(level, start[entry.seat] - spent[entry.seat])

    return fractions


def stack_to_pot(view: GameState | Observation, seat: int) -> float:
    """The chips `seat` can still put in against the deepest live opponent with chips behind,
    over the pot once it has called; 0 when nobody can bet against it."""
    level = max(view.committed_this_street)
    call = min(level - view.committed_this_street[seat], view.stacks[seat])
    others = [
        view.stacks[s] - min(level - view.committed_this_street[s], view.stacks[s])
        for s in range(view.config.num_seats)
        if s != seat and view.dealt_in[s] and not view.folded[s] and not view.all_in[s]
    ]
    return min(view.stacks[seat] - call, max(others, default=0)) / (view.pot + call)


def defense_share(view: Observation) -> float | None:
    """Minimum defense frequency against this street's last bet or raise, for each player
    facing it: together they must continue often enough that a pure bluff of that size breaks
    even, so each continues 1 - (b / (1 + b)) ** (1 / n). None when nobody has bet, or when
    someone has already called, since the bluff has then already failed."""
    bet = last_bet(view)
    if bet is None:
        return None
    street = [e for e in view.history if e.street == view.street]
    last = max(i for i, e in enumerate(street) if e.action.type in (ActionType.BET, ActionType.RAISE))
    if any(e.action.type == ActionType.CALL for e in street[last + 1 :]):
        return None
    pot, risked = bet
    facing = sum(
        1
        for s in range(view.config.num_seats)
        if view.dealt_in[s]
        and not view.folded[s]
        and not view.all_in[s]
        and view.committed_this_street[s] < view.current_bet
    )
    return 1 - (risked / (pot + risked)) ** (1 / max(1, facing))


def players_behind(view: Observation) -> int:
    """Live seats that act after this seat preflop, up to and including the big blind."""
    count = 0
    seat = view.seat
    while seat != view.big_blind_seat:
        seat = (seat + 1) % view.config.num_seats
        count += view.dealt_in[seat] and not view.folded[seat] and not view.all_in[seat]
    return count


def position_width(average: float, behind: int) -> float:
    """The share of hands to play first in from a seat with `behind` players left to act, for a
    style that plays `average` of its hands overall. The constant is calibrated so measured
    six-handed VPIP/PFR match the style targets (DESIGN.md Section 14, decision 7)."""
    return min(0.95, average * _WIDTH_SCALE / (behind + 1.5))


def effective_stack(view: Observation, seat: int | None = None) -> int:
    """A seat's stack (this seat's by default) against the deepest opponent still in when it
    first acted preflop, or now if it has not acted: a 100bb button facing 10bb blinds plays
    10bb deep, whatever the players who folded before it held. Stacks as the hand started."""
    seat = view.seat if seat is None else seat
    folded = set()
    for entry in view.history:
        if entry.street != Street.PREFLOP or entry.seat == seat:
            break
        if entry.action.type == ActionType.FOLD:
            folded.add(entry.seat)
    others = [
        view.starting_stacks[s]
        for s in range(view.config.num_seats)
        if s != seat and view.dealt_in[s] and s not in folded
    ]
    return min(view.starting_stacks[seat], max(others, default=view.starting_stacks[seat]))


def usual_raises(view: Observation) -> list[tuple[int, int, int]]:
    """For each preflop bet or raise so far: (seat, chips it raised to, the most the bots
    themselves would raise to there). The bots open to 2.5bb plus 1bb per limper and re-raise to
    3x in position or 4x out of it plus 1x per caller, so larger sizes are unusual."""
    big_blind = view.config.big_blind
    level, callers, raised = big_blind, 0, False
    result = []
    for entry in view.history:
        if entry.street != Street.PREFLOP:
            continue
        if entry.action.type in (ActionType.BET, ActionType.RAISE):
            amount = entry.action.amount
            assert amount is not None
            usual = level * (4 + callers) if raised else round(2.5 * big_blind) + callers * big_blind
            result.append((entry.seat, amount, usual))
            level, callers, raised = amount, 0, True
        elif entry.action.type == ActionType.CALL:
            callers += 1
    return result


def legal_abstract_actions(view: Observation) -> list[AbstractAction]:
    """Abstract actions available to the seat to act, in enum order."""
    legal = legal_actions(view)
    actions = []
    if legal.can_fold:
        actions.append(AbstractAction.FOLD)
    if legal.can_check:
        actions.append(AbstractAction.CHECK)
    if legal.can_call:
        actions.append(AbstractAction.CALL)
    if legal.can_bet or legal.can_raise:
        if view.street == Street.PREFLOP:
            if effective_stack(view) > PUSH_FOLD_BIG_BLINDS * view.config.big_blind:
                opened = raises_this_street(view) > 0
                actions.append(AbstractAction.RERAISE if opened else AbstractAction.OPEN)
        else:
            actions.extend(POT_FRACTIONS)
        actions.append(AbstractAction.ALL_IN)
    return actions


def fit_to_legal(
    weights: dict[AbstractAction, float],
    view: Observation,
    legal: list[AbstractAction] | None = None,
) -> dict[AbstractAction, float]:
    """Weights over exactly the legal abstract actions, in enum order. Raise mass moves to the
    first legal aggressive action (all-in when short stacks may only shove), or to call or
    check when raising is not allowed. `legal` saves recomputing `legal_abstract_actions`."""
    if legal is None:
        legal = legal_abstract_actions(view)
    targets = _targets(tuple(legal))
    result: dict[AbstractAction, float] = dict.fromkeys(legal, 0.0)

    for action, weight in weights.items():
        if weight > 0:
            result[targets[action]] += weight
    return result


@lru_cache(maxsize=64)
def _targets(legal: tuple[AbstractAction, ...]) -> Mapping[AbstractAction, AbstractAction]:
    """Where `fit_to_legal` moves each abstract action's weight (read-only: the cache shares it)."""
    aggressive_legal = [a for a in legal if a in AGGRESSIVE]
    passive = AbstractAction.CALL if AbstractAction.CALL in legal else AbstractAction.CHECK
    return MappingProxyType(
        {
            action: action
            if action in legal
            else aggressive_legal[0]
            if action in AGGRESSIVE and aggressive_legal
            else passive
            for action in AbstractAction
        }
    )


# Facing a bet, hands at least this far ahead of their range never fold.
NEVER_FOLD_EQUITY = 0.8
# Actions below this share of the mix are dropped: softmax tails would otherwise let aces fold.
MIN_SHARE = 0.01


def finish(
    weights: dict[AbstractAction, float],
    view: Observation,
    legal: list[AbstractAction] | None = None,
) -> dict[AbstractAction, float]:
    """Legal actions only, dropping mixes under MIN_SHARE."""
    fitted = fit_to_legal(weights, view, legal)
    total = sum(fitted.values())
    kept = {a: w for a, w in fitted.items() if w >= MIN_SHARE * total}
    kept_total = sum(kept.values())
    return {a: w / kept_total for a, w in kept.items()}


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-50.0, min(50.0, x))))


def acts_last(state: GameState | Observation, seat: int) -> bool:
    """Whether `seat` acts after every other live player on the streets after the flop."""
    n = state.config.num_seats
    mine = (seat - state.button - 1) % n
    return all(
        mine > (s - state.button - 1) % n for s in range(n) if s != seat and state.dealt_in[s] and not state.folded[s]
    )


def bet_sizes(sizing_preference: float) -> dict[AbstractAction, float]:
    """Shares of a bet among pot fractions: 0 means all small (1/3, 1/2), 1 all large (3/4, 1)."""
    return {
        AbstractAction.BET_33: 0.5 * (1 - sizing_preference),
        AbstractAction.BET_50: 0.5 * (1 - sizing_preference),
        AbstractAction.BET_75: 0.6 * sizing_preference,
        AbstractAction.BET_100: 0.4 * sizing_preference,
    }


def _in_position(view: Observation, seat: int, other: int) -> bool:
    n = view.config.num_seats
    return (seat - view.button - 1) % n > (other - view.button - 1) % n


def _matching_seats(view: Observation, excluded: set[int | None]) -> int:
    """Live seats, other than `excluded`, whose street commitment matches the current bet."""
    return sum(
        1
        for s in range(view.config.num_seats)
        if s not in excluded
        and view.dealt_in[s]
        and not view.folded[s]
        and view.committed_this_street[s] == view.current_bet
    )


def _target(abstract: AbstractAction, view: Observation, legal: LegalActions) -> int:
    seat = view.seat
    big_blind = view.config.big_blind
    if abstract == AbstractAction.ALL_IN:
        return legal.max_raise_to
    if abstract == AbstractAction.OPEN:
        limpers = _matching_seats(view, {seat, view.big_blind_seat})
        return round(2.5 * big_blind) + limpers * big_blind
    if abstract == AbstractAction.RERAISE:
        aggressor = view.last_aggressor
        callers = _matching_seats(view, {seat, aggressor})
        in_position = aggressor is not None and _in_position(view, seat, aggressor)
        return view.current_bet * (3 if in_position else 4) + callers * view.current_bet
    fraction = POT_FRACTIONS[abstract]
    to_call = view.current_bet - view.committed_this_street[seat]
    return view.current_bet + round(fraction * (view.pot + to_call))


def to_action(abstract: AbstractAction, view: Observation) -> Action:
    """The concrete legal action for an abstract one. Sizes are clamped to the legal range and
    become all-in when the chips left behind would be small relative to the pot. Raises
    `ValueError` if the abstract action is not available."""
    if abstract not in legal_abstract_actions(view):
        raise ValueError(f"{abstract.value} is not available to seat {view.seat}")
    if abstract == AbstractAction.FOLD:
        return Action(ActionType.FOLD)
    if abstract == AbstractAction.CHECK:
        return Action(ActionType.CHECK)
    if abstract == AbstractAction.CALL:
        return Action(ActionType.CALL)
    legal = legal_actions(view)
    target = min(max(_target(abstract, view, legal), legal.min_raise_to), legal.max_raise_to)
    pot_after = view.pot + target - view.committed_this_street[view.seat]
    if legal.max_raise_to - target < _ALL_IN_REMAINDER * pot_after:
        target = legal.max_raise_to
    return Action(ActionType.BET if legal.can_bet else ActionType.RAISE, target)
