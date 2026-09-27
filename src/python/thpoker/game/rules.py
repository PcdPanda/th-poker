"""Betting rules: who must act and which actions are legal.

The functions read only public betting fields, so they accept a GameState or an Observation.
"""

from   collections.abc          import Sequence
from   typing                   import Protocol

from   thpoker.game.state       import (Action, ActionType, GameConfig,
                                        LegalActions)


class TableView(Protocol):
    """The public betting fields the rules read (GameState, Observation, or the engine's
    working copy)."""

    @property
    def hand_id(self) -> str: ...
    @property
    def config(self) -> GameConfig: ...
    @property
    def to_act(self) -> int | None: ...
    @property
    def current_bet(self) -> int: ...
    @property
    def last_raise_size(self) -> int: ...
    @property
    def stacks(self) -> Sequence[int]: ...
    @property
    def committed_this_street(self) -> Sequence[int]: ...
    @property
    def acted_at(self) -> Sequence[int | None]: ...
    @property
    def dealt_in(self) -> Sequence[bool]: ...
    @property
    def folded(self) -> Sequence[bool]: ...
    @property
    def all_in(self) -> Sequence[bool]: ...


class IllegalActionError(ValueError):
    """Raised when an action is not legal for the seat to act."""


def can_act(view: TableView, seat: int) -> bool:
    """The seat is still in the hand and has chips to bet."""
    return view.dealt_in[seat] and not view.folded[seat] and not view.all_in[seat]


def needs_to_act(view: TableView, seat: int) -> bool:
    """Whether the betting round still needs a decision from this seat.

    A seat that is the only one left able to bet only acts to match chips another live player
    actually committed; the nominal big blind alone does not force a decision when that
    blind was posted short.
    """
    if not can_act(view, seat):
        return False
    others = [s for s in range(view.config.num_seats) if s != seat]
    if not any(can_act(view, s) for s in others):
        live_commitments = [view.committed_this_street[s] for s in others if view.dealt_in[s] and not view.folded[s]]
        return view.committed_this_street[seat] < max(live_commitments, default=0)
    return view.acted_at[seat] is None or view.committed_this_street[seat] < view.current_bet


def legal_actions(view: TableView) -> LegalActions:
    """Legal actions for `view.to_act`. Raises IllegalActionError if no seat is to act.

    A raise is allowed only if another player can still respond, and, for a seat that has
    already acted this street, only if the total increase it faces since then is at least a
    full raise (so several short all-ins can add up and reopen betting).
    """
    seat = view.to_act
    if seat is None:
        raise IllegalActionError(f"No seat is to act in hand {view.hand_id}")
    stack = view.stacks[seat]
    committed = view.committed_this_street[seat]
    to_call = max(0, view.current_bet - committed)
    all_in_to = committed + stack
    others_can_respond = any(can_act(view, s) for s in range(view.config.num_seats) if s != seat)
    can_bet = can_raise = False
    min_to = max_to = 0
    if others_can_respond and stack > to_call:
        if view.current_bet == 0:
            can_bet = True
            min_to = min(view.config.big_blind, all_in_to)
            max_to = all_in_to
        else:
            acted_at = view.acted_at[seat]
            if acted_at is None or view.current_bet - acted_at >= view.last_raise_size:
                can_raise = True
                min_to = min(view.current_bet + view.last_raise_size, all_in_to)
                max_to = all_in_to
    return LegalActions(
        can_fold=to_call > 0,
        can_check=to_call == 0,
        can_call=to_call > 0,
        call_amount=min(to_call, stack),
        can_bet=can_bet,
        can_raise=can_raise,
        min_raise_to=min_to,
        max_raise_to=max_to,
    )


def check_action(view: TableView, action: Action) -> LegalActions:
    """Raise `IllegalActionError` naming the violated rule unless the action is legal."""
    legal = legal_actions(view)
    allowed = {
        ActionType.FOLD: legal.can_fold,
        ActionType.CHECK: legal.can_check,
        ActionType.CALL: legal.can_call,
        ActionType.BET: legal.can_bet,
        ActionType.RAISE: legal.can_raise,
    }[action.type]
    if not allowed:
        raise IllegalActionError(f"{action.type.value} is not legal for seat {view.to_act}")
    if action.type in (ActionType.BET, ActionType.RAISE):
        if action.amount is None or not legal.min_raise_to <= action.amount <= legal.max_raise_to:
            raise IllegalActionError(
                f"{action.type.value} to {action.amount} is outside "
                f"[{legal.min_raise_to}, {legal.max_raise_to}] for seat {view.to_act}"
            )
    elif action.amount is not None:
        raise IllegalActionError(f"{action.type.value} takes no amount, got {action.amount}")
    return legal
