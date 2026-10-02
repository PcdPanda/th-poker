"""Bayesian range tracking (DESIGN.md Section 6.2): what every player can hold after its actions.

Bots are queried for the probability of their observed action with every hand they could hold,
so their ranges are exact given their policies. The user's range is tracked with a reference
policy (how a good opponent would read the user), where every bet or raise counts as one action
whatever its size and a small floor keeps unusual plays from emptying the range; the same reading
can be applied to every seat.
"""

from   dataclasses              import dataclass

import numpy as np

from   thpoker.bots.abstraction import legal_abstract_actions, to_action
from   thpoker.bots.bot         import Bot
from   thpoker.game.cards       import COMBOS
from   thpoker.game.engine      import observation, replay_states
from   thpoker.game.state       import Action, GameState

REFERENCE_FLOOR = 0.02
_HOLDING = [[i for i, combo in enumerate(COMBOS) if card in combo] for card in range(52)]


@dataclass(frozen=True)
class Snapshot:
    """Ranges just before a history action (the last snapshot follows the last action):
    normalized weights over the 1326 combos for every seat still in the hand."""

    state: GameState
    ranges: dict[int, np.ndarray]


def _likelihoods(bot: Bot, state: GameState, seat: int, holes: list[int], taken: Action, exact: bool) -> np.ndarray:
    """Chance that `seat` would take `taken` holding each combo in `holes`. `exact` matches the
    chip amount; otherwise every bet or raise counts as the same action."""
    view = observation(state, seat)
    concrete = {a: to_action(a, view) for a in legal_abstract_actions(view)}
    policies = bot.policies(view, holes)
    result = np.zeros(len(COMBOS))
    for hole, distribution in policies.items():
        total = 0.0
        for abstract, probability in distribution.items():
            action = concrete[abstract]
            # In one spot every bet or raise size shares its action type.
            matches = (action == taken) if exact else (action.type == taken.type)
            if matches:
                total += probability
        result[hole] = total
    return result


def track(
    hand: GameState,
    user_seat: int,
    bots: dict[int, Bot],
    reference: Bot,
    reference_view: bool = False,
) -> list[Snapshot]:
    """Snapshots before every action of a finished hand and after its last one, seen from
    `user_seat`: opponents' ranges exclude the user's cards and the board as it appears.

    Bots are read exactly with their own policies; the user is read with `reference`. With
    `reference_view`, every seat is read with `reference` the way the user is, which is how a
    good opponent would see the hand. Raises `ValueError` if a bot's observed action is
    impossible under its own policy (the hand was not played by `bots`).
    """
    states = replay_states(hand)
    user_cards = hand.hole_cards[user_seat]
    assert user_cards is not None
    live = [s for s in range(hand.config.num_seats) if hand.dealt_in[s]]
    ranges = {}
    for seat in live:
        weights = np.ones(len(COMBOS))
        if seat != user_seat:
            for card in user_cards:
                weights[_HOLDING[card]] = 0.0
        ranges[seat] = weights
    snapshots = []
    for index, state in enumerate(states):
        for weights in ranges.values():
            for card in state.board:
                weights[_HOLDING[card]] = 0.0
        in_hand = [s for s in live if not state.folded[s]]
        snapshots.append(Snapshot(state, {s: ranges[s] / ranges[s].sum() for s in in_hand}))
        if index == len(hand.history):
            break
        entry = hand.history[index]
        seat = entry.seat
        holes = [int(i) for i in np.flatnonzero(ranges[seat])]
        if seat == user_seat or reference_view:
            likelihood = _likelihoods(reference, state, seat, holes, entry.action, exact=False)
            likelihood = REFERENCE_FLOOR + (1 - REFERENCE_FLOOR) * likelihood
        else:
            likelihood = _likelihoods(bots[seat], state, seat, holes, entry.action, exact=True)
        updated = ranges[seat] * likelihood
        if updated.sum() <= 0:
            raise ValueError(f"seat {seat}'s action {entry.action} is impossible under its policy")
        ranges[seat] = updated
    return snapshots
