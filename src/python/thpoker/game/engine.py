"""The hand engine: a pure state machine from `new_hand` through actions to a settled hand, with
main and side pot construction and pot splitting among tied winners.

Events carry full information (including every seat's hole cards) because they are what the
log stores for review. Interfaces show a player only `observation(...)` during play.
"""

from __future__ import annotations

from   dataclasses              import dataclass

from   thpoker.game.evaluator   import describe, evaluate
from   thpoker.game.rng         import Rng
from   thpoker.game.rules       import (IllegalActionError, check_action,
                                        needs_to_act)
from   thpoker.game.state       import (Action, ActionType, AnteType,
                                        ConfigError, Event, GameConfig,
                                        GameState, HistoryEntry, Observation,
                                        PotAward, Street)

_NEXT_STREET = {Street.PREFLOP: Street.FLOP, Street.FLOP: Street.TURN, Street.TURN: Street.RIVER}
_CARDS_DEALT = {Street.FLOP: 3, Street.TURN: 1, Street.RIVER: 1}


def new_hand(
    config: GameConfig,
    seed: int,
    button: int,
    stacks: tuple[int, ...],
    dealt_in: tuple[bool, ...] | None = None,
    hand_id: str = "hand",
    deck: tuple[int, ...] | None = None,
) -> tuple[GameState, list[Event]]:
    """Post antes and blinds, deal hole cards, and return the state waiting for the first action.

    `dealt_in` defaults to every seat with chips. `hand_id` is visible to every player, so it
    must not reveal the seed. `deck` fixes the card order (for replay and tests); otherwise it
    is shuffled from `seed`. Hole cards are dealt one at a time starting
    left of the button, then the board comes off the top of the remaining deck.
    Raises `ConfigError` for inconsistent seats, stacks, button, or deck.
    """
    n = config.num_seats
    if len(stacks) != n or any(s < 0 for s in stacks):
        raise ConfigError(f"need {n} non-negative stacks, got {stacks}")
    dealt = tuple(dealt_in) if dealt_in is not None else tuple(s > 0 for s in stacks)
    if len(dealt) != n or any(d and s == 0 for d, s in zip(dealt, stacks)):
        raise ConfigError(f"dealt-in seats {dealt} must have chips {stacks}")
    if sum(dealt) < 2:
        raise ConfigError(f"a hand needs at least two dealt-in seats, got {dealt}")
    if not 0 <= button < n or not dealt[button]:
        raise ConfigError(f"button seat {button} is not dealt in")
    cards = list(deck) if deck is not None else _shuffled(seed)
    if sorted(cards) != list(range(52)):
        raise ConfigError("deck must be an ordering of all 52 cards")

    order = _deal_order(button, dealt)
    if len(order) == 2:
        small_blind_seat, big_blind_seat = button, order[0]
    else:
        small_blind_seat, big_blind_seat = order[0], order[1]

    hand = _Hand.start(
        config,
        seed,
        hand_id,
        (button, small_blind_seat, big_blind_seat),
        stacks,
        dealt,
        cards,
    )
    events = [
        Event(
            "HandStarted",
            {
                "hand_id": hand.hand_id,
                "seed": seed,
                "config": config.to_dict(),
                "button": button,
                "stacks": list(stacks),
                "dealt_in": list(dealt),
            },
        )
    ]
    antes = [0] * n
    if config.ante_type == AnteType.PER_PLAYER:
        for seat in order:
            antes[seat] = hand.post_ante(seat, config.ante)
    small_blind = hand.post_blind(small_blind_seat, config.small_blind)
    big_blind = hand.post_blind(big_blind_seat, config.big_blind)
    if config.ante_type == AnteType.BIG_BLIND_ANTE:
        # The big blind takes priority over the ante when the stack cannot cover both.
        antes[big_blind_seat] = hand.post_big_blind_ante(big_blind_seat, config.ante)
    if any(antes):
        events.append(Event("AntesPosted", {"amounts": antes}))
    events.append(
        Event(
            "BlindsPosted",
            {
                "small_blind_seat": small_blind_seat,
                "small_blind": small_blind,
                "big_blind_seat": big_blind_seat,
                "big_blind": big_blind,
            },
        )
    )
    hand.deal_hole_cards(order)
    events.append(Event("HoleCardsDealt", {"hole_cards": [_card_list(h) for h in hand.hole_cards]}))
    first = small_blind_seat if len(order) == 2 else (big_blind_seat + 1) % n
    hand.advance(first, events)
    return hand.freeze(), events


def _shuffled(seed: int) -> list[int]:
    return Rng(seed).derive("deck").permutation(52)


def _deal_order(button: int, dealt: tuple[bool, ...]) -> list[int]:
    """Dealt-in seats from the one left of the button round to the button."""
    n = len(dealt)
    return [(button + k) % n for k in range(1, n + 1) if dealt[(button + k) % n]]


def dealt_hole_cards(seed: int, button: int, dealt_in: tuple[bool, ...], seat: int) -> tuple[int, int]:
    """The hole cards `new_hand` deals `seat` from `seed`'s deck, without dealing the hand."""
    order = _deal_order(button, dealt_in)
    cards = _shuffled(seed)
    position = order.index(seat)
    return cards[position], cards[position + len(order)]


def apply_action(state: GameState, action: Action) -> tuple[GameState, list[Event]]:
    """Apply the action of `state.to_act` and return the new state; `state` is not modified.

    Closing a betting round deals the next street automatically; when no further betting is
    possible the board runs out and the hand settles. Raises `IllegalActionError`.
    """
    if state.street == Street.COMPLETE:
        raise IllegalActionError(f"hand {state.hand_id} is complete")
    check_action(state, action)
    hand = _Hand(state)
    events: list[Event] = []
    seat = hand.act(action, events)
    live = [s for s in range(state.config.num_seats) if hand.dealt_in[s] and not hand.folded[s]]
    if len(live) == 1:
        hand.finish_uncontested(live[0], events)
    else:
        hand.advance((seat + 1) % state.config.num_seats, events)
    return hand.freeze(), events


def observation(state: GameState, seat: int) -> Observation:
    """The seat's view: other hole cards hidden unless shown at showdown; no deck or seed."""
    hole_cards = tuple(cards if s == seat or s in state.shown else None for s, cards in enumerate(state.hole_cards))
    return Observation(
        seat=seat,
        hand_id=state.hand_id,
        config=state.config,
        button=state.button,
        small_blind_seat=state.small_blind_seat,
        big_blind_seat=state.big_blind_seat,
        street=state.street,
        dealt_in=state.dealt_in,
        starting_stacks=state.starting_stacks,
        stacks=state.stacks,
        committed_this_street=state.committed_this_street,
        committed_total=state.committed_total,
        dead_money=state.dead_money,
        board=state.board,
        hole_cards=hole_cards,
        to_act=state.to_act,
        current_bet=state.current_bet,
        last_raise_size=state.last_raise_size,
        last_aggressor=state.last_aggressor,
        acted_at=state.acted_at,
        folded=state.folded,
        all_in=state.all_in,
        history=state.history,
        awards=state.awards,
        shown=state.shown,
    )


def is_terminal(state: GameState) -> bool:
    return state.street == Street.COMPLETE


def net_results(state: GameState) -> tuple[int, ...]:
    """Chips won or lost per seat. Raises `IllegalActionError` before the hand is complete."""
    if state.street != Street.COMPLETE:
        raise IllegalActionError(f"hand {state.hand_id} is not complete")
    return tuple(end - start for end, start in zip(state.stacks, state.starting_stacks))


def _card_list(cards: tuple[int, int] | None) -> list[int] | None:
    return None if cards is None else list(cards)


class _Hand:
    """Mutable working copy of a GameState used within one engine call."""

    def __init__(self, state: GameState):
        self.hand_id = state.hand_id
        self.seed = state.seed
        self.config = state.config
        self.button = state.button
        self.small_blind_seat = state.small_blind_seat
        self.big_blind_seat = state.big_blind_seat
        self.street = state.street
        self.dealt_in = state.dealt_in
        self.starting_stacks = state.starting_stacks
        self.stacks = list(state.stacks)
        self.committed_this_street = list(state.committed_this_street)
        self.committed_total = list(state.committed_total)
        self.dead_money = state.dead_money
        self.board = list(state.board)
        self.hole_cards = list(state.hole_cards)
        self.deck = list(state.deck)
        self.to_act = state.to_act
        self.current_bet = state.current_bet
        self.last_raise_size = state.last_raise_size
        self.last_aggressor = state.last_aggressor
        self.acted_at = list(state.acted_at)
        self.folded = list(state.folded)
        self.all_in = list(state.all_in)
        self.history = list(state.history)
        self.awards = list(state.awards)
        self.shown = state.shown

    @classmethod
    def start(
        cls,
        config: GameConfig,
        seed: int,
        hand_id: str,
        positions: tuple[int, int, int],
        stacks: tuple[int, ...],
        dealt: tuple[bool, ...],
        deck: list[int],
    ) -> _Hand:
        n = config.num_seats
        button, small_blind_seat, big_blind_seat = positions
        return cls(
            GameState(
                hand_id=hand_id,
                seed=seed,
                config=config,
                button=button,
                small_blind_seat=small_blind_seat,
                big_blind_seat=big_blind_seat,
                street=Street.PREFLOP,
                dealt_in=dealt,
                starting_stacks=tuple(stacks),
                stacks=tuple(stacks),
                committed_this_street=(0,) * n,
                committed_total=(0,) * n,
                dead_money=0,
                board=(),
                hole_cards=(None,) * n,
                deck=tuple(deck),
                to_act=None,
                current_bet=config.big_blind,
                last_raise_size=config.big_blind,
                last_aggressor=None,
                acted_at=(None,) * n,
                folded=(False,) * n,
                all_in=(False,) * n,
                history=(),
            )
        )

    def freeze(self) -> GameState:
        return GameState(
            hand_id=self.hand_id,
            seed=self.seed,
            config=self.config,
            button=self.button,
            small_blind_seat=self.small_blind_seat,
            big_blind_seat=self.big_blind_seat,
            street=self.street,
            dealt_in=self.dealt_in,
            starting_stacks=self.starting_stacks,
            stacks=tuple(self.stacks),
            committed_this_street=tuple(self.committed_this_street),
            committed_total=tuple(self.committed_total),
            dead_money=self.dead_money,
            board=tuple(self.board),
            hole_cards=tuple(self.hole_cards),
            deck=tuple(self.deck),
            to_act=self.to_act,
            current_bet=self.current_bet,
            last_raise_size=self.last_raise_size,
            last_aggressor=self.last_aggressor,
            acted_at=tuple(self.acted_at),
            folded=tuple(self.folded),
            all_in=tuple(self.all_in),
            history=tuple(self.history),
            awards=tuple(self.awards),
            shown=self.shown,
        )

    def _move_chips(self, seat: int, amount: int, live: bool):
        self.stacks[seat] -= amount
        self.committed_total[seat] += amount
        if live:
            self.committed_this_street[seat] += amount
        self.all_in[seat] = self.stacks[seat] == 0

    def post_ante(self, seat: int, amount: int) -> int:
        """A per-player ante counts toward pots by contribution, not toward the street's bet."""
        posted = min(amount, self.stacks[seat])
        self._move_chips(seat, posted, live=False)
        return posted

    def post_big_blind_ante(self, seat: int, amount: int) -> int:
        posted = min(amount, self.stacks[seat])
        self.stacks[seat] -= posted
        self.dead_money += posted
        self.all_in[seat] = self.stacks[seat] == 0
        return posted

    def post_blind(self, seat: int, amount: int) -> int:
        posted = min(amount, self.stacks[seat])
        self._move_chips(seat, posted, live=True)
        return posted

    def deal_hole_cards(self, order: list[int]):
        first, second = self.deck[: len(order)], self.deck[len(order) : 2 * len(order)]
        for seat, a, b in zip(order, first, second):
            self.hole_cards[seat] = (a, b)
        del self.deck[: 2 * len(order)]

    def act(self, action: Action, events: list[Event]) -> int:
        seat = self.to_act
        assert seat is not None  # guaranteed by check_action
        if action.type == ActionType.FOLD:
            self.folded[seat] = True
        elif action.type == ActionType.CHECK:
            self.acted_at[seat] = self.current_bet
        elif action.type == ActionType.CALL:
            owed = self.current_bet - self.committed_this_street[seat]
            self._move_chips(seat, min(owed, self.stacks[seat]), live=True)
            self.acted_at[seat] = self.current_bet
        else:
            target = action.amount
            assert target is not None  # guaranteed by check_action
            self._move_chips(seat, target - self.committed_this_street[seat], live=True)
            if target - self.current_bet >= self.last_raise_size:
                self.last_raise_size = target - self.current_bet
            self.current_bet = target
            self.last_aggressor = seat
            self.acted_at[seat] = target
        self.history.append(HistoryEntry(self.street, seat, action))
        events.append(
            Event(
                "ActionTaken",
                {
                    "street": self.street.value,
                    "seat": seat,
                    "action": action.to_dict(),
                    "committed_this_street": self.committed_this_street[seat],
                    "stack": self.stacks[seat],
                },
            )
        )
        return seat

    def advance(self, start: int, events: list[Event]):
        """Give the action to the next seat that must act, or close the betting round."""
        n = self.config.num_seats
        for k in range(n):
            seat = (start + k) % n
            if needs_to_act(self, seat):
                self.to_act = seat
                return
        self.to_act = None
        self._end_street(events)

    def _return_uncalled(self, events: list[Event]):
        top = max(self.committed_this_street)
        leaders = [s for s, c in enumerate(self.committed_this_street) if c == top]
        if top == 0 or len(leaders) != 1:
            return
        leader = leaders[0]
        excess = top - max(c for s, c in enumerate(self.committed_this_street) if s != leader)
        self.stacks[leader] += excess
        self.committed_this_street[leader] -= excess
        self.committed_total[leader] -= excess
        self.all_in[leader] = self.stacks[leader] == 0
        events.append(Event("UncalledBetReturned", {"seat": leader, "amount": excess}))

    def _end_street(self, events: list[Event]):
        self._return_uncalled(events)
        if self.street == Street.RIVER:
            self._showdown(events)
            return
        n = self.config.num_seats
        self.street = _NEXT_STREET[self.street]
        self.committed_this_street = [0] * n
        self.current_bet = 0
        self.last_raise_size = self.config.big_blind
        self.acted_at = [None] * n
        dealt = self.deck[: _CARDS_DEALT[self.street]]
        del self.deck[: len(dealt)]
        self.board.extend(dealt)
        events.append(
            Event(
                "StreetDealt",
                {"street": self.street.value, "cards": dealt, "board": list(self.board)},
            )
        )
        self.advance((self.button + 1) % n, events)

    def finish_uncontested(self, winner: int, events: list[Event]):
        self._return_uncalled(events)
        amount = sum(self.committed_total) + self.dead_money
        self.stacks[winner] += amount
        award = PotAward(amount, (winner,), (winner,), (amount,))
        self.awards.append(award)
        events.append(Event("PotAwarded", award.to_dict()))
        self._complete(events)

    def _showdown(self, events: list[Event]):
        n = self.config.num_seats
        live = tuple(self.dealt_in[s] and not self.folded[s] for s in range(n))
        values = {}
        for seat in range(n):
            cards = self.hole_cards[seat]
            if live[seat] and cards is not None:
                values[seat] = evaluate(list(cards) + self.board)
        events.append(
            Event(
                "Showdown",
                {
                    "hands": [
                        {
                            "seat": seat,
                            "cards": _card_list(self.hole_cards[seat]),
                            "value": value,
                            "description": describe(value),
                        }
                        for seat, value in values.items()
                    ]
                },
            )
        )
        self.shown = tuple(values)
        # Side pots are settled before the main pot, last one first.
        for pot in reversed(build_pots(tuple(self.committed_total), live, self.dead_money)):
            best = max(values[s] for s in pot.eligible)
            winners = [s for s in pot.eligible if values[s] == best]
            shares = split_pot(pot.amount, winners, self.button, n)
            for seat, share in zip(winners, shares):
                self.stacks[seat] += share
            award = PotAward(pot.amount, pot.eligible, tuple(winners), tuple(shares))
            self.awards.append(award)
            events.append(Event("PotAwarded", award.to_dict()))
        self._complete(events)

    def _complete(self, events: list[Event]):
        self.street = Street.COMPLETE
        self.to_act = None
        net = [end - start for end, start in zip(self.stacks, self.starting_stacks)]
        events.append(Event("HandComplete", {"stacks": list(self.stacks), "net": net}))


def replay_states(hand: GameState) -> list[GameState]:
    """Every state of a hand: before each action in `hand.history`, then the final state.

    Rebuilds the starting deck from the dealt hole cards, the board, and the undealt rest, so
    the replay repeats the hand exactly. Raises `IllegalActionError` for an illegal action in
    the history and `ConfigError` if the history does not replay to `hand` (both `ValueError`s:
    a state that was not produced by this engine).
    """
    n = hand.config.num_seats
    order = [(hand.button + k) % n for k in range(1, n + 1) if hand.dealt_in[(hand.button + k) % n]]
    holes = [hand.hole_cards[s] for s in order]
    if any(h is None for h in holes):
        raise ConfigError(f"hand {hand.hand_id} lacks hole cards for a dealt-in seat")
    deck = tuple(h[0] for h in holes if h) + tuple(h[1] for h in holes if h) + hand.board + hand.deck
    state, _ = new_hand(hand.config, hand.seed, hand.button, hand.starting_stacks, hand.dealt_in, hand.hand_id, deck)
    states = [state]
    for entry in hand.history:
        state, _ = apply_action(state, entry.action)
        states.append(state)
    if state != hand:
        raise ConfigError(f"hand {hand.hand_id} does not replay to the same final state")
    return states


@dataclass(frozen=True)
class Pot:
    amount: int
    eligible: tuple[int, ...]


def build_pots(committed_total: tuple[int, ...], live: tuple[bool, ...], dead_money: int = 0) -> list[Pot]:
    """Split total commitments into a main pot and side pots, smallest cap first.

    Each live player's total commitment caps a pot that only players who put in at least that
    much can win. Chips from folded players count toward the pots they reached, and
    `dead_money` (the big-blind ante) goes to the main pot. Uncalled bets must already have
    been returned, so no chips sit above the largest live commitment.
    """
    caps = sorted({c for c, is_live in zip(committed_total, live) if is_live and c > 0})
    pots = []
    previous = 0
    for cap in caps:
        amount = sum(min(c, cap) - min(c, previous) for c in committed_total)
        eligible = tuple(seat for seat, c in enumerate(committed_total) if live[seat] and c >= cap)
        pots.append(Pot(amount, eligible))
        previous = cap
    leftover = sum(c - previous for c in committed_total if c > previous)
    if leftover or not pots:
        raise ValueError(f"chips {committed_total} do not fit under live commitments {live}")
    pots[0] = Pot(pots[0].amount + dead_money, pots[0].eligible)
    return pots


def split_pot(amount: int, winners: list[int], button: int, num_seats: int) -> list[int]:
    """Shares aligned with `winners`. Odd chips go one at a time to the winners in seat order
    starting left of the button."""
    ordered = sorted(winners, key=lambda seat: (seat - button - 1) % num_seats)
    base, odd = divmod(amount, len(winners))
    extra = {seat: 1 for seat in ordered[:odd]}
    return [base + extra.get(seat, 0) for seat in winners]
