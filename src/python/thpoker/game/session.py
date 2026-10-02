"""Session rules between hands: blinds and levels, button movement, rebuys, eliminations, payouts.

Pure like the engine: `begin_hand` turns a SessionState into the inputs of `engine.new_hand`,
and `end_hand` folds a finished hand back into the session.
"""

from   dataclasses              import dataclass, replace
import enum
from   thpoker.game.cards       import (COMBO_CLASS, PREFLOP_CLASSES,
                                        combo_index)
from   thpoker.game.engine      import dealt_hole_cards
from   thpoker.game.rng         import Rng
from   thpoker.game.state       import (AnteType, ConfigError, Event,
                                        GameConfig, GameState, Street)


class Mode(enum.StrEnum):
    CASH = "CASH"
    TOURNAMENT = "TOURNAMENT"
    TRAINING = "TRAINING"


class SessionError(ValueError):
    """Raised for a session operation that the current session state does not allow."""


@dataclass(frozen=True)
class BlindLevel:
    small_blind: int
    big_blind: int
    ante: int


# Big blind per level as a multiple of the level-1 big blind; levels past the end double.
_LEVEL_MULTIPLES = (1, 1.5, 2, 3, 4, 6, 8, 10, 15, 20, 30, 40, 60, 80, 100)
_FIRST_ANTE_LEVEL = 4  # zero-based: antes start at level 5
MAX_DEALS = 20_000
# preset name -> (starting stack in level-1 big blinds, hands per level)
TOURNAMENT_PRESETS = {"turbo": (50, 8), "regular": (100, 15), "deep": (150, 25)}


def preset_schedule(first_big_blind: int) -> tuple[BlindLevel, ...]:
    """The standard schedule: SB = BB / 2, a big-blind ante equal to the BB from level 5."""
    levels = []
    for index, multiple in enumerate(_LEVEL_MULTIPLES):
        big_blind = round(first_big_blind * multiple)
        ante = big_blind if index >= _FIRST_ANTE_LEVEL else 0
        levels.append(BlindLevel(big_blind // 2, big_blind, ante))
    return tuple(levels)


def blind_level(schedule: tuple[BlindLevel, ...], index: int) -> BlindLevel:
    if index < len(schedule):
        return schedule[index]
    factor = 2 ** (index - len(schedule) + 1)
    last = schedule[-1]
    return BlindLevel(last.small_blind * factor, last.big_blind * factor, last.ante * factor)


def default_payouts(num_players: int) -> tuple[float, ...]:
    """Standard sit-and-go splits: winner-take-all up to 3 players, 65/35 up to 6, else 50/30/20."""
    if num_players <= 3:
        return (1.0,)
    if num_players <= 6:
        return (0.65, 0.35)
    return (0.5, 0.3, 0.2)


@dataclass(frozen=True)
class TournamentConfig:
    """Exactly one of `hands_per_level` or `minutes_per_level` sets the level length. An empty
    `payouts` means `default_payouts` for the number of entrants."""

    blind_schedule: tuple[BlindLevel, ...]
    hands_per_level: int | None = 15
    minutes_per_level: float | None = None
    ante_type: AnteType = AnteType.BIG_BLIND_ANTE
    payouts: tuple[float, ...] = ()
    buy_in: int = 0

    def __post_init__(self):
        if not self.blind_schedule:
            raise ConfigError("blind schedule is empty")
        if (self.hands_per_level is None) == (self.minutes_per_level is None):
            raise ConfigError("set exactly one of hands_per_level and minutes_per_level")
        if self.hands_per_level is not None and self.hands_per_level < 1:
            raise ConfigError(f"hands_per_level must be >= 1, got {self.hands_per_level}")
        if self.minutes_per_level is not None and self.minutes_per_level <= 0:
            raise ConfigError(f"minutes_per_level must be > 0, got {self.minutes_per_level}")
        if self.payouts and (any(p <= 0 for p in self.payouts) or abs(sum(self.payouts) - 1.0) > 1e-9):
            raise ConfigError(f"payouts must be positive and sum to 1, got {self.payouts}")
        for level in self.blind_schedule:
            GameConfig(2, level.small_blind, level.big_blind, level.ante, self._ante_type(level))

    def _ante_type(self, level: BlindLevel) -> AnteType:
        return self.ante_type if level.ante else AnteType.NONE

    def game_config(self, num_seats: int, level_index: int) -> GameConfig:
        level = blind_level(self.blind_schedule, level_index)
        return GameConfig(num_seats, level.small_blind, level.big_blind, level.ante, self._ante_type(level))


@dataclass(frozen=True)
class SessionConfig:
    """Rules of a session. `user_seat` None means a bot-only session. Cash blinds live in
    `cash_blinds`; a tournament takes its blinds from `tournament` instead. Training may keep
    the user `user_position` dealt-in seats left of the button every hand (0 is the button) and
    deal it only the classes in `user_hands` ("AA", "AKs", ...; empty for any hand)."""

    mode: Mode
    seed: int
    num_seats: int
    starting_stack: int
    user_seat: int | None = 0
    cash_blinds: GameConfig | None = None
    reset_stacks_each_hand: bool = False
    auto_rebuy: bool = False
    rebuy_threshold: int = 0
    tournament: TournamentConfig | None = None
    user_position: int | None = None
    user_hands: tuple[str, ...] = ()

    def __post_init__(self):
        if not 2 <= self.num_seats <= 8:
            raise ConfigError(f"num_seats must be 2..8, got {self.num_seats}")
        if self.user_seat is not None and not 0 <= self.user_seat < self.num_seats:
            raise ConfigError(f"user seat {self.user_seat} is outside 0..{self.num_seats - 1}")
        if self.starting_stack <= 0:
            raise ConfigError(f"starting stack must be > 0, got {self.starting_stack}")
        if self.user_position is not None or self.user_hands:
            if self.mode != Mode.TRAINING or self.user_seat is None:
                raise ConfigError("a fixed seat and chosen hands need a training session's user")
            if self.user_position is not None and not 0 <= self.user_position < self.num_seats:
                raise ConfigError(f"user position must be 0..{self.num_seats - 1}")
            unknown = set(self.user_hands) - set(PREFLOP_CLASSES)
            if unknown:
                raise ConfigError(f"unknown starting hands {sorted(unknown)}")
        if self.mode != Mode.TOURNAMENT:
            if self.mode == Mode.TRAINING and not self.reset_stacks_each_hand:
                raise ConfigError("training resets stacks every hand")
            if self.cash_blinds is None or self.tournament is not None:
                raise ConfigError("a cash session needs cash_blinds and no tournament")
            if self.cash_blinds.num_seats != self.num_seats:
                raise ConfigError("cash_blinds.num_seats must equal num_seats")
            if self.auto_rebuy and not 0 < self.rebuy_threshold <= self.starting_stack:
                raise ConfigError(f"rebuy threshold must be in (0, starting stack], got {self.rebuy_threshold}")
            if self.auto_rebuy and self.reset_stacks_each_hand:
                raise ConfigError("auto-rebuy has no effect when stacks reset every hand")
        elif self.tournament is None or self.cash_blinds is not None:
            raise ConfigError("a tournament session needs a tournament config and no cash_blinds")
        elif self.reset_stacks_each_hand or self.auto_rebuy:
            raise ConfigError("stack resets and rebuys are cash-only")
        elif len(self.tournament.payouts) > self.num_seats:
            raise ConfigError(
                f"payouts have {len(self.tournament.payouts)} paid places but only {self.num_seats} entrants"
            )


@dataclass(frozen=True)
class HandSetup:
    """Arguments for `engine.new_hand`."""

    config: GameConfig
    seed: int
    button: int
    stacks: tuple[int, ...]
    dealt_in: tuple[bool, ...]
    hand_id: str


@dataclass(frozen=True)
class SessionState:
    """`active` marks seats still in the session (not eliminated). `places` and `prizes` (a
    fraction of the prize pool) are set as tournament players finish; tied finishers share
    the best of their places and the average of those places' prizes. `buy_ins` counts chips
    bought per seat, so cash results are stacks minus buy-ins."""

    config: SessionConfig
    hand_number: int
    button: int | None
    stacks: tuple[int, ...]
    buy_ins: tuple[int, ...]
    active: tuple[bool, ...]
    level: int
    places: tuple[int | None, ...]
    prizes: tuple[float, ...]
    last_big_blind_seat: int | None
    last_dealt_count: int
    awaiting_rebuy: bool
    finished: bool


def start_session(config: SessionConfig) -> tuple[SessionState, list[Event]]:
    n = config.num_seats
    state = SessionState(
        config=config,
        hand_number=0,
        button=None,
        stacks=(config.starting_stack,) * n,
        buy_ins=(config.starting_stack,) * n,
        active=(True,) * n,
        level=0,
        places=(None,) * n,
        prizes=(0.0,) * n,
        last_big_blind_seat=None,
        last_dealt_count=0,
        awaiting_rebuy=False,
        finished=False,
    )
    return state, [Event("SessionStarted", {"mode": config.mode.value, "seed": config.seed})]


def begin_hand(state: SessionState, elapsed_minutes: float = 0.0) -> tuple[SessionState, HandSetup, list[Event]]:
    """Apply between-hand changes (rebuys, blind level) and set up the next hand.

    `elapsed_minutes` matters only for minute-based tournament levels, which never go back
    down. The hand id never contains the seed, since players see it. Raises `SessionError`
    if the session is finished or waiting for the user to rebuy.
    """
    if state.finished:
        raise SessionError("the session is finished")
    if state.awaiting_rebuy:
        raise SessionError("the user is busted; rebuy or end the session first")
    config = state.config
    n = config.num_seats
    events: list[Event] = []
    stacks = list(state.stacks)
    buy_ins = list(state.buy_ins)
    level = state.level
    if config.mode != Mode.TOURNAMENT:
        assert config.cash_blinds is not None  # checked by SessionConfig
        game_config = config.cash_blinds
        for seat in range(n):
            if config.reset_stacks_each_hand:
                # Count the reset as a buy-in change so the session net still sums every hand.
                buy_ins[seat] += config.starting_stack - stacks[seat]
                stacks[seat] = config.starting_stack
                continue
            bot_busted = seat != config.user_seat and stacks[seat] == 0
            below_threshold = config.auto_rebuy and stacks[seat] < config.rebuy_threshold
            if bot_busted or below_threshold:
                amount = config.starting_stack - stacks[seat]
                stacks[seat] += amount
                buy_ins[seat] += amount
                events.append(Event("Rebuy", {"seat": seat, "amount": amount}))
    else:
        tournament = config.tournament
        assert tournament is not None  # checked by SessionConfig
        if tournament.hands_per_level is not None:
            level = state.hand_number // tournament.hands_per_level
        else:
            assert tournament.minutes_per_level is not None
            level = max(state.level, int(elapsed_minutes // tournament.minutes_per_level))
        game_config = tournament.game_config(n, level)
        if level != state.level:
            events.append(
                Event(
                    "BlindLevelChanged",
                    {
                        "level": level + 1,
                        "small_blind": game_config.small_blind,
                        "big_blind": game_config.big_blind,
                        "ante": game_config.ante,
                    },
                )
            )
    dealt_in = tuple(state.active[s] and stacks[s] > 0 for s in range(n))
    button = _next_button(state, dealt_in)
    number = state.hand_number + 1
    setup = HandSetup(
        config=game_config,
        seed=_hand_seed(config, number, button, dealt_in),
        button=button,
        stacks=tuple(stacks),
        dealt_in=dealt_in,
        hand_id=f"hand-{number}",
    )
    new_state = replace(
        state,
        hand_number=number,
        button=button,
        stacks=tuple(stacks),
        buy_ins=tuple(buy_ins),
        level=level,
    )
    return new_state, setup, events


def _hand_seed(config: SessionConfig, number: int, button: int, dealt_in: tuple[bool, ...]) -> int:
    """The seed of hand `number`. With chosen hands it is redrawn until the user is dealt one;
    every try derives from the session seed, so the session still replays from it."""
    rng = Rng(config.seed)
    seed = rng.derive("hand", number).randbelow(1 << 63)
    user = config.user_seat
    if not config.user_hands or user is None or not dealt_in[user]:
        return seed
    wanted = {PREFLOP_CLASSES.index(name) for name in config.user_hands}
    tries = 1
    while COMBO_CLASS[combo_index(*dealt_hole_cards(seed, button, dealt_in, user))] not in wanted:
        if tries == MAX_DEALS:
            raise SessionError(f"no hand from {len(wanted)} chosen classes in {MAX_DEALS} deals")
        seed = rng.derive("hand", number, tries).randbelow(1 << 63)
        tries += 1
    return seed


def _next_button(state: SessionState, dealt_in: tuple[bool, ...]) -> int:
    n = state.config.num_seats
    user = state.config.user_seat
    if state.config.user_position is not None and user is not None and dealt_in[user]:
        # The button sits `user_position` dealt-in seats to the user's right.
        right = [s for s in ((user - k) % n for k in range(n)) if dealt_in[s]]
        return right[state.config.user_position % len(right)]
    if state.button is None:
        start = Rng(state.config.seed).derive("button").randbelow(n)
        return next(s for s in ((start + k) % n for k in range(n)) if dealt_in[s])
    last_big_blind = state.last_big_blind_seat
    if sum(dealt_in) == 2 and state.last_dealt_count >= 3 and last_big_blind is not None and dealt_in[last_big_blind]:
        # Going heads-up: last hand's big blind takes the button (and small blind) so no
        # player posts the big blind twice in a row.
        return last_big_blind
    return next(s for s in ((state.button + k) % n for k in range(1, n + 1)) if dealt_in[s])


def end_hand(state: SessionState, hand: GameState) -> tuple[SessionState, list[Event]]:
    """Record a finished hand: stacks, eliminations and places, and the end of a tournament.

    Raises `SessionError` if the hand is not complete.
    """
    if hand.street != Street.COMPLETE:
        raise SessionError(f"hand {hand.hand_id} is not complete")
    config = state.config
    events: list[Event] = []
    stacks = hand.stacks
    active = list(state.active)
    places = list(state.places)
    prizes = list(state.prizes)
    finished = False
    awaiting_rebuy = False
    if config.mode == Mode.TOURNAMENT:
        assert config.tournament is not None
        payouts = config.tournament.payouts or default_payouts(config.num_seats)
        busted = [s for s in range(config.num_seats) if active[s] and stacks[s] == 0]
        worst_place = sum(active)
        # Players busted on the same hand finish in order of their stacks at its start.
        for start_stack in sorted({hand.starting_stacks[s] for s in busted}):
            group = [s for s in busted if hand.starting_stacks[s] == start_stack]
            best_place = worst_place - len(group) + 1
            shared = [payouts[p - 1] if p <= len(payouts) else 0.0 for p in range(best_place, worst_place + 1)]
            for seat in group:
                active[seat] = False
                places[seat] = best_place
                prizes[seat] = sum(shared) / len(group)
                events.append(
                    Event(
                        "PlayerEliminated",
                        {"seat": seat, "place": best_place, "prize": prizes[seat]},
                    )
                )
            worst_place = best_place - 1
        if sum(active) == 1:
            winner = active.index(True)
            places[winner] = 1
            prizes[winner] = payouts[0]
            finished = True
            events.append(Event("TournamentFinished", {"places": places, "prizes": prizes}))
    elif (
        config.user_seat is not None
        and stacks[config.user_seat] == 0
        and not config.auto_rebuy
        and not config.reset_stacks_each_hand
    ):
        awaiting_rebuy = True
        events.append(Event("UserBusted", {"seat": config.user_seat}))
    new_state = replace(
        state,
        stacks=stacks,
        active=tuple(active),
        places=tuple(places),
        prizes=tuple(prizes),
        last_big_blind_seat=hand.big_blind_seat,
        last_dealt_count=sum(hand.dealt_in),
        awaiting_rebuy=awaiting_rebuy,
        finished=finished,
    )
    return new_state, events


def rebuy(state: SessionState, seat: int) -> tuple[SessionState, list[Event]]:
    """Top a cash seat back up to the starting stack. Raises `SessionError` in a tournament or
    when the seat is already at or above the starting stack."""
    config = state.config
    if config.mode != Mode.CASH:
        raise SessionError("rebuys are cash-only")
    amount = config.starting_stack - state.stacks[seat]
    if amount <= 0:
        raise SessionError(f"seat {seat} already has {state.stacks[seat]} chips")
    stacks = list(state.stacks)
    buy_ins = list(state.buy_ins)
    stacks[seat] += amount
    buy_ins[seat] += amount
    new_state = replace(
        state,
        stacks=tuple(stacks),
        buy_ins=tuple(buy_ins),
        awaiting_rebuy=state.awaiting_rebuy and seat != config.user_seat,
    )
    return new_state, [Event("Rebuy", {"seat": seat, "amount": amount})]


def end_session(state: SessionState) -> tuple[SessionState, list[Event]]:
    """Close the session and report each seat's net chips (stack minus buy-ins)."""
    net = [stack - bought for stack, bought in zip(state.stacks, state.buy_ins)]
    ended = replace(state, finished=True, awaiting_rebuy=False)
    return ended, [Event("SessionEnded", {"hands": state.hand_number, "net": net, "places": list(state.places)})]
