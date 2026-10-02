"""Game data types. All are immutable and round-trip through JSON via to_dict / from_dict."""

from __future__ import annotations

from   dataclasses              import dataclass, fields
import enum
from   typing                   import Any


class ActionType(enum.StrEnum):
    FOLD = "FOLD"
    CHECK = "CHECK"
    CALL = "CALL"
    BET = "BET"
    RAISE = "RAISE"


class Street(enum.StrEnum):
    PREFLOP = "PREFLOP"
    FLOP = "FLOP"
    TURN = "TURN"
    RIVER = "RIVER"
    COMPLETE = "COMPLETE"


class AnteType(enum.StrEnum):
    NONE = "NONE"
    PER_PLAYER = "PER_PLAYER"
    BIG_BLIND_ANTE = "BIG_BLIND_ANTE"


class ConfigError(ValueError):
    """Raised for an invalid game or session configuration."""


@dataclass(frozen=True)
class Action:
    """A player action. For BET and RAISE, `amount` is the player's total commitment on this
    street after the action ("raise to"); it is None for the other types."""

    type: ActionType
    amount: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type.value, "amount": self.amount}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Action:
        return cls(ActionType(data["type"]), data["amount"])


@dataclass(frozen=True)
class GameConfig:
    """Per-hand table rules. With BIG_BLIND_ANTE, `ante` is the total the big blind posts for
    the table; with PER_PLAYER it is what each dealt-in player posts."""

    num_seats: int
    small_blind: int = 50
    big_blind: int = 100
    ante: int = 0
    ante_type: AnteType = AnteType.NONE

    def __post_init__(self):
        if not 2 <= self.num_seats <= 8:
            raise ConfigError(f"num_seats must be 2..8, got {self.num_seats}")
        if not 0 < self.small_blind <= self.big_blind:
            raise ConfigError(
                f"blinds must satisfy 0 < small blind <= big blind, got {self.small_blind}/{self.big_blind}"
            )
        if self.ante < 0:
            raise ConfigError(f"ante must be >= 0, got {self.ante}")
        if (self.ante > 0) != (self.ante_type != AnteType.NONE):
            raise ConfigError(f"ante {self.ante} does not match ante type {self.ante_type.value}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_seats": self.num_seats,
            "small_blind": self.small_blind,
            "big_blind": self.big_blind,
            "ante": self.ante,
            "ante_type": self.ante_type.value,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GameConfig:
        return cls(
            data["num_seats"],
            data["small_blind"],
            data["big_blind"],
            data["ante"],
            AnteType(data["ante_type"]),
        )


@dataclass(frozen=True)
class LegalActions:
    """What the seat to act may do. `min_raise_to`/`max_raise_to` bound BET or RAISE amounts and
    are 0 when neither is allowed."""

    can_fold: bool
    can_check: bool
    can_call: bool
    call_amount: int
    can_bet: bool
    can_raise: bool
    min_raise_to: int
    max_raise_to: int


@dataclass(frozen=True)
class Event:
    """A JSON-ready record of something that happened; `kind` names it, `data` holds details."""

    kind: str
    data: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "data": self.data}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Event:
        return cls(data["kind"], data["data"])


@dataclass(frozen=True)
class PotAward:
    """One pot (main or side) and who won it; `shares` align with `winners`."""

    amount: int
    eligible: tuple[int, ...]
    winners: tuple[int, ...]
    shares: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "amount": self.amount,
            "eligible": list(self.eligible),
            "winners": list(self.winners),
            "shares": list(self.shares),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PotAward:
        return cls(
            data["amount"],
            tuple(data["eligible"]),
            tuple(data["winners"]),
            tuple(data["shares"]),
        )


@dataclass(frozen=True)
class HistoryEntry:
    street: Street
    seat: int
    action: Action

    def to_dict(self) -> dict[str, Any]:
        return {"street": self.street.value, "seat": self.seat, "action": self.action.to_dict()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HistoryEntry:
        return cls(Street(data["street"]), data["seat"], Action.from_dict(data["action"]))


@dataclass(frozen=True)
class GameState:
    """Full information about one hand. Per-seat tuples have `config.num_seats` entries.

    `acted_at[seat]` is the street's `current_bet` when the seat last acted, or None if it has
    not acted this street (posting a blind is not acting). It decides whether a raise reopens
    betting for a seat. `hole_cards[seat]` is None for seats not dealt in. `committed_total`
    counts chips that build main and side pots by contribution (bets, blinds, per-player antes);
    `dead_money` is the big-blind ante, which belongs to the main pot.
    """

    hand_id: str
    seed: int
    config: GameConfig
    button: int
    small_blind_seat: int
    big_blind_seat: int
    street: Street
    dealt_in: tuple[bool, ...]
    starting_stacks: tuple[int, ...]
    stacks: tuple[int, ...]
    committed_this_street: tuple[int, ...]
    committed_total: tuple[int, ...]
    dead_money: int
    board: tuple[int, ...]
    hole_cards: tuple[tuple[int, int] | None, ...]
    deck: tuple[int, ...]
    to_act: int | None
    current_bet: int
    last_raise_size: int
    last_aggressor: int | None
    acted_at: tuple[int | None, ...]
    folded: tuple[bool, ...]
    all_in: tuple[bool, ...]
    history: tuple[HistoryEntry, ...]
    awards: tuple[PotAward, ...] = ()
    shown: tuple[int, ...] = ()

    @property
    def pot(self) -> int:
        """Chips committed by all seats this hand, before any award."""
        return sum(self.committed_total) + self.dead_money

    def to_dict(self) -> dict[str, Any]:
        return {f.name: _encode(getattr(self, f.name)) for f in fields(self)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GameState:
        return cls(**_decode_hand_fields(data))


@dataclass(frozen=True)
class Observation:
    """What one seat may see: the GameState without other seats' hole cards (unless shown at
    showdown), the remaining deck, or the seed (which would reveal the deck)."""

    seat: int
    hand_id: str
    config: GameConfig
    button: int
    small_blind_seat: int
    big_blind_seat: int
    street: Street
    dealt_in: tuple[bool, ...]
    starting_stacks: tuple[int, ...]
    stacks: tuple[int, ...]
    committed_this_street: tuple[int, ...]
    committed_total: tuple[int, ...]
    dead_money: int
    board: tuple[int, ...]
    hole_cards: tuple[tuple[int, int] | None, ...]
    to_act: int | None
    current_bet: int
    last_raise_size: int
    last_aggressor: int | None
    acted_at: tuple[int | None, ...]
    folded: tuple[bool, ...]
    all_in: tuple[bool, ...]
    history: tuple[HistoryEntry, ...]
    awards: tuple[PotAward, ...] = ()
    shown: tuple[int, ...] = ()

    @property
    def pot(self) -> int:
        return sum(self.committed_total) + self.dead_money

    @property
    def my_cards(self) -> tuple[int, int]:
        cards = self.hole_cards[self.seat]
        if cards is None:
            raise ValueError(f"seat {self.seat} is not dealt in")
        return cards

    def to_dict(self) -> dict[str, Any]:
        return {f.name: _encode(getattr(self, f.name)) for f in fields(self)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Observation:
        return cls(**_decode_hand_fields(data))


def _encode(value: Any) -> Any:
    if isinstance(value, enum.Enum):
        return value.value
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, tuple):
        return [_encode(v) for v in value]
    return value


def _decode_hand_fields(data: dict[str, Any]) -> dict[str, Any]:
    decoded = dict(data)
    decoded["config"] = GameConfig.from_dict(data["config"])
    decoded["street"] = Street(data["street"])
    for name in (
        "dealt_in",
        "starting_stacks",
        "stacks",
        "committed_this_street",
        "committed_total",
        "board",
        "acted_at",
        "folded",
        "all_in",
        "shown",
    ):
        decoded[name] = tuple(data[name])
    if "deck" in data:
        decoded["deck"] = tuple(data["deck"])
    decoded["hole_cards"] = tuple(None if h is None else (h[0], h[1]) for h in data["hole_cards"])
    decoded["history"] = tuple(HistoryEntry.from_dict(h) for h in data["history"])
    decoded["awards"] = tuple(PotAward.from_dict(a) for a in data["awards"])
    return decoded
