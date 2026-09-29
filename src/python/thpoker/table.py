"""Runs a session at one table: bots act on their own, the user acts through `act`.

No I/O. Every bot decision is appended to the event stream as a `BotDecision` event, so a log
of the events holds everything the review needs. Per-seat statistics collected from finished
hands are shown next to each bot; with styles hidden, the user guesses each bot's style from
them, then sees the answer, which style the numbers pointed to, and how to beat it (DESIGN.md
Section 7.6).
"""

from __future__ import annotations

from   dataclasses              import dataclass
from   typing                   import Any

from   thpoker.bots.bot         import Bot, PANELS, PRESETS, Style, draw_style
from   thpoker.bots.equity_bot  import EquityBot
from   thpoker.bots.range_bot   import RangeBot
from   thpoker.bots.rule_bot    import RuleBot
from   thpoker.game.engine      import (apply_action, is_terminal, new_hand,
                                        observation)
from   thpoker.game.rng         import Rng
from   thpoker.game.rules       import legal_actions
from   thpoker.game.session     import (BlindLevel, Mode, SessionConfig,
                                        SessionError, TournamentConfig,
                                        begin_hand, default_payouts, end_hand,
                                        end_session, rebuy, start_session)
from   thpoker.game.state       import (Action, ActionType, AnteType,
                                        ConfigError, Event, GameConfig,
                                        GameState, LegalActions, Observation,
                                        Street)

_AGGRESSIVE = (ActionType.BET, ActionType.RAISE)


@dataclass
class SeatStats:
    hands: int = 0
    voluntary: int = 0  # put chips in preflop by choice (VPIP)
    preflop_raises: int = 0  # PFR
    three_bet_chances: int = 0  # faced exactly one preflop raise
    three_bets: int = 0
    postflop_aggressive: int = 0  # bets and raises after the flop
    postflop_calls: int = 0

    def summary(self) -> str:
        """Short HUD text, e.g. "VPIP 24 PFR 18 3B 6 AF 2.1 (140h)"."""
        if not self.hands:
            return "no hands yet"
        vpip = round(100 * self.voluntary / self.hands)
        pfr = round(100 * self.preflop_raises / self.hands)
        three_bet = round(100 * self.three_bets / self.three_bet_chances) if self.three_bet_chances else "-"
        factor = f"{self.postflop_aggressive / self.postflop_calls:.1f}" if self.postflop_calls else "-"
        return f"VPIP {vpip} PFR {pfr} 3B {three_bet} AF {factor} ({self.hands}h)"


class Hud:
    def __init__(self, num_seats: int):
        self.seats = [SeatStats() for _ in range(num_seats)]

    def record(self, hand: GameState):
        """Add a finished hand to every dealt-in seat's statistics."""
        voluntary, raised, chances, three_bets = set(), set(), set(), set()
        raises = 0
        for entry in hand.history:
            seat, kind = entry.seat, entry.action.type
            stats = self.seats[seat]
            if entry.street != Street.PREFLOP:
                stats.postflop_aggressive += kind in _AGGRESSIVE
                stats.postflop_calls += kind == ActionType.CALL
                continue
            if raises == 1:
                chances.add(seat)
            if kind in (ActionType.CALL, *_AGGRESSIVE):
                voluntary.add(seat)
            if kind in _AGGRESSIVE:
                raised.add(seat)
                if raises == 1:
                    three_bets.add(seat)
                raises += 1
        for seat, stats in enumerate(self.seats):
            if hand.dealt_in[seat]:
                stats.hands += 1
                stats.voluntary += seat in voluntary
                stats.preflop_raises += seat in raised
                stats.three_bet_chances += seat in chances
                stats.three_bets += seat in three_bets


@dataclass(frozen=True)
class TableConfig:
    """A session plus who sits in each bot seat. `bot_styles[seat]` names a preset, or is None
    to draw one from `population_panel`; the entry for the user's seat is ignored. Every bot
    plays at difficulty `tier`."""

    session: SessionConfig
    bot_styles: tuple[str | None, ...]
    population_panel: str = "online_micro"
    hide_styles: bool = False
    tier: int = 1

    def __post_init__(self):
        if len(self.bot_styles) != self.session.num_seats:
            raise ConfigError(f"need {self.session.num_seats} bot style entries, got {len(self.bot_styles)}")
        unknown = {s for s in self.bot_styles if s is not None and s not in PRESETS}
        if unknown:
            raise ConfigError(f"unknown styles {sorted(unknown)}; choose from {sorted(PRESETS)}")
        if self.population_panel not in PANELS:
            raise ConfigError(f"unknown panel {self.population_panel!r}; choose from {sorted(PANELS)}")
        if self.tier not in BOT_TIERS:
            raise ConfigError(f"tier must be one of {sorted(BOT_TIERS)}, got {self.tier}")

    @property
    def payouts(self) -> tuple[float, ...] | None:
        """Prize-pool shares by place in a tournament, None in a cash game."""
        tournament = self.session.tournament
        if tournament is None:
            return None
        return tournament.payouts or default_payouts(self.session.num_seats)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TableConfig:
        """Rebuild a config logged as `dataclasses.asdict` after a JSON round trip."""
        session = dict(data["session"])
        session["mode"] = Mode(session["mode"])
        if session["cash_blinds"] is not None:
            session["cash_blinds"] = GameConfig.from_dict(session["cash_blinds"])
        if session["tournament"] is not None:
            tournament = dict(session["tournament"])
            tournament["blind_schedule"] = tuple(BlindLevel(**level) for level in tournament["blind_schedule"])
            tournament["ante_type"] = AnteType(tournament["ante_type"])
            tournament["payouts"] = tuple(tournament["payouts"])
            session["tournament"] = TournamentConfig(**tournament)
        return cls(
            SessionConfig(**session),
            tuple(data["bot_styles"]),
            data["population_panel"],
            data["hide_styles"],
            data["tier"],
        )


BOT_TIERS: dict[int, type[RuleBot] | type[EquityBot] | type[RangeBot]] = {
    1: RuleBot,
    2: EquityBot,
    3: RangeBot,
}
# One name per seat, unrelated to style, so hidden styles stay hidden and no two bots match.
_BOT_NAMES = ("Alex", "Blake", "Casey", "Devon", "Emery", "Finley", "Harper", "Jordan")
# The chance, by tier, that a bot winning without a shodown shows its cards: easier bots show
# more, which gives a newcomer more hands to learn from.
SHOW_CHANCE = {1: 0.5, 2: 0.25, 3: 0.1}


def _seat_bots(config: TableConfig) -> dict[int, Bot]:
    rng = Rng(config.session.seed).derive("seats")
    bots: dict[int, Bot] = {}
    for seat, style_name in enumerate(config.bot_styles):
        if seat == config.session.user_seat:
            continue
        style: Style = PRESETS[style_name] if style_name else draw_style(config.population_panel, rng)
        bots[seat] = BOT_TIERS[config.tier](_BOT_NAMES[seat], style)
    return bots


class TableRunner:
    def __init__(self, config: TableConfig):
        self.config = config
        self.bots = _seat_bots(config)
        self.session, self.opening_events = start_session(config.session)
        self.hand: GameState | None = None
        self.hud = Hud(config.session.num_seats)

    @property
    def user_seat(self) -> int | None:
        return self.config.session.user_seat

    def start_hand(self, elapsed_minutes: float = 0.0) -> list[Event]:
        """Deal the next hand and let bots act until the user must act or the hand ends.
        Raises `SessionError` if a hand is in progress or the session cannot deal one."""
        if self.hand is not None and not is_terminal(self.hand):
            raise SessionError(f"hand {self.hand.hand_id} is still in progress")
        self.session, setup, events = begin_hand(self.session, elapsed_minutes)
        self.hand, dealt = new_hand(setup.config, setup.seed, setup.button, setup.stacks, setup.dealt_in, setup.hand_id)
        events.extend(dealt)
        events.extend(self._run_bots())
        return events

    def user_to_act(self) -> bool:
        return self.hand is not None and self.hand.to_act is not None and self.hand.to_act == self.user_seat

    def user_view(self) -> Observation:
        """The user's observation of the current (or last) hand. Raises `SessionError` without
        a user seat or before the first hand."""
        if self.hand is None or self.user_seat is None:
            raise SessionError("no hand or no user seat to observe")
        return observation(self.hand, self.user_seat)

    def user_legal_actions(self) -> LegalActions:
        return legal_actions(self.user_view())

    def act(self, action: Action) -> list[Event]:
        """Apply the user's action, then let bots act. Raises `SessionError` when it is not the
        user's turn and `IllegalActionError` for an illegal action."""
        if not self.user_to_act():
            raise SessionError("it is not the user's turn")
        assert self.hand is not None
        self.hand, events = apply_action(self.hand, action)
        events.extend(self._run_bots())
        return events

    def _run_bots(self) -> list[Event]:
        assert self.hand is not None
        events: list[Event] = []
        while not is_terminal(self.hand) and self.hand.to_act != self.user_seat:
            seat = self.hand.to_act
            assert seat is not None
            rng = Rng(self.hand.seed).derive("bot", seat, len(self.hand.history))
            decision = self.bots[seat].decide(observation(self.hand, seat), rng)
            events.append(
                Event(
                    "BotDecision",
                    {"hand_id": self.hand.hand_id, "seat": seat, **decision.to_dict()},
                )
            )
            self.hand, applied = apply_action(self.hand, decision.action)
            events.extend(applied)
        if is_terminal(self.hand):
            events.extend(self._show_uncontested())
            self.hud.record(self.hand)
            self.session, ended = end_hand(self.session, self.hand)
            events.extend(ended)
        return events

    def _show_uncontested(self) -> list[Event]:
        """With a user at the table, a bot that won without a showdown may show its cards. The
        engine's `shown` is left alone, so no bot, statistic or review ever sees them."""
        hand = self.hand
        assert hand is not None
        awards = hand.awards
        if self.user_seat is None or len(awards) != 1 or len(awards[0].eligible) != 1:
            return []
        seat = awards[0].eligible[0]
        cards = hand.hole_cards[seat]
        assert cards is not None  # a winner was dealt in
        if seat not in self.bots:
            return []
        if Rng(hand.seed).derive("show", seat).random() >= SHOW_CHANCE[self.config.tier]:
            return []
        return [Event("CardsShown", {"hand_id": hand.hand_id, "seat": seat, "cards": list(cards)})]

    def user_rebuy(self) -> list[Event]:
        if self.user_seat is None:
            raise SessionError("bot-only sessions have no user to rebuy")
        self.session, events = rebuy(self.session, self.user_seat)
        return events

    def user_active(self) -> bool:
        return self.user_seat is not None and self.session.active[self.user_seat]

    def fast_forward(self, elapsed_minutes: float = 0.0, max_hands: int = 10_000) -> list[Event]:
        """Play bot-only hands until the session finishes, for a user who is out of a
        tournament. Minute-based levels advance at the pace of the hands played so far.
        Raises `SessionError` if the user is still playing or the cap is reached."""
        if self.user_active():
            raise SessionError("fast-forward is only for a user who is out of the tournament")
        minutes_per_hand = elapsed_minutes / max(1, self.session.hand_number)
        events: list[Event] = []
        for _ in range(max_hands):
            if self.session.finished:
                return events
            elapsed_minutes += minutes_per_hand
            events.extend(self.start_hand(elapsed_minutes))
        raise SessionError(f"session did not finish within {max_hands} hands")

    def finish(self) -> list[Event]:
        self.session, events = end_session(self.session)
        return events

    def seat_label(self, seat: int) -> str:
        if seat == self.user_seat:
            return "You"
        bot = self.bots[seat]
        return bot.name if self.config.hide_styles else f"{bot.name} ({bot.style.name})"

    def bot_labels(self) -> dict[int, str]:
        return {seat: self.seat_label(seat) for seat in self.bots}


MIN_HANDS = 20  # hands before the numbers are worth reading
STYLE_WORDS = {
    "nit": "very tight",
    "tag": "tight-aggressive",
    "lag": "loose-aggressive",
    "calling_station": "calls a lot",
    "maniac": "wild",
    "balanced": "balanced",
}
EXPLOITS = {
    "nit": "Steal its blinds often and respect its raises: when it bets, it has it.",
    "tag": "Solid; look for spots where it folds too much, such as to 3-bets and late-street pressure.",
    "lag": "Call down a little lighter and 3-bet your value wider; it bets and raises a lot.",
    "calling_station": "Value bet thinner and bigger, and almost never bluff: it does not fold.",
    "maniac": "Let it bluff: call down with medium hands and slow-play strong ones.",
    "balanced": "No obvious leak; play solid poker and avoid paying off its value bets.",
}


def pointed_style(stats: SeatStats) -> str | None:
    """The preset whose VPIP and PFR targets are nearest what the seat has shown, or None
    before MIN_HANDS hands."""
    if stats.hands < MIN_HANDS:
        return None
    vpip = stats.voluntary / stats.hands
    pfr = stats.preflop_raises / stats.hands
    distances = {name: (vpip - s.vpip_target) ** 2 + (pfr - s.pfr_target) ** 2 for name, s in PRESETS.items()}
    return min(distances, key=distances.__getitem__)
