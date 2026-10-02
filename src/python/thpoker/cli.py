"""Command-line table: `bin/run_thpoker.py` starts a quick cash game with sensible defaults."""

import argparse
from   collections.abc          import Callable
from   dataclasses              import asdict, replace
from   datetime                 import date
from   functools                import partial
import math
from   operator                 import attrgetter
from   pathlib                  import Path
import sys
from   thpoker.analysis.drills  import (ChartQuestion, DrillResult, ICM,
                                        IcmQuestion, MIXED, PREFLOP, PUSHFOLD,
                                        RIGHT, THRESHOLDS, ThresholdQuestion,
                                        WRONG, boxes, chart_question, due_keys,
                                        grade_chart, grade_icm, grade_share,
                                        icm_question, new_chart_question,
                                        new_icm_question,
                                        new_threshold_question,
                                        threshold_question)
from   thpoker.analysis.ev      import PROFILES, Profile, pick_profile
from   thpoker.analysis.review  import (DecisionReview, HandReview, REFERENCE,
                                        hint, review_decision, review_hand,
                                        summarize, triaged_review)
from   thpoker.analysis.stats   import (DecisionRecord, hand_rows, loss_by_tag,
                                        patterns, progress, record,
                                        write_hands_csv)
from   thpoker.analysis.training \
                                import (BEST_OPTION, Estimate, MISTAKE,
                                        best_action, calibration, grade_option,
                                        mistake_keys, parse_mistake_key,
                                        questions, score)
from   thpoker.bots.bot         import PRESETS
from   thpoker.charts           import seats_from_button
from   thpoker.game.cards       import cards_str
from   thpoker.game.engine      import is_terminal
from   thpoker.game.rng         import Rng, new_seed
from   thpoker.game.rules       import IllegalActionError
from   thpoker.game.session     import (Mode, SessionConfig, SessionError,
                                        TOURNAMENT_PRESETS, TournamentConfig,
                                        preset_schedule)
from   thpoker.game.state       import (Action, ActionType, AnteType,
                                        ConfigError, Event, GameConfig,
                                        GameState, Street)
from   thpoker.odds             import hand_range
from   thpoker.storage          import (SessionLog, default_decisions_log,
                                        default_log_dir, default_training_log,
                                        read_session_log)
from   thpoker.table            import (EXPLOITS, STYLE_WORDS, TableConfig,
                                        TableRunner, pointed_style)
from   thpoker.text             import (describe_action, format_chips,
                                        hand_history_text, narrate,
                                        render_calibration, render_decision,
                                        render_hand, render_summary)
import time
from   typing                   import Any

HELP = """Actions:
  f            fold
  x            check            c   call (or check when free)
  b 50%        bet/raise a fraction of the pot
  b 300        bet/raise to 300 (your chip units)
  b 3bb        bet/raise to 3 big blinds       b 2.5x  raise to 2.5x the current bet
  a            all-in
  h            coach hint (the five steps for this decision)
  ?            show legal actions              q   quit"""


def full_review(
    runner: TableRunner, hand: GameState, profile: Profile, reviewed: dict[int, DecisionReview] | None = None
) -> HandReview:
    user = runner.user_seat
    assert user is not None
    return review_hand(
        hand,
        user,
        runner.bots,
        REFERENCE,
        runner.config.payouts,
        profile.min_branch,
        profile.solver_seconds,
        reviewed,
    )


class PlaySession:
    """What the command line and the web table share around a TableRunner: narration, the
    session log, the level clock, hints and reviews. `scale` is internal chip units per
    displayed chip, so user-entered blinds like 1/2 keep enough precision for pot-sized bets."""

    plain = False

    def __init__(self, runner: TableRunner, scale: int, log: SessionLog | None, profile: Profile):
        self.runner = runner
        self.scale = scale
        self.log = log
        self.profile = profile
        self.started = time.monotonic()  # minute-based tournament levels follow the clock
        self.finished: GameState | None = None  # the user's last finished hand, for review
        if log is not None:
            log.append("table_config", {"config": asdict(runner.config), "scale": scale})
        self.record(runner.opening_events)

    def say(self, line: str):
        print(line)

    def label(self, seat: int) -> str:
        return self.runner.seat_label(seat)

    def chips(self, amount: int) -> str:
        return format_chips(amount, self.scale)

    def minutes(self) -> float:
        return (time.monotonic() - self.started) / 60

    def record(self, events: list[Event]):
        hand = self.runner.hand
        for event in events:
            if self.log is not None:
                self.log.append("event", {"hand_id": hand.hand_id if hand else None, "event": event.to_dict()})
            for line in narrate(event, self.label, self.chips, self.runner.user_seat, self.plain):
                self.say(line)

    def hand_over(self):
        """Keep and log the user's hand once it ends (the log a review reads)."""
        hand = self.runner.hand
        if hand is None or not is_terminal(hand):
            return
        self.finished = hand
        if self.log is not None:
            self.log.append("hand", {"hand": hand.to_dict()})

    def knocked_out(self) -> bool:
        runner = self.runner
        return (
            runner.config.session.mode == Mode.TOURNAMENT and not runner.session.finished and not runner.user_active()
        )

    def fast_forward(self):
        """Play a knocked-out user's tournament to the end, telling how the others finish
        rather than every bot-only hand."""
        events = self.runner.fast_forward(self.minutes())
        self.record([e for e in events if e.kind in ("PlayerEliminated", "TournamentFinished")])

    def hint_lines(self) -> list[str]:
        return render_decision(self.hint_review(), self.runner.bot_labels(), self.scale)

    def hint_review(self) -> DecisionReview:
        """The five steps for the decision the user faces now (DESIGN.md Section 7.8); logged,
        so the decision can be left out of the statistics."""
        hand, user = self.runner.hand, self.runner.user_seat
        assert hand is not None and user is not None
        decision = hint(
            hand,
            user,
            self.runner.bots,
            REFERENCE,
            self.runner.config.payouts,
            self.profile.triage_branch,
        )
        if self.log is not None:
            self.log.append("hint", {"hand_id": hand.hand_id, "index": len(hand.history)})
        return decision

    def review_lines(self, hand: GameState) -> list[str]:
        review = full_review(self.runner, hand, self.profile)
        return render_hand(review, self.runner.bot_labels(), self.scale, grids=True)


class Table(PlaySession):
    """Text rendering and input parsing for play at the command line."""

    def __init__(
        self,
        runner: TableRunner,
        scale: int,
        log: SessionLog | None,
        show_hud: bool,
        hints: bool,
        profile: Profile,
    ):
        self.show_hud = show_hud
        self.hints = hints
        super().__init__(runner, scale, log, profile)

    def with_bb(self, amount: int, big_blind: int) -> str:
        return f"{self.chips(amount)} ({amount / big_blind:.1f}bb)"

    def render(self):
        view = self.runner.user_view()
        big_blind = view.config.big_blind
        board = cards_str(view.board) if view.board else "-"
        print(f"\nBoard: {board}    Pot: {self.with_bb(view.pot, big_blind)}")
        for seat in range(view.config.num_seats):
            if not view.dealt_in[seat]:
                continue
            marks = ("D" if seat == view.button else " ") + (">" if seat == view.to_act else " ")
            status = "folded" if view.folded[seat] else ("all-in" if view.all_in[seat] else "")
            bet = view.committed_this_street[seat]
            bet_text = f"bet {self.chips(bet)}" if bet else ""
            cards = view.hole_cards[seat]
            card_text = f"[{cards_str(cards)}]" if cards else ""
            if self.show_hud and seat != view.seat:
                card_text = card_text or self.runner.hud.seats[seat].summary()
            print(
                f" {marks} {self.runner.seat_label(seat):<28} {self.with_bb(view.stacks[seat], big_blind):>18}"
                f"  {bet_text:<12} {status:<7} {card_text}"
            )

    def parse(self, text: str) -> Action | None:
        """Turn a typed command into an Action; None means the command was handled (help).
        Raises `ValueError` with a message for the user."""
        view = self.runner.user_view()
        legal = self.runner.user_legal_actions()
        parts = text.strip().lower().split()
        if not parts:
            raise ValueError("type an action, or ? for help")
        command = parts[0]
        if command == "?":
            self.show_legal()
            return None
        if command == "h":
            self.show_hint()
            return None
        if command == "f":
            return Action(ActionType.FOLD)
        if command in ("x", "k"):
            return Action(ActionType.CHECK)
        if command == "c":
            return Action(ActionType.CHECK if legal.can_check else ActionType.CALL)
        if command == "a":
            if legal.can_bet or legal.can_raise:
                return Action(ActionType.BET if legal.can_bet else ActionType.RAISE, legal.max_raise_to)
            return Action(ActionType.CALL)
        if command in ("b", "r"):
            if len(parts) != 2:
                raise ValueError("give a size, e.g. b 50%, b 300, b 3bb, b 2.5x")
            if not (legal.can_bet or legal.can_raise):
                raise ValueError("you cannot bet or raise here")
            size = parts[1]
            to_call = view.current_bet - view.committed_this_street[view.seat]
            if size.endswith("%"):
                target = view.current_bet + round(float(size[:-1]) / 100 * (view.pot + to_call))
            elif size.endswith("bb"):
                target = round(float(size[:-2]) * view.config.big_blind)
            elif size.endswith("x"):
                target = round(float(size[:-1]) * view.current_bet)
            else:
                target = round(float(size) * self.scale)
            if not legal.min_raise_to <= target <= legal.max_raise_to:
                raise ValueError(
                    f"size must be between {self.chips(legal.min_raise_to)} and {self.chips(legal.max_raise_to)}"
                )
            return Action(ActionType.BET if legal.can_bet else ActionType.RAISE, target)
        raise ValueError(f"unknown command {command!r}; ? for help")

    def show_legal(self):
        legal = self.runner.user_legal_actions()
        options = []
        if legal.can_fold:
            options.append("f fold")
        if legal.can_check:
            options.append("x check")
        if legal.can_call:
            options.append(f"c call {self.chips(legal.call_amount)}")
        if legal.can_bet or legal.can_raise:
            word = "bet" if legal.can_bet else "raise to"
            options.append(f"b {word} {self.chips(legal.min_raise_to)}..{self.chips(legal.max_raise_to)}")
        print("Legal: " + " | ".join(options))
        print(HELP)

    def play_hand(self) -> bool:
        """Play one hand; False when the user quits."""
        self.record(self.runner.start_hand(self.minutes()))
        view = self.runner.user_view()
        if view.dealt_in[view.seat]:
            print(
                f"\n=== Hand {self.runner.session.hand_number} · blinds {self.chips(view.config.small_blind)}/"
                f"{self.chips(view.config.big_blind)} · you hold {cards_str(view.my_cards)}"
            )
        while self.runner.user_to_act():
            self.render()
            text = input("Your action (? for help): ")
            if text.strip().lower() == "q":
                return False
            try:
                action = self.parse(text)
            except ValueError as error:
                print(f"  {error}")
                continue
            if action is None:
                continue
            try:
                events = self.runner.act(action)
            except IllegalActionError as error:
                print(f"  {error}")
                continue
            if self.log is not None:
                self.log.append(
                    "user_action",
                    {"hand_id": view.hand_id, "action": action.to_dict(), "timestamp": time.time()},
                )
            self.record(events)
        self.hand_over()
        if self.runner.hand is not None and self.runner.hand.street == Street.COMPLETE and view.dealt_in[view.seat]:
            final = self.runner.user_view()
            net = final.stacks[final.seat] - final.starting_stacks[final.seat]
            print(
                f"  You {'won' if net >= 0 else 'lost'} {self.chips(abs(net))}; "
                f"stack {self.with_bb(final.stacks[final.seat], final.config.big_blind)}"
            )
        return True

    def guess_styles(self):
        """Opponent identification (DESIGN.md Section 7.6): guess each hidden style, then see
        the answer, what the numbers pointed to, and how to beat it."""
        names = list(PRESETS)
        print("\n  Guess each opponent's style: " + ", ".join(f"{i} {name}" for i, name in enumerate(names, 1)))
        right = 0
        for seat, bot in self.runner.bots.items():
            stats = self.runner.hud.seats[seat]
            text = input(f"  {bot.name} ({stats.summary()}): ").strip()
            guess = names[int(text) - 1] if text.isdigit() and 1 <= int(text) <= len(names) else None
            truth = bot.style.name
            right += guess == truth
            pointed = pointed_style(stats)
            clue = f" The numbers pointed to {pointed}." if pointed else ""
            print(f"   {'Right' if guess == truth else f'It is {truth}'}.{clue} {EXPLOITS[truth]}")
            if self.log is not None:
                self.log.append(
                    "style_guess",
                    {"seat": seat, "guess": guess, "style": truth, "hands": stats.hands},
                )
        print(f"  {right} of {len(self.runner.bots)} right.")

    def show_hint(self):
        if not self.hints:
            print("  Hints are off for this session (--hints turns them on).")
            return
        print("  Thinking...")
        for line in self.hint_lines():
            print("  " + line)

    def review_last_hand(self, quiz: bool, training: SessionLog | None):
        """Review the hand just played; with `quiz`, ask for estimates before each reveal."""
        hand, user = self.runner.hand, self.runner.user_seat
        if hand is None or user is None or not any(e.seat == user for e in hand.history):
            print("  Nothing to review: you made no decision in that hand.")
            return
        print("  Reviewing (a few seconds)...")
        if quiz:
            review = full_review(self.runner, hand, self.profile)
            decisions = [(hand.hand_id, d) for d in review.decisions]
            quiz_decisions(decisions, self.runner.bot_labels(), self.scale, training)
            return
        for line in self.review_lines(hand):
            print("  " + line)


def record_decisions(
    reviews: list[HandReview],
    session: str,
    paid_places: int | None,
    path: Path,
    hinted: set[tuple[str, int]],
    training: bool = False,
):
    """Append each reviewed decision's record for leak statistics, once per decision; decisions
    made after a coach hint are left out, and training decisions are tagged as such."""
    known = {(r.session, r.hand_id, r.index) for r in DecisionRecord.read(path, "decision")}
    log = SessionLog(path)
    for review in reviews:
        for decision in review.decisions:
            key = (review.hand.hand_id, decision.index)
            if (session, *key) not in known and key not in hinted:
                known.add((session, *key))
                log.append(
                    "decision",
                    record(decision, session, review.hand.hand_id, paid_places, training).to_dict(),
                )


def _ask_share(question: str, empty: str = "skip") -> float | None:
    """A percentage typed as 35 or 35%, returned as a share; None for Enter."""
    while True:
        text = input(f"  {question} (%, Enter to {empty}): ").strip().rstrip("%")
        if not text:
            return None
        try:
            value = float(text)
        except ValueError:
            print("  Type a number such as 35.")
            continue
        if 0 <= value <= 100:
            return value / 100
        print("  Type a number from 0 to 100.")


def _ask_option(decision: DecisionReview, scale: int, empty: str = "skip") -> Action | None:
    """One of the decision's options by number; None for Enter."""
    options = [o.action for o in decision.reference]
    for number, action in enumerate(options, 1):
        print(f"   {number}. {describe_action(action, scale)}")
    while True:
        text = input(f"  Best option (number, Enter to {empty}): ").strip()
        if not text:
            return None
        if text.isdigit() and 1 <= int(text) <= len(options):
            return options[int(text) - 1]
        print(f"  Type a number from 1 to {len(options)}.")


_QUESTIONS = {
    "equity": "Your equity against their range",
    "required_equity": "Equity needed to call",
}


def quiz_decisions(
    decisions: list[tuple[str, DecisionReview]],
    labels: dict[int, str],
    scale: int,
    training: SessionLog | None,
) -> list[Estimate]:
    """Predict-then-reveal: show each decision's situation and ranges, take the user's
    estimates, then reveal the full review. Estimates are appended to `training`."""
    estimates = []
    for hand_id, decision in decisions:
        print()
        for line in render_decision(decision, labels, scale, reveal=False):
            print("  " + line)
        for kind in questions(decision):
            if kind == BEST_OPTION:
                chosen = _ask_option(decision, scale)
                if chosen is None:
                    continue
                estimate = score(decision, kind, chosen, hand_id)
                best = describe_action(best_action(decision), scale)
                print("  Right." if estimate.actual else f"  The best option against the reference is {best}.")
            else:
                guess = _ask_share(_QUESTIONS[kind])
                if guess is None:
                    continue
                estimate = score(decision, kind, guess, hand_id)
                print(f"  You said {estimate.guess:.0%}; it is {estimate.actual:.0%}.")
            estimates.append(estimate)
            if training is not None:
                training.append("estimate", estimate.to_dict())
        for line in render_decision(decision, labels, scale):
            print("  " + line)
    return estimates


def build_config(args: argparse.Namespace) -> tuple[TableConfig, int]:
    """Table configuration and display scale from command-line arguments. Raises `ConfigError`."""
    seed = args.seed if args.seed is not None else new_seed()
    small_text, _, big_text = args.blinds.partition("/")
    user_small, user_big = float(small_text), float(big_text or small_text)
    if not 0 < user_small <= user_big:
        raise ConfigError(f"blinds must satisfy 0 < small <= big, got {args.blinds}")
    scale = max(1, math.ceil(100 / user_big))
    small_blind, big_blind = round(user_small * scale), round(user_big * scale)
    cash_only = {
        "--ante": args.ante,
        "--ante-type": args.ante_type,
        "--auto-rebuy": args.auto_rebuy,
        "--reset-stacks": args.reset_stacks,
    }
    tournament_only = {
        "--preset": args.preset,
        "--hands-per-level": args.hands_per_level,
        "--minutes-per-level": args.minutes_per_level,
    }
    training = args.mode == "training"
    if not training and (args.position or args.hands):
        raise ConfigError("--position and --hands apply to training only")

    if args.mode == "tournament":
        misplaced = [name for name, value in cash_only.items() if value]
        if misplaced:
            raise ConfigError(f"{', '.join(misplaced)} apply to cash games only")
        if small_blind * 2 != big_blind:
            raise ConfigError("a tournament's small blind is always half its big blind")
        if args.hands_per_level and args.minutes_per_level:
            raise ConfigError("give either --hands-per-level or --minutes-per-level, not both")
        preset_stack, preset_hands = TOURNAMENT_PRESETS[args.preset or "regular"]
        stack_bb = args.stack if args.stack is not None else preset_stack
        minutes = args.minutes_per_level
        tournament = TournamentConfig(
            blind_schedule=preset_schedule(big_blind),
            hands_per_level=None if minutes else (args.hands_per_level or preset_hands),
            minutes_per_level=minutes,
        )
        session = SessionConfig(Mode.TOURNAMENT, seed, args.seats, round(stack_bb * big_blind), tournament=tournament)
    else:
        misplaced = [name for name, value in tournament_only.items() if value]
        if misplaced:
            raise ConfigError(f"{', '.join(misplaced)} apply to tournaments only")
        if args.ante_type and not args.ante:
            raise ConfigError("--ante-type needs --ante")
        ante = round(args.ante * scale)
        ante_type = AnteType.NONE
        if ante:
            bb_ante = args.ante_type == "bb-ante"
            ante_type = AnteType.BIG_BLIND_ANTE if bb_ante else AnteType.PER_PLAYER
        stack = round((args.stack if args.stack is not None else 100) * big_blind)
        position = None
        if args.position:
            names = seats_from_button(args.seats)
            if args.position.upper() not in names:
                raise ConfigError(f"--position with {args.seats} players is noe of {', '.join(names)}")
            position = names.index(args.position.upper())
        session = SessionConfig(
            Mode.TRAINING if training else Mode.CASH,
            seed,
            args.seats,
            stack,
            cash_blinds=GameConfig(args.seats, small_blind, big_blind, ante, ante_type),
            reset_stacks_each_hand=args.reset_stacks or training,
            auto_rebuy=args.auto_rebuy is not None,
            rebuy_threshold=round((args.auto_rebuy or 0) * big_blind),
            user_position=position,
            user_hands=hand_range(args.hands) if args.hands else (),
        )
    styles = tuple(args.style for _ in range(args.seats))
    return TableConfig(session, styles, args.panel, args.hide_styles, args.tier), scale


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="bin/run_thpoker.py", description="No-Limit Hold'em practice table.")
    parser.add_argument("--mode", choices=("cash", "tournament", "training"), default="cash")
    parser.add_argument("--seats", type=int, default=6, help="2 to 8 seats including you (default 6)")
    parser.add_argument("--blinds", default="50/100", help="SB/BB in your chip units (default 50/100)")
    parser.add_argument("--ante", type=float, default=0, help="ante in your chip units (cash)")
    parser.add_argument("--ante-type", choices=("per-player", "bb-ante"), help="cash (default per-player)")
    parser.add_argument("--stack", type=float, help="starting stack in big blinds (default 100; tournament: preset)")
    parser.add_argument("--preset", choices=sorted(TOURNAMENT_PRESETS), help="tournament (default regular)")
    parser.add_argument("--hands-per-level", type=int, help="tournament level length in hands")
    parser.add_argument("--position", help="training: always sit here (BTN, SB, BB, UTG, ..., CO)")
    parser.add_argument(
        "--hands",
        help="training: deal you only X-Y (the top X%% to Y%% of hands), pairs, small-aces or suited-connectors",
    )
    parser.add_argument("--minutes-per-level", type=float, help="tournament level length in minutes")
    parser.add_argument("--style", help="give every bot this style instead of drawing from the panel")
    parser.add_argument("--panel", default="online_micro", help="population panel for bot styles")
    parser.add_argument("--tier", type=int, choices=(1, 2, 3), default=2, help="bot difficulty (default 2)")
    parser.add_argument(
        "--hud",
        action=argparse.BooleanOptionalAction,
        help="show bot statistics (default on, off with --hide-styles)",
    )
    parser.add_argument("--hide-styles", action="store_true", help="hide bot styles (opponent reading practice)")
    parser.add_argument(
        "--guess-every",
        type=int,
        default=40,
        metavar="HANDS",
        help="with --hide-styles, guess the bots' styles every this many hands (0: never)",
    )
    parser.add_argument(
        "--hints",
        action=argparse.BooleanOptionalAction,
        help="allow the coach hint (h) during play (default on in cash, off in tournaments)",
    )
    add_device_argument(parser)
    parser.add_argument("--reset-stacks", action="store_true", help="cash: reset stacks every hand")
    parser.add_argument("--auto-rebuy", type=float, metavar="BB", help="cash: rebuy when below this many big blinds")
    parser.add_argument("--seed", type=int, help="session seed for an exactly repeatable session")
    parser.add_argument("--log-dir", type=Path, default=default_log_dir())
    parser.add_argument("--no-log", action="store_true", help="keep no session or training log")
    parser.add_argument("--training-log", type=Path, default=default_training_log())
    return parser.parse_args(argv)


def load_session(
    path: Path,
) -> tuple[TableConfig, int, TableRunner, list[GameState], list[dict[str, Any]]]:
    """A logged session with a user: its configuration, display scale, a runner that rebuilds
    the same bots, its hands, and all its records. Raises `OSError` or `ValueError`."""
    records = read_session_log(path)
    try:
        logged = next((r for r in records if r["type"] == "table_config"), None)
        if logged is None:
            raise ValueError("no table configuration in the log")
        config = TableConfig.from_dict(logged["config"])
        hands = [GameState.from_dict(r["hand"]) for r in records if r["type"] == "hand"]
        scale = logged["scale"]
    except (KeyError, TypeError) as error:
        raise ValueError(f"damaged log: {error!r}") from error
    if config.session.user_seat is None:
        raise ValueError("bot-only sessions have no decisions to review")
    return config, scale, TableRunner(config), hands, records


def seat_labels(runner: TableRunner, styled: bool) -> dict[int, str]:
    """Seat names in plain words, with each bot's style when `styled` and styles are shown."""
    labels = {} if runner.user_seat is None else {runner.user_seat: "You"}
    for seat, bot in runner.bots.items():
        style = styled and not runner.config.hide_styles
        labels[seat] = f"{bot.name} ({STYLE_WORDS[bot.style.name]})" if style else bot.name
    return labels


def game_name(runner: TableRunner) -> str:
    mode = runner.config.session.mode
    return "tournament" if mode == Mode.TOURNAMENT else f"{mode.value.lower()} game"


def collect_shows(events: list[Event], shows: dict[str, dict[int, tuple[int, int]]]):
    """Collect cards a bot showed without a showdown, by hand id."""
    for event in events:
        if event.kind == "CardsShown":
            cards = tuple(event.data["cards"])
            shows.setdefault(event.data["hand_id"], {})[event.data["seat"]] = (cards[0], cards[1])


def logged_shows(records: list[dict[str, Any]]) -> dict[str, dict[int, tuple[int, int]]]:
    """The cards bots showed without a showdown in a session log. Raises `ValueError`."""
    try:
        events = [Event.from_dict(r["event"]) for r in records if r["type"] == "event"]
    except (KeyError, TypeError) as error:
        raise ValueError(f"damaged log: {error!r}") from error
    shows: dict[str, dict[int, tuple[int, int]]] = {}
    collect_shows(events, shows)
    return shows


def plain_history(runner: TableRunner, hand: GameState, scale: int, shown: dict[int, tuple[int, int]]) -> list[str]:
    """The hand as the user saw it, with no styles: what the page copies and the CSV keeps."""
    user = runner.user_seat
    assert user is not None
    return hand_history_text(hand, user, seat_labels(runner, False), scale, game_name(runner), shown)


def session_histories(
    runner: TableRunner,
    scale: int,
    hands: list[GameState],
    shows: dict[str, dict[int, tuple[int, int]]],
) -> dict[str, str]:
    """Each hand the user was dealt into, as the user saw it, by hand id: the CSV's history."""
    user = runner.user_seat
    assert user is not None
    return {
        h.hand_id: "\n".join(plain_history(runner, h, scale, shows.get(h.hand_id, {})))
        for h in hands
        if h.dealt_in[user]
    }


def add_device_argument(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--device",
        choices=("auto", *PROFILES),
        default="auto",
        help="how much work reviews and hints may do (default: time this machine)",
    )


def review_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="bin/run_thpoker.py review",
        description="Review a logged session: its largest mistakes, or one hand in full.",
    )
    parser.add_argument("log", type=Path, help="a session log (JSONL) written while playing")
    parser.add_argument("--hand", type=int, help="show this hand number in full")
    parser.add_argument("--top", type=int, default=5, help="mistakes to list (default 5)")
    parser.add_argument("--grids", action="store_true", help="show range grids in a full hand")
    parser.add_argument(
        "--quiz",
        action="store_true",
        help="estimate before each reveal (the hand, or the top mistakes)",
    )
    parser.add_argument("--training-log", type=Path, default=default_training_log())
    parser.add_argument("--decisions-log", type=Path, default=default_decisions_log())
    add_device_argument(parser)
    args = parser.parse_args(argv)
    profile = pick_profile(args.device)
    try:
        config, scale, runner, hands, records = load_session(args.log)
    except (OSError, ValueError) as error:
        print(f"Cannot review {args.log}: {error}", file=sys.stderr)
        return 2
    user = config.session.user_seat
    assert user is not None  # checked by load_session
    bots = runner.bots
    played = [h for h in hands if any(e.seat == user for e in h.history)]
    labels = runner.bot_labels()
    if args.hand is not None:
        chosen = [h for h in hands if h.hand_id == f"hand-{args.hand}"]
        if not chosen:
            print(f"No hand {args.hand} in {args.log}", file=sys.stderr)
            return 2
        review = full_review(runner, chosen[0], profile)
        if args.quiz:
            quiz_decisions(
                [(chosen[0].hand_id, d) for d in review.decisions],
                labels,
                scale,
                SessionLog(args.training_log),
            )
        else:
            print("\n".join(render_hand(review, labels, scale, args.grids)))
        return 0
    reviews = []
    for number, hand in enumerate(played, 1):
        print(f"\rReviewing hand {number} of {len(played)}...", end="", file=sys.stderr, flush=True)
        reviews.append(triaged_review(hand, user, bots, REFERENCE, config.payouts, profile))
    print(file=sys.stderr)
    payouts = config.payouts
    hinted = {(r["hand_id"], r["index"]) for r in records if r["type"] == "hint"}
    record_decisions(
        reviews,
        args.log.stem,
        len(payouts) if payouts else None,
        args.decisions_log,
        hinted,
        config.session.mode == Mode.TRAINING,
    )
    summary = summarize(reviews, args.top)
    print("\n".join(render_summary(summary, scale)))
    if args.quiz:
        quiz_decisions(summary.mistakes, labels, scale, SessionLog(args.training_log))
    return 0


def leaks_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="bin/run_thpoker.py leaks",
        description="Patterns across every reviewed decision, and the spots that cost the most.",
    )
    parser.add_argument("--decisions-log", type=Path, default=default_decisions_log())
    parser.add_argument("--top", type=int, default=10, help="costly spots to list (default 10)")
    args = parser.parse_args(argv)
    records = DecisionRecord.read(args.decisions_log, "decision")
    found = patterns(records)
    if not found:
        cash = sum(r.tags.get("mode") == "cash" for r in records)
        print(
            f"{len(records)} reviewed decisions, {cash} of them in cash games. The patterns read cash "
            f"decisions and need a few of each kind: review more cash sessions to see them."
        )
        return 0
    print(f"{len(records)} reviewed decisions.")
    print("\n".join(found))
    print("\nLoss per decision by spot (bb):")
    for key, value, count, mean in loss_by_tag([r for r in records if r.tags.get("mode") == "cash"])[: args.top]:
        print(f"  {key:<12} {value:<16} {mean:6.2f}  ({count} decisions)")
    return 0


def progress_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="bin/run_thpoker.py progress",
        description="Reference EV lost per 100 hands, whatever the cards did, next to the bots.",
    )
    parser.add_argument("--decisions-log", type=Path, default=default_decisions_log())
    args = parser.parse_args(argv)
    records = DecisionRecord.read(args.decisions_log, "decision")
    result = progress(records)
    if result is None:
        print("No reviewed cash decisions yet: review a session first (bin/run_thpoker.py review LOG).")
        return 0
    print(
        f"{result.loss_per_100:.1f} ± {result.stderr_per_100:.1f} bb lost per 100 hands against the "
        f"reference over {result.hands} hands ({result.decisions} decisions): {result.comparison}."
    )
    return 0


def calibration_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="bin/run_thpoker.py calibration",
        description="How close your predict-then-reveal estimates have been.",
    )
    parser.add_argument("--training-log", type=Path, default=default_training_log())
    args = parser.parse_args(argv)
    estimates = Estimate.read(args.training_log, "estimate")
    print("\n".join(render_calibration(calibration(estimates))))
    return 0


_GRADE_WORDS = {RIGHT: "Right.", MIXED: "Mixed: the chart sometimes plays this.", WRONG: "Wrong."}
_SHARE_WORDS = {RIGHT: "Right.", MIXED: "Close.", WRONG: "Wrong."}
_FRESH_TRIES = 50  # attempts at a question neither scheduled nor asked this session
_OPTION_WORDS = {
    RIGHT: "Right.",
    MIXED: "Close: no mistake, but not the best option.",
    WRONG: "Wrong.",
}
MISTAKES = "mistakes"


def drill_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="bin/run_thpoker.py drill",
        description="Drills with spaced repetition: preflop charts, push/fold, bet-size "
        "thresholds, ICM push/fold, and your own reviewed mistakes.",
    )
    parser.add_argument("kind", choices=(PREFLOP, PUSHFOLD, THRESHOLDS, ICM, MISTAKES))
    parser.add_argument("--count", type=int, default=10, help="questions (default 10)")
    parser.add_argument("--seed", type=int, help="seed for repeatable new questions")
    parser.add_argument("--training-log", type=Path, default=default_training_log())
    parser.add_argument(
        "--decisions-log",
        type=Path,
        default=default_decisions_log(),
        help="mistakes: the reviewed decisions",
    )
    parser.add_argument("--log-dir", type=Path, default=default_log_dir(), help="mistakes: the session logs")
    add_device_argument(parser)
    args = parser.parse_args(argv)
    results = DrillResult.read(args.training_log, "drill")
    today = date.today().toordinal()
    if args.kind == MISTAKES:
        _print_grades(mistake_drill(args, results, today))
        return 0
    new: Callable[[Rng], ChartQuestion | ThresholdQuestion | IcmQuestion]
    rebuild: Callable[[str], ChartQuestion | ThresholdQuestion | IcmQuestion]
    if args.kind == THRESHOLDS:
        new, rebuild = new_threshold_question, threshold_question
    elif args.kind == ICM:
        new, rebuild = new_icm_question, icm_question
    else:
        new, rebuild = partial(new_chart_question, args.kind), chart_question
    due = due_keys(results, today, args.kind)
    scheduled = set(boxes(results))
    asked: set[str] = set()
    rng = Rng(new_seed() if args.seed is None else args.seed)
    log = SessionLog(args.training_log)
    grades = []
    for number in range(1, args.count + 1):
        key = due.pop(0) if due else None
        if key is None:
            # A new question should be new: not a spot already scheduled or asked today.
            for attempt in range(_FRESH_TRIES):
                fresh = new(rng.derive(number, attempt)).key
                if fresh not in scheduled and fresh not in asked:
                    break
            key = fresh
        asked.add(key)
        try:
            question = rebuild(key)
        except ValueError as error:  # a logged key the current charts cannot rebuild
            print(f"Skipping {key}: {error}", file=sys.stderr)
            continue
        print(f"\n{number}. {question.describe()}")
        if isinstance(question, ThresholdQuestion):
            guess = _ask_share("Your answer", empty="stop")
            if guess is None:
                break
            grade = grade_share(question, guess)
            print(f"  {_SHARE_WORDS[grade]} It is {question.answer:.1%}.")
        else:
            for index, option in enumerate(question.options, 1):
                print(f"   {index}. {option}")
            text = input("  Your choice (number, Enter to stop): ").strip()
            if not text.isdigit() or not 1 <= int(text) <= len(question.options):
                break
            if isinstance(question, ChartQuestion):
                grade = grade_chart(question, int(text) - 1)
                shares = ", ".join(f"{o} {f:.0%}" for o, f in zip(question.options, question.frequencies))
                print(f"  {_GRADE_WORDS[grade]} The chart: {shares}.")
            else:
                grade = grade_icm(question, int(text) - 1)
                print(f"  {_OPTION_WORDS[grade]} {question.verdicts()}")
        grades.append(grade)
        log.append("drill", DrillResult(question.key, grade, today).to_dict())
    _print_grades(grades)
    return 0


def _print_grades(grades: list[str]):
    if grades:
        print(f"\n{grades.count(RIGHT)} right, {grades.count(MIXED)} mixed, {grades.count(WRONG)} wrong.")


def mistake_drill(args: argparse.Namespace, results: list[DrillResult], today: int) -> list[str]:
    """Re-ask reviewed mistakes (DESIGN.md Section 7.4): the due ones first, then the costliest
    not yet drilled. Each is rebuilt from its session log and asked as "the best option?", then
    graded and scheduled like the other drills. Returns the grades."""
    records = DecisionRecord.read(args.decisions_log, "decision")
    mistakes = sorted((r for r in records if r.verdict == "mistake"), key=attrgetter("loss"), reverse=True)
    by_key = dict(zip(mistake_keys(mistakes), mistakes))
    scheduled = set(boxes(results))
    queue = due_keys(results, today, MISTAKE) + [k for k in by_key if k not in scheduled]
    if not queue:
        print("No mistakes to drill: review a session first (bin/run_thpoker.py review LOG).")
        return []
    profile = pick_profile(args.device)
    log = SessionLog(args.training_log)
    sessions: dict[str, tuple[TableConfig, int, TableRunner, list[GameState], list[dict[str, Any]]]] = {}
    grades: list[str] = []
    for key in queue:
        if len(grades) == args.count:
            break
        session, hand_id, index = parse_mistake_key(key)
        if session not in sessions:
            try:
                sessions[session] = load_session(args.log_dir / f"{session}.jsonl")
            except (OSError, ValueError) as error:
                print(f"Skipping {session}: {error}", file=sys.stderr)
                continue
        config, scale, runner, hands, _ = sessions[session]
        hand = next((h for h in hands if h.hand_id == hand_id), None)
        user = config.session.user_seat
        if hand is None or user is None:
            print(f"Skipping {key}: the hand is not in its session log", file=sys.stderr)
            continue
        print(f"\n{len(grades) + 1}. From {session}, {hand_id}. Thinking...")
        try:
            decision = review_decision(
                hand,
                user,
                runner.bots,
                REFERENCE,
                index,
                config.payouts,
                profile.min_branch,
                profile.solver_seconds,
            )
        except ValueError as error:  # the log changed since the review, e.g. a reused seed
            print(f"Skipping {key}: {error}", file=sys.stderr)
            continue
        labels = runner.bot_labels()
        # The question hides what was played, including a size the abstraction lacks; the
        # reveal shows it.
        question = replace(
            decision,
            chosen=None,
            reference=[o for o in decision.reference if o.abstract is not None],
        )
        for line in render_decision(question, labels, scale, reveal=False):
            print("  " + line)
        chosen = _ask_option(question, scale, empty="stop")
        if chosen is None:
            break
        grade = grade_option(question, chosen)
        print(f"  {_OPTION_WORDS[grade]}")
        for line in render_decision(decision, labels, scale):
            print("  " + line)
        grades.append(grade)
        log.append("drill", DrillResult(key, grade, today).to_dict())
    return grades


def _ask(prompt: str) -> str:
    return input(prompt).strip().lower()


def session_rows(
    path: Path,
    config: TableConfig,
    scale: int,
    hands: list[GameState],
    decisions: list[DecisionRecord],
    histories: dict[str, str],
) -> list[dict[str, Any]]:
    """One CSV row per hand of a logged session, dated by the log's last write."""
    user = config.session.user_seat
    assert user is not None  # checked by load_session
    mode = config.session.mode.value.lower()
    day = date.fromtimestamp(path.stat().st_mtime).isoformat()
    return hand_rows(hands, user, scale, path.stem, day, mode, decisions, histories)


def export_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="thpoker export",
        description="Write one CSV row per hand of saved sessions, for a spreadsheet or pandas.",
    )
    parser.add_argument("logs", nargs="*", type=Path, help="session logs (default: every saved session)")
    parser.add_argument("--out", type=Path, help="the CSV file to write (default: print it)")
    parser.add_argument("--decisions-log", type=Path, default=default_decisions_log())
    args = parser.parse_args(argv)
    try:
        decisions = DecisionRecord.read(args.decisions_log, "decision")
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Leaving the review columns blank: {args.decisions_log}: {error}", file=sys.stderr)
        decisions = []
    rows: list[dict[str, Any]] = []
    for path in args.logs or sorted(default_log_dir().glob("session-*.jsonl")):
        try:
            config, scale, runner, hands, records = load_session(path)
            histories = session_histories(runner, scale, hands, logged_shows(records))
        except (OSError, ValueError) as error:
            print(f"Skipping {path}: {error}", file=sys.stderr)
            continue
        rows += session_rows(path, config, scale, hands, decisions, histories)
    if args.out is None:
        write_hands_csv(rows, sys.stdout)
    else:
        with args.out.open("w", encoding="utf-8", newline="") as handle:
            write_hands_csv(rows, handle)
        print(f"Wrote {len(rows)} hands to {args.out}")
    return 0


COMMANDS = {
    "drill": drill_main,
    "export": export_main,
    "review": review_main,
    "calibration": calibration_main,
    "leaks": leaks_main,
    "progress": progress_main,
}


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] in COMMANDS:
        return COMMANDS[argv[0]](argv[1:])
    args = parse_args(argv)
    try:
        config, scale = build_config(args)
        runner = TableRunner(config)
    except (ConfigError, ValueError) as error:
        print(f"Invalid setup: {error}", file=sys.stderr)
        return 2
    session = config.session
    log = training = None
    if not args.no_log:
        log = SessionLog(args.log_dir / f"session-{session.seed}.jsonl")
        training = SessionLog(args.training_log)
        print(f"Logging to {log.path}")
    show_hud = args.hud if args.hud is not None else not args.hide_styles
    hints = args.hints if args.hints is not None else session.mode != Mode.TOURNAMENT
    table = Table(runner, scale, log, show_hud, hints, pick_profile(args.device))
    print(f"{session.mode.value.title()} session, {session.num_seats} seats, seed {session.seed}. Type ? for help.")
    try:
        while not runner.session.finished:
            if runner.session.awaiting_rebuy:
                if _ask("You are out of chips. Rebuy? [y/N] ") != "y":
                    break
                table.record(runner.user_rebuy())
            if table.knocked_out():
                if _ask("You are out. Fast-forward to the end? [y/N] ") == "y":
                    table.fast_forward()
                break
            if not table.play_hand():
                break
            if runner.session.finished:
                break
            if args.hide_styles and args.guess_every and runner.session.hand_number % args.guess_every == 0:
                table.guess_styles()
            answer = _ask("\nEnter for next hand, r to review this hand, p to predict then review, q to quit: ")
            if answer in ("r", "p"):
                table.review_last_hand(answer == "p", training)
                answer = _ask("\nEnter for next hand, q to quit: ")
            if answer == "q":
                break
    except (KeyboardInterrupt, EOFError):
        print()
    except SessionError as error:
        print(f"Session stopped: {error}", file=sys.stderr)
    ended = runner.finish()
    table.record(ended)
    net = ended[0].data["net"]
    if session.user_seat is not None and session.mode != Mode.TOURNAMENT:
        print(
            f"Session over after {runner.session.hand_number} hands. Net: {table.chips(net[session.user_seat])} chips."
        )
    return 0
