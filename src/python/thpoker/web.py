"""A local web table (DESIGN.md Section 9.2) on the standard-library HTTP server:
`thpoker web` serves play, the coach, hand review and past sessions in a browser at
http://127.0.0.1:8000.

It listens on this machine only, keeps its sessions in memory, and logs hands like the command
line. During play it only ever returns the user's observation, never the full state; the coach
and the history show only the user's cards and cards that were shown.
"""

import argparse
from   collections              import defaultdict
from   collections.abc          import Callable
from   concurrent.futures       import CancelledError
from   contextlib               import suppress
from   dataclasses              import dataclass
from   datetime                 import date
from   functools                import partial
import http.server
import io
import json
import math
from   pathlib                  import Path
import re
import secrets
from   thpoker.analysis.ev      import (OptionValue, PROFILES, Profile,
                                        pick_profile)
from   thpoker.analysis.review  import (DecisionReview, HandRating, HandReview,
                                        REFERENCE, best_option, hand_rating,
                                        move_ratings, position_names,
                                        range_view, rating_band, thresholds)
from   thpoker.analysis.stats   import (DecisionRecord, hand_rows,
                                        write_hands_csv)
from   thpoker.analysis.tracking \
                                import track
from   thpoker.bots.abstraction import to_action
from   thpoker.bots.bot         import Bot
from   thpoker.bots.equity_bot  import public_seed
from   thpoker.charts           import seats_from_button
from   thpoker.cli              import (PlaySession, Task, add_device_argument,
                                        background, build_config,
                                        collect_shows, full_review, game_name,
                                        logged_ratings, logged_shows,
                                        parse_args, plain_history, result_or,
                                        seat_labels, session_histories)
from   thpoker.game.cards       import PREFLOP_CLASSES, card_str
from   thpoker.game.engine      import is_terminal, observation, replay_states
from   thpoker.game.evaluator   import describe, evaluate
from   thpoker.game.rng         import Rng
from   thpoker.game.rules       import IllegalActionError
from   thpoker.game.session     import Mode
from   thpoker.game.state       import Action, ActionType, Event, GameState
from   thpoker.odds             import (hand_equity, hand_range, hand_rank,
                                        hand_window, range_share)
from   thpoker.storage          import (SessionLog, default_decisions_log,
                                        default_log_dir, read_records)
from   thpoker.table            import (EXPLOITS, MIN_HANDS, STYLE_WORDS,
                                        TableConfig, TableRunner)
from   thpoker.text             import (chips_shown, decision_details,
                                        decision_headline, decision_summary,
                                        describe_action, hand_history_text,
                                        hand_result_text, percent_text,
                                        plain_decision_text, rated_hand_text)
import threading
from   typing                   import Any, NamedTuple


STATIC = Path(__file__).resolve().parent / "web_static"
ASSETS = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
}
MAX_BODY = 16_384
MAX_SESSIONS = 8  # the oldest session is dropped beyond this
RECENT_HANDS = 20  # rows of "Your hands" sent with the state
HISTORY_LIMIT = 50  # newest saved sessions listed
_KINDS = {kind.value.lower(): kind for kind in ActionType}
_PAST = {
    ActionType.FOLD: "folded",
    ActionType.CHECK: "checked",
    ActionType.CALL: "called",
    ActionType.BET: "bet",
    ActionType.RAISE: "raised to",
}
_BLINDS = re.compile(r"\d+(\.\d+)?/\d+(\.\d+)?")
_POSITION = re.compile(r"[A-Z]{2,3}(\+\d)?")
_HANDS = re.compile(r"[a-z]+(-[a-z]+)*|\d+(\.\d+)?-\d+(\.\d+)?")
_SET_WORDS = {
    "pairs": "pairs",
    "small-aces": "small aces",
    "suited-connectors": "suited connectors",
}


class WebError(ValueError):
    """A request the table cannot serve: bad input, or not the moment for it."""


class Download(NamedTuple):
    """A file answer rather than JSON."""

    filename: str
    text: str


def _csv(rows: list[dict[str, Any]], filename: str) -> Download:
    stream = io.StringIO()
    write_hands_csv(rows, stream)
    return Download(filename, stream.getvalue())


def _hands_label(classes: tuple[str, ...]) -> str:
    """A training session's dealt hands in plain words: a named set, a few classes, a share of
    the strength order, or a count."""
    for name, words in _SET_WORDS.items():
        if set(classes) == set(hand_range(name)):
            return words
    if len(classes) <= 3:
        return ", ".join(classes) + " only"
    window = hand_window(classes)
    if window is None:
        return f"{len(classes)} hand types"
    low, high = window
    if low == 0:
        return "any hand" if high == 1 else f"best {percent_text(high)}%"
    if high == 1:
        return f"worst {percent_text(1 - low)}%"
    return f"top {percent_text(low)}-{percent_text(high)}%"


def _training(config: TableConfig) -> dict[str, str | None] | None:
    """A training session's fixed seat (a position code) and dealt hands, for the page."""
    session = config.session
    if session.mode != Mode.TRAINING:
        return None
    position = session.user_position
    return {
        "position": None if position is None else seats_from_button(session.num_seats)[position],
        "hands": _hands_label(session.user_hands) if session.user_hands else None,
    }


def _hands_json(spec: str) -> dict[str, Any]:
    """The hands a spec deals, for the setup page's preview."""
    classes = hand_range(spec)
    chosen = set(classes)
    return {
        "classes": [1.0 if c in chosen else 0.0 for c in PREFLOP_CLASSES],
        "count": len(classes),
        "share": round(range_share(classes), 4),
        "label": _hands_label(classes),
    }


def _result(hand: GameState, user: int, scale: int, all_in_bb: float | None) -> str | None:
    if not is_terminal(hand) or not hand.dealt_in[user]:
        return None
    net = (hand.stacks[user] - hand.starting_stacks[user]) / hand.config.big_blind
    return hand_result_text(net, all_in_bb, scale, hand.config.big_blind)


def _player_view(history: list[str], rated: list[str], summary: str, result: str | None) -> str:
    """The hand as the user saw it, to copy for a chat assistant: no styles and no analysis,
    only the moves rated 0 to 1 and the hand's three numbers."""
    lines = history + [""] + (rated + [""] if rated else []) + [summary]
    return "\n".join(lines + ([result] if result else [])) + "\n"


def _page_rows(
    rows: list[dict[str, Any]],
    hands: list[GameState],
    user: int,
    scale: int,
    ratings: dict[str, HandRating],
    pending: set[str],
    settled: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """CSV rows with what the hands tables also show: a rated hand's moves and summary are made
    once and kept in `settled`, an unrated hand's summary each time, as its rating may still
    come; `review` is None while the rating is being worked out."""
    result = []
    for row, hand in zip(rows, hands):
        hole = hand.hole_cards[user]
        assert hole is not None
        page = {
            "strength": percent_text(hand_rank(hole)[1]) + "%",
            "acted": any(e.seat == user for e in hand.history),
            "pending": hand.hand_id in pending,
            "review": None,
        }
        rated = ratings.get(hand.hand_id)
        if rated is not None and hand.hand_id not in settled:
            rating = rated.rating()
            lines, summary = rated_hand_text(hand, user, rated.moves, rated.chance, rating, scale)
            each = [
                f"{hand.history[m.history_index].street.title()} {m.rating:.2f}"
                + (" (after a hint)" if m.hinted else "")
                for m in rated.moves
            ]
            settled[hand.hand_id] = {
                "review": lines + [summary],
                "band": None if rating is None else rating_band(rating),
                "chance": percent_text(rated.chance) + "%",
                "moves": " · ".join(each),
            }
        if rated is not None:
            page.update(settled[hand.hand_id])
        elif hand.hand_id not in pending:
            page["review"] = [rated_hand_text(hand, user, (), None, None, scale)[1]]
        result.append({**row, **page})
    return result


def _chance(hand: GameState, user: int, bots: dict[int, Bot]) -> dict[str, Any]:
    """The user's chance to win now against the hands the others likely hold, read from their
    play with their own policies as the coach reads them, overall and against each opponent
    alone, with each one's likely hands."""
    hole = hand.hole_cards[user]
    assert hole is not None
    ranges = track(hand, user, bots, REFERENCE)[-1].ranges
    opponents = {seat: weights for seat, weights in ranges.items() if seat != user}
    seed = public_seed(observation(hand, user))
    acted = {entry.seat for entry in hand.history}
    entries = []
    for seat, weights in opponents.items():
        view = range_view(weights, hand.board, make_up=False)
        entries.append(
            {
                "seat": seat,
                "chance": round(hand_equity(hole, hand.board, [weights], Rng(seed)).value, 3),
                "acted": seat in acted,
                "width": round(view.width, 3),
                "classes": [round(w, 3) for w in view.classes],
            }
        )
    overall = hand_equity(hole, hand.board, list(opponents.values()), Rng(seed))
    return {"chance": round(overall.value, 3), "opponents": entries}


def _strong_mix(hand: GameState, user: int, index: int) -> dict[Action, float]:
    """How often the strong reference player takes each option at history action `index` (past
    the end: now), holding the user's cards."""
    view = observation(replay_states(hand)[index], user)
    mix: dict[Action, float] = defaultdict(float)
    for abstract, share in REFERENCE.action_probabilities(view).items():
        mix[to_action(abstract, view)] += share
    return dict(mix)


def _worth(option: OptionValue, tournament: bool, big_blind: int, scale: int) -> tuple[float, float]:
    """An option's average result and its noise: chips shown on the table, or percent of the
    prize pool in a tournament."""
    if tournament:
        assert option.icm is not None
        return round(100 * option.icm, 2) + 0.0, round(100 * (option.icm_stderr or 0.0), 2)
    noise = 0.0 if option.exact else chips_shown(option.stderr, big_blind, scale)
    return chips_shown(option.ev, big_blind, scale), noise


def _decision_json(
    review: DecisionReview,
    labels: dict[int, str],
    scale: int,
    big_blind: int,
    mix: dict[Action, float],
) -> dict[str, Any]:
    tournament = review.tournament
    best = best_option(review.reference, tournament).action
    reference = {o.action: o for o in review.reference}
    options = []
    for option in review.exploitative:
        vs_bots, vs_bots_noise = _worth(option, tournament, big_blind, scale)
        vs_strong, vs_strong_noise = _worth(reference[option.action], tournament, big_blind, scale)
        options.append(
            {
                "action": describe_action(option.action, scale),
                "vs_bots": vs_bots,
                "vs_bots_noise": vs_bots_noise,
                "vs_strong": vs_strong,
                "vs_strong_noise": vs_strong_noise,
                "strong_share": round(mix.get(option.action, 0.0), 3),
                "chosen": option.action == review.chosen,
                "best": option.action == best,
            }
        )
    return {
        "street": review.situation.street.value.lower(),
        "hole": [card_str(c) for c in review.hole],
        "board": [card_str(c) for c in review.board],
        "chosen": None if review.chosen is None else describe_action(review.chosen, scale),
        "verdict": None if review.chosen is None else review.verdict(),
        "headline": decision_headline(review, scale, big_blind, mix),
        "summary": decision_summary(review, scale, big_blind),
        "details": decision_details(review, labels, scale, big_blind),
        "unit": "% of the prize pool" if tournament else "chips",
        "options": options,
        "ranges": [
            {
                "name": labels[seat],
                "width": round(view.width, 3),
                "classes": [round(w, 3) for w in view.classes],
            }
            for seat, view in review.ranges.items()
        ],
    }


def _coach_json(
    runner: TableRunner,
    scale: int,
    hand: GameState,
    reviews: list[DecisionReview],
    shows: dict[int, tuple[int, int]],
    review: HandReview | None = None,
    hinted: set[int] | None = None,
    kept: HandRating | None = None,
) -> dict[str, Any]:
    """A coach answer: the hand so far, each decision for the page, the user's result once the
    hand is over, and two plain texts to paste into a chat assistant: the full analysis, and the
    hand as the user saw it with each move rated. `review` is the whole hand's, for its rating
    and its expected result from when the chips went in; a saved hand's rating as `kept` is
    shown instead, so the texts match its row."""
    user = runner.user_seat
    assert user is not None
    result = _result(hand, user, scale, review.all_in_net_bb if review else None)
    plain, styled = seat_labels(runner, False), seat_labels(runner, True)
    big_blind = hand.config.big_blind
    moves = move_ratings(reviews, hinted or set())
    chance: float | None
    rating: float | None
    if kept is not None:
        moves, chance, rating = kept.moves, kept.chance, kept.rating()
    elif review is not None:
        chance, rating = review.chance(), review.rating()
    else:  # a move checked has its chance but no rating of moves not shown; a hint has neither
        chance, rating = (reviews[-1].reference_equity.value if moves else None), None
    rated, summary = rated_hand_text(hand, user, moves, chance, rating, scale)
    copy = hand_history_text(hand, user, styled, scale, game_name(runner), shows) + [""]
    decisions = []
    for number, decision in enumerate(reviews, 1):
        mix = _strong_mix(hand, user, decision.index)
        decisions.append(_decision_json(decision, plain, scale, big_blind, mix))
        copy += plain_decision_text(number, decision, styled, scale, big_blind, mix) + [""]
    copy += (rated + [""] if rated else []) + [summary] + ([result] if result else [])
    history = plain_history(runner, hand, scale, shows)
    return {
        "history": history,
        "decisions": decisions,
        "result": result,
        "summary": summary if review is not None else None,
        "rating": rating,
        "band": None if rating is None else rating_band(rating),
        "copy_text": "\n".join(copy).strip() + "\n",
        "hand_text": _player_view(history, rated, summary, _result(hand, user, scale, None)),
    }


@dataclass(frozen=True)
class Later:
    """An answer finished in two steps around the table lock: `work` runs without the lock, as
    it may wait for or make a review, then `finish` builds the answer from what it returned
    under the lock again, unless `table` has gone meanwhile (deleted or dropped)."""

    work: Callable[[], Any]
    finish: Callable[[Any], dict[str, Any]] | None = None  # None: what `work` returned
    table: "WebTable | None" = None


def _wait_for(tasks: list[Task[Any]]):
    """Wait for background reviews, running any not started yet; a failed one is left to the
    answer, which shows what it has."""
    for task in tasks:
        with suppress(Exception):
            task.result()


class WebTable(PlaySession):
    """One session: the runner, the narration so far, its finished hands, and the session log."""

    plain = True

    def __init__(
        self,
        config: TableConfig,
        scale: int,
        log: SessionLog | None,
        profile: Profile,
        coach: bool,
    ):
        self.lines: list[str] = []
        self.shows: dict[str, dict[int, tuple[int, int]]] = {}
        self.hands: dict[str, GameState] = {}  # finished hands the user was dealt into, by id
        self.histories: dict[str, str] = {}  # each finished hand's moves, for its row
        self.settled: dict[str, dict[str, Any]] = {}  # page fields of rated hands, by id
        self.chance_task: tuple[tuple[str, int], Task[dict[str, Any]]] | None = None
        self.day = date.today().isoformat()
        self.coach = coach
        super().__init__(TableRunner(config), scale, log, profile)
        self.name = log.path.stem if log is not None else f"web-{config.session.seed}"

    def say(self, line: str):
        self.lines.append(line)

    def label(self, seat: int) -> str:
        return "You" if seat == self.runner.user_seat else self.runner.bots[seat].name

    def record(self, events: list[Event]):
        collect_shows(events, self.shows)
        super().record(events)

    def hand_over(self):
        super().hand_over()
        hand, user = self.finished, self.runner.user_seat
        if hand is None or user is None or hand.hand_id in self.hands or not hand.dealt_in[user]:
            return
        self.hands[hand.hand_id] = hand
        self.histories.update(session_histories(self.runner, self.scale, [hand], self.shows))

    def rating_pending(self, hand_id: str) -> bool:
        jobs = self.jobs.get(hand_id)
        return jobs is not None and jobs.rating is not None and not jobs.rating.future.done()

    def rows(self, hands: list[GameState]) -> list[dict[str, Any]]:
        """The hands' rows for the page and the CSV, built when asked from what is kept, so a
        rating landing later shows at the next ask."""
        user = self.runner.user_seat
        assert user is not None
        mode = self.runner.config.session.mode.value.lower()
        # Pending first: a rating landing meanwhile is then read as in, never as missing.
        pending = {h.hand_id for h in hands if self.rating_pending(h.hand_id)}
        rows = hand_rows(hands, user, self.scale, self.name, self.day, mode, [], self.histories, self.ratings)
        return _page_rows(rows, hands, user, self.scale, self.ratings, pending, self.settled)

    def turn_started(self, hand: GameState, user: int):
        """In training, the chance to win is worked out first on the review thread, so it never
        competes with the review of the decision it is shown for."""
        if self.runner.config.session.mode == Mode.TRAINING:
            work = partial(_chance, hand, user, self.runner.bots)
            self.chance_task = ((hand.hand_id, len(hand.history)), background(work))

    def next_hand(self):
        """Deal the next hand, or for a user out of a tournament play it out to the end."""
        runner = self.runner
        if runner.session.finished:
            raise WebError("the session is over")
        if runner.hand is not None and not is_terminal(runner.hand):
            raise WebError("the hand is still being played")
        if runner.session.awaiting_rebuy:
            raise WebError("you are out of chips: rebuy or start a new game")
        if self.knocked_out():
            self.fast_forward()
            return
        self.lines.append(f"=== Hand {runner.session.hand_number + 1}")
        self.record(runner.start_hand(self.minutes()))
        self.hand_over()

    def rebuy(self):
        if not self.runner.session.awaiting_rebuy:
            raise WebError("a rebuy is only for a cash player out of chips")
        self.record(self.runner.user_rebuy())

    def act(self, payload: dict[str, Any]):
        if not self.runner.user_to_act():
            raise WebError("it is not your turn")
        kind = payload.get("kind")
        if not isinstance(kind, str):
            raise WebError("the action needs a kind")
        legal = self.runner.user_legal_actions()
        if kind == "allin":
            action = (
                Action(ActionType.RAISE if legal.can_raise else ActionType.BET, legal.max_raise_to)
                if legal.max_raise_to
                else Action(ActionType.CALL)
            )
        elif kind in ("bet", "raise"):
            amount = payload.get("amount")
            if not isinstance(amount, (int, float)) or isinstance(amount, bool) or not math.isfinite(amount):
                raise WebError("a bet or raise needs an amount")
            action = Action(_KINDS[kind], round(float(amount) * self.scale))
            assert action.amount is not None
            if (legal.can_bet or legal.can_raise) and not (legal.min_raise_to <= action.amount <= legal.max_raise_to):
                low, high = self.chips(legal.min_raise_to), self.chips(legal.max_raise_to)
                raise WebError(f"a {kind} must be between {low} and {high}")
        elif kind in _KINDS:
            action = Action(_KINDS[kind])
        else:
            raise WebError(f"unknown action {kind!r}")
        try:
            self.record(self.runner.act(action))
        except IllegalActionError as error:
            raise WebError(str(error)) from error
        self.hand_over()

    def state(self) -> dict[str, Any]:
        runner = self.runner
        user = runner.user_seat
        hand = runner.hand
        view = runner.user_view() if hand is not None else None
        hand_over = hand is None or is_terminal(hand)
        positions = position_names(view) if view is not None else {}
        shown = self.shows.get(hand.hand_id, {}) if hand is not None else {}
        last: dict[int, str] = {}
        won: dict[int, int] = defaultdict(int)
        for award in view.awards if view is not None else ():
            for winner, share in zip(award.winners, award.shares):
                won[winner] += share
        if view is not None and view.history:
            street = view.history[-1].street if hand_over else view.street
            for move in view.history:
                if move.street == street:
                    amount = move.action.amount
                    words = _PAST[move.action.type]
                    last[move.seat] = words if amount is None else f"{words} {self.chips(amount)}"
        seats = []
        for seat in range(runner.config.session.num_seats):
            entry: dict[str, Any] = {
                "seat": seat,
                "name": self.label(seat),
                "is_user": seat == user,
                "stack": runner.session.stacks[seat] / self.scale,
            }
            bot = runner.bots.get(seat)
            if bot is not None and not runner.config.hide_styles:
                stats = runner.hud.seats[seat]
                entry.update(
                    style=STYLE_WORDS[bot.style.name],
                    tip=EXPLOITS[bot.style.name],
                    hud={
                        "hands": stats.hands,
                        "plays": stats.voluntary / stats.hands if stats.hands else None,
                        "raises": stats.preflop_raises / stats.hands if stats.hands else None,
                        "reraises": stats.three_bets / stats.three_bet_chances if stats.three_bet_chances else None,
                        "aggression": stats.postflop_aggressive / stats.postflop_calls
                        if stats.postflop_calls
                        else None,
                    },
                )
            if view is not None:
                cards = view.hole_cards[seat] or shown.get(seat)
                # Blinds, antes and bets, less any uncalled bet returned: the stack's drop,
                # before the winnings of a finished hand.
                in_pot = view.starting_stacks[seat] - view.stacks[seat] + won[seat]
                entry.update(
                    committed=view.committed_this_street[seat] / self.scale,
                    in_pot=in_pot / self.scale,
                    folded=view.folded[seat],
                    all_in=view.all_in[seat],
                    dealt_in=view.dealt_in[seat],
                    button=seat == view.button,
                    position=positions.get(seat),
                    last_action=last.get(seat),
                    # Between hands the session's stacks, which count a rebuy.
                    stack=(runner.session.stacks if hand_over else view.stacks)[seat] / self.scale,
                    cards=[card_str(c) for c in cards] if cards else None,
                )
            seats.append(entry)
        cash = runner.config.session.mode != Mode.TOURNAMENT
        shown_hands = self.rows(list(self.hands.values())[-RECENT_HANDS:])
        result: dict[str, Any] = {
            "seats": seats,
            "log": self.lines[-80:],
            "hand_over": hand_over,
            "session_over": runner.session.finished,
            "awaiting_rebuy": runner.session.awaiting_rebuy,
            "knocked_out": self.knocked_out(),
            "your_turn": runner.user_to_act(),
            "step": 1 / self.scale,  # the smallest chip amount, in the units shown
            "coach": self.coach,
            "hands": shown_hands,
            "pending": sum(r["pending"] for r in shown_hands),
            "hud_min_hands": MIN_HANDS,
            "acted": view is not None and any(e.seat == user for e in view.history),
            "training": _training(runner.config),
        }
        if user is not None and cash:
            result["session_net"] = (runner.session.stacks[user] - runner.session.buy_ins[user]) / self.scale
        if view is not None:
            result.update(
                board=[card_str(c) for c in view.board],
                pot=view.pot / self.scale,
                big_blind=view.config.big_blind / self.scale,
            )
            hole = view.hole_cards[view.seat]
            if hole is not None and len(view.board) >= 3:
                name = describe(evaluate(list(hole) + list(view.board)))
                result["your_hand"] = name[0].upper() + name[1:]
            if hand_over and view.dealt_in[view.seat]:
                net = view.stacks[view.seat] - view.starting_stacks[view.seat]
                result["hand_result"] = net / self.scale
            if view.dealt_in[view.seat]:
                result["hand_number"] = int(view.hand_id.rpartition("-")[2])

        if runner.user_to_act():
            assert view is not None
            legal = runner.user_legal_actions()
            result["legal"] = {
                "fold": legal.can_fold,
                "check": legal.can_check,
                "call": legal.call_amount / self.scale if legal.can_call else None,
                "raise": {
                    "kind": "raise" if legal.can_raise else "bet",
                    "min": legal.min_raise_to / self.scale,
                    "max": legal.max_raise_to / self.scale,
                }
                if legal.can_bet or legal.can_raise
                else None,
            }
            result["call_needs"] = thresholds(view).required_equity
        return result

    def _answer(self, hand: GameState, found: DecisionReview | HandReview) -> dict[str, Any]:
        """The coach's answer for one decision, or for the whole hand with its rating."""
        shows = self.shows.get(hand.hand_id, {})
        jobs = self.jobs.get(hand.hand_id)
        hinted = jobs.hinted if jobs is not None else set()
        if isinstance(found, HandReview):
            return _coach_json(self.runner, self.scale, hand, found.decisions, shows, found, hinted)
        return _coach_json(self.runner, self.scale, hand, [found], shows, None, hinted)

    def hint(self) -> Later:
        if not self.coach:
            raise WebError("the coach is off for this game")
        if not self.runner.user_to_act():
            raise WebError("a hint needs your turn to act")
        hand, work = self.hint_work()
        return Later(work, partial(self._answer, hand), self)

    def analyze(self) -> Later:
        """The user's latest decision in the current or just-finished hand, judged as the hand
        review judges it. Asked mid-hand, it counts as a hint: what it shows carries into the
        decision the user faces next."""
        if not self.coach:
            raise WebError("the coach is off for this game")
        hand, user = self.runner.hand, self.runner.user_seat
        mine = [i for i, e in enumerate(hand.history) if e.seat == user] if hand else []
        if hand is None or user is None or not mine:
            raise WebError("you haven't acted in this hand yet")
        if self.runner.user_to_act():
            self.note_hint(hand.hand_id, len(hand.history))
        return Later(self.move_work(hand, mine[-1]), partial(self._answer, hand), self)

    def review(self) -> Later:
        """The user's last finished hand, which after a fast-forward is not the hand shown."""
        hand, user, current = self.finished, self.runner.user_seat, self.runner.hand
        playing = current is not None and not is_terminal(current)
        if hand is None or user is None or playing:
            raise WebError("a review needs a finished hand")
        return Later(self.review_work(hand), partial(self._answer, hand), self)

    def copy(self, number: int) -> Later:
        """Hand `number` as the user saw it with its moves rated, once the reviews already
        under way are done: a finished hand, or the hand being played so far."""
        hand_id, current = f"hand-{number}", self.runner.hand
        hand = self.hands.get(hand_id)
        if hand is None and current is not None and current.hand_id == hand_id:
            hand = current
        if hand is None:
            raise WebError(f"no hand {number} to copy")
        jobs = self.jobs.get(hand_id)
        tasks: list[Task[Any]] = [] if jobs is None else list(jobs.moves.values())
        if jobs is not None and jobs.rating is not None:
            tasks.append(jobs.rating)
        return Later(partial(_wait_for, tasks), partial(self._copy_answer, hand), self)

    def _copy_answer(self, hand: GameState, waited: object) -> dict[str, Any]:
        user = self.runner.user_seat
        assert user is not None
        rated = self.ratings.get(hand.hand_id)
        chance: float | None
        if rated is not None:
            moves, chance = rated.moves, rated.chance
        else:  # mid-hand, or a hand whose rating failed: the moves reviewed so far
            jobs = self.jobs.get(hand.hand_id)
            futures = [jobs.moves[i].future for i in sorted(jobs.moves)] if jobs is not None else []
            done = [f.result() for f in futures if f.done() and not f.cancelled() and not f.exception()]
            moves = move_ratings(done, jobs.hinted if jobs is not None else set())
            chance = done[-1].reference_equity.value if done else None
        rating = hand_rating([(m.rating, m.stake) for m in moves]) if moves else None
        lines, summary = rated_hand_text(hand, user, moves, chance, rating, self.scale)
        history = plain_history(self.runner, hand, self.scale, self.shows.get(hand.hand_id, {}))
        result = _result(hand, user, self.scale, None)
        return {"text": _player_view(history, lines, summary, result)}

    def chance(self) -> Later:
        """A training helper: the user's chance to win now, normally already worked out when
        the turn began (`turn_started`)."""
        runner = self.runner
        if runner.config.session.mode != Mode.TRAINING:
            raise WebError("the chance to win is shown in training only")
        hand, user = runner.hand, runner.user_seat
        if hand is None or user is None or is_terminal(hand) or hand.folded[user]:
            raise WebError("the chance to win needs you in a hand being played")
        turn, task = self.chance_task or (None, None)
        now = task if turn == (hand.hand_id, len(hand.history)) else None
        return Later(partial(result_or, now, partial(_chance, hand, user, runner.bots)), table=self)

    def hands_csv(self) -> Download:
        return _csv(self.rows(list(self.hands.values())), f"{self.name}.csv")


def _session_argv(options: dict[str, Any]) -> list[str]:
    """Command-line arguments for a new session, from checked JSON fields."""
    argv = ["--no-log"]
    fields: dict[str, type | tuple[type, ...]] = {
        "mode": str,
        "seats": int,
        "tier": int,
        "blinds": str,
        "stack": (int, float),
        "seed": int,
        "position": str,
        "hands": str,
    }
    for name, kind in fields.items():
        value = options.get(name)
        if value is None:
            continue
        if (
            not isinstance(value, kind)
            or isinstance(value, bool)
            or (isinstance(value, float) and not math.isfinite(value))
        ):
            raise WebError(f"{name} has the wrong type")
        argv += [f"--{name}", str(value)]
    blinds = options.get("blinds")
    if isinstance(blinds, str):
        if not _BLINDS.fullmatch(blinds):
            raise WebError("blinds look like 50/100: the small blind, a slash, then the big blind")
        small, big = map(float, blinds.split("/"))
        if not 0 < small <= big:  # the command line's own check words this for its users
            raise WebError("the small blind must be above 0 and no bigger than the big blind")
    position, hands = options.get("position"), options.get("hands")
    if isinstance(position, str) and not _POSITION.fullmatch(position):
        raise WebError("pick a seat from the list")
    if isinstance(hands, str) and not _HANDS.fullmatch(hands):
        raise WebError("hands look like 5-25 (the top 5% to 25% of hands) or pairs")
    for flag, name in (("--hide-styles", "hide_styles"), ("--hints", "coach")):
        if options.get(name) is True:
            argv.append(flag)
    if options.get("coach") is False:
        argv.append("--no-hints")
    return argv


class _Saved:
    """A saved session as read so far. The log is read on from where the last read stopped, so
    a log that grows, as a session being played does with every hand and rating, costs only its
    new lines."""

    def __init__(self, path: Path):
        stat = path.stat()
        self.inode, self.offset = stat.st_ino, 0
        self.config: TableConfig | None = None
        self.scale = 1
        self.runner: TableRunner | None = None
        self.hands: list[GameState] = []
        self.shows: dict[str, dict[int, tuple[int, int]]] = {}
        self.ratings: dict[str, HandRating] = {}
        self.histories: dict[str, str] = {}  # filled when the session's hands are first asked for
        self.settled: dict[str, dict[str, Any]] = {}  # page fields of rated hands, by id
        self.read(path)

    def read(self, path: Path):
        """Read the log's new records. Raises `OSError` or `ValueError`."""
        records, offset = read_records(path, self.offset)
        try:
            if self.config is None:
                logged = next((r for r in records if r["type"] == "table_config"), None)
                if logged is None:
                    raise ValueError("no table configuration in the log")
                self.config, self.scale = TableConfig.from_dict(logged["config"]), logged["scale"]
                if self.config.session.user_seat is None:
                    raise ValueError("bot-only sessions have no decisions to review")
                self.runner = TableRunner(self.config)
            hands = [GameState.from_dict(r["hand"]) for r in records if r["type"] == "hand"]
            assert self.runner is not None
            self.runner.replay(records)
        except (KeyError, TypeError) as error:
            raise ValueError(f"damaged log: {error!r}") from error
        shows, ratings = logged_shows(records), logged_ratings(records)
        # Kept only once every new line has parsed, so a damaged one is met again, not skipped.
        self.offset = offset
        self.hands += hands
        for hand_id, shown in shows.items():
            self.shows.setdefault(hand_id, {}).update(shown)
        self.ratings.update(ratings)

    @property
    def user(self) -> int:
        assert self.config is not None and self.config.session.user_seat is not None
        return self.config.session.user_seat


class App:
    """All sessions, and the saved ones in the log folder, served one request at a time: a
    request waiting for a review waits without holding the others up."""

    def __init__(
        self,
        log_dir: Path | None,
        profile: Profile = PROFILES["pc"],
        decisions_log: Path | None = None,
    ):
        self.log_dir = log_dir
        self.profile = profile
        self.decisions_log = decisions_log  # reviewed decisions, for the exported review columns
        self.tables: dict[str, WebTable] = {}
        self.saved: dict[Path, _Saved] = {}
        self.decisions: tuple[tuple[int, int], list[DecisionRecord]] | None = None
        self.lock = threading.Lock()

    def handle(self, method: str, parts: list[str], payload: dict[str, Any]) -> dict[str, Any] | Download:
        """Route an API request (the path after /api/, split). Raises `WebError` or `KeyError`."""
        with self.lock:
            answer = self._route(method, parts, payload)
        if not isinstance(answer, Later):
            return answer
        try:
            done = answer.work()
        except CancelledError as error:  # the table was deleted or dropped meanwhile
            raise KeyError(parts) from error
        with self.lock:
            if answer.table is not None and answer.table not in self.tables.values():
                raise KeyError(parts)
            return done if answer.finish is None else answer.finish(done)

    def _route(self, method: str, parts: list[str], payload: dict[str, Any]) -> dict[str, Any] | Download | Later:
        if parts and parts[0] in ("history", "history.csv"):
            return self._history(method, parts, payload)
        if method == "POST" and parts == ["sessions"]:
            return self._create(payload)
        if method == "GET" and len(parts) == 2 and parts[0] == "hands":
            return _hands_json(parts[1])
        if len(parts) < 2 or parts[0] != "sessions":
            raise KeyError(parts)
        table = self.tables[parts[1]]
        command = parts[2] if len(parts) > 2 else None
        if method == "GET" and command is None:
            return table.state()
        if method == "GET" and command == "hands.csv":
            return table.hands_csv()
        if method == "GET" and command == "chance":
            return table.chance()
        if method == "GET" and command == "hands" and len(parts) == 5 and parts[4] == "copy":
            if not parts[3].isdigit():
                raise WebError("the hand number must be a whole number")
            return table.copy(int(parts[3]))
        if method == "POST" and command == "action":
            table.act(payload)
            return table.state()
        if method == "POST" and command == "next":
            table.next_hand()
            return table.state()
        if method == "POST" and command == "rebuy":
            table.rebuy()
            return table.state()
        if method == "POST" and command == "hint":
            return table.hint()
        if method == "POST" and command == "analyze":
            return table.analyze()
        if method == "POST" and command == "review":
            return table.review()
        raise KeyError(parts)

    def _create(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            args = parse_args(_session_argv(payload))
            config, scale = build_config(args)
            coach = args.hints if args.hints is not None else config.session.mode != Mode.TOURNAMENT
            log = SessionLog(self.log_dir / f"session-{config.session.seed}.jsonl") if self.log_dir else None
            table = WebTable(config, scale, log, self.profile, coach)
            table.next_hand()
        except WebError:
            raise
        except SystemExit as error:  # argparse refused a value
            raise WebError("can't start that game: a setting is out of range") from error
        except (ValueError, ArithmeticError) as error:
            raise WebError(f"can't start that game: {error}") from error
        key = secrets.token_hex(8)
        self.tables[key] = table
        while len(self.tables) > MAX_SESSIONS:
            self.tables.pop(next(iter(self.tables))).close(wait=False)
        return {"id": key, "state": table.state()}

    def _logs(self) -> dict[str, Path]:
        """Saved sessions by name, newest first. Names are only ever looked up here, never
        joined into a path, so a request cannot reach outside the log folder."""
        if self.log_dir is None or not self.log_dir.is_dir():
            return {}
        paths = sorted(self.log_dir.glob("session-*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
        return {p.stem: p for p in paths}

    def _load(self, path: Path) -> _Saved:
        """The session read up to date: read on from the last read while the log only grew, and
        again from the start when it was replaced or cut (a same-seed session after a delete)."""
        stat = path.stat()
        saved = self.saved.get(path)
        if saved is None or saved.inode != stat.st_ino or stat.st_size < saved.offset:
            self.saved.pop(path, None)
            saved = _Saved(path)
            self.saved[path] = saved
        elif stat.st_size > saved.offset:
            saved.read(path)
        return saved

    def _decisions(self) -> list[DecisionRecord]:
        """Reviewed decisions for the review columns, read again only when that log changes;
        they stay blank if it is damaged."""
        if self.decisions_log is None:
            return []
        try:
            stat = self.decisions_log.stat()
        except OSError:
            return []
        key = (stat.st_mtime_ns, stat.st_size)
        if self.decisions is None or self.decisions[0] != key:
            try:
                records = DecisionRecord.read(self.decisions_log, "decision")
            except (OSError, ValueError, KeyError, TypeError):
                records = []
            self.decisions = (key, records)
        return self.decisions[1]

    def _live(self, path: Path) -> list["WebTable"]:
        return [t for t in self.tables.values() if t.log is not None and t.log.path == path]

    def _rows(self, path: Path) -> list[dict[str, Any]]:
        """A saved session's rows, built when asked; a hand still being rated by the table that
        writes the log shows as pending. Raises `OSError` or `ValueError`."""
        # Pending first, then the log: a rating landing meanwhile is then read as in.
        rating = {h for t in self._live(path) for h in t.jobs if t.rating_pending(h)}
        saved = self._load(path)
        assert saved.config is not None and saved.runner is not None
        user, mode = saved.user, saved.config.session.mode.value.lower()
        day = date.fromtimestamp(path.stat().st_mtime).isoformat()
        hands = [h for h in saved.hands if h.dealt_in[user]]
        missing = [h for h in hands if h.hand_id not in saved.histories]
        if missing:
            saved.histories.update(session_histories(saved.runner, saved.scale, missing, saved.shows))
        rows = hand_rows(
            hands,
            user,
            saved.scale,
            path.stem,
            day,
            mode,
            self._decisions(),
            saved.histories,
            saved.ratings,
        )
        pending = rating - set(saved.ratings)
        return _page_rows(rows, hands, user, saved.scale, saved.ratings, pending, saved.settled)

    def _history(self, method: str, parts: list[str], payload: dict[str, Any]) -> dict[str, Any] | Download | Later:
        logs = self._logs()
        if method == "GET" and parts == ["history"]:
            listed = []
            for name, path in list(logs.items())[:HISTORY_LIMIT]:
                try:
                    saved = self._load(path)
                except (OSError, ValueError):
                    continue  # bot-only or damaged logs have nothing to show
                assert saved.config is not None
                user = saved.user
                played = [h for h in saved.hands if h.dealt_in[user]]
                net = sum(h.stacks[user] - h.starting_stacks[user] for h in played)
                listed.append(
                    {
                        "name": name,
                        "date": date.fromtimestamp(path.stat().st_mtime).isoformat(),
                        "mode": saved.config.session.mode.value.lower(),
                        "training": _training(saved.config),
                        "players": saved.config.session.num_seats,
                        "difficulty": saved.config.tier,
                        "hands": len(played),
                        "net": net / saved.scale,
                    }
                )
            return {"sessions": listed, "logging": self.log_dir is not None}
        if method == "GET" and parts == ["history.csv"]:
            rows = []
            for path in logs.values():
                try:
                    rows += self._rows(path)
                except (OSError, ValueError):
                    continue
            return _csv(rows, "thpoker-hands.csv")
        if method == "POST" and parts == ["history", "delete"]:
            names = payload.get("names")
            if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
                raise WebError("the delete needs a list of session names")
            return self._delete([logs[name] for name in names])
        if len(parts) < 2:
            raise KeyError(parts)
        path = logs[parts[1]]
        command = parts[2] if len(parts) > 2 else None
        if method == "GET" and command in (None, "hands.csv"):
            rows = self._rows(path)
            if command is None:
                return {"name": path.stem, "rows": rows, "pending": sum(r["pending"] for r in rows)}
            return _csv(rows, f"{path.stem}.csv")
        if method == "POST" and command == "review":
            saved = self._load(path)
            number = payload.get("hand")
            if not isinstance(number, int) or isinstance(number, bool):
                raise WebError("the review needs a hand number")
            hand = next((h for h in saved.hands if h.hand_id == f"hand-{number}"), None)
            if hand is None:
                raise WebError(f"no hand {number} in that session")
            assert saved.runner is not None
            work = partial(full_review, saved.runner, hand, self.profile)
            return Later(work, partial(self._saved_answer, saved, hand))
        raise KeyError(parts)

    def _saved_answer(self, saved: _Saved, hand: GameState, review: HandReview) -> dict[str, Any]:
        assert saved.runner is not None
        shows = saved.shows.get(hand.hand_id, {})
        rated = saved.ratings.get(hand.hand_id)
        return _coach_json(saved.runner, saved.scale, hand, review.decisions, shows, review, kept=rated)

    def _delete(self, paths: list[Path]) -> dict[str, Any]:
        """Delete saved sessions. A table still writing one is dropped first, its reviews
        cancelled, so nothing brings the file back."""
        for path in paths:
            for key, table in list(self.tables.items()):
                if table in self._live(path):
                    for jobs in table.jobs.values():
                        jobs.cancel()
                    assert table.log is not None
                    table.log.delete()
                    del self.tables[key]
            path.unlink(missing_ok=True)
            self.saved.pop(path, None)
        listed = self._history("GET", ["history"], {})
        assert isinstance(listed, dict)
        return listed


class _Handler(http.server.BaseHTTPRequestHandler):
    app: App

    def do_GET(self):
        if not self._local_host():
            return
        if self.path in ASSETS:
            name, kind = ASSETS[self.path]
            self._send(200, (STATIC / name).read_bytes(), kind)
        elif self.path.startswith("/api/"):
            self._api("GET")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if not self._local_host():
            return
        if not self.path.startswith("/api/"):
            self._send(404, b"not found", "text/plain")
        elif self.headers.get_content_type() != "application/json":
            # Browsers send JSON across sites only after a preflight this server never grants.
            self._json(415, {"error": "the request body must be JSON"})
        else:
            self._api("POST")

    def _local_host(self) -> bool:
        """Serve only requests addressed to this machine by name, so a web page on another host
        cannot reach the table through a name that resolves here (DNS rebinding)."""
        assert isinstance(self.server, http.server.HTTPServer)
        port = self.server.server_port
        if self.headers.get("Host") in (f"127.0.0.1:{port}", f"localhost:{port}"):
            return True
        self._json(403, {"error": "the table only serves 127.0.0.1"})
        return False

    def _api(self, method: str):
        try:
            length = int(self.headers.get("Content-Length") or 0)
            if not 0 <= length <= MAX_BODY:
                self._json(413 if length > MAX_BODY else 400, {"error": "bad request length"})
                return
            payload = json.loads(self.rfile.read(length) or b"{}") if method == "POST" else {}
            if not isinstance(payload, dict):
                raise WebError("the request body must be a JSON object")
            answer = self.app.handle(method, [p for p in self.path[5:].split("/") if p], payload)
            if isinstance(answer, Download):
                disposition = f'attachment; filename="{answer.filename}"'
                self._send(200, answer.text.encode(), "text/csv; charset=utf-8", disposition)
            else:
                self._json(200, answer)
        except KeyError:
            self._json(404, {"error": "no such session or command"})
        except (ValueError, ArithmeticError, TypeError) as error:
            self._json(400, {"error": str(error)})
        except Exception as error:  # noqa: BLE001 - answer rather than drop the connection
            self._json(500, {"error": f"internal error: {type(error).__name__}"})

    def _json(self, status: int, payload: dict[str, Any]):
        self._send(status, json.dumps(payload).encode(), "application/json")

    def _send(self, status: int, body: bytes, kind: str, disposition: str | None = None):
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        if disposition is not None:
            self.send_header("Content-Disposition", disposition)
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any):  # noqa: A002 - the base class's name
        """Quiet: the table is local and every request would otherwise be printed."""


class _Server(http.server.ThreadingHTTPServer):
    app: App


def serve(
    port: int = 8000,
    log_dir: Path | None = None,
    profile: Profile = PROFILES["pc"],
    decisions_log: Path | None = None,
) -> _Server:
    """A server for the web table on this machine only; call `serve_forever` to run it."""
    app = App(log_dir, profile, decisions_log)
    server = _Server(("127.0.0.1", port), type("Handler", (_Handler,), {"app": app}))
    server.app = app
    return server


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="thpoker web", description="Play in a browser on this machine.")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--log-dir", type=Path, default=default_log_dir())
    parser.add_argument("--no-log", action="store_true")
    add_device_argument(parser)
    args = parser.parse_args(argv)
    log_dir = None if args.no_log else args.log_dir
    server = serve(args.port, log_dir, pick_profile(args.device), default_decisions_log())
    print(f"Open http://127.0.0.1:{args.port} in a browser (Ctrl-C stops the table).")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        with server.app.lock:  # stop the reviews nobody will read
            for table in server.app.tables.values():
                table.close(wait=False)
    return 0
