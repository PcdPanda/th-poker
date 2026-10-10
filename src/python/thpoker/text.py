"""Text for the command line and the web table: one line of narration per thing that happened in
a hand or a session, and the rendering of reviews (the data comes from `analysis/review.py`).
"""

from   collections              import defaultdict
from   collections.abc          import Callable, Sequence
import functools
import math
from   thpoker.analysis.ev      import OptionValue
from   thpoker.analysis.review  import (DecisionReview, GodView, HandReview,
                                        MoveRating, RATING_CLOSE, RangeView,
                                        SessionSummary, best_option,
                                        hindsight_best, position_names)
from   thpoker.analysis.training \
                                import Calibration, RECENT
from   thpoker.game.cards       import RANKS, cards_str
from   thpoker.game.engine      import (is_terminal, new_hand, observation,
                                        replay_states)
from   thpoker.game.evaluator   import describe, evaluate
from   thpoker.game.state       import (Action, ActionType, AnteType, Event,
                                        GameState, Street)
from   thpoker.odds             import hand_rank


def format_chips(amount: float, scale: int) -> str:
    shown = amount / scale
    return f"{shown:,.0f}" if shown == int(shown) else f"{shown:,.2f}"


def _conjugate(verb: str, is_user: bool) -> str:
    """ "raise to 300" for the user, "raises to 300" for everyone else."""
    if is_user:
        return verb
    first, _, rest = verb.partition(" ")
    return f"{first}s {rest}".rstrip()


def narrate(
    event: Event,
    label: Callable[[int], str],
    chips: Callable[[int], str],
    user: int | None,
    plain: bool = False,
) -> list[str]:
    """Lines describing `event`: `label` names a seat, `chips` formats an amount. `plain` words
    the pot line for newcomers ("Casey wins 300") rather than as a pot and its shares."""
    data = event.data
    if event.kind == "ActionTaken":
        action = Action.from_dict(data["action"])
        verb = {
            ActionType.FOLD: "fold",
            ActionType.CHECK: "check",
            ActionType.CALL: "call",
            ActionType.BET: f"bet {chips(action.amount or 0)}",
            ActionType.RAISE: f"raise to {chips(action.amount or 0)}",
        }[action.type]
        all_in = " (all-in)" if data["stack"] == 0 and action.type != ActionType.FOLD else ""
        return [f"  {label(data['seat'])} {_conjugate(verb, data['seat'] == user)}{all_in}"]
    if event.kind == "StreetDealt":
        return [f"--- {data['street'].title()}: {cards_str(data['board'])}"]
    if event.kind == "UncalledBetReturned":
        return [f"  {chips(data['amount'])} returned to {label(data['seat'])}"]
    if event.kind == "Showdown":
        return [
            f"  {label(hand['seat'])} {_conjugate('show', hand['seat'] == user)} {cards_str(hand['cards'])}: {hand['description']}"
            for hand in data["hands"]
        ]
    if event.kind == "CardsShown":
        return [f"  {label(data['seat'])} shows {cards_str(data['cards'])}"]
    if event.kind == "PotAwarded":
        if plain:
            return [
                "  "
                + ", ".join(
                    f"{label(s)} {_conjugate('win', s == user)} {chips(x)}"
                    for s, x in zip(data["winners"], data["shares"])
                )
            ]
        winners = ", ".join(f"{label(s)} {chips(x)}" for s, x in zip(data["winners"], data["shares"]))
        return [f"  Pot {chips(data['amount'])} -> {winners}"]
    if event.kind == "BlindLevelChanged":
        return [
            f"*** Level {data['level']}: blinds {chips(data['small_blind'])}/{chips(data['big_blind'])}, ante {chips(data['ante'])}"
        ]
    if event.kind == "PlayerEliminated":
        return [f"*** {label(data['seat'])} finishes in place {data['place']}"]
    if event.kind == "Rebuy":
        return [f"  {label(data['seat'])} rebuys {chips(data['amount'])}"]
    if event.kind == "TournamentFinished":
        order = sorted(range(len(data["places"])), key=data["places"].__getitem__)
        return ["*** Tournament over"] + [
            f"    {data['places'][seat]}. {label(seat)}  prize {data['prizes'][seat]:.0%}" for seat in order
        ]
    return []


_VERBS = {
    ActionType.FOLD: "fold",
    ActionType.CHECK: "check",
    ActionType.CALL: "call",
    ActionType.BET: "bet",
    ActionType.RAISE: "raise to",
}


def describe_action(action: Action, scale: int) -> str:
    verb = _VERBS[action.type]
    return verb if action.amount is None else f"{verb} {format_chips(action.amount, scale)}"


def _share(value: float) -> str:
    return f"{100 * value:.0f}%"


def range_grid(view: RangeView) -> list[str]:
    """13x13 grid, pairs on the diagonal and suited hands above it: '#' nearly always in the
    range, '+' often, '.' sometimes, blank rarely or never."""
    lines = ["   " + " ".join(RANKS[::-1])]
    for row in range(13):
        cells = []
        for column in range(13):
            weight = view.classes[row * 13 + column]
            cells.append("#" if weight >= 0.75 else "+" if weight >= 0.4 else "." if weight >= 0.1 else " ")
        lines.append(f" {RANKS[::-1][row]} " + " ".join(cells))
    return lines


def _option_rows(review: DecisionReview, scale: int, bot_label: str) -> list[str]:
    reference = {o.action: o for o in review.reference}
    tournament = review.tournament
    best_bot = best_option(review.exploitative, tournament).action
    best_reference = best_option(review.reference, tournament).action
    unit = "prize pool %" if tournament else "bb"
    rows = [f"   EV ({unit}):  vs {bot_label} | vs reference"]
    for option in review.exploitative:
        other = reference[option.action]
        marks = []
        if option.action == review.chosen:
            marks.append("your choice")
        if option.action == best_reference:
            marks.append("best vs reference")
        if option.action == best_bot and best_bot != best_reference:
            marks.append(f"best vs {bot_label}")
        rows.append(
            f"   {describe_action(option.action, scale):<16} {_value(option, tournament):>13} | {_value(other, tournament):<13} {', '.join(marks)}"
        )
    return rows


def _value(option: OptionValue, tournament: bool) -> str:
    if tournament and option.icm is not None:
        return f"{100 * option.icm:.2f} ± {100 * (option.icm_stderr or 0):.2f}"
    if option.exact:
        return f"{option.ev:+.2f}"
    return f"{option.ev:+.2f} ± {option.stderr:.2f}"


def render_decision(
    review: DecisionReview,
    labels: dict[int, str],
    scale: int,
    grids: bool = False,
    reveal: bool = True,
) -> list[str]:
    """The five steps of DESIGN.md Section 7.1 for one decision. `labels` names the seats.
    Without `reveal` only the situation and the ranges are shown, for predict-then-reveal."""
    spot = review.situation
    board = f" on {cards_str(review.board)}" if review.board else ""
    chose = "your move" if review.chosen is None else f"you chose {describe_action(review.chosen, scale)}"
    lines = [f"{spot.street.value.title()}: {cards_str(review.hole)}{board}; {chose}"]
    where = "in position" if spot.in_position else "out of position"
    spr = f" (SPR {spot.spr:.1f})" if spot.spr is not None else ""
    lines.append(
        f" 1 Situation   {spot.position}, {spot.players} players, {where}, {spot.pot_type} pot {spot.pot_bb:.1f}bb, "
        f"to call {spot.to_call_bb:.1f}bb, effective {spot.effective_bb:.1f}bb{spr}"
    )
    if spot.texture is not None:
        tags = spot.texture
        change = f", {tags.change}" if tags.change else ""
        lines.append(f"               board: {tags.high}, {tags.pairing}, {tags.suits}, {tags.connectivity}{change}")
    for seat, view in review.ranges.items():
        groups = ""
        if view.groups is not None:
            groups = "; " + ", ".join(f"{name} {_share(share)}" for name, share in view.groups.items())
        lines.append(f" 2 Range       {labels[seat]}: {_share(view.width)} of hands{groups}")
        if grids:
            lines.extend("               " + row for row in range_grid(view))
    if not reveal:
        return lines
    percentile = f"; top {_share(1 - review.percentile)} of your range" if review.percentile is not None else ""
    error = "" if review.equity.exact else f" ± {_share(review.equity.stderr)}"
    lines.append(
        f" 3 Equity      {_share(review.equity.value)}{error} against their range{'s' if len(review.ranges) > 1 else ''}{percentile}"
    )
    limits = []
    if review.thresholds.required_equity is not None:
        limits.append(f"calling needs {_share(review.thresholds.required_equity)} equity")
    if review.thresholds.defense_share is not None:
        limits.append(f"continue with {_share(review.thresholds.defense_share)} of your range")
    for action, folds in review.thresholds.bluff_break_even.items():
        limits.append(f"a bluff {describe_action(action, scale)} needs {_share(folds)} folds")
    lines.append(" 4 Thresholds  " + ("; ".join(limits) if limits else "nothing to call"))
    bot_labels = [labels[s] for s in review.ranges]
    lines.append(" 5 Options")
    lines.extend(_option_rows(review, scale, bot_labels[0] if len(bot_labels) == 1 else "the bots"))
    if review.reference_note is not None:
        lines.append(f"   Reference: {review.reference_note}")
    if review.chosen is not None:  # a coach hint has nothing chosen to judge yet
        loss, stderr = review.loss(review.reference)
        bot_loss, _ = review.loss(review.exploitative)
        verdict = review.verdict()
        unit = "% of the prize pool" if review.tournament else "bb"
        factor = 100 if review.tournament else 1
        if verdict == "mistake":
            lines.append(
                f"   Mistake: {factor * loss:.2f}{unit} ± {factor * stderr:.2f} against the reference ({factor * bot_loss:.2f} against the bots)"
            )
        elif verdict == "close":
            lines.append(
                f"   Close: {factor * loss:.2f}{unit} ± {factor * stderr:.2f} behind the best option, within the threshold or the noise"
            )
        else:
            lines.append("   Best option.")
        lines.append(f"   Rated {review.rating():.2f} of 1 (1 is the best option you had).")

    if review.exploit_spot():
        lines.append(
            "   Exploit spot: the best play against these bots differs from the play against a strong opponent."
        )
    return lines


def render_hand(review: HandReview, labels: dict[int, str], scale: int, grids: bool = False) -> list[str]:
    lines = []
    for number, decision in enumerate(review.decisions, 1):
        lines.append(f"-- Decision {number}")
        lines.extend(render_decision(decision, labels, scale, grids))
    result = f"Result: {review.net_bb:+.1f}bb"
    if review.all_in_net_bb is not None:
        result += f" (all-in expectation {review.all_in_net_bb:+.1f}bb)"
    lines.append(result)
    return lines


def render_summary(summary: SessionSummary, scale: int) -> list[str]:
    lines = [
        f"{summary.hands} hands, {summary.decisions} decisions: {len(summary.mistakes)} mistakes shown, {summary.close} close calls",
        f"Net {summary.net_bb:+.1f}bb; with all-ins at their expectation {summary.all_in_adjusted_bb:+.1f}bb",
    ]
    for rank, (hand_id, decision) in enumerate(summary.mistakes, 1):
        assert decision.chosen is not None  # mistakes are chosen actions
        loss, _ = decision.loss(decision.reference)
        unit = "% pool" if decision.tournament else "bb"
        amount = 100 * loss if decision.tournament else loss
        board = f" on {cards_str(decision.board)}" if decision.board else ""
        lines.append(
            f" {rank}. {hand_id}, {decision.situation.street.value.lower()}: {cards_str(decision.hole)}{board}, "
            f"{describe_action(decision.chosen, scale)} cost {amount:.2f}{unit}"
        )
    return lines


_KIND_NAMES = {
    "equity": "Equity against the range",
    "required_equity": "Equity needed to call",
    "best_option": "Best option",
}


def render_calibration(summary: list[Calibration]) -> list[str]:
    if not summary:
        return ["No estimates yet: press p after a hand, or run review with --quiz."]
    lines = []
    for item in summary:
        name = _KIND_NAMES[item.kind]
        window = min(item.count, RECENT)
        if item.bias is None:
            lines.append(
                f"{name}: right {_share(item.error)} of {item.count} times (last {window}: {_share(item.recent_error)})"
            )
        else:
            lean = "high" if item.bias > 0 else "low"
            lines.append(
                f"{name}: off by {_share(item.error)} on average over {item.count} "
                f"(last {window}: {_share(item.recent_error)}); your guesses run {_share(abs(item.bias))} {lean}"
            )
    return lines


# Plain words, for the web page and for pasting a hand into a chat assistant.

_SUIT_SYMBOLS = {"c": "♣", "d": "♦", "h": "♥", "s": "♠"}
_POSITION_WORDS = {
    "HJ": "in the hijack (two seats before the dealer)",
    "CO": "in the cutoff (just before the dealer)",
    "BTN": "on the dealer button (last to act after the flop)",
    "SB": "in the small blind",
    "BB": "in the big blind",
}
_SEAT_WORDS = {
    "HJ": "Hijack",
    "CO": "Cutoff",
    "BTN": "Dealer",
    "SB": "Small blind",
    "BB": "Big blind",
}
_POT_TYPE_WORDS = {
    "unopened": "Nobody has raised yet.",
    "limped": "Players have only called the big blind so far.",
    "single-raised": "There has been one raise.",
    "3-bet": "There has been a raise and a re-raise.",
    "4-bet+": "There have been three or more raises.",
}
_GROUP_WORDS = {
    "two pair+": "two pair or better",
    "one pair": "one pair",
    "draw": "a draw",
    "air": "nothing yet",
}
_TEXTURE_WORDS = {
    "A-high": "ace-high",
    "K-high": "king-high",
    "broadway": "high cards (ten to ace)",
    "middle": "middle cards",
    "low": "low cards",
    "unpaired": "no pair",
    "paired": "a pair",
    "trips": "three of a kind",
    "rainbow": "all different suits",
    "two-tone": "two cards of one suit",
    "monotone": "all one suit",
    "flush possible": "a flush possible",
    "no flush": "no flush possible",
    "connected": "cards close in rank (straights likely)",
    "semi-connected": "cards fairly close in rank",
    "disconnected": "cards far apart in rank",
    "flush completed": "the last card made a flush possible",
    "straight completed": "the last card made a straight possible",
    "board paired": "the last card paired the board",
    "overcard": "the last card is higher than the rest",
    "brick": "the last card changed little",
}
_STREET_WORDS = {
    Street.PREFLOP: "Before the flop",
    Street.FLOP: "Flop",
    Street.TURN: "Turn",
    Street.RIVER: "River",
}


def pretty_cards(cards: tuple[int, ...] | list[int]) -> str:
    """ "A♠ 10♥" rather than "As Th"."""
    return " ".join(("10" if text[0] == "T" else text[0]) + _SUIT_SYMBOLS[text[1]] for text in cards_str(cards).split())


def chips_shown(bb: float, big_blind: int, scale: int) -> float:
    """Big blinds as the chips shown on the table, rounded to the smallest chip."""
    return round(bb * big_blind / scale, 0 if scale == 1 else 2) + 0.0  # + 0.0: no "-0"


def _chips(bb: float, big_blind: int, scale: int, signed: bool = False) -> str:
    places = 0 if scale == 1 else 2
    value = chips_shown(bb, big_blind, scale)
    return f"{value:+,.{places}f}" if signed and value else f"{value:,.{places}f}"


def _worth(option: OptionValue, review: DecisionReview, big_blind: int, scale: int) -> str:
    if review.tournament:
        assert option.icm is not None
        noise = f" ± {100 * option.icm_stderr:.2f}" if option.icm_stderr else ""
        return f"{100 * option.icm:.2f}%{noise}"
    noise = "" if option.exact else f" ± {_chips(option.stderr, big_blind, scale)}"
    return _chips(option.ev, big_blind, scale, signed=True) + noise


def decision_headline(review: DecisionReview, scale: int, big_blind: int, mix: dict[Action, float]) -> str:
    """One line: the verdict on a chosen move, or the suggestion for a coach hint. The best move
    is the best against a strong player, the standard the verdict uses. When a hint's averages
    are within their noise of each other, it names what a strong player does (`mix`) instead."""
    tournament = review.tournament
    best = best_option(review.reference, tournament)
    best_text = describe_action(best.action, scale)
    if review.chosen is None:
        value, clear = _icm_or_ev(best, tournament), True
        for other in review.reference:
            if other.action == best.action:
                continue
            if tournament:
                noise = math.hypot(best.icm_stderr or 0.0, other.icm_stderr or 0.0)
            else:
                noise = math.hypot(best.stderr, other.stderr)
            clear = clear and value - _icm_or_ev(other, tournament) > 2 * noise
        if clear or not mix:
            return f"Suggested: {best_text}."
        usual = max(mix, key=mix.__getitem__)
        return (
            f"Too close to tell from the averages. A strong player would "
            f"{describe_action(usual, scale)} here {_share(mix[usual])} of the time."
        )
    chosen = describe_action(review.chosen, scale)
    verdict = review.verdict()
    if verdict == "best":
        return f"Good move. Best option: {chosen}."
    if verdict == "close":
        return f"Close. Best option: {best_text}; your {chosen} was nearly as good."
    loss, _ = review.loss(review.reference)
    cost = f"{100 * loss:.2f}% of the prize pool" if tournament else f"{_chips(loss, big_blind, scale)} chips"
    return f"Costly. Best option: {best_text}; your {chosen} gave up about {cost} on average."


def _icm_or_ev(option: OptionValue, tournament: bool) -> float:
    if tournament:
        assert option.icm is not None
        return option.icm
    return option.ev


def decision_summary(review: DecisionReview, scale: int, big_blind: int) -> list[str]:
    """The two or three lines a newcomer reads under the headline, from what the user could know:
    the others' hands as a strong player reads them, not the bots' styles."""
    equity = review.reference_equity
    error = "" if equity.exact else f" (± {_share(equity.stderr)})"
    lines = [] if review.chosen is None else [f"Rated {review.rating():.2f} of 1 (1 is the best option you had)."]
    lines.append(
        "Your chance at showdown against the hands a strong player would put them on: about "
        f"{_share(equity.value)}{error}."
    )
    required = review.thresholds.required_equity
    if required is not None:
        to_call = _chips(review.situation.to_call_bb, big_blind, scale)
        lines.append(f"Calling costs {to_call}: it pays if you win at least {_share(required)} of the time.")
    return lines


def god_summary(review: DecisionReview, god: GodView | None, scale: int) -> list[str]:
    """God's view lines under the summary: the chance against the hands the bots' styles would
    hold, the better play against these bots, and with their cards known (`god`), the chance
    against those cards and the move that would have won more in hindsight."""
    error = "" if review.equity.exact else f" (± {_share(review.equity.stderr)})"
    lines = [f"Against the hands these players' styles would hold here: about {_share(review.equity.value)}{error}."]
    if review.exploit_spot():
        best_bot = best_option(review.exploitative, review.tournament)
        lines.append(
            f"Against these particular opponents, {describe_action(best_bot.action, scale)} does better than the standard play."
        )
    if god is None:
        return lines
    about = "" if god.equity.exact else "about "
    lines.append(f"Against the cards they really held: {about}{_share(god.equity.value)}.")
    best = hindsight_best(review, god) if review.chosen is not None else None
    if best is not None:
        chosen = next(o for o in god.options if o.action == review.chosen)
        estimate = "" if best.exact and chosen.exact else " (an estimate)"
        lines.append(
            f"Seeing their cards, {describe_action(best.action, scale)} would have won more"
            f"{estimate}. You couldn't see them, so the rating stands."
        )
    return lines


def range_lines(ranges: dict[int, RangeView], labels: dict[int, str], lead: str = "") -> list[str]:
    """What each opponent likely holds, opponents who read the same named together."""
    alike: dict[str, list[str]] = {}
    for seat, view in ranges.items():
        groups = ""
        if view.groups is not None:
            groups = (
                " (" + ", ".join(f"{_GROUP_WORDS[name]} {_share(share)}" for name, share in view.groups.items()) + ")"
            )
        alike.setdefault(f"about {_share(view.width)} of all starting hands{groups}", []).append(labels[seat])
    lines = []
    for holding, names in alike.items():
        who = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
        lines.append(f"{lead}{who} likely {'holds' if len(names) == 1 else 'each hold'} {holding}.")
    return lines


def decision_details(review: DecisionReview, labels: dict[int, str], scale: int, big_blind: int) -> list[str]:
    """Everything else, in words: the spot, the board, what each opponent likely holds as a strong
    player reads them, and the break-even numbers."""
    spot = review.situation
    after = "last" if spot.in_position else "first"
    where = (
        "in early position (among the first to act)"
        if spot.position.startswith("UTG")
        else _POSITION_WORDS[spot.position]
    )
    lines = [
        f"You are {where}, with {spot.players} players in the hand; you act {after} after the flop.",
        f"{_POT_TYPE_WORDS[spot.pot_type]} Pot {_chips(spot.pot_bb, big_blind, scale)}, "
        f"{_chips(spot.to_call_bb, big_blind, scale)} to call, {_chips(spot.effective_bb, big_blind, scale)} left to play for.",
    ]
    if spot.spr is not None:
        lines.append(f"Stack-to-pot ratio {spot.spr:.1f}: the chips left to play for, per chip already in the pot.")
    if spot.texture is not None:
        tags = spot.texture
        words = [_TEXTURE_WORDS[t] for t in (tags.high, tags.pairing, tags.suits, tags.connectivity)]
        change = f"; {_TEXTURE_WORDS[tags.change]}" if tags.change else ""
        lines.append(f"The board: {', '.join(words)}{change}.")
    lines += range_lines(review.reference_ranges, labels)
    if review.percentile is not None:
        lines.append(f"Your hand is stronger than {_share(review.percentile)} of the hands you would likely have here.")
    defense = review.thresholds.defense_share
    if defense is not None:
        lines.append(
            f"Folding more than {_share(1 - defense)} of your hands here would let any bluff against you profit."
        )
    for action, folds in review.thresholds.bluff_break_even.items():
        lines.append(
            f"A bluff {describe_action(action, scale)} pays off if they fold more than {_share(folds)} of the time."
        )
    if review.reference_note is not None:
        lines.append(f"The strong-player numbers here come from a {review.reference_note}.")
    if not all(o.exact for o in review.reference):
        lines.append("Numbers with ± are estimates from sampling: options closer than that are too close to call.")
    if not review.tournament:
        lines.append(f"1 big blind = {_chips(1, big_blind, scale)} chips.")
    return lines


def solver_decision_text(
    number: int,
    review: DecisionReview,
    labels: dict[int, str],
    scale: int,
    big_blind: int,
    mix: dict[Action, float],
    styled: dict[int, str] | None,
    god: GodView | None,
) -> list[str]:
    """One decision with the coach's numbers, from what the user could know; with `styled`
    (God's view, seat names with styles) also the bots' styles and, from `god`, their cards."""
    board = pretty_cards(review.board) if review.board else "none yet"
    chose = "" if review.chosen is None else f"; you chose {describe_action(review.chosen, scale)}"
    lines = [
        f"Decision {number} ({_STREET_WORDS[review.situation.street].lower()}): your cards {pretty_cards(review.hole)}, board {board}{chose}.",
        decision_headline(review, scale, big_blind, mix),
    ]
    lines += decision_summary(review, scale, big_blind)
    if styled is not None:
        lines += god_summary(review, god, scale)
    lines += decision_details(review, labels, scale, big_blind)
    if styled is not None:
        lines += range_lines(review.ranges, styled, "By their styles, ")
    unit = "share of the prize pool" if review.tournament else "chips"
    columns = ["against a strong player"]
    if styled is not None:
        columns += ["against these players' styles", "against the cards they held, in hindsight"]
    lines.append(f"Options (average result in {unit}: {' | '.join(columns)} | how often a strong player does it):")
    best = best_option(review.reference, review.tournament).action
    bots = {o.action: o for o in review.exploitative}
    cards = {o.action: o for o in god.options} if god is not None else {}
    for option in review.reference:
        marks = [
            m
            for m, on in (
                ("your choice", option.action == review.chosen),
                ("best against a strong player", option.action == best),
            )
            if on
        ]
        values = [_worth(option, review, big_blind, scale)]
        if styled is not None:
            values.append(_worth(bots[option.action], review, big_blind, scale))
            held = cards.get(option.action)
            values.append("-" if held is None else _worth(held, review, big_blind, scale))
        lines.append(
            f"  {describe_action(option.action, scale)}: {' | '.join(values)} | {_share(mix.get(option.action, 0.0))}"
            + (f" ({', '.join(marks)})" if marks else "")
        )
    return lines


def held_text(hand: GameState, user: int, labels: dict[int, str], shown: dict[int, tuple[int, int]]) -> list[str]:
    """For God's view of a finished hand: the cards each opponent held that the user never saw."""
    lines = []
    for seat, cards in enumerate(hand.hole_cards):
        if seat == user or not hand.dealt_in[seat] or seat in hand.shown or seat in shown:
            continue
        assert cards is not None
        folds = [e.street for e in hand.history if e.seat == seat and e.action.type == ActionType.FOLD]
        how = (
            "didn't have to show"
            if not folds
            else "folded before the flop"
            if folds[0] == Street.PREFLOP
            else f"folded on the {_STREET_WORDS[folds[0]].lower()}"
        )
        lines.append(f"{labels[seat]} held {pretty_cards(cards)} ({how}).")
    return lines


def percent_text(share: float) -> str:
    """A whole percent, or one decimal where that would read as none or all."""
    whole = round(100 * share)
    return str(whole) if 0 < whole < 100 or share in (0, 1) else f"{100 * share:.1f}"


def rated_moves_text(hand: GameState, user: int, moves: Sequence[MoveRating], scale: int) -> list[str]:
    """The user's moves with their 0-1 ratings, under a header giving the scale in the coach's
    words."""
    states = replay_states(hand)
    chips = functools.partial(format_chips, scale=scale)
    close = f"{RATING_CLOSE:.2f}"
    lines = [
        f"Your moves, rated from 0 to 1: 1.00 is the best option you had, {close} to 0.99 is close "
        f"to it, and below {close} is a costly mistake; the more of the pot a move gives up, the "
        "lower it goes."
    ]
    for move in moves:
        entry = hand.history[move.history_index]
        words = _move_text(entry.action, user, states[move.history_index], "you", True, chips)
        marks = " (after a hint)" if move.hinted else ""
        marks += " (time ran out)" if move.timed_out else ""
        lines.append(f"  {_STREET_WORDS[entry.street]}: {words}, rated {move.rating:.2f}{marks}")
    return lines


def rated_hand_text(
    hand: GameState,
    user: int,
    moves: Sequence[MoveRating],
    chance: float | None,
    rating: float | None,
    scale: int,
) -> tuple[list[str], str]:
    """The rated moves and the summary sentence, as the copy texts and the hands tables give
    them. Without moves, the summary counts the user's moves in the hand."""
    hole = hand.hole_cards[user]
    assert hole is not None
    lines = rated_moves_text(hand, user, moves, scale) if moves else []
    made = len(moves) or sum(e.seat == user for e in hand.history)
    return lines, hand_summary_text(hole, chance, rating, made, is_terminal(hand))


def hand_summary_text(hole: tuple[int, int], chance: float | None, rating: float | None, moves: int, over: bool) -> str:
    """The hand's three numbers in words: the starting hand's rank, the chance to win at the
    last (or, mid-hand, latest) move, and the rating of `moves` rated moves."""
    stronger, through = hand_rank(hole)
    rank = f"best {percent_text(through)}%" if through <= 0.5 else f"worst {percent_text(1 - stronger)}%"
    parts = [f"Your cards ({pretty_cards(hole)}) are in the {rank} of starting hands."]
    if not moves:
        if over:
            parts.append("You made no decision this hand.")
        return " ".join(parts)
    if chance is not None:
        parts.append(
            f"At your {'last' if over else 'latest'} move your chance to win at showdown was about "
            f"{percent_text(chance)}%, reading the others' hands from their play the way a strong "
            "player would."
        )
    if rating is not None:
        span = "this hand" if over else "so far"
        if moves == 1:
            parts.append(f"Your move {span} rates {rating:.2f}.")
        else:
            parts.append(f"Your moves {span} rate {rating:.2f} overall, with moves in bigger pots counting for more.")
    return " ".join(parts)


def hand_result_text(net_bb: float, all_in_net_bb: float | None, scale: int, big_blind: int) -> str:
    if abs(net_bb) < 1e-9:
        text = "You broke even this hand."
    else:
        verb = "won" if net_bb > 0 else "lost"
        blinds = f"{abs(net_bb):.1f}".removesuffix(".0")
        unit = "big blind" if blinds == "1" else "big blinds"
        text = f"You {verb} {_chips(abs(net_bb), big_blind, scale)} chips ({blinds} {unit}) this hand."
    if all_in_net_bb is not None:
        text += f" With the cards as they were when the chips went in, you would average {_chips(all_in_net_bb, big_blind, scale, signed=True)}."
    return text


def hand_history_text(
    hand: GameState,
    user: int,
    labels: dict[int, str],
    scale: int,
    game: str,
    shown: dict[int, tuple[int, int]],
) -> list[str]:
    """The hand as the user saw it, street by street, up to its result (which the caller adds):
    only the user's cards and cards that were shown. `game` names it ("cash game",
    "tournament"); `shown` holds cards a bot showed without a showdown."""
    config = hand.config
    chips = functools.partial(format_chips, scale=scale)
    states = replay_states(hand)
    ante = ""
    if config.ante_type == AnteType.BIG_BLIND_ANTE:
        ante = f", big blind ante {chips(config.ante)}"
    elif config.ante_type == AnteType.PER_PLAYER:
        ante = f", ante {chips(config.ante)} each"
    lines = [
        f"No-limit Texas Hold'em {game}, {sum(hand.dealt_in)} players, blinds "
        f"{chips(config.small_blind)}/{chips(config.big_blind)}{ante}. "
        f"Hand {hand.hand_id.rpartition('-')[2]}.",
        "Seats in the order they act before the flop, with their chips at the start of the hand:",
    ]
    # What each seat posted, from the deal's own events: the amounts depend only on the stacks,
    # and a later state has already returned any uncalled part.
    posted: dict[int, int] = defaultdict(int)
    for event in new_hand(config, hand.seed, hand.button, hand.starting_stacks, hand.dealt_in, hand.hand_id)[1]:
        if event.kind == "AntesPosted":
            for seat, amount in enumerate(event.data["amounts"]):
                posted[seat] += amount
        elif event.kind == "BlindsPosted":
            posted[event.data["small_blind_seat"]] += event.data["small_blind"]
            posted[event.data["big_blind_seat"]] += event.data["big_blind"]
    for seat, code in position_names(observation(hand, user)).items():
        extra = f", posts {chips(posted[seat])}" if posted[seat] else ""
        mine = f", your cards {pretty_cards(hand.hole_cards[seat] or ())}" if seat == user else ""
        seat_name = _SEAT_WORDS.get(code, "Early position")
        lines.append(f"  {seat_name} ({code}): {labels[seat]}, {chips(hand.starting_stacks[seat])}{extra}{mine}")
    street = None
    moves: list[str] = []
    for index, before in enumerate(states):
        # A street's header comes with its first action, or at the end for a street just dealt.
        now = hand.history[index].street if index < len(hand.history) else before.street
        if now != street and now != Street.COMPLETE:
            if moves:
                lines.append("  " + ", ".join(moves) + ".")
            moves = []
            street = now
            header = _STREET_WORDS[street]
            if street != Street.PREFLOP:
                new = before.board[-1:] if street != Street.FLOP else before.board
                header += f" {pretty_cards(new)} (pot {chips(before.pot)})"
            lines.append(header + ":")
        if index < len(hand.history):
            entry = hand.history[index]
            moves.append(_move_text(entry.action, entry.seat, before, labels[entry.seat], entry.seat == user, chips))
    if moves:
        lines.append("  " + ", ".join(moves) + ".")
    final = states[-1]
    if final.street == Street.COMPLETE:
        runout = final.board[len(states[-2].board) :] if hand.history else final.board
        if runout:
            lines.append(f"The rest of the board, with no more betting: {pretty_cards(runout)}.")
        for seat in final.shown:
            cards = final.hole_cards[seat]
            assert cards is not None
            value = describe(evaluate(list(cards) + list(final.board)))
            lines.append(f"{labels[seat]} {_conjugate('show', seat == user)} {pretty_cards(cards)}: {value}.")
        for seat, cards in shown.items():
            lines.append(f"{labels[seat]} shows {pretty_cards(cards)} without being called.")
        widest = max((len(a.eligible) for a in final.awards), default=0)
        for award in final.awards:
            side = " (side pot)" if len(award.eligible) < widest else ""
            lines.append(
                ", ".join(
                    f"{labels[s]} {_conjugate('win', s == user)} {chips(x)}"
                    for s, x in zip(award.winners, award.shares)
                )
                + side
                + "."
            )
    elif final.to_act == user:
        lines.append("The hand is still being played, and it is your turn.")
    else:
        lines.append("The hand is still being played.")
    return lines


def _move_text(
    action: Action,
    seat: int,
    before: GameState,
    name: str,
    is_user: bool,
    chips: Callable[[int], str],
) -> str:
    """A call shows the chips it adds; a bet or raise the street total it makes."""
    committed = before.committed_this_street[seat]
    everything = committed + before.stacks[seat]
    if action.type == ActionType.CALL:
        verb = f"call {chips(min(before.current_bet, everything) - committed)}"
        all_in = before.current_bet >= everything
    elif action.amount is not None:
        verb = f"{_VERBS[action.type]} {chips(action.amount)}"
        all_in = action.amount == everything
    else:
        verb, all_in = _VERBS[action.type], False
    return f"{name} {_conjugate(verb, is_user)}" + (" (all-in)" if all_in else "")
