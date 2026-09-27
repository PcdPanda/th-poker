"""Text for the command line and the web table: one line of narration per thing that happened in
a hand or a session, and the rendering of reviews (the data comes from `analysis/review.py`).
"""

from   collections.abc          import Callable

from   thpoker.analysis.ev      import OptionValue
from   thpoker.analysis.review  import (DecisionReview, HandReview, RangeView,
                                        SessionSummary, best_option)
from   thpoker.analysis.training \
                                import Calibration, RECENT
from   thpoker.game.cards       import RANKS, cards_str
from   thpoker.game.state       import Action, ActionType, Event


def format_chips(amount: float, scale: int) -> str:
    shown = amount / scale
    return f"{shown:,.0f}" if shown == int(shown) else f"{shown:,.2f}"


def _conjugate(verb: str, is_user: bool) -> str:
    """ "raise to 300" for the user, "raises to 300" for everyone else."""
    if is_user:
        return verb
    first, _, rest = verb.partition(" ")
    return f"{first}s {rest}".rstrip()


def narrate(event: Event, label: Callable[[int], str], chips: Callable[[int], str], user: int | None) -> list[str]:
    """Lines describing `event`: `label` names a seat, `chips` formats an amount."""
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
    if event.kind == "PotAwarded":
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
