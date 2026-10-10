"""Leak statistics (DESIGN.md Section 7.5): patterns across every reviewed decision, compared to
the theory baselines of Section 7.4, and the spots that cost the most; and the skill metrics of
Section 7.7.

Each reviewed decision is kept as a small record (tags, loss, what was chosen, the thresholds),
so statistics over many sessions do not need the sessions reviewed again. Spot tags are a fixed
taxonomy, so the statistics compare like with like across sessions.

The skill metric is the reference EV lost per 100 hands, independent of how the cards ran, next
to what the bots lose when their own play is reviewed the same way. Losses are weighted by how
recent their hand is (a half-life of `HALF_LIFE_HANDS`), so the measure follows improvement. The
0-3000 rating of DESIGN.md Section 14 is not given yet: the tiers' own losses (measured by
`analysis/rating_calibration.py`) do not separate Tier 2 from Tier 3, so anchoring a scale on
them would mislead.
"""

from   bisect                   import bisect_left, bisect_right
from   collections              import defaultdict
from   collections.abc          import Mapping
import csv
from   dataclasses              import dataclass
import math
from   thpoker.analysis.review  import (DecisionReview, HandRating, hand_group,
                                        position_names)
from   thpoker.game.cards       import cards_str
from   thpoker.game.engine      import observation
from   thpoker.game.state       import GameState, Street
from   thpoker.odds             import hand_rank
from   thpoker.storage          import Record
from   typing                   import Any, TextIO

MIN_SAMPLE = 5  # decisions a pattern needs before it is reported
LARGE_FACING = ("medium", "large", "overbet")  # bets of more than 40% of the pot


@dataclass(frozen=True)
class DecisionRecord(Record):
    """What the statistics keep of a reviewed decision. `loss` is the reference loss (big
    blinds, or prize-pool share in tournaments); `bet_fraction` is the user's own bet or raise
    as a share of the pot before it."""

    session: str
    hand_id: str
    index: int
    tags: dict[str, str]
    loss: float
    stderr: float
    verdict: str
    action: str
    defense: float | None
    bet_fraction: float | None


def record(
    review: DecisionReview,
    session: str,
    hand_id: str,
    paid_places: int | None = None,
    training: bool = False,
) -> DecisionRecord:
    loss, stderr = review.loss(review.reference)
    assert review.chosen is not None
    return DecisionRecord(
        session,
        hand_id,
        review.index,
        tags(review, review.tournament, paid_places, training),
        loss,
        stderr,
        review.verdict(),
        review.chosen.type.value.lower(),
        review.thresholds.defense_share,
        review.chosen_bet,
    )


def loss_by_tag(records: list[DecisionRecord], min_count: int = MIN_SAMPLE) -> list[tuple[str, str, int, float]]:
    """(tag, value, decisions, mean loss) for every tag value seen at least `min_count` times,
    the costliest first."""
    groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    for item in records:
        for key, value in item.tags.items():
            groups[(key, value)].append(item.loss)
    rows = [
        (key, value, len(losses), sum(losses) / len(losses))
        for (key, value), losses in groups.items()
        if len(losses) >= min_count
    ]
    return sorted(rows, key=_mean_loss, reverse=True)


def _mean_loss(row: tuple[str, str, int, float]) -> float:
    return row[3]


def patterns(records: list[DecisionRecord], min_count: int = MIN_SAMPLE) -> list[str]:
    """Findings phrased as patterns, for cash decisions (chip losses) with enough samples."""
    cash = [r for r in records if r.tags.get("mode") == "cash"]
    found = []
    facing = [
        r for r in cash if r.tags["street"] == "river" and r.tags["facing"] in LARGE_FACING and r.defense is not None
    ]
    if len(facing) >= min_count:
        folded = sum(r.action == "fold" for r in facing) / len(facing)
        defense = sum(r.defense or 0.0 for r in facing) / len(facing)
        gap = (1 - folded) - defense
        direction = "less" if gap < 0 else "more"
        found.append(
            f"Facing river bets over 40% of the pot you continue {1 - folded:.0%} of the time against a "
            f"minimum defense of {defense:.0%} ({len(facing)} decisions): {100 * abs(gap):.0f} points {direction} than the theory."
        )
    river_bets = [
        r for r in cash if r.tags["street"] == "river" and r.bet_fraction is not None and r.tags["facing"] == "unopened"
    ]
    if len(river_bets) >= min_count:
        bluffs = sum(r.tags["hand"] == "air" for r in river_bets) / len(river_bets)
        balanced = sum(b / (1 + 2 * b) for b in (r.bet_fraction or 0.0 for r in river_bets)) / len(river_bets)
        found.append(
            f"Your river bets are {bluffs:.0%} bluffs; a balanced range at your sizes has {balanced:.0%} ({len(river_bets)} bets)."
        )
    costly = [row for row in loss_by_tag(cash, min_count) if row[0] != "mode"][:3]
    for key, value, count, mean in costly:
        found.append(f"Costly spot: {key} {value}, {mean:.2f}bb lost per decision over {count} decisions.")
    return found


# Buckets of the bet faced, as a share of the pot before it.
FACING_BUCKETS = ((0.4, "small"), (0.7, "medium"), (1.05, "large"))


def facing_bucket(facing: float | None) -> str:
    if facing is None:
        return "unopened"
    for limit, name in FACING_BUCKETS:
        if facing <= limit:
            return name
    return "overbet"


def _preflop_facing(review: DecisionReview) -> str:
    """Preflop sizes are about the pot type, not the pot share (an open is 1.3-1.7 pots)."""
    spot = review.situation
    if spot.to_call_bb == 0:
        return "unopened"
    return {
        "unopened": "blinds",
        "limped": "limpers",
        "single-raised": "open",
        "3-bet": "3-bet",
    }.get(spot.pot_type, "4-bet+")


def hand_class(review: DecisionReview) -> str:
    """Where the hand stands: preflop by its place in the user's own range; after the flop
    value (top quarter of the range with a made hand), draw, marginal, or air."""
    percentile = review.percentile if review.percentile is not None else 0.5
    if not review.board:
        return ("weak", "playable", "strong", "premium")[bisect_right((0.3, 0.6, 0.9), percentile)]
    group = hand_group(review.hole, review.board)
    made = group in ("two pair+", "one pair")
    if made and percentile >= 0.75:
        return "value"
    if group == "draw":
        return "draw"
    return "marginal" if made or percentile >= 0.4 else "air"


def tags(
    review: DecisionReview,
    tournament: bool,
    paid_places: int | None = None,
    training: bool = False,
) -> dict[str, str]:
    """The decision's tags. `paid_places` (tournaments) sets the stage. Training decisions get
    their own mode, so the cash measures leave them out."""
    spot = review.situation
    table = "heads-up" if spot.dealt == 2 else "3-5" if spot.dealt <= 5 else "6-8"
    result = {
        "mode": "training" if training else "tournament" if tournament else "cash",
        "table": table,
        "pot": "heads-up" if spot.players == 2 else "multiway",
        "street": spot.street.value.lower(),
        "position": spot.position,
        "in_position": "IP" if spot.in_position else "OOP",
        "pot_type": spot.pot_type,
        "role": spot.role,
        "facing": _preflop_facing(review) if spot.street == Street.PREFLOP else facing_bucket(spot.facing),
        "hand": hand_class(review),
    }
    if spot.street != Street.PREFLOP and spot.texture is not None and spot.spr is not None:
        result["texture"] = "wet" if spot.texture.wet else "dry"
        result["spr"] = "under 2" if spot.spr < 2 else "2-6" if spot.spr <= 6 else "over 6"
    if tournament:
        stack = spot.stack_bb
        result["stack"] = ("push/fold", "short", "medium", "deep")[bisect_left((15, 25, 50), stack)]
        if paid_places is not None:
            left = spot.dealt
            result["stage"] = (
                "heads-up"
                if left == 2
                else "in the money"
                if left <= paid_places
                else "near payouts"
                if left == paid_places + 1
                else "early"
            )
    return result


HALF_LIFE_HANDS = 2000
# Below this many hands one big pot can swing the measure by more than the gap between tiers.
MIN_HANDS = 200
# (reference bb lost per 100 hands, standard error) of each tier playing the user's seat at
# six-handed 100bb cash tables (rating_calibration, about 1,200 hands each).
BOT_LOSSES = {3: (199.7, 18.6), 2: (207.1, 20.7), 1: (258.4, 25.1)}


@dataclass(frozen=True)
class Progress:
    hands: int
    decisions: int
    loss_per_100: float
    stderr_per_100: float
    comparison: str  # against the bots' own losses, allowing two standard errors


def _tiers(tiers: list[int]) -> str:
    names = [str(t) for t in tiers]
    return "the Tier " + (names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]) + " bots"


def compare(loss: float, stderr: float, hands: int) -> str:
    """Where a loss per 100 hands over `hands` hands stands against each tier, allowing two
    standard errors of the user's measure and the tier's together."""
    if hands < MIN_HANDS:
        return f"too few hands to compare: review at least {MIN_HANDS}"
    better, behind, level = [], [], []
    for tier, (mean, bot_stderr) in sorted(BOT_LOSSES.items()):
        margin = 2 * math.hypot(stderr, bot_stderr)
        if loss + margin < mean:
            better.append(tier)
        elif loss - margin > mean:
            behind.append(tier)
        else:
            level.append(tier)
    if not better and not behind:
        return "not yet told apart from any tier: review more hands"
    parts = [
        f"{word} {_tiers(tiers)}"
        for word, tiers in (("better than", better), ("behind", behind), ("level with", level))
        if tiers
    ]
    return ", ".join(parts)


def progress(records: list[DecisionRecord]) -> Progress | None:
    """The measure over cash decisions (tournament losses are prize-pool shares, another unit),
    or None without any. Hands count in the order their decisions were recorded."""
    by_hand: dict[tuple[str, str], float] = {}
    decisions = 0
    for record in records:
        if record.tags.get("mode") != "cash":
            continue
        key = (record.session, record.hand_id)
        by_hand[key] = by_hand.get(key, 0.0) + record.loss
        decisions += 1
    if not by_hand:
        return None
    losses = list(by_hand.values())
    count = len(losses)
    weights = [0.5 ** ((count - 1 - age) / HALF_LIFE_HANDS) for age in range(count)]
    total = sum(weights)
    mean = sum(w * loss for w, loss in zip(weights, losses)) / total
    stderr = sum((w * (loss - mean)) ** 2 for w, loss in zip(weights, losses)) ** 0.5 / total
    return Progress(count, decisions, 100 * mean, 100 * stderr, compare(100 * mean, 100 * stderr, count))


HAND_COLUMNS = (
    "session",
    "date",
    "mode",
    "players",
    "big_blind",
    "ante",
    "ante_type",
    "hand",
    "position",
    "cards",
    "board",
    "put_in",
    "result",
    "result_bb",
    "showdown",
    "hand_rank",
    "win_chance",
    "rating",
    "decisions",
    "mistakes",
    "loss_bb",
    "loss_prize_pct",
    "history",
)


def _number(amount: float) -> float | int:
    return int(amount) if amount == int(amount) else round(amount, 2)


def hand_rows(
    hands: list[GameState],
    user: int,
    scale: int,
    session: str,
    date: str,
    mode: str,
    decisions: list[DecisionRecord],
    histories: dict[str, str],
    ratings: Mapping[str, HandRating],
) -> list[dict[str, Any]]:
    """One row per hand the user was dealt into, in chips as shown on the table, for a
    spreadsheet; the big blind and ante are the hand's own, as they rise in a tournament. 
    Only the user's own cards appear. `hand_rank` is the share of starting hands
    at least as strong; the chance to win at the last move and the hand's rating come from
    `ratings`, by hand id, and stay blank for hands never rated. The review columns come from
    `decisions` (records written by `thpoker review`) and stay blank for hands never reviewed;
    `histories` holds each hand's moves as the user saw them, by hand id."""
    reviewed: dict[str, list[DecisionRecord]] = defaultdict(list)
    for decision in decisions:
        if decision.session == session:
            reviewed[decision.hand_id].append(decision)
    rows = []
    for hand in hands:
        if not hand.dealt_in[user]:
            continue
        won = sum(x for a in hand.awards for s, x in zip(a.winners, a.shares) if s == user)
        result = hand.stacks[user] - hand.starting_stacks[user]
        records = reviewed.get(hand.hand_id)
        loss = sum(r.loss for r in records) if records else None
        tournament = mode == "tournament"
        hole = hand.hole_cards[user]
        assert hole is not None
        rated = ratings.get(hand.hand_id)
        rating = rated.rating() if rated is not None else None
        rows.append(
            {
                "session": session,
                "date": date,
                "mode": mode,
                "players": sum(hand.dealt_in),
                "big_blind": _number(hand.config.big_blind / scale),
                "ante": _number(hand.config.ante / scale),
                "ante_type": hand.config.ante_type.lower(),
                "hand": int(hand.hand_id.rpartition("-")[2]),
                "position": position_names(observation(hand, user))[user],
                "cards": cards_str(hole),
                "board": cards_str(hand.board),
                "put_in": _number((won - result) / scale),
                "result": _number(result / scale),
                "result_bb": round(result / hand.config.big_blind, 2),
                "showdown": "yes" if user in hand.shown else "no",
                "hand_rank": round(hand_rank(hole)[1], 4),
                "win_chance": round(rated.chance, 3) if rated is not None else "",
                "rating": round(rating, 2) if rating is not None else "",
                "decisions": len(records) if records else "",
                "mistakes": sum(r.verdict == "mistake" for r in records) if records else "",
                "loss_bb": round(loss, 2) if loss is not None and not tournament else "",
                "loss_prize_pct": round(100 * loss, 3) if loss is not None and tournament else "",
                "history": histories.get(hand.hand_id, ""),
            }
        )
    return rows


def write_hands_csv(rows: list[dict[str, Any]], stream: TextIO):
    writer = csv.DictWriter(stream, fieldnames=HAND_COLUMNS, lineterminator="\n", extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
