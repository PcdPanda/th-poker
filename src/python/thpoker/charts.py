"""Lookups into the shipped strategy tables: preflop strategies for 20bb-and-deeper stacks, from
`thpoker/data/preflop_ranges.json.gz`, and chip-EV push/fold charts for first-in shoves, from
`thpoker/data/pushfold.json.gz` (built by `bin/generate_data.py`; the models are in the
docstrings of `thpoker/generators`).
"""

from   functools                import cache
import gzip
import json
from   pathlib                  import Path
from   thpoker.game.cards       import PREFLOP_CLASSES
from   typing                   import Any

PREFLOP_RANGES = Path(__file__).resolve().parent / "data" / "preflop_ranges.json.gz"
PUSHFOLD_CHARTS = Path(__file__).resolve().parent / "data" / "pushfold.json.gz"
STACK_BUCKETS = (20.0, 40.0, 100.0)


@cache
def _ranges() -> dict[str, Any]:
    data = json.loads(gzip.decompress(PREFLOP_RANGES.read_bytes()))
    if tuple(data["classes"]) != PREFLOP_CLASSES:
        raise ValueError(f"{PREFLOP_RANGES!r} lists classes in a different order")
    return data


def stack_bucket(effective_big_blinds: float) -> float:
    """The solved stack depth closest on a ratio scale (20 below 28bb, 40 below 63bb, else 100)."""
    for low, high in zip(STACK_BUCKETS, STACK_BUCKETS[1:]):
        if effective_big_blinds < (low * high) ** 0.5:
            return low
    return STACK_BUCKETS[-1]


NODES = ("open", "respond", "versus_3bet", "versus_allin")


def strategy(node: str, num_players: int, seats: str, stack: float, bb_ante: bool) -> tuple[tuple[float, ...], ...]:
    """Per hand class (in PREFLOP_CLASSES order), action probabilities with fold first:
    open (fold, open) for seat "p"; respond (fold, call, 3-bet), versus_3bet (fold, call,
    all-in) and versus_allin (fold, call) for seats "p-q" (opener p, responder q). Raises
    `ValueError` for an unknown node, seats without a strategy, or a table size outside 2-8."""
    if node not in NODES:
        raise ValueError(f"unknown node {node!r}; choose from {NODES}")
    if not 2 <= num_players <= 8:
        raise ValueError(f"preflop ranges cover 2 to 8 players, got {num_players}")
    ante = "bb_ante" if bb_ante else "no_ante"
    config = _ranges()["configs"][f"{num_players}/{stack_bucket(stack):g}/{ante}"]
    if seats not in config[node]:
        raise ValueError(f"no {node} strategy for seats {seats} with {num_players} players")
    rows = config[node][seats]
    if node == "open":
        return tuple((1 - p / 100, p / 100) for p in rows)
    return tuple((max(0.0, 1 - sum(r) / 100), *(p / 100 for p in r)) for r in rows)


def seat_names(num_players: int) -> list[str]:
    """Position names in preflop order (first to act first): UTG, UTG+1, ..., HJ, CO, then BTN,
    SB, BB; heads-up the button is the small blind."""
    if num_players == 2:
        return ["BTN", "BB"]
    others = num_players - 3
    late = ["HJ", "CO"][:others] if others else []
    early = ["UTG" if i == 0 else f"UTG+{i}" for i in range(others - len(late))]
    return early + late + ["BTN", "SB", "BB"]


def seats_from_button(num_players: int) -> list[str]:
    """Position names by how many dealt-in seats left of the button they sit (0 is BTN)."""
    names = seat_names(num_players)
    return names if num_players == 2 else names[-3:] + names[:-3]


@cache
def _charts() -> dict[str, Any]:
    data = json.loads(gzip.decompress(PUSHFOLD_CHARTS.read_bytes()))
    if tuple(data["classes"]) != PREFLOP_CLASSES:
        raise ValueError(f"{PUSHFOLD_CHARTS!r} lists classes in a different order")
    return data


def _chart(num_players: int, bb_ante: bool) -> dict[str, dict[str, list[float]]]:
    if not 2 <= num_players <= 8:
        raise ValueError(f"push/fold charts cover 2 to 8 players, got {num_players}")
    return _charts()["tables"][str(num_players)]["bb_ante" if bb_ante else "no_ante"]


def push_chart(num_players: int, pusher: int, bb_ante: bool) -> tuple[float, ...]:
    """Per hand class (in PREFLOP_CLASSES order), the largest effective stack at which it shoves
    first in from seat `pusher`. Raises ValueError for a seat that cannot be first in (the big
    blind) or is out of range."""
    if not 0 <= pusher < num_players - 1:
        raise ValueError(f"seat {pusher} cannot shove first in with {num_players} players")
    return tuple(_chart(num_players, bb_ante)["push"][str(pusher)])


def call_chart(num_players: int, pusher: int, caller: int, bb_ante: bool) -> tuple[float, ...]:
    """Per hand class (in PREFLOP_CLASSES order), the largest effective stack at which seat `caller` calls a first in shove
    from seat `pusher`, everyone between them having folded. Raises ValueError unless the
    caller acts after the pusher."""
    if not 0 <= pusher < caller < num_players:
        raise ValueError(f"seat {caller} does not act after seat {pusher} with {num_players} players")
    return tuple(_chart(num_players, bb_ante)["call"][f"{pusher}-{caller}"])
