"""Best-five-card hand values for 5 to 7 cards.

A hand value is an int where higher is stronger: category << 20 plus up to five tie-break ranks
in 4-bit slots. Equal values are exact ties.

Each card has a precomputed key: 5**rank in the low 32 bits (a base-5 count per rank, unique
because no rank appears more than 4 times) and a 4-bit counter for its suit above bit 32. A hand's
key is the sum of its card keys. A suit counter reaching 5 means a flush, which with at most 7
cards rules out quads and full houses, so the flush table alone decides the value.
"""

from   thpoker.game.cards       import rank_of, suit_of

HIGH_CARD, PAIR, TWO_PAIR, TRIPS, STRAIGHT, FLUSH, FULL_HOUSE, QUADS, STRAIGHT_FLUSH = range(9)
CATEGORY_NAMES = (
    "high card",
    "pair",
    "two pair",
    "three of a kind",
    "straight",
    "flush",
    "full house",
    "four of a kind",
    "straight flush",
)

_RANK_BITS = (1 << 32) - 1
_CARD_KEYS = tuple(5 ** rank_of(c) + (1 << (32 + 4 * suit_of(c))) for c in range(52))
# Adding 3 to each 4-bit suit counter sets its top bit exactly when the count is >= 5.
_FLUSH_ADD = 0x3333
_FLUSH_TEST = 0x8888


def _value(category: int, ranks: list[int]) -> int:
    value = category
    for slot in range(5):
        value = (value << 4) | (ranks[slot] if slot < len(ranks) else 0)
    return value


def _straight_high(rank_mask: int) -> int:
    """Top rank of the best straight in a 13-bit rank mask, or -1. The wheel's top is the five."""
    extended = (rank_mask << 1) | ((rank_mask >> 12) & 1)  # bit 0 is the ace played low
    for top in range(13, 3, -1):
        if (extended >> (top - 4)) & 31 == 31:
            return top - 1
    return -1


def _flush_value(rank_mask: int) -> int:
    high = _straight_high(rank_mask)
    if high >= 0:
        return _value(STRAIGHT_FLUSH, [high])
    ranks = [r for r in range(12, -1, -1) if rank_mask >> r & 1]
    return _value(FLUSH, ranks[:5])


def _counts_value(counts: list[int]) -> int:
    """Value of a non-flush hand from its per-rank counts."""
    groups = sorted(((c, r) for r, c in enumerate(counts) if c), reverse=True)
    top_count = groups[0][0]
    second_count = groups[1][0] if len(groups) > 1 else 0
    ranks_by_group = [r for _, r in groups]
    if top_count == 4:
        quad = ranks_by_group[0]
        return _value(QUADS, [quad, max(r for r in ranks_by_group if r != quad)])
    if top_count == 3 and second_count >= 2:
        return _value(FULL_HOUSE, ranks_by_group[:2])
    rank_mask = sum(1 << r for r in ranks_by_group)
    high = _straight_high(rank_mask)
    if high >= 0:
        return _value(STRAIGHT, [high])
    if top_count == 3:
        return _value(TRIPS, ranks_by_group[:3])
    if top_count == 2 and second_count == 2:
        return _value(TWO_PAIR, ranks_by_group[:2] + [max(ranks_by_group[2:])])
    if top_count == 2:
        return _value(PAIR, ranks_by_group[:4])
    return _value(HIGH_CARD, ranks_by_group[:5])


def _rank_part_value(rank_part: int) -> int:
    counts = []
    for _ in range(13):
        rank_part, count = divmod(rank_part, 5)
        counts.append(count)
    return _counts_value(counts)


_FLUSH_VALUES = tuple(_flush_value(m) if m.bit_count() >= 5 else 0 for m in range(1 << 13))
# Filled on demand; at most ~73,000 rank multisets of 5-7 cards exist.
_NON_FLUSH_VALUES: dict[int, int] = {}


def _value_from_key(key: int, cards: tuple[int, ...] | list[int]) -> int:
    suit_counts = key >> 32
    if (suit_counts + _FLUSH_ADD) & _FLUSH_TEST:
        for suit in range(4):
            if (suit_counts >> (4 * suit)) & 15 >= 5:
                break
        rank_mask = 0
        for card in cards:
            if card & 3 == suit:
                rank_mask |= 1 << (card >> 2)
        return _FLUSH_VALUES[rank_mask]
    rank_part = key & _RANK_BITS
    value = _NON_FLUSH_VALUES.get(rank_part)
    if value is None:
        value = _NON_FLUSH_VALUES[rank_part] = _rank_part_value(rank_part)
    return value


def evaluate(cards: tuple[int, ...] | list[int]) -> int:
    """Value of the best five-card hand among 5 to 7 distinct cards."""
    if not 5 <= len(cards) <= 7:
        raise ValueError(f"evaluate needs 5 to 7 cards, got {len(cards)}")
    key = 0
    for card in cards:
        key += _CARD_KEYS[card]
    return _value_from_key(key, cards)


def evaluate_combos(board: tuple[int, ...] | list[int], combos: list[tuple[int, int]]) -> list[int]:
    """Values of board + each two-card combo. The board key is computed once for the batch.

    The board must have 3 to 5 cards; combos that share a card with the board are the caller's
    responsibility to exclude.
    """
    if not 3 <= len(board) <= 5:
        raise ValueError(f"board must have 3 to 5 cards, got {len(board)}")
    board_key = 0
    for card in board:
        board_key += _CARD_KEYS[card]
    board_cards = list(board)
    keys = _CARD_KEYS
    return [_value_from_key(board_key + keys[a] + keys[b], board_cards + [a, b]) for a, b in combos]


def category(value: int) -> int:
    return value >> 20


_RANK_NAMES = tuple("two three four five six seven eight nine ten jack queen king ace".split())


def _plural(rank: int) -> str:
    return _RANK_NAMES[rank] + ("es" if rank == 4 else "s")


def describe(value: int) -> str:
    """Short text such as "pair of aces" or "flush, king high"."""
    cat = value >> 20
    first = (value >> 16) & 15
    second = (value >> 12) & 15
    if cat == PAIR:
        return f"pair of {_plural(first)}"
    if cat == TWO_PAIR:
        return f"two pair, {_plural(first)} and {_plural(second)}"
    if cat == TRIPS:
        return f"three {_plural(first)}"
    if cat == FULL_HOUSE:
        return f"full house, {_plural(first)} full of {_plural(second)}"
    if cat == QUADS:
        return f"four {_plural(first)}"
    return f"{CATEGORY_NAMES[cat]}, {_RANK_NAMES[first]} high"
