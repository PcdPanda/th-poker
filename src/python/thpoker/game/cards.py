"""Card, hole-card combo, and preflop class encodings.

A card is an int 0..51 with rank = card // 4 (0 = deuce ... 12 = ace) and suit = card % 4.
A combo is an unordered pair of distinct cards, indexed 0..1325 in the order of `COMBOS`.
"""

import numpy as np

RANKS = "23456789TJQKA"
SUITS = "cdhs"


class CardParseError(ValueError):
    """Raised for text that is not a valid card or card list."""


def rank_of(card: int) -> int:
    return card >> 2


def suit_of(card: int) -> int:
    return card & 3


def parse_card(text: str) -> int:
    """Parse two characters such as "As" or "td". Raises `CardParseError`."""
    if len(text) != 2 or text[0].upper() not in RANKS or text[1].lower() not in SUITS:
        raise CardParseError(f"not a card: {text!r}")
    return RANKS.index(text[0].upper()) * 4 + SUITS.index(text[1].lower())


def parse_cards(text: str) -> list[int]:
    """Parse "AsKd", "As Kd", or "As,Kd". Raises `CardParseError`, including for duplicates."""
    compact = text.replace(" ", "").replace(",", "")
    if len(compact) % 2:
        raise CardParseError(f"not a card list: {text!r}")
    cards = [parse_card(compact[i : i + 2]) for i in range(0, len(compact), 2)]
    if len(set(cards)) != len(cards):
        raise CardParseError(f"duplicate card in {text!r}")
    return cards


def card_str(card: int) -> str:
    return RANKS[card >> 2] + SUITS[card & 3]


def cards_str(cards: tuple[int, ...] | list[int]) -> str:
    return " ".join(card_str(c) for c in cards)


COMBOS: tuple[tuple[int, int], ...] = tuple((a, b) for a in range(52) for b in range(a + 1, 52))
COMBO_CARDS = np.array(COMBOS, dtype=np.int64)  # the two cards of each combo
_COMBO_INDEX = {pair: i for i, pair in enumerate(COMBOS)}


def combo_index(a: int, b: int) -> int:
    """Index of the combo {a, b} in `COMBOS`, independent of argument order."""
    return _COMBO_INDEX[(a, b) if a < b else (b, a)]


def preflop_class(a: int, b: int) -> str:
    """Preflop class name such as "AA", "AKs", or "T9o"."""
    high, low = (a, b) if rank_of(a) >= rank_of(b) else (b, a)
    name = RANKS[rank_of(high)] + RANKS[rank_of(low)]
    if rank_of(high) == rank_of(low):
        return name
    return name + ("s" if suit_of(high) == suit_of(low) else "o")


def _grid_order() -> tuple[str, ...]:
    # 13x13 grid, aces first: diagonal pairs, suited above the diagonal, offsuit below.
    names = []
    for row in range(12, -1, -1):
        for col in range(12, -1, -1):
            if row == col:
                names.append(RANKS[row] * 2)
            elif col < row:
                names.append(RANKS[row] + RANKS[col] + "s")
            else:
                names.append(RANKS[col] + RANKS[row] + "o")
    return tuple(names)


def _combos_by_class() -> dict[str, tuple[int, ...]]:
    grouped: dict[str, list[int]] = {name: [] for name in PREFLOP_CLASSES}
    for index, (a, b) in enumerate(COMBOS):
        grouped[preflop_class(a, b)].append(index)
    return {name: tuple(indices) for name, indices in grouped.items()}


PREFLOP_CLASSES = _grid_order()
COMBOS_OF_CLASS = _combos_by_class()
# Index into PREFLOP_CLASSES of each combo, by combo index.
COMBO_CLASS = tuple(PREFLOP_CLASSES.index(preflop_class(a, b)) for a, b in COMBOS)
