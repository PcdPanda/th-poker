"""Generate `thpoker/data/preflop_equity.json.gz`: each preflop class's equity against every other.

Run `bin/generate_data.py preflop_equity`.

Each random board is evaluated for all 1326 combos at once, and every pair of card-disjoint
combos that avoids the board adds a win, tie, or loss to its class pair, so one board refines
all 169 x 169 entries together. The standard error comes from the spread between batches.
"""

import numpy as np
from   thpoker.game.cards       import COMBOS, PREFLOP_CLASSES
from   thpoker.game.evaluator   import evaluate_combos
from   thpoker.game.rng         import Rng
from   thpoker.odds             import combo_matrices

_BATCHES = 10


def generate(boards: int, seed: int) -> dict[str, object]:
    """Class-versus-class equity from `boards` random boards. Raises `ValueError` if `boards`
    is not a positive multiple of the batch count."""
    if boards <= 0 or boards % _BATCHES:
        raise ValueError(f"boards must be a positive multiple of {_BATCHES}, got {boards}")
    disjoint, membership = combo_matrices()
    rng = Rng(seed)
    batch_equities = []
    total_won = np.zeros((len(PREFLOP_CLASSES),) * 2)
    total_pairs = np.zeros_like(total_won)
    for _ in range(_BATCHES):
        won = np.zeros((len(COMBOS),) * 2, np.float32)
        pairs = np.zeros_like(won)
        for _ in range(boards // _BATCHES):
            board = rng.permutation(52)[:5]
            usable = np.array([a not in board and b not in board for a, b in COMBOS])
            indices = np.flatnonzero(usable)
            values = np.full(len(COMBOS), -1, np.int64)
            values[indices] = evaluate_combos(board, [COMBOS[i] for i in indices])
            valid = disjoint & usable[:, None] & usable[None, :]
            ahead = values[:, None] > values[None, :]
            tied = values[:, None] == values[None, :]
            won += (ahead + 0.5 * tied) * valid
            pairs += valid
        class_won = membership @ won @ membership.T
        class_pairs = membership @ pairs @ membership.T
        batch_equities.append(class_won / class_pairs)
        total_won += class_won
        total_pairs += class_pairs
    stderr = np.std(batch_equities, axis=0, ddof=1) / np.sqrt(_BATCHES)
    return {
        "boards": boards,
        "seed": seed,
        "stderr": round(float(stderr.max()), 4),
        "classes": list(PREFLOP_CLASSES),
        "equity": np.round(total_won / total_pairs, 4).tolist(),
    }
