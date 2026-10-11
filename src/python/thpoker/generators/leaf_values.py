"""Generate `thpoker/data/leaf_values.json.gz`: what a hand wins from a closed street to the end
of the hand, for the review's end-of-street values (`analysis.ev.future_share`).

Run `bin/generate_data.py leaf_values`.

Balanced Hard plays six-handed hands against itself, every seat as deep as the others, at a depth
drawn evenly on a log scale from 20 to 250 big blinds. At each flop or turn that closes with two
or more players in and nobody all in, one player still in is picked, and its
equity against the others' ranges (tracked exactly from their play), the stack-to-pot ratio, the
streets left, its position and role, and whether it holds a draw are recorded with what it went
on to win as a share of the pot. Per streets left and heads-up or multiway, weighted least
squares (weights 1/(1+SPR)^2, since the outcome's spread grows with the stacks) fits
`won = equity + log(1+SPR) * terms @ coef`, with sandwich standard errors; the terms' coefficients
are set at a few stack-to-pot ratios and move linearly in log(1 + SPR) between them. The table must be
rebuilt whenever Hard's postflop play changes.
"""

from   concurrent.futures       import ProcessPoolExecutor
import math
import numpy as np
from   thpoker.analysis.ev      import (DRAW_GAIN, STREETS_LEFT, last_raiser,
                                        leaf_group, leaf_key, leaf_terms,
                                        showdown_share)
from   thpoker.analysis.tracking \
                                import track
from   thpoker.bots.abstraction import acts_last, stack_to_pot
from   thpoker.bots.bot         import Bot, PRESETS
from   thpoker.bots.range_bot   import RangeBot
from   thpoker.game.engine      import (apply_action, is_terminal, new_hand,
                                        observation)
from   thpoker.game.rng         import Rng
from   thpoker.game.state       import GameConfig
from   thpoker.odds             import hand_equity
from   typing                   import Any

SEATS = 6
DEPTHS_BB = (20, 250)
MIN_SAMPLES = 1000  # a group with fewer has no fit, and its leaves keep the realization model
# A position and role within a fit needs this many samples to be used, between its equities'
# 1st and 99th percentiles.
MIN_GROUP = 200
_CHUNK = 100


def sample(
    task: tuple[int, int, int],
) -> list[tuple[str, list[float], float, float, float, str, float]]:
    """(fit key, terms, log(1 + SPR), won over the pot minus equity, SPR, position and role,
    equity) for each sample in the
    hands numbered `start` to `start + count` of `seed`'s run."""
    seed, start, count = task
    hard = RangeBot("hard", PRESETS["balanced"])
    bots: dict[int, Bot] = dict.fromkeys(range(SEATS), hard)
    config = GameConfig(SEATS)
    rows = []
    for number in range(start, start + count):
        rng = Rng(seed).derive(number)
        low, high = DEPTHS_BB
        depth = round(low * (high / low) ** rng.random())
        stacks = (depth * config.big_blind,) * SEATS
        state, _ = new_hand(config, rng.randbelow(1 << 30), number % SEATS, stacks, hand_id=f"leaf{number}")
        while not is_terminal(state):
            seat = state.to_act
            assert seat is not None
            decision = hard.decide(observation(state, seat), rng.derive(len(state.history)))
            state, _ = apply_action(state, decision.action)
        snapshots = track(state, None, bots, None)
        for before, after in zip(snapshots, snapshots[1:]):
            closed = before.state.street
            if after.state.street == closed or closed not in STREETS_LEFT:
                continue
            spot = after.state
            live = [s for s in range(SEATS) if spot.dealt_in[s] and not spot.folded[s]]
            if len(live) < 2 or any(spot.all_in[s] for s in live):
                continue
            seat = live[rng.derive("seat", closed).randbelow(len(live))]
            hole = state.hole_cards[seat]
            assert hole is not None
            board = before.state.board
            ranges = [after.ranges[s] for s in live if s != seat]
            equity = hand_equity(hole, board, ranges, rng.derive("equity", closed)).value
            spr = stack_to_pot(spot, seat)
            raiser = last_raiser(spot, closed)
            role = "none" if raiser is None else "bettor" if raiser == seat else "caller"
            draw = equity - showdown_share(hole, board, ranges) > DRAW_GAIN
            won = (state.stacks[seat] - spot.stacks[seat]) / spot.pot
            in_position = acts_last(spot, seat)
            terms, _ = leaf_terms(equity, spr, in_position, role, draw)
            key = leaf_key(STREETS_LEFT[closed], len(live) > 2)
            group = leaf_group(in_position, role)
            rows.append((key, terms.tolist(), math.log1p(spr), won - equity, spr, group, equity))
    return rows


def fit(
    rows: list[tuple[str, list[float], float, float, float, str, float]],
) -> dict[str, dict[str, Any]]:
    """Per fit key with at least `MIN_SAMPLES` samples: coefficients, their sandwich
    covariance, the sample count, the 99th percentile of SPR seen and the equities seen per
    position and role (leaves outside them fall back)."""
    fits = {}
    for key in sorted({row[0] for row in rows}):
        group = [row for row in rows if row[0] == key]
        if len(group) < MIN_SAMPLES:
            continue
        x = np.array([row[2] * np.array(row[1]) for row in group])
        y = np.array([row[3] for row in group])
        spr = np.array([row[4] for row in group])
        w = 1 / (1 + spr) ** 2
        bread = np.linalg.pinv((x.T * w) @ x)
        coef = bread @ ((x.T * w) @ y)
        residual = y - x @ coef
        cov = bread @ ((x.T * (w * residual) ** 2) @ x) @ bread
        equities = {}
        for name in sorted({row[5] for row in group}):
            seen = [row[6] for row in group if row[5] == name]
            if len(seen) >= MIN_GROUP:
                equities[name] = [float(f"{q:.4g}") for q in np.percentile(seen, [1, 99])]
        fits[key] = {
            "equities": equities,
            "coef": [float(f"{c:.6g}") for c in coef],
            "cov": [[float(f"{c:.6g}") for c in line] for line in cov],
            "samples": len(group),
            "max_spr": float(f"{np.percentile(spr, 99):.4g}"),
        }
    return fits


def generate(hands: int, workers: int, seed: int = 1) -> dict[str, Any]:
    tasks = [(seed, start, min(_CHUNK, hands - start)) for start in range(0, hands, _CHUNK)]
    rows = []
    with ProcessPoolExecutor(workers) as pool:
        for chunk in pool.map(sample, tasks):
            rows.extend(chunk)
    return {"hands": hands, "seed": seed, "samples": len(rows), "fits": fit(rows)}
