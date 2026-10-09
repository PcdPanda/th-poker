"""Calibrate the decision rating (DESIGN.md Section 7.7): review each bot tier's own play the way
the user's is reviewed and measure its reference loss per 100 hands.

Run `bin/rating_calibration.py [hands] [workers]`. Each tier takes the user's seat at six-handed
100bb cash tables of Tier 2 bots drawn from the default panel, and its decisions are reviewed
with the triage pass (a full pass where the triage asks for one).
"""

from   concurrent.futures       import ProcessPoolExecutor

from   thpoker.analysis.ev      import PROFILES
from   thpoker.analysis.review  import REFERENCE, triaged_review
from   thpoker.bots.bot         import PRESETS
from   thpoker.game.rng         import Rng
from   thpoker.game.session     import Mode, SessionConfig
from   thpoker.game.state       import GameConfig
from   thpoker.table            import BOT_TIERS, TableConfig, TableRunner

SEATS = 6
HANDS_PER_TABLE = 25
TIERS = (1, 2, 3)  # Expert plays like Hard without a user to read


def _table(task: tuple[int, int]) -> tuple[int, list[float]]:
    """(tier, reference loss in big blinds of each hand) for one table of `HANDS_PER_TABLE`
    dealt hands, counting only hands where the player acted, as the user's records do."""
    tier, seed = task
    session = SessionConfig(Mode.CASH, seed, SEATS, 10_000, 0, GameConfig(SEATS), reset_stacks_each_hand=True)
    runner = TableRunner(TableConfig(session, (None,) * SEATS, tier=2))
    player = BOT_TIERS[tier]("player", PRESETS["balanced"])
    rng = Rng(seed).derive("player")
    losses = []
    for number in range(HANDS_PER_TABLE):
        runner.start_hand()
        while runner.user_to_act():
            view = runner.user_view()
            runner.act(player.decide(view, rng.derive(number, len(view.history))).action)
        hand = runner.hand
        assert hand is not None
        if not any(e.seat == 0 for e in hand.history):
            continue
        review = triaged_review(hand, 0, runner.bots, REFERENCE, None, PROFILES["pc"])
        losses.append(sum(d.loss(d.reference)[0] for d in review.decisions))
    return tier, losses


def calibrate(hands: int, workers: int) -> dict[int, tuple[float, float, int]]:
    """(reference loss per 100 hands, its standard error, hands counted) for each tier."""
    tables = max(1, hands // HANDS_PER_TABLE)
    tasks = [(tier, 1000 * tier + table) for tier in TIERS for table in range(tables)]
    losses: dict[int, list[float]] = {tier: [] for tier in TIERS}
    with ProcessPoolExecutor(workers) as pool:
        for tier, table_losses in pool.map(_table, tasks):
            losses[tier].extend(table_losses)
    result = {}
    for tier, values in losses.items():
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / max(1, len(values) - 1)
        result[tier] = (100 * mean, 100 * (variance / len(values)) ** 0.5, len(values))
    return result
