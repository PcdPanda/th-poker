"""Decision review (DESIGN.md Section 7): the five-step report for every decision the user made in
a hand (situation, ranges, equity, thresholds, options), all-in adjusted results, and the session
summary of the largest mistakes.

A decision counts as a mistake only when its EV loss is above the threshold and above twice its
standard error, and the strong player rarely makes it; otherwise it is "close". Mistakes are ranked by the loss against the reference
opponent (a balanced Tier 3 bot reading the same actions), so the lesson carries over to other
opponents; the loss against the actual bot is shown next to it, and a large gap between the two
marks an exploit spot. Tournament decisions are ranked by ICM loss instead of chips.
"""

from   collections.abc          import Mapping, Sequence
from   dataclasses              import dataclass, replace
from   functools                import lru_cache
import itertools
import math
import numpy as np
from   thpoker.analysis.ev      import (MIN_BRANCH, OptionValue, PROFILES,
                                        Profile, icm_value_of, option_values)
from   thpoker.analysis.solver  import solve_river
from   thpoker.analysis.tracking \
                                import Snapshot, track
from   thpoker.bots.abstraction import (acts_last, call_price, defense_share,
                                        effective_stack, last_bet,
                                        legal_abstract_actions,
                                        raises_this_street, to_action)
from   thpoker.bots.bot         import Bot, PRESETS
from   thpoker.bots.equity_bot  import public_seed
from   thpoker.bots.range_bot   import RangeBot, preflop_order
from   thpoker.charts           import seat_names
from   thpoker.game.cards       import (COMBOS, COMBO_CLASS, PREFLOP_CLASSES,
                                        combo_index, rank_of, suit_of)
from   thpoker.game.engine      import (build_pots, observation, replay_states,
                                        split_pot)
from   thpoker.game.evaluator   import (HIGH_CARD, PAIR, QUADS, STRAIGHT,
                                        TRIPS, TWO_PAIR, category, evaluate)
from   thpoker.game.rng         import Rng
from   thpoker.game.state       import (Action, ActionType, GameState,
                                        Observation, Street)
from   thpoker.odds             import (Equity, Texture, hand_equity,
                                        preflop_class_equities, range_equities,
                                        settled_equity, texture)

MISTAKE_BB = 0.5
USUAL_SHARE = 0.1
RATING_POT_SHARE = 0.4
RATING_CLOSE = 0.75
RATING_CORRECT_WEIGHT = 0.7
RATING_SCALE_BB = 100.0
SOLVER_BUDGET = PROFILES["pc"].solver_seconds  # for each heads-up river decision in a full review
# The default reference opponent (DESIGN.md Section 6.2).
REFERENCE = RangeBot("reference", PRESETS["balanced"])
# An exploit spot: the best option against the actual bot beats the reference's best by this much.
EXPLOIT_GAP_BB = 1.0
GROUPS = ("two pair+", "one pair", "draw", "air")
ALL_IN_SAMPLES = 3000  # runouts sampled for a preflop all-in; later all-ins are enumerated
# A session is reviewed in two passes (DESIGN.md Section 6.5): every hand with this coarser
# pruning first, then in full only the hands where a decision might be a mistake.
TRIAGE_BRANCH = PROFILES["pc"].triage_branch


@dataclass(frozen=True)
class Situation:
    street: Street
    players: int  # still in the hand
    position: str
    in_position: bool  # acts last after the flop among the players still in
    pot_type: str  # "unopened", "limped", "single-raised", "3-bet", "4-bet+"
    pot_bb: float
    to_call_bb: float
    effective_bb: float  # chips behind that can still go in, against the deepest opponent
    spr: float | None  # effective stack over the pot, after the flop
    texture: Texture | None
    dealt: int  # players dealt in
    role: str  # preflop "aggressor" (made the last raise), "caller" of someone's raise, or "none"
    facing: float | None  # the street's last bet or raise as a share of the pot before it
    stack_bb: float  # effective stack at the start of the hand
    owed_bb: float


@dataclass(frozen=True)
class RangeView:
    """A range for display: per preflop class, how often the class is in the range relative to
    the likeliest hand (13x13 in `PREFLOP_CLASSES` order), its width as a share of the hands
    still possible, and after the flop its make-up by `GROUPS`."""

    classes: tuple[float, ...]
    width: float
    groups: dict[str, float] | None


@dataclass(frozen=True)
class Thresholds:
    """Required equity to call; the share of its range a player must continue with against the
    bet (minimum defense frequency, split across the players facing it); and for each bet or
    raise option, how often everyone must fold for a pure bluff to break even."""

    required_equity: float | None
    defense_share: float | None
    bluff_break_even: dict[Action, float]


@dataclass(frozen=True)
class DecisionReview:
    index: int  # position of the decision in the hand's history
    hole: tuple[int, int]
    board: tuple[int, ...]
    situation: Situation
    ranges: dict[int, RangeView]  # by opponent seat
    reference_ranges: dict[int, RangeView]
    equity: Equity
    reference_equity: Equity
    percentile: float | None  # where the hand sits in the user's own range, 1 = strongest
    thresholds: Thresholds
    chosen: Action | None  # None for a coach hint, asked before acting
    chosen_bet: float | None  # a chosen bet or raise: the chips it adds as a share of the pot
    exploitative: list[OptionValue]
    reference: list[OptionValue]
    tournament: bool
    icm_threshold: float | None  # ICM value of MISTAKE_BB at the user's stack
    reference_mix: dict[Action, float]
    reference_note: str | None = None  # set when the reference is not the Tier 3 bot

    def loss(self, options: list[OptionValue]) -> tuple[float, float]:
        """EV loss of the chosen option against the best one (big blinds, or prize-pool share
        in tournaments) and its standard error."""
        assert self.chosen is not None
        chosen = next(o for o in options if o.action == self.chosen)
        if self.tournament:
            best = max(options, key=_icm_of)
            gap = _icm_of(best) - _icm_of(chosen)
            return gap, math.hypot(best.icm_stderr or 0.0, chosen.icm_stderr or 0.0)
        best = max(options, key=_ev_of)
        return best.ev - chosen.ev, math.hypot(best.stderr, chosen.stderr)

    def verdict(self) -> str:
        """ "mistake", "close" (a loss within noise or under the threshold), or "best"."""
        loss, stderr = self.loss(self.reference)
        threshold = self.icm_threshold if self.tournament else MISTAKE_BB
        assert threshold is not None and self.chosen is not None
        if loss > threshold and loss > 2 * stderr:
            return "close" if self.reference_mix.get(self.chosen, 0.0) >= USUAL_SHARE else "mistake"
        return "close" if loss > 1e-9 else "best"

    def rating(self) -> float:
        """The move from 0 to 1 (DESIGN.md Section 7.7): mostly the share of the pot being played
        for that it gives up beyond the noise, then the big blinds given up on a log scale, held
        inside its verdict's band so the number never contradicts the verdict."""
        verdict = self.verdict()
        if verdict == "best":
            return 1.0
        loss, stderr = self.loss(self.reference)
        if self.tournament:  # prize-pool share back to big blinds at the user's stack
            assert self.icm_threshold
            loss, stderr = (
                loss * MISTAKE_BB / self.icm_threshold,
                stderr * MISTAKE_BB / self.icm_threshold,
            )
        given = max(0.0, loss - 2 * stderr)
        # The pot being played for: less any bet faced, whose uncallable part goes back to the
        # bettor.
        spot = self.situation
        played_for = spot.pot_bb - spot.owed_bb if spot.facing is not None else spot.pot_bb
        correct = 1 - min(1.0, given / (RATING_POT_SHARE * max(played_for, MISTAKE_BB)))
        size = 1 - min(1.0, math.log1p(given / MISTAKE_BB) / math.log1p(RATING_SCALE_BB / MISTAKE_BB))
        raw = RATING_CORRECT_WEIGHT * correct + (1 - RATING_CORRECT_WEIGHT) * size
        if verdict == "close":
            return min(max(raw, RATING_CLOSE), 0.99)
        return min(raw, RATING_CLOSE - 0.01)

    def exploit_spot(self) -> bool:
        """The actual bot is beaten by a different option than the reference, by a clear margin."""
        best_here = max(self.exploitative, key=_ev_of)
        best_reference = max(self.reference, key=_ev_of)
        against_bot = {o.action: o.ev for o in self.exploitative}
        margin = best_here.ev - against_bot.get(best_reference.action, best_here.ev)
        return best_here.action != best_reference.action and margin > EXPLOIT_GAP_BB


def _ev_of(option: OptionValue) -> float:
    return option.ev


def _icm_of(option: OptionValue) -> float:
    assert option.icm is not None
    return option.icm


def best_option(options: list[OptionValue], tournament: bool) -> OptionValue:
    """The option with the most ICM equity in a tournament, the most chips otherwise."""
    return max(options, key=_icm_of if tournament else _ev_of)


def hand_rating(moves: Sequence[tuple[float, float]]) -> float | None:
    """A hand's rating from its moves' (rating, chips at stake): the ratings weighted by the
    stakes, so a big pot counts for more than a preflop fold."""
    total = sum(stake for _, stake in moves)
    return sum(rating * stake for rating, stake in moves) / total if total > 0 else None


def rating_band(rating: float) -> str:
    """The verdict whose band holds `rating` as shown, to two decimals: "best", "close" or
    "mistake"."""
    shown = round(rating, 2)
    return "best" if shown >= 1 else "close" if shown >= RATING_CLOSE else "mistake"


@dataclass(frozen=True)
class MoveRating:
    history_index: int  # the move's place in the hand's history
    rating: float
    stake: float  # big blinds at stake once called, less what the user cannot cover
    hinted: bool  # made after a coach hint
    timed_out: bool = False


@dataclass(frozen=True)
class HandRating:
    """What a session keeps of a reviewed hand: the chance to win at the last move and each
    move's rating."""

    chance: float
    moves: tuple[MoveRating, ...]

    def rating(self) -> float | None:
        return hand_rating([(m.rating, m.stake) for m in self.moves])


def move_ratings(decisions: Sequence[DecisionReview], hinted: set[int], timed_out: set[int]) -> tuple[MoveRating, ...]:
    """The rating of each move made; a coach hint, asked before acting, has none."""
    return tuple(
        MoveRating(
            d.index,
            d.rating(),
            d.situation.pot_bb + 2 * d.situation.to_call_bb - d.situation.owed_bb,
            d.index in hinted,
            d.index in timed_out,
        )
        for d in decisions
        if d.chosen is not None
    )


@dataclass(frozen=True)
class HandReview:
    hand: GameState
    decisions: list[DecisionReview]
    net_bb: float
    all_in_net_bb: float | None  # expected result at the moment all chips went in

    def rating(self) -> float | None:
        return hand_rating([(m.rating, m.stake) for m in move_ratings(self.decisions, set(), set())])

    def chance(self) -> float | None:
        """The chance to win at the user's last decision, as a strong player reads the others."""
        return self.decisions[-1].reference_equity.value if self.decisions else None


@dataclass(frozen=True)
class SessionSummary:
    hands: int
    decisions: int
    close: int
    net_bb: float
    all_in_adjusted_bb: float  # net with every all-in replaced by its expected result
    mistakes: list[tuple[str, DecisionReview]]  # (hand id, review), largest loss first


def position_names(view: Observation) -> dict[int, str]:
    """Seat names by the players dealt in (see `seat_names`)."""
    order = preflop_order(view)
    return dict(zip(order, seat_names(len(order))))


def situation(view: Observation) -> Situation:
    big_blind = view.config.big_blind
    live = [s for s in range(view.config.num_seats) if view.dealt_in[s] and not view.folded[s]]
    raises = sum(
        1 for e in view.history if e.street == Street.PREFLOP and e.action.type in (ActionType.BET, ActionType.RAISE)
    )
    limped = any(e.street == Street.PREFLOP and e.action.type == ActionType.CALL for e in view.history)
    if raises:
        pot_type = ("single-raised", "3-bet")[raises - 1] if raises <= 2 else "4-bet+"
    else:
        pot_type = "limped" if limped or view.street != Street.PREFLOP else "unopened"
    deepest = max(view.stacks[s] + view.committed_this_street[s] for s in live if s != view.seat)
    mine = view.stacks[view.seat] + view.committed_this_street[view.seat]
    effective = min(mine, deepest) - view.committed_this_street[view.seat]
    to_call = min(view.current_bet - view.committed_this_street[view.seat], view.stacks[view.seat])
    postflop = view.street != Street.PREFLOP
    raisers = [
        e.seat
        for e in view.history
        if e.street == Street.PREFLOP and e.action.type in (ActionType.BET, ActionType.RAISE)
    ]
    if raisers and raisers[-1] == view.seat:
        role = "aggressor"
    elif raisers and any(
        e.seat == view.seat and e.street == Street.PREFLOP and e.action.type == ActionType.CALL for e in view.history
    ):
        role = "caller"
    else:
        role = "none"
    bet = last_bet(view) if to_call > 0 else None
    return Situation(
        view.street,
        len(live),
        position_names(view)[view.seat],
        acts_last(view, view.seat),
        pot_type,
        view.pot / big_blind,
        to_call / big_blind,
        effective / big_blind,
        effective / view.pot if postflop else None,
        texture(view.board) if postflop else None,
        sum(view.dealt_in),
        role,
        bet[1] / bet[0] if bet else None,
        effective_stack(view) / big_blind,
        (view.current_bet - view.committed_this_street[view.seat]) / big_blind,
    )


@lru_cache(maxsize=1 << 15)
def hand_group(hole: tuple[int, int], board: tuple[int, ...]) -> str:
    """The make-up group of a hand: a made hand counts only when a hole card improves on the
    board, so a pair on the board is not everyone's pair."""
    value = evaluate(list(hole) + list(board))
    made = category(value)
    if len(board) == 5:
        board_value = evaluate(list(board))
        on_board = category(board_value)
        # A better straight or flush than the board's own is still the hand's.
        improved = made > on_board or (made == on_board >= STRAIGHT and value > board_value)
    else:
        counts = sorted((sum(rank_of(c) == rank_of(d) for d in board) for c in board), reverse=True)
        if counts[0] == 4:
            on_board = QUADS
        elif counts[0] == 3:
            on_board = TRIPS
        elif counts.count(2) >= 4:
            on_board = TWO_PAIR
        else:
            on_board = PAIR if counts[0] == 2 else HIGH_CARD
        improved = made > on_board
    if improved:
        return "two pair+" if made >= TWO_PAIR else "one pair"
    if len(board) < 5:
        cards = list(hole) + list(board)
        flush_draw = any(
            sum(suit_of(c) == suit for c in cards) == 4 and any(suit_of(h) == suit for h in hole) for suit in range(4)
        )
        ranks = {rank_of(c) for c in cards} | ({-1} if 12 in {rank_of(c) for c in cards} else set())
        hole_ranks = {rank_of(h) for h in hole}
        straight_draw = any(
            {low, low + 1, low + 2, low + 3} <= ranks and hole_ranks & {low, low + 1, low + 2, low + 3}
            for low in range(-1, 10)
        )
        if flush_draw or straight_draw:
            return "draw"
    return "air"


def range_view(weights: np.ndarray, board: tuple[int, ...], make_up: bool = True) -> RangeView:
    possible = np.array([not (a in board or b in board) for a, b in COMBOS])
    top = weights.max()
    relative = weights / top if top > 0 else weights
    classes_of = np.asarray(COMBO_CLASS)[possible]
    by_class = np.bincount(classes_of, weights=relative[possible], minlength=len(PREFLOP_CLASSES))
    counts = np.bincount(classes_of, minlength=len(PREFLOP_CLASSES))
    classes = tuple(float(v) for v in np.divide(by_class, counts, out=np.zeros_like(by_class), where=counts > 0))
    groups = None
    if board and make_up:
        groups = dict.fromkeys(GROUPS, 0.0)
        total = weights.sum()
        for index in np.flatnonzero(weights):
            groups[hand_group(COMBOS[index], board)] += weights[index] / total
    return RangeView(classes, float(relative[possible].sum() / possible.sum()), groups)


def thresholds(view: Observation) -> Thresholds:
    """The Section 7.4 numbers for this decision. The required equity counts only chips the
    user can win; the defense share is the minimum defense frequency against the street's last
    bet or raise (None preflop against the blinds alone)."""
    to_call = min(view.current_bet - view.committed_this_street[view.seat], view.stacks[view.seat])
    required = call_price(view) if to_call > 0 else None
    defense = defense_share(view) if to_call > 0 else None
    bluffs = {}
    for abstract in legal_abstract_actions(view):
        action = to_action(abstract, view)
        if action.type in (ActionType.BET, ActionType.RAISE):
            assert action.amount is not None
            risked = action.amount - view.committed_this_street[view.seat]
            bluffs[action] = risked / (view.pot + risked)
    return Thresholds(required, defense, bluffs)


def _percentile(view: Observation, own: np.ndarray, opponents: list[np.ndarray]) -> float | None:
    """Share of the user's own range (as the reference reads it) that is weaker, counting ties
    as half, by equity against the opponents' ranges."""
    hero = combo_index(*view.my_cards)
    combos = sorted({int(i) for i in np.flatnonzero(own)} | {hero})
    if view.board:
        equities = range_equities(view.board, combos, opponents, Rng(public_seed(view)))
    else:
        by_class = preflop_class_equities(opponents)
        equities = {c: by_class[COMBO_CLASS[c]] for c in combos}
    if hero not in equities:
        return None
    mine = equities[hero]
    total = below = 0.0
    for combo, equity in equities.items():
        weight = own[combo]
        total += weight
        below += weight * ((equity < mine) + 0.5 * (equity == mine))
    return below / total if total else None


def all_in_net(hand: GameState, user: int) -> float | None:
    """The user's expected net (chips) at the moment all its chips went in before the river,
    from the shown hands over every runout (sampled for a preflop all-in); None without such an
    all-in, or when others kept betting on a later street. Folded hands count as unknown cards."""
    states = replay_states(hand)
    if user not in hand.shown or len(hand.shown) < 2:
        return None
    went_in = next((i for i, state in enumerate(states) if state.all_in[user]), None)
    if went_in is None:
        return None
    board = states[max(0, went_in - 1)].board  # the board when the user's last chips went in
    before_last = states[-2] if len(states) > 1 else states[-1]
    if len(board) >= 5 or len(before_last.board) != len(board):
        return None
    n = hand.config.num_seats
    live = tuple(s in hand.shown for s in range(n))
    holes = {s: hand.hole_cards[s] for s in hand.shown}
    known = set(board) | {c for h in holes.values() if h for c in h}
    deck = [c for c in range(52) if c not in known]
    missing = 5 - len(board)
    if missing <= 2:
        runouts = list(itertools.combinations(deck, missing))
    else:
        rng = Rng(public_seed(observation(hand, user)))
        runouts = []
        for _ in range(ALL_IN_SAMPLES):
            picked = [deck[i] for i in rng.permutation(len(deck))[:missing]]
            runouts.append(tuple(picked))
    pots = build_pots(hand.committed_total, live, hand.dead_money)
    won = 0.0
    for runout in runouts:
        full = list(board) + list(runout)
        values = {s: evaluate(list(h) + full) for s, h in holes.items() if h}
        for pot in pots:
            best = max(values[s] for s in pot.eligible)
            winners = [s for s in pot.eligible if values[s] == best]
            if user in winners:
                won += split_pot(pot.amount, winners, hand.button, n)[winners.index(user)]
    awarded = sum(share for a in hand.awards for s, share in zip(a.winners, a.shares) if s == user)
    behind = hand.stacks[user] - awarded
    return behind + won / len(runouts) - hand.starting_stacks[user]


def _review_decision(
    index: int,
    snapshot: Snapshot,
    reference_snapshot: Snapshot,
    user: int,
    bots: dict[int, Bot],
    reference: Bot,
    payouts: Sequence[float] | None,
    min_branch: float,
    solver_seconds: float | None,
    chosen: Action | None,
) -> DecisionReview:
    state = snapshot.state
    hole = state.hole_cards[user]
    assert hole is not None
    view = observation(state, user)
    opponents = {s: w for s, w in snapshot.ranges.items() if s != user}
    exploitative = option_values(state, bots, reference, opponents, chosen, payouts, min_branch)
    reference_values = option_values(
        state,
        {s: reference for s in bots},
        reference,
        {s: w for s, w in reference_snapshot.ranges.items() if s != user},
        chosen,
        payouts,
        min_branch,
    )
    note = None
    if payouts is None and solver_seconds is not None:
        solved = _solved_river(state, user, reference_snapshot.ranges, reference_values, solver_seconds)
        if solved is not None:
            reference_values, note = solved
    icm_threshold = None
    if payouts is not None:
        icm_threshold = icm_value_of(MISTAKE_BB * state.config.big_blind, state.starting_stacks, user, payouts)
    read = [w for s, w in reference_snapshot.ranges.items() if s != user]
    mix: dict[Action, float] = {}
    for abstract, share in reference.action_probabilities(view).items():
        action = to_action(abstract, view)
        mix[action] = mix.get(action, 0.0) + share
    return DecisionReview(
        index,
        hole,
        state.board,
        situation(view),
        {s: range_view(w, state.board) for s, w in opponents.items()},
        {s: range_view(w, state.board) for s, w in reference_snapshot.ranges.items() if s != user},
        hand_equity(hole, state.board, list(opponents.values()), Rng(public_seed(view))),
        hand_equity(hole, state.board, read, Rng(public_seed(view))),
        _percentile(view, snapshot.ranges[user], list(opponents.values())),
        thresholds(view),
        chosen,
        _bet_share(chosen, view),
        exploitative,
        reference_values,
        payouts is not None,
        icm_threshold,
        mix,
        note,
    )


def _solved_river(
    state: GameState,
    user: int,
    ranges: dict[int, np.ndarray],
    options: list[OptionValue],
    budget: float,
) -> tuple[list[OptionValue], str] | None:
    """Reference values from the river subgame solver (DESIGN.md Section 5.3) in a heads-up
    pot on the river where both players can still bet; None elsewhere. The solver prices exactly
    the options under review; its exploitability stands in for the error."""
    n = state.config.num_seats
    live = [s for s in range(n) if state.dealt_in[s] and not state.folded[s]]
    if state.street != Street.RIVER or len(live) != 2 or any(state.all_in[s] for s in live):
        return None
    order = sorted(live, key=lambda s: (s - state.button - 1) % n)  # first to act first
    committed = (state.committed_this_street[order[0]], state.committed_this_street[order[1]])
    behind = (state.stacks[order[0]], state.stacks[order[1]])
    pot = state.pot - sum(committed)  # a folded player's river chips stay in as dead money
    # The most either can put in: a bet the other cannot cover plays as a bet of this size.
    cap = min(c + b for c, b in zip(committed, behind))
    bets = raises_this_street(state)
    targets = [min(o.action.amount, cap) for o in options if o.action.amount is not None]
    hole = state.hole_cards[user]
    assert hole is not None
    hero = combo_index(*hole)
    solution = solve_river(
        state.board,
        pot,
        committed,
        behind,
        order.index(user),
        bets,
        (ranges[order[0]], ranges[order[1]]),
        keep=hero,
        budget=budget,
        root_targets=targets,
    )
    values = solution.root_values(hero)
    mine = state.committed_this_street[user]
    big_blind = state.config.big_blind
    solved = []
    for option in options:
        kind = option.action.type
        if kind == ActionType.FOLD:
            target = -1
        elif kind == ActionType.CHECK:
            target = mine
        elif kind == ActionType.CALL:  # all in for less when the bet exceeds the user's stack
            target = min(state.current_bet, mine + state.stacks[user])
        else:
            assert option.action.amount is not None
            target = min(option.action.amount, cap)
        if target not in values:
            return None
        # Zero-sum values count from the street's start less half its pot; the review counts from now.
        ev = (values[target] + pot / 2 + mine) / big_blind
        solved.append(replace(option, ev=ev, stderr=solution.exploitability / big_blind, exact=False))
    share = solution.exploitability / (pot + sum(committed))
    return solved, f"river solver, {share:.1%} of the pot from equilibrium"


def _bet_share(chosen: Action | None, view: Observation) -> float | None:
    if chosen is None or chosen.type not in (ActionType.BET, ActionType.RAISE) or chosen.amount is None:
        return None
    return (chosen.amount - view.committed_this_street[view.seat]) / view.pot


def hint(
    hand: GameState,
    user: int,
    bots: dict[int, Bot],
    reference: Bot,
    payouts: Sequence[float] | None,
    min_branch: float,
    solver_seconds: float | None,
) -> DecisionReview:
    """The five steps for the decision `user` faces now in an unfinished `hand`, before it acts
    (the coach hint of DESIGN.md Section 7.8)."""
    return _review_at(hand, user, bots, reference, len(hand.history), payouts, min_branch, solver_seconds)


def with_choice(review: DecisionReview, state: GameState, chosen: Action) -> DecisionReview | None:
    """A review made before acting at `state`, completed with the action then taken, when that
    action is one it priced (always so for fold, check, call and the abstract sizes): a review
    made after acting would value the same options. None when the size needs pricing of its own."""
    if chosen not in {o.action for o in review.exploitative}:
        return None
    assert state.to_act is not None
    view = observation(state, state.to_act)
    return replace(review, chosen=chosen, chosen_bet=_bet_share(chosen, view))


def review_hand(
    hand: GameState,
    user: int,
    bots: dict[int, Bot],
    reference: Bot,
    payouts: Sequence[float] | None = None,
    min_branch: float = MIN_BRANCH,
    solver_seconds: float | None = SOLVER_BUDGET,
    reviewed: Mapping[int, DecisionReview] | None = None,
) -> HandReview:
    """Review every decision `user` made in the finished `hand` played against `bots`. With a
    tournament's `payouts`, decisions are judged in ICM equity; `min_branch` is passed to
    `option_values`; `solver_seconds` is the time for each heads-up cash river solve (None for
    the walk alone, as in the triage pass). `reviewed` holds decisions already reviewed with
    the same settings, by history index (a decision depends only on the actions before it)."""
    indices = [index for index, entry in enumerate(hand.history) if entry.seat == user]
    found = dict(reviewed or {})
    missing = [index for index in indices if index not in found]
    if missing:
        exploitative_track = track(hand, user, bots, reference)
        reference_track = track(hand, user, bots, reference, reference_view=True)
        for index in missing:
            found[index] = _review_decision(
                index,
                exploitative_track[index],
                reference_track[index],
                user,
                bots,
                reference,
                payouts,
                min_branch,
                solver_seconds,
                hand.history[index].action,
            )
    decisions = [found[index] for index in indices]
    big_blind = hand.config.big_blind
    net = hand.stacks[user] - hand.starting_stacks[user]
    expected = all_in_net(hand, user)
    return HandReview(hand, decisions, net / big_blind, None if expected is None else expected / big_blind)


@dataclass(frozen=True)
class GodView:
    """A decision of a finished hand valued knowing the bots' styles and the cards they held."""

    options: list[OptionValue]
    equity: Equity  # against the cards still in the hand


def god_views(
    hand: GameState,
    user: int,
    bots: dict[int, Bot],
    reference: Bot,
    payouts: Sequence[float] | None,
    min_branch: float,
) -> dict[int, GodView]:
    """Each decision of `user` in the finished `hand`, by history index, with every opponent still
    in holding the cards it was dealt and answering with its own policy. Hindsight: it never
    changes a rating. As in `option_values`, the walk ends with the street and later streets
    come from the realization model, so before the river the values are estimates."""
    hole = hand.hole_cards[user]
    assert hole is not None
    states = replay_states(hand)
    views = {}
    for index, entry in enumerate(hand.history):
        if entry.seat != user:
            continue
        state = states[index]
        held = {}
        for seat, cards in enumerate(hand.hole_cards):
            if seat != user and state.dealt_in[seat] and not state.folded[seat]:
                assert cards is not None
                weights = np.zeros(len(COMBOS))
                weights[combo_index(*cards)] = 1.0
                held[seat] = weights
        options = option_values(state, bots, reference, held, entry.action, payouts, min_branch)
        seed = Rng(public_seed(observation(state, user)))
        views[index] = GodView(options, settled_equity(hole, state.board, list(held.values()), seed))
    return views


def hindsight_best(review: DecisionReview, god: GodView) -> OptionValue | None:
    """The best option knowing the cards, when it beats the move made by more than the noise and
    the mistake threshold, as `DecisionReview.verdict` judges a move."""
    loss, stderr = review.loss(god.options)
    threshold = review.icm_threshold if review.tournament else MISTAKE_BB
    assert threshold is not None
    if loss > threshold and loss > 2 * stderr:
        return best_option(god.options, review.tournament)
    return None


def review_decision(
    hand: GameState,
    user: int,
    bots: dict[int, Bot],
    reference: Bot,
    index: int,
    payouts: Sequence[float] | None = None,
    min_branch: float = MIN_BRANCH,
    solver_seconds: float | None = SOLVER_BUDGET,
) -> DecisionReview:
    """One decision of `review_hand`: the hand's action at `index`, which must be `user`'s.
    Raises `ValueError` otherwise."""
    if not 0 <= index < len(hand.history) or hand.history[index].seat != user:
        raise ValueError(f"action {index} of {hand.hand_id} is not a decision of seat {user}")
    return _review_at(hand, user, bots, reference, index, payouts, min_branch, solver_seconds)


def _review_at(
    hand: GameState,
    user: int,
    bots: dict[int, Bot],
    reference: Bot,
    index: int,
    payouts: Sequence[float] | None,
    min_branch: float,
    solver_seconds: float | None,
) -> DecisionReview:
    """The decision before history action `index`; past the end, the decision the user faces now."""
    chosen = hand.history[index].action if index < len(hand.history) else None
    return _review_decision(
        index,
        track(hand, user, bots, reference)[index],
        track(hand, user, bots, reference, reference_view=True)[index],
        user,
        bots,
        reference,
        payouts,
        min_branch,
        solver_seconds,
        chosen,
    )


def triaged_review(
    hand: GameState,
    user: int,
    bots: dict[int, Bot],
    reference: Bot,
    payouts: Sequence[float] | None,
    profile: Profile,
) -> HandReview:
    """The triage pass, then the full review where a decision might be a mistake."""
    review = review_hand(hand, user, bots, reference, payouts, profile.triage_branch, None)
    if needs_full_review(review):
        review = review_hand(hand, user, bots, reference, payouts, profile.min_branch, profile.solver_seconds)
    return review


def needs_full_review(review: HandReview) -> bool:
    """Whether a triage-pass review has a decision whose loss could be half the mistake
    threshold or more once its error (including what the coarse pruning dropped) is counted."""
    for decision in review.decisions:
        threshold = decision.icm_threshold if decision.tournament else MISTAKE_BB
        assert threshold is not None
        loss, stderr = decision.loss(decision.reference)
        if loss + 2 * stderr >= threshold / 2:
            return True
    return False


def summarize(reviews: list[HandReview], top: int = 5) -> SessionSummary:
    """Session totals and the `top` largest mistakes by loss against the reference."""
    decisions = [(r.hand.hand_id, d) for r in reviews for d in r.decisions]
    mistakes = [(hand_id, d) for hand_id, d in decisions if d.verdict() == "mistake"]
    mistakes.sort(key=_reference_loss, reverse=True)
    return SessionSummary(
        len(reviews),
        len(decisions),
        sum(1 for _, d in decisions if d.verdict() == "close"),
        sum(r.net_bb for r in reviews),
        sum(r.net_bb if r.all_in_net_bb is None else r.all_in_net_bb for r in reviews),
        mistakes[:top],
    )


def _reference_loss(item: tuple[str, DecisionReview]) -> float:
    return item[1].loss(item[1].reference)[0]
