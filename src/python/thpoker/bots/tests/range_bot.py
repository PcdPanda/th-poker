from   dataclasses              import replace
import pytest
from   thpoker.bots             import range_bot
from   thpoker.bots.abstraction import AbstractAction, legal_abstract_actions
from   thpoker.bots.bot         import PRESETS
from   thpoker.bots.equity_bot  import EquityBot, bet_shifted, preflop_range
from   thpoker.bots.range_bot   import (ExpertBot, NEUTRAL, RangeBot,
                                        UserTally, chart_range, seat_range,
                                        stretched)
from   thpoker.charts           import push_chart, strategy
from   thpoker.game.cards       import (COMBOS, COMBOS_OF_CLASS, COMBO_CLASS,
                                        PREFLOP_CLASSES, combo_index)
from   thpoker.game.engine      import (apply_action, is_terminal, new_hand,
                                        observation)
from   thpoker.game.rng         import Rng
from   thpoker.game.state       import (Action, ActionType, GameConfig,
                                        Observation)
from   thpoker.game.tests.decks import (CALL, CHECK, FOLD, heads_up, play,
                                        stacked_deck)
from   thpoker.odds             import ranked_range

TAG = RangeBot("probe", PRESETS["tag"])
# The stacked deck needs cards for every dealt seat; the tests override the seats they study.
SIX_SEATS = {0: "KdQd", 1: "9s9c", 2: "Th6h", 3: "5d4d", 4: "Jc3s", 5: "8h8s"}
BALANCED = PRESETS["balanced"]
# One combo of each class, by class index.
ONE_PER_CLASS = [COMBOS_OF_CLASS[name][0] for name in PREFLOP_CLASSES]
SIZES = [len(COMBOS_OF_CLASS[name]) for name in PREFLOP_CLASSES]


def test_policy_is_a_legal_distribution_that_never_folds_when_checking_is_free():
    rng = Rng(21)
    views: list[Observation] = []
    while len(views) < 150:
        seats = 2 + rng.randbelow(7)
        stacks = tuple(500 + rng.randbelow(15_000) for _ in range(seats))
        state, _ = new_hand(GameConfig(seats), rng.randbelow(1 << 30), 0, stacks)
        while not is_terminal(state):
            view = observation(state, state.to_act)
            views.append(view)
            state, _ = apply_action(state, TAG.decide(view, rng.derive(len(views))).action)
    for view in views:
        distribution, _ = TAG.policy(view)
        assert abs(sum(distribution.values()) - 1.0) < 1e-9
        assert set(distribution) <= set(legal_abstract_actions(view))
        if AbstractAction.CHECK in legal_abstract_actions(view):
            assert AbstractAction.FOLD not in distribution


def test_first_in_follows_the_preflop_table():
    deck = stacked_deck(0, 6, SIX_SEATS | {3: "AsAh", 4: "7c2d"}, "")
    state, _ = new_hand(GameConfig(6), 1, 0, (10_000,) * 6, deck=deck)
    assert TAG.action_probabilities(observation(state, 3)) == {AbstractAction.OPEN: 1.0}
    state = play(state, FOLD)
    assert TAG.action_probabilities(observation(state, 4)) == {AbstractAction.FOLD: 1.0}


@pytest.mark.parametrize("hole, action", [("Jh8c", AbstractAction.ALL_IN), ("7c2d", AbstractAction.FOLD)])
def test_short_stack_first_in_follows_the_push_fold_chart(hole, action):
    # Heads-up at 8bb: J8o shoves up to 13bb on the chart, 72o only up to 1.5bb.
    deck = stacked_deck(0, 2, {0: hole, 1: "AsKs"}, "")
    state, _ = new_hand(GameConfig(2), 1, 0, (800, 800), deck=deck)
    assert TAG.action_probabilities(observation(state, 0)) == {action: 1.0}


@pytest.mark.parametrize("stack, action", [(500, AbstractAction.CALL), (900, AbstractAction.FOLD)])
def test_short_stack_facing_a_shove_follows_the_call_chart(stack, action):
    # The big blind calls a heads-up shove with J8o up to 7bb on the chart.
    deck = stacked_deck(0, 2, {0: "AsKs", 1: "Jh8c"}, "")
    state, _ = new_hand(GameConfig(2), 1, 0, (stack, stack), deck=deck)
    state = play(state, Action(ActionType.RAISE, stack))
    assert TAG.action_probabilities(observation(state, 1)) == {action: 1.0}


def _big_blind_facing_a_pot_bet() -> Observation:
    state = heads_up("QsJs", "8c8d", "Kh7d2c3s9h")
    state = play(state, Action(ActionType.RAISE, 250), CALL, CHECK, Action(ActionType.BET, 500))
    return observation(state, 1)


def test_facing_a_pot_bet_it_defends_at_least_the_minimum_defense_frequency():
    view = _big_blind_facing_a_pot_bet()
    own = seat_range(view, 1)
    holes = [i for i, (a, b) in enumerate(COMBOS) if own[i] > 0 and a not in view.board and b not in view.board]
    decisions = TAG.policies(view, holes)
    kept = sum(own[h] * (1 - decisions[h].get(AbstractAction.FOLD, 0.0)) for h in holes) / sum(own[h] for h in holes)
    _, rationale = TAG.policy(view)
    # A pot-size bet needs 50% defense (1/(1+b)); the style shades it slightly.
    assert rationale["defense_share"] == pytest.approx(0.5, abs=0.05)
    assert kept >= rationale["defense_share"] - 0.03


def test_policy_and_policies_agree_on_the_hand_held():
    view = _big_blind_facing_a_pot_bet()
    hero = combo_index(*view.my_cards)
    assert TAG.policies(view, [hero])[hero] == TAG.action_probabilities(view)


def test_a_limp_leaves_the_charts_and_falls_back_to_the_tier_2_bands():
    deck = stacked_deck(0, 6, SIX_SEATS, "")
    state, _ = new_hand(GameConfig(6), 1, 0, (10_000,) * 6, deck=deck)
    state = play(state, CALL)  # first to act limps
    view = observation(state, 4)
    assert chart_range(view, 3) is None
    assert seat_range(view, 3) == preflop_range(view, 3)


def test_a_hand_without_an_equity_falls_back_the_same_way_when_played_and_tracked(monkeypatch):
    # Card removal can leave an opponent range empty for a hand; both paths then use Tier 2.
    view = _big_blind_facing_a_pot_bet()
    hero = combo_index(*view.my_cards)
    computed = range_bot.range_equities

    def without_hero(*args, **kwargs):
        return {h: e for h, e in computed(*args, **kwargs).items() if h != hero}

    monkeypatch.setattr(range_bot, "range_equities", without_hero)
    tier_2 = EquityBot("probe", PRESETS["tag"]).action_probabilities(view)
    assert TAG.action_probabilities(view) == tier_2
    assert TAG.policies(view, [hero])[hero] == tier_2


def test_an_oversized_deep_open_leaves_the_charts():
    # The charts were solved for a 2.5bb open; a 100bb open-shove goes to the size-aware Tier 2.
    state, _ = new_hand(GameConfig(2), 1, 0, (10_000, 10_000))
    shoved = play(state, Action(ActionType.RAISE, 10_000))
    assert chart_range(observation(shoved, 1), 0) is None
    assert seat_range(observation(shoved, 1), 0) == preflop_range(observation(shoved, 1), 0)
    opened = play(state, Action(ActionType.RAISE, 250))
    assert chart_range(observation(opened, 1), 0) is not None
    # The charts' own all-in 4-bet after a 3-bet stays on them.
    four_bet = play(
        state,
        Action(ActionType.RAISE, 250),
        Action(ActionType.RAISE, 1_000),
        Action(ActionType.RAISE, 10_000),
    )
    assert chart_range(observation(four_bet, 1), 0) is not None


def test_a_short_shove_reads_and_is_answered_from_the_push_fold_charts_at_its_depth():
    # Three-handed, a 12bb button shove against 100bb blinds: the shover's depth governs, so its
    # range is the 12bb push chart and the small blind answers from the 12bb call chart.
    state, _ = new_hand(GameConfig(3), 1, 0, (1_200, 10_000, 10_000))
    view = observation(play(state, Action(ActionType.RAISE, 1_200)), 1)
    pushes = [range_bot._soft_threshold(t, 12.0) for t in push_chart(3, 0, False)]
    assert chart_range(view, 0) == tuple(pushes[c] for c in COMBO_CLASS)
    _, rationale = TAG.policy(view)
    assert rationale["rule_triggered"] == "push/fold chart: facing a shove"
    assert rationale["stack_bb"] == 12.0


def test_a_deep_shove_leaves_the_charts_even_for_a_short_viewer():
    # A 100bb shove into a 12bb small blind and a 100bb big blind is a genuine 100bb shove.
    state, _ = new_hand(GameConfig(3), 1, 0, (10_000, 1_200, 10_000))
    view = observation(play(state, Action(ActionType.RAISE, 10_000)), 1)
    assert chart_range(view, 0) is None
    assert seat_range(view, 0) == preflop_range(view, 0)


def test_a_short_stack_calling_an_open_leaves_the_charts():
    # The push/fold call chart is for calling a shove, not a 2.5bb open.
    state, _ = new_hand(GameConfig(3), 1, 0, (10_000, 1_200, 10_000))
    view = observation(play(state, Action(ActionType.RAISE, 250), CALL), 2)
    assert chart_range(view, 1) is None


def test_an_opener_facing_a_short_three_bet_plays_at_the_three_bettors_depth():
    # The 100bb button opens and the 20bb small blind 3-bets all in (its usual 4x size doubled,
    # so still on the charts): the button answers from the 20bb chart, not the 100bb one.
    state, _ = new_hand(GameConfig(3), 1, 0, (10_000, 2_000, 10_000))
    state = play(state, Action(ActionType.RAISE, 250), Action(ActionType.RAISE, 2_000), FOLD)
    _, rationale = TAG.policy(observation(state, 0))
    assert rationale["stack_bb"] == 20.0


def test_an_open_is_read_at_the_depth_it_was_made_before_a_short_three_bet():
    # Six-handed, 100bb except a 20bb big blind: UTG opens, the big blind 3-bets and UTG calls.
    # The open comes from the 100bb chart and the call from the 20bb one.
    state, _ = new_hand(GameConfig(6), 1, 0, (10_000, 10_000, 2_000, 10_000, 10_000, 10_000))
    state = play(state, Action(ActionType.RAISE, 250), FOLD, FOLD, FOLD, FOLD)
    state = play(state, Action(ActionType.RAISE, 1_000), CALL)
    opens = strategy("open", 6, "0", 100.0, False)
    calls = strategy("versus_3bet", 6, "0-5", 20.0, False)
    expected = tuple(opens[c][1] * calls[c][1] for c in COMBO_CLASS)
    assert chart_range(observation(state, 2), 3) == pytest.approx(expected)


def _share(weights) -> float:
    return sum(w * n for w, n in zip(weights, SIZES)) / 1326


def test_with_a_neutral_read_expert_plays_and_reads_exactly_like_hard():
    rng = Rng(5)
    expert = ExpertBot("probe", PRESETS["tag"], {}, 0)
    compared = 0
    while compared < 40:
        seats = 2 + rng.randbelow(5)
        state, _ = new_hand(GameConfig(seats), rng.randbelow(1 << 30), 0, (10_000,) * seats)
        while not is_terminal(state):
            view = observation(state, state.to_act)
            board = set(view.board)
            holes = [i for i, c in enumerate(COMBOS) if not board.intersection(c)]
            assert expert.decisions(view, holes) == TAG.decisions(view, holes)
            compared += 1
            state, _ = apply_action(state, TAG.decide(view, rng.derive(compared)).action)


def test_a_stretch_fills_the_strongest_classes_first_and_empties_the_weakest_first():
    aces, kings = PREFLOP_CLASSES.index("AA"), PREFLOP_CLASSES.index("KK")
    only_aces = [0.0] * 169
    only_aces[aces] = 1.0
    both = list(only_aces)
    both[kings] = 1.0
    assert stretched(only_aces, 2.0) == both  # 6 combos more: all the kings
    assert stretched(only_aces, 1.5)[kings] == 0.5
    assert stretched(both, 0.5) == only_aces
    # Facing an open, continuing 0.5 times and re-raising 1.5 times as often: the continuing
    # kings go first, and the re-raise is held within what still continues.
    rows = [(1.0, 0.0, 0.0)] * 169
    rows[aces], rows[kings] = (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)
    stretched_rows = range_bot._stretched_rows("respond", tuple(rows), (0.5, 1.5))
    assert stretched_rows[aces] == (0.0, 0.5, 0.5) and stretched_rows[kings] == (1.0, 0.0, 0.0)
    assert all(min(row) >= 0 for row in stretched_rows)


def _first_in_under_the_gun():
    state, _ = new_hand(GameConfig(6), 1, 0, (10_000,) * 6, deck=stacked_deck(0, 6, SIX_SEATS, ""))
    return state


def test_expert_steals_more_against_a_big_blind_who_defends_too_little():
    # Six-handed with the button in seat 0, the user in the big blind (seat 2) defends half as
    # often as a solid player: under the gun opens 1/sqrt(0.5) times its chart share.
    view = observation(_first_in_under_the_gun(), 3)
    reads = {view.hand_id: replace(NEUTRAL, defends=0.5)}
    expert = ExpertBot("probe", PRESETS["tag"], reads, 2)
    opens = expert.policies(view, ONE_PER_CLASS)
    share = _share([opens[h].get(AbstractAction.OPEN, 0.0) for h in ONE_PER_CLASS])
    chart = _share([row[1] for row in strategy("open", 6, "0", 100.0, False)])
    assert share == pytest.approx(chart * 0.5**-0.5)
    # Another Expert seat reads the wider open as it was played.
    opened = observation(play(_first_in_under_the_gun(), Action(ActionType.RAISE, 250)), 4)
    other = ExpertBot("other", BALANCED, reads, 2)
    read = chart_range(opened, 3, other.chart_factors)
    assert read is not None
    played = [opens[h].get(AbstractAction.OPEN, 0.0) for h in ONE_PER_CLASS]
    assert [read[h] for h in ONE_PER_CLASS] == pytest.approx(played)


def test_expert_reads_and_answers_a_user_who_opens_or_limps_more_than_expected():
    # Under the gun, the user (seat 3) opens twice as often as the chart: its open reads as twice
    # the chart's share, and the next seat continues and re-raises sqrt(2) times as often.
    opened = observation(play(_first_in_under_the_gun(), Action(ActionType.RAISE, 250)), 4)
    expert = ExpertBot("probe", PRESETS["tag"], {opened.hand_id: replace(NEUTRAL, opens=2.0)}, 3)
    read = expert.seat_range(opened, 3)
    chart = _share([row[1] for row in strategy("open", 6, "0", 100.0, False)])
    assert _share([read[h] for h in ONE_PER_CLASS]) == pytest.approx(2 * chart)
    answers = expert.policies(opened, ONE_PER_CLASS)
    going = _share([1 - answers[h].get(AbstractAction.FOLD, 0.0) for h in ONE_PER_CLASS])
    rows = strategy("respond", 6, "0-1", 100.0, False)
    assert going == pytest.approx(_share([r[1] + r[2] for r in rows]) * 2**0.5)
    # A user who limps 29% of first-in spots is read with a limp band that wide.
    limped = observation(play(_first_in_under_the_gun(), CALL), 4)
    reads = {limped.hand_id: replace(NEUTRAL, limp_width=0.29)}
    limps = ExpertBot("probe", PRESETS["tag"], reads, 3).seat_range(limped, 3)
    assert limps == ranked_range(0.1, 0.1 + 0.29)


def test_expert_calls_down_a_frequent_bettor_more():
    # The big blind faces the user's pot-size flop bet; the user bets twice as often as expected.
    view = _big_blind_facing_a_pot_bet()
    hard = TAG.policy(view)[1]["defense_share"]
    expert = ExpertBot("probe", PRESETS["tag"], {view.hand_id: replace(NEUTRAL, bets_flop=2.0)}, 0)
    assert expert.policy(view)[1]["defense_share"] == pytest.approx(hard + 0.15, abs=2e-4)
    # Its bet says less: the range shifts with a floor of 1 - 0.75 / 2 on the flop.
    charted = chart_range(view, 0)
    assert charted is not None
    assert expert.seat_range(view, 0) == bet_shifted(view, 0, charted, (0.625, 0.25))
    assert expert.seat_range(view, 0) != seat_range(view, 0)


def test_expert_bluffs_less_and_bets_thinner_against_a_user_who_rarely_folds():
    # Checked to on the river in position by a user in the big blind who folds half as often.
    state = heads_up("QsJs", "8c8d", "Kh7d2c3s9h")
    state = play(state, Action(ActionType.RAISE, 250), CALL, CHECK, CHECK, CHECK, CHECK, CHECK)
    view = observation(state, 0)
    hard = TAG.policy(view)[1]
    reads = {view.hand_id: replace(NEUTRAL, folds_late=0.5)}
    expert = ExpertBot("probe", PRESETS["tag"], reads, 1).policy(view)[1]
    assert expert["bluff_share"] == pytest.approx(hard["bluff_share"] * 0.5, abs=2e-4)
    assert expert["value_share"] > hard["value_share"]


def test_from_the_users_own_seat_expert_only_reads_the_user_differently():
    # The pass that finds how often a solid player would bet holds the user's range as Expert
    # reads it, and plays it as Hard would: no adjustment aimed at the user fires.
    state = play(heads_up("QsJs", "8c8d", "Kh7d2c3s9h"), Action(ActionType.RAISE, 250), CALL)
    view = observation(state, 1)
    read = replace(NEUTRAL, defends=0.6, three_bets=1.5, bets_flop=2.0, folds_flop=0.5)
    solid = ExpertBot("solid", BALANCED, {view.hand_id: read}, 1)
    assert solid.seat_range(view, 1) != seat_range(view, 1)

    class Reading(RangeBot):
        def seat_range(self, view, seat):
            return solid.seat_range(view, seat)

    holes = [i for i, w in enumerate(solid.seat_range(view, 1)) if w > 0]
    assert solid.decisions(view, holes) == Reading("solid", BALANCED).decisions(view, holes)


def test_the_tally_counts_each_spot_against_what_a_solid_player_would_do():
    tally = UserTally()
    # Under the gun 100bb deep: an open, then (another hand) a limp.
    opened = play(_first_in_under_the_gun(), Action(ActionType.RAISE, 250), *[FOLD] * 5)
    tally.count(opened, 3, NEUTRAL)
    opens = _share([row[1] for row in strategy("open", 6, "0", 100.0, False)])
    assert (tally.observed["opens"], tally.expected["opens"]) == (1, pytest.approx(opens))
    limped = play(_first_in_under_the_gun(), CALL, *[FOLD] * 4, CHECK)
    tally.count(limped, 3, NEUTRAL)
    assert (tally.spots, tally.limps, tally.expected["opens"]) == (2, 1, pytest.approx(opens))
    # Behind someone else's limp the next seat is not first in: nothing is counted.
    tally.count(limped, 4, NEUTRAL)
    assert (tally.spots, tally.expected["opens"]) == (2, pytest.approx(opens))
    # The next seat calls an open: it defended, and did not re-raise.
    called = play(_first_in_under_the_gun(), Action(ActionType.RAISE, 250), CALL, *[FOLD] * 4)
    tally.count(called, 4, NEUTRAL)
    rows = strategy("respond", 6, "0-1", 100.0, False)
    assert tally.expected["defends"] == pytest.approx(_share([r[1] + r[2] for r in rows]))
    assert tally.expected["three_bets"] == pytest.approx(_share([r[2] for r in rows]))
    assert (tally.observed["defends"], tally.observed["three_bets"]) == (1, 0)
    # At 10bb the push/fold charts govern, and nothing is counted.
    short, _ = new_hand(GameConfig(6), 1, 0, (1_000,) * 6, deck=stacked_deck(0, 6, SIX_SEATS, ""))
    tally.count(play(short, Action(ActionType.RAISE, 1_000), *[FOLD] * 5), 3, NEUTRAL)
    assert (tally.spots, tally.observed["opens"]) == (2, 1)
    # The big blind checks the flop, then folds to a pot-size bet: a solid player folds half
    # the time there, since a pot-size bluff needs to work half the time.
    flop = play(heads_up("QsJs", "8c8d", "Kh7d2c3s9h"), Action(ActionType.RAISE, 250), CALL)
    folded = play(flop, CHECK, Action(ActionType.BET, 500), FOLD)
    tally.count(folded, 1, NEUTRAL)
    assert (tally.observed["folds_flop"], tally.expected["folds_flop"]) == (1, 0.5)
    assert tally.observed["bets_flop"] == 0 and 0 < tally.expected["bets_flop"] < 1
    assert tally.read().folds_flop == (1 + 4) / (0.5 + 4)
    assert tally.read().limp_width == (1 + 5) / (2 + 10)
