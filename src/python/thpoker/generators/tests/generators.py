import numpy as np
import pytest
from   thpoker.analysis.ev      import LEAF_KNOTS, leaf_group, leaf_terms
from   thpoker.game.cards       import COMBOS_OF_CLASS, PREFLOP_CLASSES
from   thpoker.game.rng         import Rng
from   thpoker.generators       import leaf_values
from   thpoker.generators.preflop_equity \
                                import generate
from   thpoker.generators.preflop_ranges \
                                import _pair, solve as solve_preflop
from   thpoker.generators.pushfold \
                                import (_Spot, _accumulate, _current,
                                        solve as solve_pushfold)
from   thpoker.odds             import Classes


@pytest.fixture(scope="module")
def classes() -> Classes:
    return Classes.load()


@pytest.mark.parametrize("ante", [0.0, 1.0])
def test_every_showdown_or_flop_splits_exactly_the_chips_in_the_pot(classes, ante):
    # Six seats (the last two are the blinds); opener seat 1 against the big blind (seat 5), who
    # also posted any ante. In every outcome both players' results add up to the dead money
    # from the other seats, less the big blind's own ante.
    blinds = np.array([0.0, 0.0, 0.0, 0.0, 0.5, 1.0])
    m = _pair(classes, blinds, ante, 1, 5, stack=40.0)
    dead_from_others = 0.5 + ante
    for opener, responder in (
        (m.opener_call, m.responder_call),
        (m.opener_call3, m.responder_call3),
        (m.opener_allin, m.responder_allin),
    ):
        assert np.allclose(opener + responder, dead_from_others - ante)
    assert m.dead == pytest.approx(dead_from_others)


def test_heads_up_solution_opens_wide_from_the_button_and_is_near_equilibrium(classes):
    strategies, gain = solve_preflop(classes, 2, 100.0, bb_ante=False)
    opens = strategies["open"]["0"]
    assert opens[PREFLOP_CLASSES.index("AA")] == 100
    assert opens[PREFLOP_CLASSES.index("72o")] == 0
    # The small blind is the button and plays after the flop in position, so it opens most
    # hands (solvers raise about 80%); played out of position it opened 55%.
    combos = np.array([len(COMBOS_OF_CLASS[c]) for c in PREFLOP_CLASSES])
    assert np.dot(combos, opens) / 100 / combos.sum() > 0.65
    assert gain < 0.01  # big blinds per hand a best response


def test_generated_table_is_complete_and_zero_sum():
    table = generate(20, seed=3)
    equity = table["equity"]
    assert table["classes"] == list(PREFLOP_CLASSES)
    assert len(equity) == 169 and all(len(row) == 169 for row in equity)
    # Every pair of combos contributes a win to one side exactly when it is a loss to the
    # other, so A-vs-B and B-vs-A sum to 1 up to the 4-decimal rounding.
    assert max(abs(equity[i][j] + equity[j][i] - 1) for i in range(169) for j in range(169)) <= 1e-4


def test_aces_against_kings_match_the_published_equity():
    table = generate(200, seed=3)
    aces, kings = PREFLOP_CLASSES.index("AA"), PREFLOP_CLASSES.index("KK")
    # Published: AA wins about 82% all-in against KK.
    assert abs(table["equity"][aces][kings] - 0.82) < 4 * table["stderr"]


def test_board_count_must_split_into_batches():
    with pytest.raises(ValueError, match="multiple of 10"):
        generate(15, seed=3)


def test_heads_up_solve_converges_to_the_published_16bb_shove_range():
    classes = Classes.load()
    push, calls, gain = solve_pushfold(classes, 2, 0, 16.0, bb_ante=False)
    # PokerStrategy.com (HoldemResources ranges): at 16bb the small blind shoves 43.3%.
    assert (push * classes.sizes).sum() / classes.sizes.sum() == pytest.approx(0.433, abs=0.02)
    assert len(calls) == 1 and gain < 0.001  # big blinds per hand a best response could gain


def test_push_fold_values_follow_the_pot_and_the_equity(classes):
    # Aces win every all-in against other hands and every other matchup is a coin flip, so each
    # value is chip arithmetic: 16bb stacks, blinds of 0.5 and 1, the small blind shoving.
    aces, kings, suited = (PREFLOP_CLASSES.index(name) for name in ("AA", "KK", "AKs"))
    equity = np.full_like(classes.equity, 0.5)
    equity[aces, :] = 1.0
    equity[:, aces] = 0.0
    equity[aces, aces] = 0.5
    spot = _Spot(Classes(equity, classes.pairs, classes.sizes), np.array([0.5, 1.0]), 0, 16.0)
    only = np.eye(len(PREFLOP_CLASSES))
    assert spot.pot(1) == 32.0
    shove, fold = spot.push_values([np.zeros(len(PREFLOP_CLASSES))])  # the big blind never calls
    assert np.allclose(shove, 1.0)
    assert np.allclose(fold, -0.5)
    # A combo shares no card with 1,225 others, so a 6-combo class meets 7,350 combos.
    shove, _ = spot.push_values([np.ones(len(PREFLOP_CLASSES))])  # the big blind always calls
    assert shove[aces] == pytest.approx(16 * (1 - 6 / 7350))  # all but the 6 other aces combos
    assert shove[kings] == pytest.approx(-16 * 36 / 7350)  # kings lose to the 36 aces combos
    # Called only by aces: kings lose 16 against 36 combos and steal the blinds from the rest.
    shove, _ = spot.push_values([only[aces]])
    assert shove[kings] == pytest.approx((36 * -16 + 7314 * 1.0) / 7350)
    call, fold = spot.call_values(1, only[aces])  # only aces shove
    assert call[kings] == pytest.approx(-16.0)
    assert call[aces] == pytest.approx(0.0)
    assert np.allclose(fold, -1.0)
    # Aces and suited ace-kings shove: kings meet 36 aces combos (lose 16) and 12 ace-kings (flip).
    call, _ = spot.call_values(1, only[aces] + only[suited])
    assert call[kings] == pytest.approx((36 * -16 + 12 * 0.0) / 48)


def test_regret_matching_plays_the_better_action_and_drops_negative_regret():
    regret = np.zeros((2, len(PREFLOP_CLASSES)))
    assert np.allclose(_current(regret), 0.5)  # no regret yet: an even mix
    shove, fold = np.full(len(PREFLOP_CLASSES), 1.0), np.full(len(PREFLOP_CLASSES), -0.5)
    _accumulate(regret, _current(regret), shove, fold)
    # Shoving beat the even mix (0.25) by 0.75; folding's regret would be negative, so it is zero.
    assert np.allclose(regret[0], 0.75) and np.allclose(regret[1], 0.0)
    assert np.allclose(_current(regret), 1.0)


def test_leaf_samples_are_repeatable_and_won_stays_within_the_stacks():
    rows = leaf_values.sample((7, 0, 30))
    assert rows and rows == leaf_values.sample((7, 0, 30))
    for key, terms, _, won_less_equity, spr, _, equity in rows:
        # The second term at each knot is the equity weighted by that knot; together, the equity.
        assert sum(terms[len(LEAF_KNOTS) : 2 * len(LEAF_KNOTS)]) == pytest.approx(equity)
        assert 0 <= equity <= 1
        if key.endswith("heads-up"):  # multiway, a winner can take several stacks
            assert -spr - 1e-9 <= won_less_equity + equity <= 1 + spr + 1e-9


def test_the_leaf_fit_recovers_known_coefficients():
    rng = Rng(3)
    truth = np.array([0.05 * ((i % 7) - 3) for i in range(7 * len(LEAF_KNOTS))])
    rows = []
    for _ in range(8_000):
        equity, spr = rng.random(), 30 * rng.random() ** 2
        # Out of position the hands are never stronger than 0.6, and callers are rare.
        in_position = rng.random() < 0.5
        equity = equity if in_position else 0.6 * equity
        role = ("none", "bettor", "caller")[rng.randbelow(3)]
        if role == "caller" and not in_position and rng.random() > 0.05:
            role = "bettor"
        terms, _ = leaf_terms(equity, spr, in_position, role, rng.random() < 0.2)
        noise = 0.01 * (rng.random() - 0.5) * (1 + spr)
        won = np.log1p(spr) * terms @ truth + noise
        group = leaf_group(in_position, role)
        rows.append(("2-heads-up", terms.tolist(), np.log1p(spr), won, spr, group, equity))
    rows.append(("1-multiway", *rows[0][1:]))
    fits = leaf_values.fit(rows)
    assert set(fits) == {"2-heads-up"}  # too few samples for the other key
    assert np.allclose(fits["2-heads-up"]["coef"], truth, atol=0.02)
    seen = fits["2-heads-up"]["equities"]
    assert "caller-out of position" not in seen  # too few samples to be used
    assert seen["bettor-out of position"][1] <= 0.6 < seen["bettor-in position"][1]
