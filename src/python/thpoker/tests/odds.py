from   itertools                import combinations
from   statistics               import fmean, stdev

import pytest

from   thpoker.game.cards       import (COMBOS, COMBOS_OF_CLASS,
                                        PREFLOP_CLASSES, combo_index,
                                        parse_cards)
from   thpoker.game.evaluator   import evaluate
from   thpoker.game.rng         import Rng
import thpoker.odds
from   thpoker.odds             import (Classes, FULL_RANGE, Range, chen_score,
                                        equity_to_reach_top, equity_vs_random,
                                        hand_equity, hand_range, hand_window,
                                        range_equities, range_share,
                                        ranked_range, representative_equities,
                                        texture)


@pytest.mark.parametrize(
    "cards, score",
    [("AsAd", 20), ("AsKs", 12), ("JsTs", 9), ("2c2d", 5), ("Ts9s", 8), ("7c2d", -1.5)],
)
def test_chen_score(cards, score):
    assert chen_score(*parse_cards(cards)) == score


def test_best_two_percent_is_jacks_or_better_and_ace_king_suited():
    # Chen scores: AA 20, KK 16, QQ 14, JJ and AKs 12; the next hand, AQs, scores 11.
    expected = {i for name in ("AA", "KK", "QQ", "JJ", "AKs") for i in COMBOS_OF_CLASS[name]}
    assert {i for i, w in enumerate(ranked_range(0.0, 0.02)) if w} == expected


def test_width_cuts_are_unbiased_across_widths():
    # Midpoint percentiles make a cut through a tie group err either way; cutting at each tie
    # group's first hand would admit whole groups and overshoot by about 50 combos on average.
    widths = [w / 100 for w in range(1, 100)]
    errors = [sum(ranked_range(0.0, w)) - w * len(COMBOS) for w in widths]
    assert abs(sum(errors) / len(errors)) < 10


def test_training_hands_come_from_a_strength_window_or_a_named_set():
    assert len(hand_range("0-100")) == len(PREFLOP_CLASSES)
    best = hand_range("0-5")
    assert best[:3] == ("AA", "KK", "QQ") and abs(range_share(best) - 0.05) < 0.01
    low, high = hand_window(hand_range("25-5"))  # either order
    assert hand_range("25-5") == hand_range("5-25") and abs(low - 0.05) < 0.01
    assert abs(high - 0.25) < 0.01
    (nearest,) = hand_range("40-40.1")  # too narrow for any class: the nearest one
    assert abs(sum(hand_window([nearest])) / 2 - 0.4) < 0.01
    assert set(hand_range("pairs")) == {rank * 2 for rank in "AKQJT98765432"}
    assert hand_window(hand_range("pairs")) is None
    assert set(hand_range("small-aces")) == {f"A{r}{s}" for r in "98765432" for s in "so"}
    assert set(hand_range("suited-connectors")) == {a + b + "s" for a, b in zip("AKQJT9876543", "KQJT98765432")}
    for bad in ("top", "5-", "-5", "5-150", "AKs"):
        with pytest.raises(ValueError):
            hand_range(bad)


def only(*hands: str) -> Range:
    chosen = {combo_index(*parse_cards(h)) for h in hands}
    return tuple(1.0 if i in chosen else 0.0 for i in range(len(COMBOS)))


def of_class(name: str) -> Range:
    chosen = set(COMBOS_OF_CLASS[name])
    return tuple(1.0 if i in chosen else 0.0 for i in range(len(COMBOS)))


def cards(text: str) -> tuple[int, ...]:
    return tuple(parse_cards(text))


def test_river_equity_is_exact_against_one_opponent():
    board = cards("2c7d9hJs3c")
    beats = hand_equity(cards("AhAd"), board, [only("KhKd")], Rng(1))
    assert (beats.value, beats.exact, beats.stderr) == (1.0, True, 0.0)
    assert hand_equity(cards("AhAd"), board, [only("AsAc")], Rng(1)).value == 0.5
    both = hand_equity(cards("QhQd"), board, [only("KhKd"), only("5h5d")], Rng(1))
    assert both.value == 0.0 and not both.exact  # the multiway product is an approximation


def mean_estimate(hole: tuple[int, ...], board: tuple[int, ...], ranges: list[Range]) -> tuple[float, float]:
    """Mean of 400 seeded estimates, and the tolerance 4 standard errors of that mean."""
    estimates = [hand_equity(hole, board, ranges, Rng(seed)) for seed in range(400)]
    return fmean(e.value for e in estimates), 4 * fmean(e.stderr for e in estimates) / 400**0.5


def test_turn_estimate_agrees_with_brute_force():
    hole, board = cards("AhKh"), cards("2h7h9cTd")
    won = pairs = 0.0
    for index in COMBOS_OF_CLASS["QQ"]:
        combo = COMBOS[index]
        for river in (c for c in range(52) if c not in hole + board + combo):
            full = list(board) + [river]
            hero, other = evaluate(list(hole) + full), evaluate(list(combo) + full)
            won += 1.0 if hero > other else 0.5 if hero == other else 0.0
            pairs += 1
    mean, tolerance = mean_estimate(hole, board, [of_class("QQ")])
    assert abs(mean - won / pairs) < tolerance


def test_two_opponent_flop_estimate_matches_the_board_weighted_product():
    # Against several opponents the estimate targets the product of per-opponent shares on each
    # board, with boards weighted by the product of the opponents' still-possible range weight.
    hole, board = cards("AhKh"), cards("2h7h9c")
    classes = ("QQ", "88")
    weighted = total = 0.0
    for runout in combinations([c for c in range(52) if c not in hole + board], 2):
        full = list(board + runout)
        hero = evaluate(list(hole) + full)
        product = weight = 1.0
        for name in classes:
            combos = [COMBOS[i] for i in COMBOS_OF_CLASS[name] if not set(COMBOS[i]) & set(full)]
            values = [evaluate(list(c) + full) for c in combos]
            product *= sum(1.0 if hero > v else 0.5 if hero == v else 0.0 for v in values) / len(values)
            weight *= len(values)
        weighted += product * weight
        total += weight
    mean, tolerance = mean_estimate(hole, board, [of_class(name) for name in classes])
    assert abs(mean - weighted / total) < tolerance


def test_sampled_boards_do_not_depend_on_the_hole_cards(monkeypatch):
    seen: list[tuple[int, ...]] = []
    original = thpoker.odds._board_equity

    def record(hole, board, ranges, live):
        seen.append(board)
        return original(hole, board, ranges, live)

    monkeypatch.setattr(thpoker.odds, "_board_equity", record)
    flop, first, second = cards("2h7h9c"), cards("AhKh"), cards("7c2d")
    hand_equity(first, flop, [FULL_RANGE], Rng(5))
    first_boards, seen[:] = list(seen), []
    hand_equity(second, flop, [FULL_RANGE], Rng(5))
    # Each hand skips only the boards that use its own cards.
    assert [b for b in first_boards if not set(b) & set(second)] == [b for b in seen if not set(b) & set(first)]


def test_equity_to_reach_top_counts_combos_in_equity_order():
    # Against any two cards aces are the best hand (6 combos) and kings the next 6.
    assert equity_to_reach_top(6 / 1326) == equity_vs_random(cards("AhAd"))
    assert equity_to_reach_top(7 / 1326) == equity_to_reach_top(12 / 1326) == equity_vs_random(cards("KhKd"))


def test_flop_estimate_agrees_with_brute_force():
    hole, board, villain = cards("AhKh"), cards("2h7h9c"), of_class("QQ")
    won = pairs = 0.0
    for index in COMBOS_OF_CLASS["QQ"]:
        combo = COMBOS[index]
        deck = [c for c in range(52) if c not in hole + board + combo]
        for runout in combinations(deck, 2):
            full = list(board + runout)
            hero, other = evaluate(list(hole) + full), evaluate(list(combo) + full)
            won += 1.0 if hero > other else 0.5 if hero == other else 0.0
            pairs += 1
    estimates = [hand_equity(hole, board, [villain], Rng(seed)) for seed in range(400)]
    typical_stderr = fmean(e.stderr for e in estimates)
    # Unbiased: the mean of 400 estimates lands within 4 standard errors of the exact value.
    # (Averaging boards without weighting them by the villain combos they leave possible is
    # off by about 0.025 here, outside this bound.)
    assert abs(fmean(e.value for e in estimates) - won / pairs) < 4 * typical_stderr / 400**0.5
    # Honest: the reported standard error matches the actual spread between seeds.
    assert 0.5 < stdev(e.value for e in estimates) / typical_stderr < 2.0


def test_preflop_equities_match_published_values():
    # Published all-in equities: AA vs KK 82%, QQ vs AKo 57%, AA vs any two cards 85%.
    assert hand_equity(cards("AhAd"), (), [of_class("KK")], Rng(1)).value == pytest.approx(0.82, abs=0.02)
    assert hand_equity(cards("QhQd"), (), [of_class("AKo")], Rng(1)).value == pytest.approx(0.57, abs=0.02)
    assert equity_vs_random(cards("AhAd")) == pytest.approx(0.85, abs=0.01)
    assert hand_equity(cards("AhAd"), (), [FULL_RANGE], Rng(1)).value == pytest.approx(0.85, abs=0.01)


@pytest.mark.parametrize(
    "ranges, board, message",
    [
        ([], (), "at least one"),
        ([FULL_RANGE], cards("2c7d"), "0, 3, 4, or 5"),
        ([only("AsAc")], cards("As7d9hJs3c"), "empty"),
    ],
)
def test_invalid_equity_requests_rejected(ranges, board, message):
    with pytest.raises(ValueError, match=message):
        hand_equity(cards("KhKd"), board, ranges, Rng(1))


def test_range_equities_on_the_river_equal_exact_single_hand_equities():
    board = cards("Kh9d4c2s7h")
    villain = of_class("KQo")
    combos = [i for i, (a, b) in enumerate(COMBOS) if i % 17 == 0]
    scored = range_equities(board, combos, [villain], Rng(3))
    for index, value in scored.items():
        assert value == pytest.approx(hand_equity(COMBOS[index], board, [villain], Rng(3)).value, abs=1e-12)


def test_a_hands_range_equity_does_not_depend_on_the_other_hands_scored():
    flop, hero = cards("2h7h9c"), combo_index(*parse_cards("AhKh"))
    alone = range_equities(flop, [hero], [FULL_RANGE], Rng(8))[hero]
    together = range_equities(flop, list(range(0, len(COMBOS), 5)) + [hero], [FULL_RANGE], Rng(8))[hero]
    assert alone == together


def test_flop_range_equities_average_to_the_brute_force_equity():
    hole, board = cards("AhKh"), cards("2h7h9c")
    won = pairs = 0.0
    for index in COMBOS_OF_CLASS["QQ"]:
        combo = COMBOS[index]
        for runout in combinations([c for c in range(52) if c not in hole + board + combo], 2):
            full = list(board + runout)
            hero, other = evaluate(list(hole) + full), evaluate(list(combo) + full)
            won += 1.0 if hero > other else 0.5 if hero == other else 0.0
            pairs += 1
    hero = combo_index(*hole)
    estimates = [range_equities(board, [hero], [of_class("QQ")], Rng(seed))[hero] for seed in range(300)]
    assert abs(fmean(estimates) - won / pairs) < 4 * stdev(estimates) / 300**0.5


def test_representative_equities_match_hand_equity_for_every_class():
    # A lopsided range (aces and broadways weighted up) so card removal matters.
    weights = tuple(1.0 + 3.0 * (max(a, b) // 4 >= 10) + 5.0 * (min(a, b) // 4 == 12) for a, b in COMBOS)
    fast = representative_equities(weights, range(len(PREFLOP_CLASSES)))
    assert representative_equities(weights, [7]) == {7: fast[7]}  # the same whatever else is asked
    for klass, name in enumerate(PREFLOP_CLASSES):
        representative = COMBOS[COMBOS_OF_CLASS[name][0]]
        assert fast[klass] == pytest.approx(hand_equity(representative, (), [weights], Rng(1)).value, abs=1e-12)


@pytest.mark.parametrize("board_size, opponents", [(3, 1), (4, 2), (5, 3)])
def test_range_equities_match_hand_equity_with_fractional_weights(monkeypatch, board_size, opponents):
    # With the same number of runouts, hand_equity draws the same boards from the same seed.
    monkeypatch.setattr(thpoker.odds, "RUNOUT_SAMPLES", thpoker.odds.RANGE_RUNOUT_SAMPLES)
    rng = Rng(board_size * 10 + opponents)
    board = tuple(rng.permutation(52)[:board_size])
    # Fractional weights, some zero, so rounding in the card-removal subtraction shows up.
    ranges = [tuple(rng.random() * (rng.random() < 0.6) for _ in COMBOS) for _ in range(opponents)]
    combos = [i for i, (a, b) in enumerate(COMBOS) if i % 3 == 0 and a not in board and b not in board]
    scored = range_equities(board, combos, ranges, Rng(5))
    assert scored.keys() == set(combos)
    for index, value in scored.items():
        assert value == pytest.approx(hand_equity(COMBOS[index], board, ranges, Rng(5)).value, abs=1e-12)


def test_a_hand_that_blocks_the_whole_range_gets_no_equity():
    # Every hand of this range shares a card with AsKs, so AsKs cannot face it at all.
    weights = [0.0] * len(COMBOS)
    for hand in ("AsQd", "KsJh", "AsJc"):
        weights[combo_index(*parse_cards(hand))] = 1 / 3  # removing thirds leaves rounding crumbs
    aces_kings, queens = combo_index(*parse_cards("AsKs")), combo_index(*parse_cards("QhQc"))
    equities = range_equities(tuple(parse_cards("2c3d4h")), [aces_kings, queens], [tuple(weights)], Rng(1))
    assert aces_kings not in equities and queens in equities


@pytest.mark.parametrize(
    "text, high, pairing, suits, connectivity, wet",
    [
        ("Ks7d2c", "K-high", "unpaired", "rainbow", "disconnected", False),
        ("9h8h7c", "middle", "unpaired", "two-tone", "connected", True),
        ("Jh8h3c", "broadway", "unpaired", "two-tone", "semi-connected", True),
        ("AsAd5c", "A-high", "paired", "rainbow", "semi-connected", False),
        ("6h4h2h", "low", "unpaired", "monotone", "connected", True),
        ("4c3d2s", "low", "unpaired", "rainbow", "connected", True),
        ("QcQdQh", "broadway", "trips", "rainbow", "disconnected", False),
    ],
)
def test_flop_tags(text, high, pairing, suits, connectivity, wet):
    tags = texture(cards(text))
    assert (tags.high, tags.pairing, tags.suits, tags.connectivity, tags.wet) == (
        high,
        pairing,
        suits,
        connectivity,
        wet,
    )
    assert tags.change is None


@pytest.mark.parametrize(
    "text, change",
    [
        ("Ks7h2hAh", "flush completed"),
        ("Ks9d2c8h", "brick"),
        ("Ks9dTc2h", "brick"),
        ("8s4d2cKh", "overcard"),
        ("9s7d2cJh", "straight completed"),
        ("Ks7d2c7h", "board paired"),
        ("Ks9d2cTh", "straight completed"),
    ],
)
def test_last_card_changes(text, change):
    assert texture(cards(text)).change == change


def test_board_size_is_checked():
    with pytest.raises(ValueError, match="3 to 5"):
        texture(cards("Ks7d"))


def test_class_tables_count_combos_and_card_removal():
    classes, index = Classes.load(), PREFLOP_CLASSES.index
    assert [classes.sizes[index(name)] for name in ("AA", "AKs", "AKo")] == [6, 4, 12]
    # Ordered combo pairs that share no card: an aces combo misses only one other aces combo,
    # aces never block kings, and each suited ace-king leaves 3 of the 6 aces combos.
    assert classes.pairs[index("AA"), index("AA")] == 6
    assert classes.pairs[index("AA"), index("KK")] == 36
    assert classes.pairs[index("AKs"), index("AA")] == 12


def test_the_shipped_equity_table_is_zero_sum_and_matches_published_equity():
    equity = Classes.load().equity
    # Both sides of a class pair come from the same boards, so they sum to one, up to the
    # four-decimal rounding of the file.
    assert abs(equity + equity.T - 1).max() <= 1e-4
    aces, kings = PREFLOP_CLASSES.index("AA"), PREFLOP_CLASSES.index("KK")
    assert equity[aces, kings] == pytest.approx(0.82, abs=0.01)  # published: about 82% all-in
