from   collections              import Counter
from   dataclasses              import replace
import hashlib
from   itertools                import combinations
import json

import pytest

from   thpoker.game.cards       import (COMBOS, COMBOS_OF_CLASS,
                                        CardParseError, PREFLOP_CLASSES,
                                        card_str, combo_index, parse_card,
                                        parse_cards, preflop_class)
from   thpoker.game.engine      import (Pot, apply_action, build_pots,
                                        is_terminal, net_results, new_hand,
                                        observation, replay_states, split_pot)
from   thpoker.game.evaluator   import (CATEGORY_NAMES, category, describe,
                                        evaluate, evaluate_combos)
from   thpoker.game.rng         import Rng
from   thpoker.game.rules       import IllegalActionError, legal_actions
from   thpoker.game.state       import (Action, ActionType, AnteType,
                                        ConfigError, Event, GameConfig,
                                        GameState, Observation, Street)
from   thpoker.game.tests.decks import CALL, CHECK, FOLD, play, stacked_deck


def raise_to(amount: int) -> Action:
    return Action(ActionType.RAISE, amount)


def test_heads_up_button_posts_small_blind_acts_first_preflop_and_last_after():
    state, _ = new_hand(GameConfig(2), seed=1, button=0, stacks=(10_000, 10_000))
    assert (state.small_blind_seat, state.big_blind_seat, state.to_act) == (0, 1, 0)
    assert state.committed_this_street == (50, 100)
    state = play(state, CALL)
    assert state.to_act == 1 and legal_actions(state).can_check  # big blind option
    state = play(state, CHECK)
    assert state.street == Street.FLOP and len(state.board) == 3 and state.to_act == 1


def test_multiway_blinds_left_of_button_and_action_order():
    state, _ = new_hand(GameConfig(4), seed=1, button=0, stacks=(10_000,) * 4)
    assert (state.small_blind_seat, state.big_blind_seat, state.to_act) == (1, 2, 3)
    state = play(state, CALL, CALL, CALL, CHECK)
    assert state.street == Street.FLOP and state.to_act == 1


def test_minimum_raise_is_the_last_raise_size():
    state, _ = new_hand(GameConfig(3), seed=1, button=0, stacks=(10_000,) * 3)
    assert (legal_actions(state).min_raise_to, legal_actions(state).max_raise_to) == (200, 10_000)
    state = play(state, raise_to(300))
    assert legal_actions(state).min_raise_to == 500


def test_short_all_in_does_not_reopen_betting():
    # Big blind (seat 2) has 450: its all-in to 450 over a raise to 300 is 150, short of 200.
    state, _ = new_hand(GameConfig(3), seed=1, button=0, stacks=(10_000, 10_000, 450))
    state = play(state, raise_to(300), CALL, raise_to(450))
    assert state.to_act == 0
    legal = legal_actions(state)
    assert not legal.can_raise and legal.can_call and legal.call_amount == 150


def test_short_all_ins_that_add_up_to_a_full_raise_reopen_betting():
    # UTG (seat 3) raises to 300; seat 4 calls; seats 0 and 1 go all-in for 400 and 520,
    # each short, but 220 in total over 300 is a full raise for UTG.
    stacks = (400, 520, 10_000, 10_000, 10_000)
    state, _ = new_hand(GameConfig(5), seed=1, button=0, stacks=stacks)
    state = play(state, raise_to(300), CALL, raise_to(400), raise_to(520), FOLD)
    assert state.to_act == 3
    legal = legal_actions(state)
    assert legal.can_raise and legal.min_raise_to == 720


def test_uncalled_bet_is_returned():
    state, _ = new_hand(GameConfig(2), seed=1, button=0, stacks=(10_000, 10_000))
    state, events = apply_action(play(state, raise_to(300)), FOLD)
    assert is_terminal(state) and net_results(state) == (100, -100)
    assert {"seat": 0, "amount": 200} in [e.data for e in events if e.kind == "UncalledBetReturned"]


def test_three_way_all_in_builds_main_and_side_pot():
    deck = stacked_deck(0, 3, {0: "AhAd", 1: "KhKd", 2: "QhQd"}, "2c7d9hJs3c")
    state, _ = new_hand(GameConfig(3), seed=1, button=0, stacks=(1_000, 3_000, 5_000), deck=deck)
    state = play(state, raise_to(1_000), raise_to(3_000), CALL)
    assert is_terminal(state)
    # Side pot 2 x 2,000 to kings, then the main pot 3 x 1,000 to aces; queens lose 3,000.
    assert [(a.amount, a.winners) for a in state.awards] == [(4_000, (1,)), (3_000, (0,))]
    assert state.stacks == (3_000, 4_000, 2_000)


def test_split_pot_odd_chip_goes_to_first_winner_left_of_button():
    config = GameConfig(3, ante=1, ante_type=AnteType.PER_PLAYER)
    deck = stacked_deck(0, 3, {0: "2c3d", 1: "6h7s", 2: "4c5d"}, "TsJdQcKhAh")
    state, _ = new_hand(config, seed=1, button=0, stacks=(10_000,) * 3, deck=deck)
    state = play(state, CALL, FOLD, CHECK, CHECK, CHECK, CHECK, CHECK, CHECK, CHECK)
    # Pot = 3 antes + 50 + 100 + 100 = 253, split 126 / 127; seat 2 is first left of button 0.
    assert state.awards[0].winners == (0, 2) and state.awards[0].shares == (126, 127)
    assert state.stacks == (10_025, 9_949, 10_026)


def test_big_blind_is_posted_before_the_big_blind_ante():
    config = GameConfig(3, ante=100, ante_type=AnteType.BIG_BLIND_ANTE)
    state, events = new_hand(config, seed=1, button=0, stacks=(10_000, 10_000, 150))
    assert state.committed_this_street[2] == 100 and state.committed_total[2] == 100
    assert state.dead_money == 50 and state.all_in[2]
    assert [e.data["amounts"] for e in events if e.kind == "AntesPosted"] == [[0, 0, 50]]


def test_big_blind_ante_goes_to_the_main_pot_when_the_big_blind_folds():
    # The big blind calls an all-in, then folds to a later bet: its ante is dead money in the
    # main pot, not a commitment layer above the live players.
    config = GameConfig(3, ante=100, ante_type=AnteType.BIG_BLIND_ANTE)
    deck = stacked_deck(0, 3, {0: "AhAd", 1: "KhKd", 2: "7c2s"}, "2c7d9hJs3c")
    state, _ = new_hand(config, seed=1, button=0, stacks=(1_000, 5_000, 5_000), deck=deck)
    state = play(state, raise_to(1_000), CALL, CALL, Action(ActionType.BET, 500), FOLD)
    assert [(a.amount, a.winners) for a in state.awards] == [(3_100, (0,))]
    assert state.stacks == (3_100, 4_000, 3_900)


def test_short_big_blind_still_sets_a_full_big_blind_to_call():
    deck = stacked_deck(0, 3, {0: "KhKd", 1: "QhQd", 2: "AhAd"}, "2c7d9hJs3c")
    state, _ = new_hand(GameConfig(3), seed=1, button=0, stacks=(10_000, 10_000, 30), deck=deck)
    assert legal_actions(state).call_amount == 100
    state = play(state, CALL, CALL, CHECK, CHECK, CHECK, CHECK, CHECK, CHECK)
    # Kings win the 2 x 70 side pot; aces win only the main pot of 3 x 30.
    assert [(a.amount, a.winners) for a in state.awards] == [(140, (0,)), (90, (2,))]
    assert state.stacks == (10_040, 9_900, 90)


def test_short_per_player_ante_caps_its_own_pot():
    # Seat 0 is all-in on a 10-chip ante, so it can win only 10 from each player.
    config = GameConfig(3, ante=25, ante_type=AnteType.PER_PLAYER)
    deck = stacked_deck(0, 3, {0: "AhAd", 1: "KhKd", 2: "QhQd"}, "2c7d9hJs3c")
    state, _ = new_hand(config, seed=1, button=0, stacks=(10, 10_000, 10_000), deck=deck)
    state = play(state, CALL, CHECK, CHECK, CHECK, CHECK, CHECK, CHECK, CHECK)
    assert [(a.amount, a.winners) for a in state.awards] == [(230, (1,)), (30, (0,))]
    assert state.stacks == (30, 10_105, 9_875)


def test_heads_up_short_big_blind_runs_out_without_action():
    state, events = new_hand(GameConfig(2), seed=1, button=0, stacks=(10_000, 30))
    assert is_terminal(state) and len(state.board) == 5
    assert {"seat": 0, "amount": 20} in [e.data for e in events if e.kind == "UncalledBetReturned"]
    assert sum(a.amount for a in state.awards) == 60


@pytest.mark.parametrize(
    "action, message",
    [
        (CHECK, "CHECK is not legal"),
        (raise_to(150), "outside"),
        (Action(ActionType.RAISE), "outside"),
        (Action(ActionType.CALL, 100), "takes no amount"),
        (Action(ActionType.BET, 300), "BET is not legal"),
    ],
)
def test_illegal_actions_are_rejected(action, message):
    state, _ = new_hand(GameConfig(3), seed=1, button=0, stacks=(10_000,) * 3)
    with pytest.raises(IllegalActionError, match=message):
        apply_action(state, action)


def test_no_action_after_the_hand_is_complete():
    state, _ = new_hand(GameConfig(2), seed=1, button=0, stacks=(10_000, 10_000))
    state = play(state, FOLD)
    with pytest.raises(IllegalActionError, match="complete"):
        apply_action(state, CHECK)


@pytest.mark.parametrize(
    "stacks, button, dealt_in, message",
    [
        ((100,), 0, None, "stacks"),
        ((100, 0, 0), 0, None, "at least two"),
        ((100, 100, 0), 2, None, "button"),
        ((100, 0, 100), 0, (True, True, True), "must have chips"),
    ],
)
def test_bad_hand_setup_rejected(stacks, button, dealt_in, message):
    with pytest.raises(ConfigError, match=message):
        new_hand(GameConfig(max(2, len(stacks))), 1, button, stacks, dealt_in)


def test_observation_hides_other_cards_deck_and_seed_until_showdown():
    deck = stacked_deck(0, 2, {0: "AhAd", 1: "KhKd"}, "2c7d9hJs3c")
    state, _ = new_hand(GameConfig(2), seed=987_654, button=0, stacks=(10_000, 10_000), deck=deck)
    view = observation(state, 0)
    assert view.hole_cards == (tuple(parse_cards("AhAd")), None)
    assert not hasattr(view, "deck") and not hasattr(view, "seed")
    assert "987654" not in json.dumps(view.to_dict())
    state = play(state, CALL, CHECK, CHECK, CHECK, CHECK, CHECK, CHECK, CHECK)
    assert observation(state, 0).hole_cards[1] == tuple(parse_cards("KhKd"))


def test_finished_hand_observation_and_events_round_trip_through_json():
    config = GameConfig(4, ante=10, ante_type=AnteType.PER_PLAYER)
    state, events = new_hand(config, 3, 1, (5_000,) * 4)
    for action in (raise_to(250), CALL, FOLD, CALL) + (CHECK,) * 9:
        state, more = apply_action(state, action)
        events += more
    assert is_terminal(state) and state.awards and state.shown
    assert GameState.from_dict(json.loads(json.dumps(state.to_dict()))) == state
    view = observation(state, 2)
    assert Observation.from_dict(json.loads(json.dumps(view.to_dict()))) == view
    assert [Event.from_dict(json.loads(json.dumps(e.to_dict()))) for e in events] == events


def _random_action(state: GameState, rng: Rng) -> Action:
    legal = legal_actions(state)
    choices = [
        t
        for t, ok in (
            (ActionType.FOLD, legal.can_fold),
            (ActionType.CHECK, legal.can_check),
            (ActionType.CALL, legal.can_call),
            (ActionType.BET, legal.can_bet),
            (ActionType.RAISE, legal.can_raise),
        )
        if ok
    ]
    kind = choices[rng.randbelow(len(choices))]
    if kind in (ActionType.BET, ActionType.RAISE):
        amount = legal.min_raise_to + rng.randbelow(legal.max_raise_to - legal.min_raise_to + 1)
        return Action(kind, amount)
    return Action(kind)


def test_random_play_conserves_chips_and_always_settles():
    rng = Rng(12345)
    for hand_number in range(400):
        seats = 2 + rng.randbelow(7)
        stacks = tuple(1 + rng.randbelow(3_000) for _ in range(seats))  # some shorter than a blind
        ante_type = (AnteType.NONE, AnteType.PER_PLAYER, AnteType.BIG_BLIND_ANTE)[rng.randbelow(3)]
        config = GameConfig(seats, 50, 100, 0 if ante_type == AnteType.NONE else 25, ante_type)
        state, _ = new_hand(config, hand_number, rng.randbelow(seats), stacks)
        total = sum(stacks)
        for _ in range(500):
            cards = [c for h in state.hole_cards if h for c in h] + list(state.board) + list(state.deck)
            assert sorted(cards) == list(range(52))
            assert all(s >= 0 for s in state.stacks)
            if is_terminal(state):
                break
            assert sum(state.stacks) + state.pot == total
            state, _ = apply_action(state, _random_action(state, rng))
        assert is_terminal(state) and state.to_act is None
        assert sum(state.stacks) == total
        assert sum(a.amount for a in state.awards) == state.pot


def test_replay_rebuilds_every_state_of_a_hand():
    config = GameConfig(4, ante=10, ante_type=AnteType.PER_PLAYER)
    state, _ = new_hand(config, 3, 1, (5_000,) * 4)
    states = [state]
    for action in (raise_to(250), CALL, FOLD, CALL) + (CHECK,) * 9:
        state, _ = apply_action(state, action)
        states.append(state)
    assert replay_states(state) == states


def test_replay_rejects_a_tampered_hand():
    state, _ = new_hand(GameConfig(2), 1, 0, (10_000, 10_000))
    state = play(state, FOLD)
    with pytest.raises(ConfigError, match="does not replay"):
        replay_states(replace(state, stacks=(0, 20_000)))
    with pytest.raises(IllegalActionError):
        replay_states(replace(state, history=(replace(state.history[0], action=CHECK),)))


def test_side_pots_are_capped_by_each_all_in():
    # Seat 3 folded after putting in 200; seats 0-2 are all-in for 100, 300, 300.
    pots = build_pots((100, 300, 300, 200), (True, True, True, False))
    assert pots == [Pot(400, (0, 1, 2)), Pot(500, (1, 2))]


def test_single_pot_when_everyone_matches():
    assert build_pots((500, 500, 0), (True, True, False)) == [Pot(1000, (0, 1))]


def test_dead_money_goes_to_the_main_pot():
    assert build_pots((100, 300, 300), (True, True, True), dead_money=50) == [
        Pot(350, (0, 1, 2)),
        Pot(400, (1, 2)),
    ]


def test_chips_above_every_live_commitment_are_an_error():
    with pytest.raises(ValueError, match="do not fit"):
        build_pots((100, 500), (True, False))


def test_odd_chips_go_left_of_the_button_first():
    # Button is seat 4 of 6: seat order after it is 5, 0, 1, ...
    assert split_pot(101, [1, 5], button=4, num_seats=6) == [50, 51]
    assert split_pot(302, [0, 1, 5], button=4, num_seats=6) == [101, 100, 101]


def test_parse_and_format_round_trip_every_card():
    assert [parse_card(card_str(c)) for c in range(52)] == list(range(52))
    assert parse_card("As") == 51 and parse_card("2c") == 0 and parse_card("td") == 33


def test_parse_cards_accepts_separators_and_rejects_duplicates():
    assert parse_cards("AsKd") == parse_cards("As Kd") == parse_cards("As,Kd") == [51, 45]
    with pytest.raises(CardParseError, match="duplicate"):
        parse_cards("AsAs")


@pytest.mark.parametrize("text", ["A", "1s", "Ax", "AsK"])
def test_parse_rejects_bad_text(text):
    with pytest.raises(CardParseError):
        parse_cards(text)


def test_combos_cover_every_pair_once():
    assert len(COMBOS) == 1326 == 52 * 51 // 2
    assert len(set(COMBOS)) == 1326
    assert all(combo_index(a, b) == combo_index(b, a) == i for i, (a, b) in enumerate(COMBOS))


def test_preflop_classes_have_the_right_combo_counts():
    assert len(PREFLOP_CLASSES) == 169
    counts = {name: len(COMBOS_OF_CLASS[name]) for name in PREFLOP_CLASSES}
    assert sum(counts.values()) == 1326
    assert all(counts[n] == (6 if len(n) == 2 else 4 if n.endswith("s") else 12) for n in counts)
    assert PREFLOP_CLASSES[:3] == ("AA", "AKs", "AQs") and PREFLOP_CLASSES[13] == "AKo"


def test_preflop_class_names():
    assert preflop_class(*parse_cards("KsAs")) == "AKs"
    assert preflop_class(*parse_cards("9dTh")) == "T9o"
    assert preflop_class(*parse_cards("7c7h")) == "77"


def test_stream_is_shake256_of_seed_and_counter():
    # The documented construction, derived here with hashlib directly, guards the promise that
    # the stream never changes across Python versions or refactors.
    key = hashlib.shake_256(b"poker.rng.v1:42").digest(32)
    block = hashlib.shake_256(key + (0).to_bytes(8, "big")).digest(512)
    expected = (int.from_bytes(block[:8], "big") >> 11) / 2**53
    assert Rng(42).random() == expected


def test_same_seed_same_stream_and_different_seed_differs():
    first, second = Rng(7), Rng(7)
    assert [first.random() for _ in range(100)] == [second.random() for _ in range(100)]
    assert Rng(7).permutation(52) != Rng(8).permutation(52)


def test_derive_depends_only_on_seed_and_labels():
    used = Rng(5)
    for _ in range(10):
        used.random()
    assert used.derive("hand", 3).random() == Rng(5).derive("hand", 3).random()
    assert Rng(5).derive("hand", 3).random() != Rng(5).derive("hand", 4).random()
    assert Rng(5).derive("3").random() != Rng(5).derive(3).random()


def test_randbelow_covers_range_evenly():
    rng = Rng(1)
    counts = [0] * 6
    for _ in range(60_000):
        counts[rng.randbelow(6)] += 1
    # Each bucket expects 10,000 with standard deviation ~91; 500 is over 5 sigma.
    assert all(abs(c - 10_000) < 500 for c in counts)


def test_permutation_is_a_permutation():
    assert sorted(Rng(3).permutation(52)) == list(range(52))


@pytest.mark.parametrize("seed", [-1, 1.5, True, "7"])
def test_invalid_seed_rejected(seed):
    with pytest.raises(ValueError, match="seed"):
        Rng(seed)


def test_randbelow_rejects_empty_range():
    with pytest.raises(ValueError, match="n >= 1"):
        Rng(1).randbelow(0)


# Known counts of each category over all 2,598,960 five-card hands.
FIVE_CARD_COUNTS = {
    "high card": 1_302_540,
    "pair": 1_098_240,
    "two pair": 123_552,
    "three of a kind": 54_912,
    "straight": 10_200,
    "flush": 5_108,
    "full house": 3_744,
    "four of a kind": 624,
    "straight flush": 40,
}


def test_every_five_card_hand_gets_the_known_category_counts_and_7462_classes():
    values = [evaluate(hand) for hand in combinations(range(52), 5)]
    counts = Counter(CATEGORY_NAMES[category(v)] for v in values)
    assert counts == FIVE_CARD_COUNTS
    assert len(set(values)) == 7462


def test_seven_card_value_is_the_best_five_card_subset():
    rng = Rng(2024)
    for _ in range(3000):
        cards = rng.permutation(52)[:7]
        assert evaluate(cards) == max(evaluate(five) for five in combinations(cards, 5))


@pytest.mark.parametrize(
    "stronger, weaker",
    [
        ("2c3d4h5s6c", "Ac2d3h4s5c"),  # six-high straight beats the wheel
        ("Ac2d3h4s5c", "AcKdQhJs9c"),  # wheel beats ace high
        ("2h4h6h8hTh", "9cTdJhQsKc"),  # flush beats straight
        ("2c2d2h3s3c", "AhKhQhJh9h"),  # full house beats flush
        ("3c3d3h3s2c", "AcAdAhKsKc"),  # quads beat full house
        ("5h6h7h8h9h", "AcAdAhAsKc"),  # straight flush beats quads
        ("AcAdKhKs2c", "AcAdQhQsKc"),  # higher second pair
        ("AcAdKhQsJc", "AcAdKhQsTc"),  # last kicker decides
        ("AhKh9h7h5h", "AhQhJhTh8h"),  # flush compares the top card first
        ("AcAdAhAs3c", "AcAdAhAs2c"),  # quads kicker
    ],
)
def test_hand_ordering(stronger, weaker):
    assert evaluate(parse_cards(stronger)) > evaluate(parse_cards(weaker))


def test_board_that_plays_is_an_exact_tie():
    board = parse_cards("TsJdQcKhAh")
    assert evaluate(board + parse_cards("2c3d")) == evaluate(board + parse_cards("4c5d"))


def test_quads_kicker_ignores_a_lower_pair():
    # AAAA with KK and Q: the kicker is the king, not the queen; with 22 and Q it is the queen.
    assert evaluate(parse_cards("AcAdAhAsKcKdQc")) > evaluate(parse_cards("AcAdAhAs2c2dQc"))


def test_evaluate_combos_matches_evaluate():
    board = parse_cards("Ah7h2hKd9s")
    combos = [c for c in COMBOS if not set(c) & set(board)]
    assert evaluate_combos(board, combos) == [evaluate(board + list(c)) for c in combos]


@pytest.mark.parametrize(
    "cards, text",
    [
        ("AcAd7h5s2c", "pair of aces"),
        ("6c6d6h9s2c", "three sixes"),
        ("KcKdKh9s9c", "full house, kings full of nines"),
        ("2h4h6h8hKh", "flush, king high"),
        ("Ac2d3h4s5c", "straight, five high"),
    ],
)
def test_describe(cards, text):
    assert describe(evaluate(parse_cards(cards))) == text


@pytest.mark.parametrize("count", [4, 8])
def test_card_count_outside_five_to_seven_rejected(count):
    with pytest.raises(ValueError, match="5 to 7"):
        evaluate(list(range(count)))
