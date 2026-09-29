import functools
from   thpoker.analysis.ev      import OptionValue
from   thpoker.analysis.review  import DecisionReview, Situation, Thresholds
from   thpoker.game.engine      import new_hand
from   thpoker.game.state       import (Action, ActionType, AnteType, Event,
                                        GameConfig, Street)
from   thpoker.game.tests.decks import CALL, CHECK, FOLD, play, stacked_deck
from   thpoker.odds             import Equity
from   thpoker.text             import (decision_headline, decision_summary,
                                        format_chips, hand_history_text,
                                        narrate, plain_decision_text)

LABELS = {0: "You", 1: "Blake"}
RAISE_300 = Action(ActionType.RAISE, 300)


def queens_against_ace_king():
    """Heads-up, the user on the button with AsKd against QhQc, blinds 50/100."""
    deck = stacked_deck(0, 2, {0: "AsKd", 1: "QhQc"}, "2c3d7h8s9c")
    return new_hand(GameConfig(2), 1, 0, (10_000, 10_000), hand_id="hand-7", deck=deck)[0]


def test_the_hand_history_tells_every_action_pot_and_card_the_user_saw():
    hand = play(queens_against_ace_king(), RAISE_300, CALL, *[CHECK] * 6)
    assert hand_history_text(hand, 0, LABELS, 1, "cash game", {}) == [
        "No-limit Texas Hold'em cash game, 2 players, blinds 50/100. Hand 7.",
        "Seats in the order they act before the flop, with their chips at the start of the hand:",
        "  Dealer (BTN): You, 10,000, posts 50, your cards A♠ K♦",
        "  Big blind (BB): Blake, 10,000, posts 100",
        "Before the flop:",
        "  You raise to 300, Blake calls 200.",
        "Flop 2♣ 3♦ 7♥ (pot 600):",
        "  Blake checks, You check.",
        "Turn 8♠ (pot 600):",
        "  Blake checks, You check.",
        "River 9♣ (pot 600):",
        "  Blake checks, You check.",
        "You show A♠ K♦: high card, ace high.",
        "Blake shows Q♥ Q♣: pair of queens.",
        "Blake wins 600.",
    ]


def test_mid_hand_the_history_hides_the_opponents_cards_and_says_whose_turn_it_is():
    hand = play(queens_against_ace_king(), RAISE_300, CALL, CHECK)
    text = hand_history_text(hand, 0, LABELS, 1, "cash game", {})
    assert text[-2:] == ["  Blake checks.", "The hand is still being played, and it is your turn."]
    assert not any("Q♥" in line or "Q♣" in line for line in text)


def test_a_bot_that_shows_without_a_showdown_is_told():
    hand = play(queens_against_ace_king(), RAISE_300, Action(ActionType.RAISE, 900), FOLD)
    text = hand_history_text(hand, 0, LABELS, 1, "cash game", {1: hand.hole_cards[1]})
    assert text[-3:] == [
        "  You raise to 300, Blake raises to 900, You fold.",
        "Blake shows Q♥ Q♣ without being called.",
        "Blake wins 600.",
    ]


def test_the_plain_pot_line_says_who_wins_while_the_command_line_keeps_its_own():
    split = Event("PotAwarded", {"amount": 600, "eligible": [0, 1], "winners": [0, 1], "shares": [300, 300]})
    chips = functools.partial(format_chips, scale=1)
    assert narrate(split, LABELS.__getitem__, chips, 0, plain=True) == ["  You win 300, Blake wins 300"]
    assert narrate(split, LABELS.__getitem__, chips, 0) == ["  Pot 600 -> You 300, Blake 300"]


def test_a_side_pot_is_named_and_a_big_blind_ante_adds_up():
    # Three-handed, the user shoves 2,000 with aces; Blake raises to 6,000 with kings and Casey
    # calls with queens: the user wins the 6,000 main pot, Blake the 8,000 side pot.
    deck = stacked_deck(0, 3, {0: "AsAd", 1: "KsKd", 2: "QsQd"}, "2c3d7h8s9c")
    start = new_hand(GameConfig(3), 1, 0, (2_000, 10_000, 10_000), hand_id="hand-3", deck=deck)[0]
    hand = play(start, Action(ActionType.RAISE, 2_000), Action(ActionType.RAISE, 6_000), CALL, *[CHECK] * 6)
    labels = {**LABELS, 2: "Casey"}
    assert hand_history_text(hand, 0, labels, 1, "cash game", {})[-2:] == [
        "Blake wins 8,000 (side pot).",
        "You win 6,000.",
    ]
    # The big blind's ante is dead money in the pot, and part of what the big blind posts.
    config = GameConfig(2, ante=100, ante_type=AnteType.BIG_BLIND_ANTE)
    deck = stacked_deck(0, 2, {0: "AsKd", 1: "QhQc"}, "2c3d7h8s9c")
    anted = play(new_hand(config, 1, 0, (10_000, 10_000), hand_id="hand-4", deck=deck)[0], RAISE_300, CALL)
    text = hand_history_text(anted, 0, LABELS, 1, "tournament", {})
    assert text[0].endswith("blinds 50/100, big blind ante 100. Hand 4.")
    assert text[3:7] == [
        "  Big blind (BB): Blake, 10,000, posts 200",
        "Before the flop:",
        "  You raise to 300, Blake calls 200.",
        "Flop 2♣ 3♦ 7♥ (pot 700):",
    ]


def test_a_hand_over_before_any_action_still_shows_its_board():
    deck = stacked_deck(0, 2, {0: "AsKd", 1: "QhQc"}, "2c3d7h8s9c")
    hand = new_hand(GameConfig(2), 1, 0, (30, 10_000), hand_id="hand-5", deck=deck)[0]
    text = hand_history_text(hand, 0, LABELS, 1, "tournament", {})
    assert "  Dealer (BTN): You, 30, posts 30, your cards A♠ K♦" in text
    assert "  Big blind (BB): Blake, 10,000, posts 100" in text  # before 70 comes back uncalled
    assert "The rest of the board, with no more betting: 2♣ 3♦ 7♥ 8♠ 9♣." in text


def fold_to_a_bet(tournament: bool) -> DecisionReview:
    """The user folded to a 2 big blind bet when calling was worth 1.5 big blinds (or 3% more of
    the prize pool): a mistake of 1.5 big blinds."""
    options = [
        OptionValue(None, Action(ActionType.FOLD), 0.0, 0.0, True, 0.30, 0.0),
        OptionValue(None, Action(ActionType.CALL), 1.5, 0.1, False, 0.33, 0.002),
        OptionValue(None, Action(ActionType.RAISE, 900), 0.25, 0.1, False, 0.31, 0.002),
    ]
    spot = Situation(
        Street.FLOP,
        2,
        "BTN",
        True,
        "single-raised",
        6.0,
        2.0,
        95.0,
        15.8,
        None,
        2,
        "caller",
        0.5,
        100.0,
    )
    return DecisionReview(
        3,
        (0, 1),
        (10, 20, 30),
        spot,
        {},
        Equity(0.4, 0.01, False),
        0.8,
        Thresholds(0.25, 0.6, {}),
        Action(ActionType.FOLD),
        None,
        options,
        options,
        tournament,
        0.01 if tournament else None,
    )


def test_coach_amounts_are_chips_on_the_table_or_shares_of_the_prize_pool():
    cash, tournament = fold_to_a_bet(False), fold_to_a_bet(True)
    # Blinds 50/100 shown as they are (scale 1); blinds 0.5/1 are kept as 50/100 internally (scale 100).
    assert decision_headline(cash, 1, 100, {}) == (
        "Costly. Best option: call; your fold gave up about 150 chips on average."
    )
    assert decision_headline(cash, 100, 100, {}).endswith("about 1.50 chips on average.")
    assert decision_headline(tournament, 1, 100, {}).endswith("about 3.00% of the prize pool on average.")
    assert decision_summary(cash, 1, 100)[1] == "Calling costs 200: it pays if you win at least 25% of the time."
    options = plain_decision_text(1, cash, {}, 1, 100, {Action(ActionType.CALL): 0.75})[-3:]
    assert options == [
        "  fold: 0 | 0 | 0% (your choice)",
        "  call: +150 ± 10 | +150 ± 10 | 75% (best)",
        "  raise to 900: +25 ± 10 | +25 ± 10 | 0%",
    ]
