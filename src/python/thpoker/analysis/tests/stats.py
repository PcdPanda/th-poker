from   dataclasses              import replace
import io
import pytest
from   thpoker.analysis.review  import HandRating, MoveRating, review_hand
import thpoker.analysis.stats
from   thpoker.analysis.stats   import (DecisionRecord, HAND_COLUMNS,
                                        MIN_HANDS, compare, facing_bucket,
                                        hand_rows, loss_by_tag, patterns,
                                        progress, record, tags,
                                        write_hands_csv)
from   thpoker.analysis.tests.review \
                                import BetsTheRiver
from   thpoker.analysis.tests.tracking \
                                import Calls, REFERENCE
from   thpoker.bots.abstraction import AbstractAction, legal_abstract_actions
from   thpoker.bots.bot         import Bot, PRESETS
from   thpoker.game.engine      import new_hand
from   thpoker.game.state       import (Action, ActionType, GameConfig, AnteType,
                                        Observation, Street)
from   thpoker.game.tests.decks import (CALL, CHECK, FOLD, heads_up, play,
                                        stacked_deck)
from   thpoker.odds             import hand_rank
from   typing                   import Any


def decision(
    action: str,
    loss: float,
    street: str = "river",
    facing: str = "large",
    hand: str = "marginal",
    **extra,
) -> DecisionRecord:
    spot = {"mode": "cash", "street": street, "facing": facing, "hand": hand, "position": "BB"}
    return DecisionRecord(
        "s",
        "hand-1",
        0,
        spot,
        loss,
        0.0,
        "close",
        action,
        extra.get("defense"),
        extra.get("bet_fraction"),
    )


def test_river_defense_is_compared_with_the_minimum_defense_frequency():
    # Facing pot-size river bets (defend 50%), the user folds 8 of 10 times.
    records = [decision("fold", 1.0, defense=0.5)] * 8 + [decision("call", 0.0, defense=0.5)] * 2
    found = patterns(records)
    assert found[0].startswith("Facing river bets over 40% of the pot you continue 20% of the time")
    assert "minimum defense of 50%" in found[0] and "30 points less than the theory" in found[0]


def test_river_bluffs_are_compared_with_a_balanced_share():
    # Pot-size bets are balanced with a third bluffs; none of these are.
    records = [decision("bet", 0.0, facing="unopened", hand="value", bet_fraction=1.0)] * 6
    assert any("0% bluffs; a balanced range at your sizes has 33%" in line for line in patterns(records))


def test_losses_are_ranked_by_spot_with_a_minimum_sample():
    records = [decision("call", 2.0, street="turn")] * 5 + [decision("call", 0.1, street="flop")] * 5
    records += [decision("call", 9.0, street="preflop")] * 4  # too few to report
    streets = [row for row in loss_by_tag(records) if row[0] == "street"]
    assert streets[0][:3] == ("street", "turn", 5) and streets[0][3] == pytest.approx(2.0)
    assert [row[1] for row in streets] == ["turn", "flop"]


def test_records_round_trip_through_the_log():
    item = decision("bet", 0.4, facing="unopened", bet_fraction=0.75)
    assert DecisionRecord.from_dict({**item.to_dict(), "type": "decision", "schema_version": 1}) == item


def test_a_bet_of_any_size_is_recorded_as_a_share_of_the_pot():
    # A 120 bet into 200 is not an abstract size, yet it is a bet of 60% of the pot.
    state = heads_up("8c8d", "KsKc")
    hand = play(state, CALL, CHECK, *[CHECK] * 4, Action(ActionType.BET, 120), CALL)
    river_bet = review_hand(hand, 1, {0: Calls()}, REFERENCE).decisions[-1]
    assert record(river_bet, "s", hand.hand_id).bet_fraction == pytest.approx(0.6)


class OpensThenFolds(Bot):
    """Opens first in and calls a 3-bet of its open; folds to anything else and to bets later."""

    name, style = "opener", PRESETS["balanced"]

    def policy(self, view: Observation) -> tuple[dict[AbstractAction, float], dict[str, Any]]:
        legal = legal_abstract_actions(view)
        raisers = [e.seat for e in view.history if e.action.type in (ActionType.BET, ActionType.RAISE)]
        if view.street == Street.PREFLOP and not raisers:
            return {AbstractAction.OPEN: 1.0}, {}
        if view.street == Street.PREFLOP and raisers[0] == view.seat:
            return {AbstractAction.CALL: 1.0}, {}
        return {AbstractAction.CHECK if AbstractAction.CHECK in legal else AbstractAction.FOLD: 1.0}, {}


@pytest.mark.parametrize(
    "facing, bucket",
    [(None, "unopened"), (0.33, "small"), (0.5, "medium"), (1.0, "large"), (2.0, "overbet")],
)
def test_bets_faced_fall_into_size_buckets(facing, bucket):
    assert facing_bucket(facing) == bucket


def test_a_river_call_in_a_limped_pot_is_tagged_by_its_spot():
    state = heads_up("8c8d", "JhTh", "AhKhQh2c3d")
    hand = play(state, CALL, CHECK, *[CHECK] * 4, CHECK, Action(ActionType.BET, 200), CALL)
    river_call = review_hand(hand, 1, {0: BetsTheRiver()}, REFERENCE).decisions[-1]
    found = tags(river_call, tournament=True, paid_places=1)
    assert found["street"] == "river" and found["facing"] == "large"  # a pot-size bet
    assert (found["pot_type"], found["role"], found["position"], found["in_position"]) == (
        "limped",
        "none",
        "BB",
        "OOP",
    )
    assert found["hand"] == "value" and found["texture"] == "wet"
    assert (found["stack"], found["stage"]) == ("deep", "heads-up")


def test_preflop_spots_are_tagged_by_the_raise_faced_and_stacks_by_the_start():
    deck = stacked_deck(0, 3, {0: "AsKs", 1: "QdQh", 2: "7h2c"}, "Js9d4c3s2d")
    state, _ = new_hand(GameConfig(3), 1, 0, (6_000, 6_000, 6_000), deck=deck)
    # The button opens, the small blind 3-bets, the big blind folds, the button calls; the
    # small blind then bets on the flop.
    hand = play(
        state,
        Action(ActionType.RAISE, 250),
        Action(ActionType.RAISE, 1_200),
        Action(ActionType.FOLD),
        CALL,
    )
    hand = play(hand, Action(ActionType.BET, 4_000), Action(ActionType.FOLD))
    decisions = review_hand(hand, 1, {0: OpensThenFolds(), 2: OpensThenFolds()}, REFERENCE).decisions
    facing_open, flop_bet = decisions
    found = tags(facing_open, tournament=False)
    assert found["facing"] == "open"
    assert found["hand"] == "premium"  # queens are in the top 10% of the small blind's range
    # 48bb are left by the flop, but the stack tag reads the 60bb the hand started with.
    assert flop_bet.situation.effective_bb == pytest.approx(48.0)
    assert tags(flop_bet, tournament=True)["stack"] == "deep"


def hand_decision(hand_id: str, loss: float, mode: str = "cash") -> DecisionRecord:
    return DecisionRecord("s", hand_id, 0, {"mode": mode}, loss, 0.0, "close", "call", None, None)


def test_recent_hands_weigh_more(monkeypatch):
    # With a half-life of one hand, an old 10bb loss counts half as much as a new clean hand.
    monkeypatch.setattr(thpoker.analysis.stats, "HALF_LIFE_HANDS", 1)
    result = progress([hand_decision("hand-1", 6.0), hand_decision("hand-1", 4.0), hand_decision("hand-2", 0.0)])
    assert result is not None and (result.hands, result.decisions) == (2, 3)
    assert result.loss_per_100 == pytest.approx(100 * (0.5 * 10 + 1 * 0) / 1.5)


def test_the_comparison_allows_for_the_error_of_both_measures(monkeypatch):
    monkeypatch.setattr(thpoker.analysis.stats, "BOT_LOSSES", {3: (100.0, 20.0), 2: (200.0, 5.0), 1: (210.0, 5.0)})
    assert compare(50, 1, 500) == "better than the Tier 1, 2 and 3 bots"
    # 80 +- 1 is clearly under 100 alone, but not once Tier 3's own error of 20 counts.
    assert compare(80, 1, 500) == "better than the Tier 1 and 2 bots, level with the Tier 3 bots"
    assert compare(300, 10, 500) == "behind the Tier 1, 2 and 3 bots"
    assert compare(195, 1, 500) == "better than the Tier 1 bots, behind the Tier 3 bots, level with the Tier 2 bots"
    assert compare(100, 200, 500).startswith("not yet told apart")


def test_a_few_hands_give_no_verdict():
    assert compare(0, 0, MIN_HANDS - 1).startswith("too few hands")
    result = progress([hand_decision("hand-1", 0.0)])
    assert result is not None and result.comparison.startswith("too few hands")


def test_only_cash_decisions_count():
    assert progress([hand_decision("hand-1", 0.01, mode="tournament")]) is None


def test_hand_rows_count_the_chips_put_in_and_join_the_review():
    deck = stacked_deck(0, 2, {0: "AsKd", 1: "QhQc"}, "2c3d7h8s9c")
    raise_300 = Action(ActionType.RAISE, 300)
    shown_down = play(
        new_hand(GameConfig(2), 1, 0, (10_000, 10_000), hand_id="hand-7", deck=deck)[0],
        raise_300,
        CALL,
        *[CHECK] * 6,
    )
    # Blake folds to the raise: 200 of the user's 300 come back, so 100 went in and won 200.
    folded_to = play(
        new_hand(GameConfig(2), 1, 0, (10_000, 10_000), hand_id="hand-8", deck=deck)[0],
        raise_300,
        FOLD,
    )
    reviewed = DecisionRecord("s", "hand-7", 0, {}, 1.5, 0.2, "mistake", "raise", None, 0.5)
    other_session = replace(reviewed, session="t", hand_id="hand-8")
    rows = hand_rows(
        [shown_down, folded_to],
        0,
        1,
        "s",
        "2026-01-02",
        "cash",
        [reviewed, other_session],
        {"hand-7": "You raise to 300."},
        # The raise was worth 1.5 big blinds at stake and rated 0.6; the call 6 and rated 1.
        {"hand-7": HandRating(0.4321, (MoveRating(0, 0.6, 1.5, False), MoveRating(2, 1.0, 6.0, True)))},
    )
    picked = [
        {
            k: row[k]
            for k in (
                "position",
                "put_in",
                "result",
                "showdown",
                "win_chance",
                "rating",
                "mistakes",
                "loss_bb",
            )
        }
        for row in rows
    ]
    assert picked == [
        {
            "position": "BTN",
            "put_in": 300,
            "result": -300,
            "showdown": "yes",
            "win_chance": 0.432,
            "rating": round((0.6 * 1.5 + 6.0) / 7.5, 2),
            "mistakes": 1,
            "loss_bb": 1.5,
        },
        {
            "position": "BTN",
            "put_in": 100,
            "result": 100,
            "showdown": "no",
            "win_chance": "",
            "rating": "",
            "mistakes": "",
            "loss_bb": "",
        },
    ]
    hole = shown_down.hole_cards[0]
    assert hole is not None and rows[0]["hand_rank"] == round(hand_rank(hole)[1], 4)
    stream = io.StringIO()
    write_hands_csv([{**rows[0], "page_only": "not written"}], stream)
    assert stream.getvalue().splitlines() == [
        ",".join(HAND_COLUMNS),
        f"s,2026-01-02,cash,2,100,0,none,7,BTN,As Kd,2c 3d 7h 8s 9c,300,-300,-3.0,yes,{rows[0]['hand_rank']},0.432,0.92,1,1,1.5,,You raise to 300.",
    ]
    # Blinds 200/400 with a 300 big-blind ante, kept at 100 internal units per chip shown.
    config = GameConfig(2, 20_000, 40_000, 30_000, AnteType.BIG_BLIND_ANTE)
    ante_hand = play(
        new_hand(config, 1, 0, (1_000_000, 1_000_000), hand_id="hand-9", deck=deck)[0], FOLD
    )
    (ante_row,) = hand_rows([ante_hand], 0, 100, "s", "2026-01-02", "tournament", [], {}, {})
    assert (ante_row["big_blind"], ante_row["ante"], ante_row["ante_type"]) == (
        400,
        300,
        "big_blind_ante",
    )
