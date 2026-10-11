import pytest

from   thpoker.bots.abstraction import (AbstractAction, bet_fractions,
                                        defense_share, last_bet,
                                        legal_abstract_actions, to_action,
                                        usual_raises)
from   thpoker.bots.bot         import PRESETS
from   thpoker.bots.range_bot   import RangeBot
from   thpoker.game.engine      import (apply_action, is_terminal, new_hand,
                                        observation)
from   thpoker.game.rng         import Rng
from   thpoker.game.rules       import check_action
from   thpoker.game.state       import Action, ActionType, AnteType, GameConfig
from   thpoker.game.tests.decks import CALL, CHECK, FOLD, play


def test_open_is_two_and_a_half_big_blinds_plus_one_per_limper():
    state, _ = new_hand(GameConfig(6), 1, 0, (10_000,) * 6)
    assert to_action(AbstractAction.OPEN, observation(state, state.to_act)) == Action(ActionType.RAISE, 250)
    state, _ = apply_action(state, CALL)  # one limper
    assert to_action(AbstractAction.OPEN, observation(state, state.to_act)) == Action(ActionType.RAISE, 350)


def test_reraise_is_three_times_in_position_and_four_times_out_of_position():
    # 3 seats, button 0: seat 0 (button) opens to 250; the blinds are out of position.
    state, _ = new_hand(GameConfig(3), 1, 0, (10_000,) * 3)
    state, _ = apply_action(state, Action(ActionType.RAISE, 250))
    assert to_action(AbstractAction.RERAISE, observation(state, 1)) == Action(ActionType.RAISE, 1_000)
    # 4 seats, button 0: UTG (seat 3) opens and the button re-raises in position.
    state, _ = new_hand(GameConfig(4), 1, 0, (10_000,) * 4)
    state, _ = apply_action(state, Action(ActionType.RAISE, 250))  # seat 3 (UTG) opens
    assert to_action(AbstractAction.RERAISE, observation(state, 0)) == Action(ActionType.RAISE, 750)


def test_pot_fraction_bet_and_all_in_when_little_would_remain():
    state, _ = new_hand(GameConfig(2), 1, 0, (10_000, 10_000))
    state, _ = apply_action(state, CALL)
    state, _ = apply_action(state, Action(ActionType.CHECK))  # flop, pot 200
    view = observation(state, state.to_act)
    assert to_action(AbstractAction.BET_50, view) == Action(ActionType.BET, 100)
    short, _ = new_hand(GameConfig(2), 1, 0, (10_000, 350))
    short, _ = apply_action(short, CALL)
    short, _ = apply_action(short, Action(ActionType.CHECK))
    # BB has 250 left: a pot-size bet of 200 would leave 50, under a quarter of the 400 pot.
    assert to_action(AbstractAction.BET_100, observation(short, 1)) == Action(ActionType.BET, 250)


def test_unavailable_abstract_action_rejected():
    state, _ = new_hand(GameConfig(2), 1, 0, (10_000, 10_000))
    with pytest.raises(ValueError, match="not available"):
        to_action(AbstractAction.CHECK, observation(state, 0))


def test_every_available_abstract_action_maps_to_a_legal_action():
    rng = Rng(99)
    for hand_number in range(150):
        seats = 2 + rng.randbelow(7)
        stacks = tuple(50 + rng.randbelow(5_000) for _ in range(seats))
        state, _ = new_hand(GameConfig(seats), hand_number, 0, stacks)
        while not is_terminal(state):
            view = observation(state, state.to_act)
            available = legal_abstract_actions(view)
            for abstract in available:
                check_action(state, to_action(abstract, view))
            state, _ = apply_action(state, to_action(available[rng.randbelow(len(available))], view))


def test_effective_stacks_of_fifteen_big_blinds_or_less_only_shove_preflop():
    shove_or_fold = [AbstractAction.FOLD, AbstractAction.CALL, AbstractAction.ALL_IN]
    short, _ = new_hand(GameConfig(3), 1, 0, (1_500, 10_000, 10_000))  # UTG has 15bb
    assert legal_abstract_actions(observation(short, 0)) == shove_or_fold
    covered, _ = new_hand(GameConfig(3), 1, 0, (10_000, 1_000, 1_000))  # opponents have 10bb
    assert legal_abstract_actions(observation(covered, 0)) == shove_or_fold
    deep, _ = new_hand(GameConfig(3), 1, 0, (1_600, 10_000, 10_000))  # 16bb
    assert AbstractAction.OPEN in legal_abstract_actions(observation(deep, 0))
    # A 100bb button behind a folded 100bb seat plays at the 10bb blinds' depth.
    folded = play(new_hand(GameConfig(4), 1, 0, (10_000, 1_000, 1_000, 10_000))[0], FOLD)
    assert legal_abstract_actions(observation(folded, 0)) == shove_or_fold


def test_usual_raise_sizes_are_the_bots_own_sizes():
    # Heads-up: a 2.5bb open, a 4x 3-bet out of position, then a shove against a usual 4x 4-bet.
    heads_up = new_hand(GameConfig(2), 1, 0, (10_000, 10_000))[0]
    state = play(heads_up, Action(ActionType.RAISE, 250), Action(ActionType.RAISE, 1_000))
    state = play(state, Action(ActionType.RAISE, 10_000))
    assert usual_raises(observation(state, 1)) == [
        (0, 250, 250),
        (1, 1_000, 1_000),
        (0, 10_000, 4_000),
    ]
    # Six-handed: a limper adds 1bb to the open and a caller adds 1x to the 3-bet.
    six = new_hand(GameConfig(6), 1, 0, (10_000,) * 6)[0]
    state = play(six, CALL, Action(ActionType.RAISE, 350), CALL, Action(ActionType.RAISE, 1_750))
    assert usual_raises(observation(state, 1)) == [(4, 350, 350), (0, 1_750, 1_750)]


def test_the_last_bet_counts_short_blinds_calls_for_less_and_antes():
    # A 40-chip big blind, a limp and a raise to 400: 190 was in the pot before the raise.
    state, _ = new_hand(GameConfig(3), 1, 0, (10_000, 10_000, 40))
    raised = play(state, Action(ActionType.CALL), Action(ActionType.RAISE, 400))
    assert last_bet(observation(raised, 0)) == (190, 350)
    # A 300 pot, a bet of 2,000, an all-in call of 900 and a raise to 6,000: 3,200 before it.
    state, _ = new_hand(GameConfig(3), 1, 0, (1_000, 20_000, 20_000))
    flop = play(state, Action(ActionType.CALL), Action(ActionType.CALL), Action(ActionType.CHECK))
    raised = play(
        flop,
        Action(ActionType.CHECK),
        Action(ActionType.BET, 2_000),
        Action(ActionType.CALL),
        Action(ActionType.RAISE, 6_000),
    )
    assert last_bet(observation(raised, 2)) == (3_200, 6_000)
    # Antes of 10 each, a 50 small blind and a big blind short at 50, then an open to 300.
    config = GameConfig(3, 50, 100, 10, AnteType.PER_PLAYER)
    state, _ = new_hand(config, 1, 0, (10_000, 10_000, 60))
    opened = play(state, Action(ActionType.RAISE, 300))
    assert last_bet(observation(opened, 1)) == (130, 300)


def test_no_defense_is_owed_once_someone_has_called():
    state, _ = new_hand(GameConfig(3), 1, 0, (10_000,) * 3)
    flop = play(state, Action(ActionType.CALL), Action(ActionType.CALL), Action(ActionType.CHECK))
    called = play(flop, Action(ActionType.BET, 300), Action(ActionType.CALL))
    assert defense_share(observation(called, 0)) is None
    facing = play(flop, Action(ActionType.BET, 300))
    assert defense_share(observation(facing, 2)) == pytest.approx(1 - 0.5**0.5)  # pot bet, two to act


def test_bet_fractions_count_a_short_all_in_call_at_what_it_could_put_in():
    # The short small blind calls a 1,000 flop bet all in for 400: the turn's reads must see a
    # 600 flop pot, not one that the full call would have shrunk to nothing.
    state, _ = new_hand(GameConfig(3), 1, 0, (10_000, 600, 10_000))
    flop = play(state, Action(ActionType.RAISE, 200), CALL, CALL)
    turn = play(flop, CHECK, Action(ActionType.BET, 1_000), CALL, CALL)
    view = observation(turn, turn.to_act)
    assert bet_fractions(view) == {len(flop.history) + 1: 1_000 / 600}
    assert RangeBot("hard", PRESETS["balanced"]).policy(view)
