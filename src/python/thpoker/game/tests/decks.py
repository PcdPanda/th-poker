from   thpoker.game.cards       import parse_cards
from   thpoker.game.engine      import apply_action, new_hand
from   thpoker.game.state       import (Action, ActionType, GameConfig,
                                        GameState)

FOLD, CHECK, CALL = Action(ActionType.FOLD), Action(ActionType.CHECK), Action(ActionType.CALL)


def stacked_deck(button: int, num_seats: int, hole: dict[int, str], board: str) -> tuple[int, ...]:
    """A deck that deals `hole[seat]` to each seat and then `board`, following the engine's
    documented order: one card at a time starting left of the button."""
    seats = [(button + k) % num_seats for k in range(1, num_seats + 1)]
    order = [s for s in seats if s in hole]
    top = [parse_cards(hole[s])[0] for s in order] + [parse_cards(hole[s])[1] for s in order]
    top += parse_cards(board)
    return tuple(top + [c for c in range(52) if c not in top])


def play(state: GameState, *actions: Action) -> GameState:
    for action in actions:
        state, _ = apply_action(state, action)
    return state


def heads_up(
    button_hole: str,
    other_hole: str,
    board: str = "Qh9d4c3s2d",
    stacks: tuple[int, int] = (10_000, 10_000),
) -> GameState:
    """A heads-up hand with the button in seat 0, 100bb deep by default."""
    deck = stacked_deck(0, 2, {0: button_hole, 1: other_hole}, board)
    return new_hand(GameConfig(2), 1, 0, stacks, deck=deck)[0]
