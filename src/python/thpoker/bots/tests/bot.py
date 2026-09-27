import json
import pytest
from   thpoker.bots.abstraction import AbstractAction
from   thpoker.bots.bot         import Bot, Decision, PRESETS
from   thpoker.game.engine      import (apply_action, is_terminal, new_hand,
                                        observation)
from   thpoker.game.rng         import Rng
from   thpoker.game.state       import (Action, ActionType, GameConfig,
                                        Observation)


class FixedBot(Bot):
    """Checks 30% and bets half pot 70%, whatever it holds."""

    name = "fixed"
    style = PRESETS["balanced"]

    def policy(self, view):
        return {AbstractAction.CHECK: 0.3, AbstractAction.BET_50: 0.7}, {"rule_triggered": "fixed"}


class FixedDraw:
    def __init__(self, value: float):
        self.value = value

    def random(self) -> float:
        return self.value


def checked_to_view():
    state, _ = new_hand(GameConfig(2), 1, 0, (10_000, 10_000))
    for action in (Action(ActionType.CALL), Action(ActionType.CHECK)):
        state, _ = apply_action(state, action)
    return observation(state, state.to_act)  # flop, pot 200, first to act


@pytest.mark.parametrize(
    "draw, chosen, action",
    [
        (0.0, AbstractAction.CHECK, Action(ActionType.CHECK)),
        (0.29, AbstractAction.CHECK, Action(ActionType.CHECK)),
        (0.31, AbstractAction.BET_50, Action(ActionType.BET, 100)),
        (0.999, AbstractAction.BET_50, Action(ActionType.BET, 100)),
    ],
)
def test_decide_samples_the_distribution_with_the_logged_draw(draw, chosen, action):
    decision = FixedBot().decide(checked_to_view(), FixedDraw(draw))
    assert (decision.random_value, decision.chosen, decision.action) == (draw, chosen, action)


def test_decision_round_trips_through_json():
    decision = FixedBot().decide(checked_to_view(), FixedDraw(0.5))
    assert Decision.from_dict(json.loads(json.dumps(decision.to_dict()))) == decision


def random_views(bot: Bot, seed: int, count: int) -> list[Observation]:
    """Observations from hands at 2-8 seats and 2bb to 200bb where `bot` plays every seat."""
    rng = Rng(seed)
    views: list[Observation] = []
    while len(views) < count:
        seats = 2 + rng.randbelow(7)
        stacks = tuple(200 + rng.randbelow(20_000) for _ in range(seats))
        state, _ = new_hand(GameConfig(seats), rng.randbelow(1 << 30), 0, stacks)
        while not is_terminal(state):
            view = observation(state, state.to_act)
            views.append(view)
            state, _ = apply_action(state, bot.decide(view, rng.derive(len(views))).action)
    return views
