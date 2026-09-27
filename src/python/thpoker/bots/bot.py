"""The bot interface, the logged record of each bot decision, and the style presets and
population panels.

Preset values are starting points; Phase 2 tunes them until each panel's measured VPIP/PFR
matches the published anchors (DESIGN.md Section 1.4, decision 7).
"""

from __future__ import annotations
from   abc                      import ABC, abstractmethod
from   dataclasses              import dataclass, replace
from   thpoker.bots.abstraction import AbstractAction, to_action
from   thpoker.game.cards       import COMBOS, COMBO_CLASS, combo_index
from   thpoker.game.rng         import Rng
from   thpoker.game.state       import Action, Observation, Street
from   typing                   import Any


@dataclass(frozen=True)
class Style:
    """`vpip_target` and `pfr_target` are fractions of hands entered and raised preflop at an
    average seat; the rest scale postflop handclasses around 1.0 (or 0...1 where noted)."""

    name: str
    vpip_target: float
    pfr_target: float
    aggression: float
    bluff_multiplier: float
    call_down_tendency: float  # 0...1
    fold_to_pressure: float  # 0...1
    sizing_preference: float  # <0: small; zero; >0: large bets
    temperature: float  # 0...1; probabilities are flattened to p ** (1 - temperature)


PRESETS = {
    style.name: style
    for style in (
        Style("nit", 0.12, 0.09, 0.3, 0.4, 0.2, 0.7, 0.4, 0.03),
        Style("tag", 0.22, 0.18, 1.2, 1.0, 0.4, 0.4, 0.0, 0.04),
        Style("lag", 0.28, 0.22, 1.8, 1.6, 0.6, 0.3, 0.2, 0.06),
        Style("calling_station", 0.41, 0.07, 0.3, 0.3, 0.9, 0.1, -0.3, 0.08),
        Style("maniac", 0.60, 0.48, 2.5, 2.5, 0.7, 0.1, 0.5, 0.10),
        Style("balanced", 0.24, 0.19, 1.1, 1.0, 0.5, 0.3, 0.1, 0.04),
    )
}

PANELS = {
    "online_micro": {
        "nit": 0.35,
        "tag": 0.30,
        "calling_station": 0.20,
        "lag": 0.10,
        "balanced": 0.04,
        "maniac": 0.01,
    },
    "live_home": {
        "calling_station": 0.35,
        "nit": 0.20,
        "tag": 0.15,
        "maniac": 0.15,
        "lag": 0.10,
        "balanced": 0.05,
    },
}


def draw_style(panel: str, rng: Rng) -> Style:
    """Draw a preset from a population panel by its weights. Raises KeyError for an unknown
    panel."""
    weights = PANELS[panel]
    draw = rng.random() * sum(weights.values())
    for name, weight in weights.items():
        draw -= weight
        if draw <= 0:
            return PRESETS[name]
    return PRESETS[next(reversed(weights))]


@dataclass(frozen=True)
class Decision:
    """Everything needed to audit a bot action: the mixed strategy, the uniform draw that
    sampled it, the chosen abstract action and its concrete amount, and why."""

    distribution: dict[AbstractAction, float]
    random_value: float
    chosen: AbstractAction
    action: Action
    rationale: str | dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "distribution": {a.value: p for a, p in self.distribution.items()},
            "random_value": self.random_value,
            "chosen": self.chosen.value,
            "action": self.action.to_dict(),
            "rationale": self.rationale,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Decision:
        return cls(
            {AbstractAction(a): p for a, p in data["distribution"].items()},
            data["random_value"],
            AbstractAction(data["chosen"]),
            Action.from_dict(data["action"]),
            data["rationale"],
        )


Policy = tuple[dict[AbstractAction, float], dict[str, Any]]  # A distribution and its rationale


class Bot(ABC):
    """A bot's probabilities are a pure function of its observation and parameters, so the
    analysis can replay its policy for hypothetical hole cards. All randomness is outside."""

    name: str
    style: Style

    @abstractmethod
    def policy(self, view: Observation) -> Policy:
        """The distribution over legal abstract actions (summing to 1) and the rationale."""

    def action_probabilities(self, view: Observation) -> dict[AbstractAction, float]:
        return self.policy(view)[0]

    def decisions(self, view: Observation, holes: list[int]) -> dict[int, Policy]:
        """The distribution and rationale this bot would use holding each combo in `holes`. By
        default, for range tracking, preflop query cut decisions by hand class, so one
        decision per class serves all its combos. Bots override this when per-hole can serve
        every hand."""
        result: dict[int, Policy] = {}
        by_class: dict[int, Policy] = {}
        for hole in holes:
            klass = COMBO_CLASS[hole] if view.street == Street.PREFLOP else None
            if klass is not None and klass in by_class:
                result[hole] = by_class[klass]
                continue
            cards = list(view.hole_cards)
            cards[view.seat] = COMBOS[hole]
            result[hole] = self.policy(replace(view, hole_cards=tuple(cards)))
            if klass is not None:
                by_class[klass] = result[hole]
        return result

    def policies(self, view: Observation, holes: list[int]) -> dict[int, dict[AbstractAction, float]]:
        """`decisions` without the rationales; hands decided alike share one distribution."""
        return {hole: decided[0] for hole, decided in self.decisions(view, holes).items()}

    def decide(self, view: Observation, rng: Rng) -> Decision:
        distribution, rationale = self.policy(view)
        draw = rng.random()
        cumulative = 0.0
        chosen = None
        for abstract, probability in distribution.items():
            cumulative += probability
            if draw < cumulative:
                chosen = abstract
                break
        if chosen is None:  # floating-point sum fell just short of the draw
            chosen = next(a for a, p in reversed(distribution.items()) if p > 0)
        return Decision(distribution, draw, chosen, to_action(chosen, view), rationale)


class TierBot(Bot):
    """A bot tier whose `decisions` serve every hand in one pass; its own hand is one of them."""

    def policy(self, view: Observation) -> Policy:
        hero = combo_index(*view.my_cards)
        return self.decisions(view, [hero])[hero]

    @abstractmethod
    def decisions(self, view: Observation, holes: list[int]) -> dict[int, Policy]:
        """See `Bot.decisions`."""
