"""Tier 1: hand-strength buckets with fixed thresholds per street, plus style and randomness.

Easy to read and to beat by design. Preflop hands are ranked by the Chen formula; the entered
range widens with fewer players left to act. Postflop hands fall into five buckets (monster,
strong, medium, draw, weak) that map to fixed action frequencies scaled by the style.
"""

from   dataclasses              import dataclass
from   functools                import lru_cache
from   thpoker.bots.abstraction import (AbstractAction, bet_sizes, call_price,
                                        fit_to_legal, legal_abstract_actions,
                                        players_behind, position_width,
                                        raises_this_street, sigmoid)
from   thpoker.bots.bot         import Policy, Style, TierBot
from   thpoker.game.cards       import COMBOS, rank_of, suit_of
from   thpoker.game.evaluator   import (HIGH_CARD, PAIR, QUADS, STRAIGHT,
                                        TRIPS, TWO_PAIR, category, evaluate)
from   thpoker.game.state       import Observation, Street
from   thpoker.odds             import PREFLOP_PERCENTILE
from   typing                   import Any

MONSTER, STRONG, MEDIUM, DRAW, WEAK = "MONSTER", "STRONG", "MEDIUM", "DRAW", "WEAK"

# The top 2% of hands by Chen score (JJ+ and AKs) never fold preflop.
_PREMIUM_PERCENTILE = 0.02


def _board_category(board: tuple[int, ...]) -> int:
    if len(board) == 5:
        return category(evaluate(board))
    counts = sorted((sum(1 for c in board if rank_of(c) == r) for r in range(13)), reverse=True)
    if counts[0] == 4:
        return QUADS
    if counts[0] == 3:
        return TRIPS
    if counts[0] == 2:
        return TWO_PAIR if counts[1] == 2 else PAIR
    return HIGH_CARD


def _has_draw(hole: tuple[int, int], board: tuple[int, ...]) -> bool:
    cards = list(hole) + list(board)
    for suit in range(4):
        if sum(1 for c in cards if suit_of(c) == suit) == 4 and any(suit_of(c) == suit for c in hole):
            return True
    ranks = {rank_of(c) for c in cards}
    hole_ranks = {rank_of(c) for c in hole}
    for low in range(1, 9):  # open-ended: both the card below and above the run exist
        window = set(range(low, low + 4))
        if window <= ranks and window & hole_ranks:
            return True
    return False


@lru_cache(maxsize=1 << 16)
def classify(hole: tuple[int, int], board: tuple[int, ...]) -> str:
    """Postflop strength bucket of the hole cards on this board. Cached: an EV walk asks for
    every hand on the same board at each of its nodes."""
    value = evaluate(list(hole) + list(board))
    cat = category(value)
    board_cat = _board_category(board)
    top_board = max(rank_of(c) for c in board)
    if cat >= STRAIGHT:
        if len(board) == 5:
            board_plays = value == evaluate(board)
        else:
            board_plays = cat == board_cat == QUADS  # quads on a four-card board
        return WEAK if board_plays else MONSTER
    if cat == TRIPS:
        return MONSTER if board_cat < TRIPS else WEAK
    if cat == TWO_PAIR and board_cat == HIGH_CARD:
        return MONSTER
    if cat in (PAIR, TWO_PAIR) and board_cat < cat:
        hole_ranks = [rank_of(c) for c in hole]
        if hole_ranks[0] == hole_ranks[1]:
            return STRONG if hole_ranks[0] > top_board else MEDIUM
        if top_board in hole_ranks:  # top pair, whichever hole card makes it
            kicker = max(r for r in hole_ranks if r != top_board)
            return STRONG if kicker >= 8 else MEDIUM
        return MEDIUM
    if len(board) < 5 and _has_draw(hole, board):
        return DRAW
    return WEAK


def _flatten(distribution: dict[AbstractAction, float], temperature: float) -> dict[AbstractAction, float]:
    # A power below 1 flattens toward uniform without giving mass to zero-probability actions,
    # so randomness never makes a bot fold a monster.
    powered = {a: p ** (1.0 - temperature) for a, p in distribution.items() if p > 0}
    total = sum(powered.values())
    return {a: p / total for a, p in powered.items()}


@dataclass
class _Bands:
    """The hand-independent part of a preflop decision: hands inside `raise_width` (as a share
    of all hands, strongest first) raise, the rest inside `play_width` call, scaled by
    `call_share` (limps are rarer than raises)."""

    raise_width: float
    play_width: float
    call_share: float
    aggressive: AbstractAction
    rationale: dict[str, Any]


class RuleBot(TierBot):
    def __init__(self, name: str, style: Style):
        self.name = name
        self.style = style

    def decisions(self, view: Observation, holes: list[int]) -> dict[int, Policy]:
        """Only the preflop percentile or the postflop bucket depends on the hand, so each is
        decided once."""
        legal = legal_abstract_actions(view)
        decided: dict[float | str, Policy] = {}
        result = {}
        if view.street == Street.PREFLOP:
            bands = self._preflop_bands(view)
            for hole in holes:
                percentile = PREFLOP_PERCENTILE[hole]
                if percentile not in decided:
                    weights = self._preflop_weights(view, bands, percentile)
                    rationale = {**bands.rationale, "hand_percentile": round(percentile, 4)}
                    decided[percentile] = self._policy(weights, rationale, view, legal)
                result[hole] = decided[percentile]
            return result
        for hole in holes:
            bucket = classify(COMBOS[hole], view.board)
            if bucket not in decided:
                decided[bucket] = self._policy(*self._postflop(view, bucket), view, legal)
            result[hole] = decided[bucket]
        return result

    def _policy(
        self,
        weights: dict[AbstractAction, float],
        rationale: dict[str, Any],
        view: Observation,
        legal: list[AbstractAction],
    ) -> Policy:
        rationale["style"] = self.style.name
        return _flatten(fit_to_legal(weights, view, legal), self.style.temperature), rationale

    def _preflop_bands(self, view: Observation) -> _Bands:
        style = self.style
        behind = players_behind(view)
        open_width = position_width(style.pfr_target, behind)
        enter_width = position_width(style.vpip_target, behind)
        raises = raises_this_street(view)
        if raises == 0:
            bands = _Bands(
                open_width,
                enter_width,
                1.0 - style.pfr_target / style.vpip_target,
                AbstractAction.OPEN,
                {"rule_triggered": "unopened"},
            )
        elif raises == 1:
            price = call_price(view)
            call_width = enter_width * (0.3 + 0.6 * style.call_down_tendency) * min(1.2, max(0.3, 1.4 - 2 * price))
            bands = _Bands(
                open_width * 0.2 * style.aggression,
                call_width,
                1.0,
                AbstractAction.RERAISE,
                {"rule_triggered": "facing raise"},
            )
        else:
            short = view.stacks[view.seat] < 3 * view.pot
            bands = _Bands(
                open_width * 0.08 * style.aggression,
                open_width * 0.25 * (0.5 + style.call_down_tendency),
                1.0,
                AbstractAction.ALL_IN if short else AbstractAction.RERAISE,
                {"rule_triggered": "facing re-raise"},
            )
        bands.rationale.update({"players_behind": behind, "open_width": round(open_width, 4)})
        return bands

    def _preflop_weights(self, view: Observation, bands: _Bands, percentile: float) -> dict[AbstractAction, float]:
        softness = 0.02 + 0.1 * self.style.temperature
        raise_weight = sigmoid((bands.raise_width - percentile) / softness)
        play = sigmoid((bands.play_width - percentile) / softness)
        call_weight = max(0.0, play - raise_weight) * bands.call_share
        fold_weight = 0.0 if percentile < _PREMIUM_PERCENTILE else max(0.0, 1.0 - raise_weight - call_weight)
        if view.current_bet == view.committed_this_street[view.seat]:
            return {AbstractAction.CHECK: fold_weight + call_weight, bands.aggressive: raise_weight}
        return {
            AbstractAction.FOLD: fold_weight,
            AbstractAction.CALL: call_weight,
            bands.aggressive: raise_weight,
        }

    def _postflop(self, view: Observation, bucket: str) -> tuple[dict[AbstractAction, float], dict[str, Any]]:
        style = self.style
        to_call = view.current_bet - view.committed_this_street[view.seat]
        large = style.sizing_preference
        sizes = bet_sizes(large)
        if to_call == 0:
            base = {
                MONSTER: 0.75,
                STRONG: 0.65,
                MEDIUM: 0.3,
                DRAW: 0.35 * style.bluff_multiplier,
                WEAK: 0.12 * style.bluff_multiplier,
            }[bucket]
            bet = min(0.95, base * style.aggression)
            weights = {AbstractAction.CHECK: 1.0 - bet}
            for size, share in sizes.items():
                weights[size] = bet * share
            return weights, {"rule_triggered": "checked to", "bucket": bucket}
        pressure = min(2.0, to_call / max(1, view.pot - to_call))
        required = call_price(view)
        if bucket == MONSTER:
            raise_share, fold = min(0.8, 0.45 * style.aggression), 0.0
        elif bucket == STRONG:
            raise_share = 0.12 * style.aggression
            fold = max(0.0, 0.05 + (pressure - 0.75) * 0.3 * style.fold_to_pressure)
        elif bucket == MEDIUM:
            raise_share = 0.03 * style.aggression
            fold = 0.35 + 0.35 * pressure * style.fold_to_pressure - 0.5 * style.call_down_tendency
        elif bucket == DRAW:
            draw_equity = 0.34 if view.street == Street.FLOP else 0.18
            raise_share = 0.12 * style.aggression * style.bluff_multiplier
            fold = 0.15 if draw_equity >= required else 0.75
        else:
            raise_share = 0.04 * style.aggression * style.bluff_multiplier
            fold = 0.85 + 0.1 * pressure - 0.4 * style.call_down_tendency
        fold = min(0.98, max(0.0 if bucket == MONSTER else 0.02, fold))
        raise_share = min(raise_share, 1.0 - fold)
        weights = {
            AbstractAction.FOLD: fold,
            AbstractAction.CALL: 1.0 - fold - raise_share,
            AbstractAction.BET_75: raise_share * (1 - large),
            AbstractAction.BET_100: raise_share * large,
        }
        rationale = {
            "rule_triggered": "facing bet",
            "bucket": bucket,
            "facing_pot_fraction": round(pressure, 3),
            "required_equity": round(required, 4),
        }
        return weights, rationale
