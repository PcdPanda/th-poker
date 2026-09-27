"""Predict-then-reveal (DESIGN.md Section 7.6): the user estimates before the numbers are shown,
and calibration is tracked per kind of estimate.

Estimates of shares (equity against the range, equity needed to call) are scored by their error;
naming the best option is scored as a hit or a miss against the reference opponent's best.
"""

from   dataclasses              import dataclass, replace

from   thpoker.analysis.drills  import MIXED, RIGHT, WRONG
from   thpoker.analysis.review  import DecisionReview, best_option
from   thpoker.analysis.stats   import DecisionRecord
from   thpoker.game.state       import Action
from   thpoker.storage          import Record

EQUITY, REQUIRED_EQUITY, BEST_OPTION = "equity", "required_equity", "best_option"
RECENT = 20  # estimates in the recent window of a calibration summary


@dataclass(frozen=True)
class Estimate(Record):
    """One answer: 'guess' and 'actual' are shares in [0, 1], or 1 and 0 for a named option
    that was or was not the best."""

    kind: str
    guess: float
    actual: float
    hand_id: str


@dataclass(frozen=True)
class Calibration:
    """Per kind of estimate. For shares: mean absolute error and bias (positive means guesses
    run high), overall and over the last `RECENT` estimates. For the best option: hit rates."""

    kind: str
    count: int
    error: float
    recent_error: float
    bias: float | None


def questions(review: DecisionReview) -> list[str]:
    """The kinds of estimate a decision asks for: required equity only when facing a bet."""
    kinds = [EQUITY]
    if review.thresholds.required_equity is not None:
        kinds.append(REQUIRED_EQUITY)
    return kinds + [BEST_OPTION]


def best_action(review: DecisionReview) -> Action:
    """The reference opponent's best option (ICM equity in tournaments, chips otherwise)."""
    return best_option(review.reference, review.tournament).action


def score(review: DecisionReview, kind: str, guess: float | Action, hand_id: str) -> Estimate:
    """An estimate from the user's answer: a share for share kinds, an action for the option."""
    if kind == BEST_OPTION:
        assert isinstance(guess, Action)
        return Estimate(kind, 1.0, 1.0 if guess == best_action(review) else 0.0, hand_id)
    assert isinstance(guess, float)
    actual = review.equity.value if kind == EQUITY else review.thresholds.required_equity
    assert actual is not None
    return Estimate(kind, guess, actual, hand_id)


def grade_option(review: DecisionReview, chosen: Action) -> str:
    """A drill grade for naming 'chosen' as the best option: right for the reference's best,
    mixed for another option that would be no mistake, wrong for a mistake."""
    verdict = replace(review, chosen=chosen).verdict()
    return RIGHT if verdict == "best" else MIXED if verdict == "close" else WRONG


def calibration(estimates: list[Estimate]) -> list[Calibration]:
    """A summary per kind of estimate, in the order kinds first appear."""
    kinds = list(dict.fromkeys(e.kind for e in estimates))
    result = []
    for kind in kinds:
        chosen = [e for e in estimates if e.kind == kind]
        option = kind == BEST_OPTION
        # A named option stores its hits; a share its absolute error.
        scores = [e.actual if option else abs(e.guess - e.actual) for e in chosen]
        recent = scores[-RECENT:]
        bias = None if option else sum(e.guess - e.actual for e in chosen) / len(chosen)
        result.append(Calibration(kind, len(chosen), sum(scores) / len(scores), sum(recent) / len(recent), bias))
    return result


MISTAKE = "mistake"


def mistake_keys(records: list[DecisionRecord]) -> list[str]:
    """Drill keys for the recorded mistakes, oldest first: 'mistake/session/hand_id/index'."""
    return [f"{MISTAKE}/{r.session}/{r.hand_id}/{r.index}" for r in records if r.verdict == "mistake"]


def parse_mistake_key(key: str) -> tuple[str, str, int]:
    """(session, hand_id, decision_index) of a mistake drill key. Raises `ValueError`."""
    kind, session, hand_id, index = key.split("/")
    if kind != MISTAKE:
        raise ValueError(f"not a mistake drill key: {key!r}")
    return session, hand_id, int(index)
