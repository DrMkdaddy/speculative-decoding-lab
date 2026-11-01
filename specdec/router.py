"""A small online policy for enabling speculation per prompt class.

The policy does not predict acceptance from the prompt text directly. It
tracks the measured speedup of the class and only keeps speculation on while
that estimate is above the break-even line. This is deliberately simple: the
artifact contains the evidence for the threshold, and the policy is small
enough to inspect.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


_JSON = re.compile(r"\b(json|schema|array of \d+ objects)\b", re.IGNORECASE)
_CODE = re.compile(
    r"\b(python|go|rust|sql|function|struct|hashmap|query|unit tests?)\b",
    re.IGNORECASE,
)
_MATH = re.compile(r"\b(solve|equation|derivative|sum of|arithmetic|tangent)\b", re.IGNORECASE)
_LIST = re.compile(r"^\s*list\b", re.IGNORECASE)


def classify_prompt(prompt: str) -> str:
    """Cheap class tags used by the policy.

    The prompt set in this repo carries a hand label; this function is the
    deployment-time approximation and is evaluated against those labels in the
    tests. It is intentionally boring: a wrong class only costs a routing
    decision, not correctness.
    """
    if _JSON.search(prompt):
        return "json"
    if _CODE.search(prompt):
        return "code"
    if _MATH.search(prompt):
        return "math"
    if _LIST.search(prompt):
        return "list"
    return "prose"


@dataclass
class AcceptanceRouter:
    alpha: float = 0.3
    threshold: float = 1.0
    initial: float = 1.0
    ewma: dict[str, float] = field(default_factory=dict)
    observations: dict[str, int] = field(default_factory=dict)

    def choose(self, prompt: str) -> str:
        cls = classify_prompt(prompt)
        return "spec" if self.ewma.get(cls, self.initial) > self.threshold else "baseline"

    def observe(self, prompt: str, speedup: float) -> None:
        cls = classify_prompt(prompt)
        previous = self.ewma.get(cls, self.initial)
        self.ewma[cls] = (1.0 - self.alpha) * previous + self.alpha * speedup
        self.observations[cls] = self.observations.get(cls, 0) + 1

    def snapshot(self) -> dict[str, dict[str, float]]:
        return {
            cls: {
                "ewma_speedup": value,
                "observations": float(self.observations.get(cls, 0)),
                "spec_enabled": value > self.threshold,
            }
            for cls, value in sorted(self.ewma.items())
        }
