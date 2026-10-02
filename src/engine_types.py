
from dataclasses import dataclass, field
from typing import Any

class PolicyError(ValueError):
    """Raised for a malformed policy set ."""

@dataclass(frozen=True)
class Rule:
    id: str
    category: str
    decision: str
    advice: str
    when: tuple[tuple[str, frozenset[str]], ...]
    universal: bool
    activities: frozenset[str]
    categories: frozenset[str]
    modifiers: frozenset[str]
    lead: bool
    fuzzy: bool
    source_file: str

@dataclass(frozen=True)
class FiredRule:
    id: str
    category: str
    decision: str
    advice: str
    lead: bool
    evidence: dict[str, str]

@dataclass
class Decision:
    outcome: str
    decision: str | None
    fired: list[FiredRule]
    unevaluated: list[str]
    baseline_id: str | None
    baseline_text: str | None
    levels: dict[str, str]

    @property
    def citations(self) -> list[str]:
        return [r.id for r in self.fired] or ([self.baseline_id] if self.baseline_id else [])

@dataclass
class PolicyBook:
    taxonomy: dict[str, Any]
    metrics: dict[str, dict[str, Any]]
    defaults: dict[str, Any]
    rules: list[Rule]
    severity: dict[str, int]
    activity_category: dict[str, str]
    modifier_ids: set[str] = field(default_factory=set)

    def rule_ids(self) -> list[str]:
        return [r.id for r in self.rules]