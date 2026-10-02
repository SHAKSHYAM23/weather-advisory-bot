
import math
from typing import Any


from src.engine_types import Decision, PolicyBook, FiredRule, PolicyError, Rule



UNKNOWN = "UNKNOWN"
MATCHED = "MATCHED"                      
COVERED_CLEAR = "COVERED_CLEAR"          
NO_COVERAGE = "NO_COVERAGE"              
INSUFFICIENT_DATA = "INSUFFICIENT_DATA"  


def bin_value(metric_def: dict[str, Any], raw: Any) -> str:
    """Raw number/bool -> level name, or UNKNOWN if missing/invalid."""
    if raw is None:
        return UNKNOWN
    if metric_def["kind"] == "flag":
        return "ACTIVE" if bool(raw) else "INACTIVE"
    if isinstance(raw, bool) or not isinstance(raw, (int, float)) or math.isnan(raw):
        return UNKNOWN
    level = metric_def["bins"][0]["level"]
    for b in metric_def["bins"]:
        if raw >= b["from"]:
            level = b["level"]
    return level

def bin_metrics(book: PolicyBook, raw: dict[str, Any]) -> dict[str, str]:
    """Every metric in the book gets a level; metrics absent from `raw` are UNKNOWN."""
    return {name: bin_value(m, raw.get(name)) for name, m in book.metrics.items()}


def _applies(rule: Rule, activity_id: str, category: str, modifiers: frozenset[str]) -> bool:
    if rule.modifiers and not (rule.modifiers & modifiers):
        return False
    return rule.universal or activity_id in rule.activities or category in rule.categories

def _gives_coverage(rule: Rule, activity_id: str, modifiers: frozenset[str]) -> bool:
    if rule.fuzzy:
        return False
    return activity_id in rule.activities or bool(rule.modifiers & modifiers)

def evaluate(book: PolicyBook, activity_id: str, modifiers: list[str] | set[str] | frozenset[str],
             levels: dict[str, str]) -> Decision:
    if activity_id not in book.activity_category:
        raise PolicyError(f"unknown activity '{activity_id}'")
    mods = frozenset(modifiers)
    if mods - book.modifier_ids:
        raise PolicyError(f"unknown modifiers {sorted(mods - book.modifier_ids)}")
    category = book.activity_category[activity_id]

    fired: list[FiredRule] = []
    unevaluated: set[str] = set()
    covered = False

    for rule in book.rules:
        if not _applies(rule, activity_id, category, mods):
            continue
        covered = covered or _gives_coverage(rule, activity_id, mods)
        results = [(m, levels.get(m, UNKNOWN), allowed) for m, allowed in rule.when]
        if any(lvl != UNKNOWN and lvl not in allowed for _, lvl, allowed in results):
            continue                              
        unknown = [m for m, lvl, _ in results if lvl == UNKNOWN]
        if unknown:
            unevaluated.update(unknown)               
            continue
        fired.append(FiredRule(rule.id, rule.category, rule.decision, rule.advice, rule.lead,
                               {m: lvl for m, lvl, _ in results}))

    fired.sort(key=lambda r: (not r.lead, -book.severity[r.decision], r.id))
    baselines = book.defaults["baseline_sops"]

    if fired:
        top = max(fired, key=lambda r: book.severity[r.decision]).decision
        return Decision(MATCHED, top, fired, sorted(unevaluated), None, None, levels)
    if unevaluated:
        return Decision(INSUFFICIENT_DATA, None, [], sorted(unevaluated), "BASE-INSUFFICIENT-DATA",
                        baselines["BASE-INSUFFICIENT-DATA"]["text"], levels)
    if covered:
        return Decision(COVERED_CLEAR, None, [], [], "BASE-CLEAR", baselines["BASE-CLEAR"]["text"], levels)
    return Decision(NO_COVERAGE, None, [], [], "BASE-NO-COVERAGE", baselines["BASE-NO-COVERAGE"]["text"], levels)