
import yaml
from pathlib import Path
from typing import Any
from src.engine_types import Rule, PolicyBook, PolicyError

def _read(path: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise PolicyError(f"{path.name}: invalid YAML: {e}") from e
    if not isinstance(data, dict):
        raise PolicyError(f"{path.name}: top level must be a mapping")
    return data

def load_policies(policy_dir: str | Path) -> PolicyBook:
    """Load + validate everything. Any sops/*.yaml dropped in is picked up automatically."""
    root = Path(policy_dir)
    errors: list[str] = []

    taxonomy = _read(root / "taxonomy.yaml")
    metrics_raw = _read(root / "metrics.yaml")
    defaults = _read(root / "defaults.yaml")

    activity_category: dict[str, str] = {}
    for cat, body in taxonomy.get("categories", {}).items():
        for act in body.get("activities", []):
            if act["id"] in activity_category:
                errors.append(f"taxonomy: duplicate activity id '{act['id']}'")
            activity_category[act["id"]] = cat
    modifier_ids = {g["id"] for g in taxonomy.get("context_modifiers", {}).get("groups", [])}

    metrics: dict[str, dict[str, Any]] = {}
    for name, m in metrics_raw.get("metrics", {}).items():
        bins = m.get("bins") or []
        if not bins:
            errors.append(f"metrics: '{name}' has no bins")
            continue
        froms = [b["from"] for b in bins]
        if froms != sorted(froms):
            errors.append(f"metrics: '{name}' bins must be in ascending 'from' order")
        metrics[name] = {**m, "kind": "bins", "levels": [b["level"] for b in bins]}
    for name, m in metrics_raw.get("flags", {}).items():
        metrics[name] = {**m, "kind": "flag", "levels": ["INACTIVE", "ACTIVE"]}

    ladder = defaults.get("severity_order", [])
    severity = {d: i for i, d in enumerate(ladder)}
    if len(severity) < 3:
        errors.append("defaults: severity_order must list at least 3 decisions")

    rules: list[Rule] = []
    seen: dict[str, str] = {}
    sop_files = sorted((root / "sops").glob("*.yaml"))
    
    for path in sop_files:
        data = _read(path)
        file_cat = data.get("category", path.stem)
        for raw in data.get("rules", []):
            rid = str(raw.get("id", "")).strip()
            where = f"{path.name}:{rid or '<no id>'}"
            if not rid:
                errors.append(f"{where}: missing id")
                continue
            seen[rid] = path.name

            ap = raw.get("applies_to") or {}
            activities = frozenset(ap.get("activities", []))
            categories = frozenset(ap.get("categories", []))
            modifiers = frozenset(ap.get("modifiers", []))
            universal = bool(ap.get("any", False))

            when_raw = raw.get("when") or []
            when: list[tuple[str, frozenset[str]]] = []
            for cond in when_raw:
                metric, allowed = cond.get("metric"), cond.get("in") or []
                when.append((metric, frozenset(allowed)))

            rules.append(Rule(
                id=rid, category=file_cat, decision=raw.get("decision", ""),
                advice=str(raw.get("advice", "")).strip(), when=tuple(when),
                universal=universal, activities=activities, categories=categories,
                modifiers=modifiers, lead=bool(raw.get("lead", False)),
                fuzzy=bool(raw.get("fuzzy", False)), source_file=path.name,
            ))

    if errors:
        raise PolicyError("Policy validation failed:\n  - " + "\n  - ".join(errors))
    return PolicyBook(taxonomy, metrics, defaults, rules, severity, activity_category, modifier_ids)