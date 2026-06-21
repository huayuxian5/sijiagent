from __future__ import annotations

from typing import Any


def is_persistent_preference_instruction(instruction: dict[str, Any]) -> bool:
    """Use the parsed instruction field instead of keyword heuristics."""
    return bool(instruction.get("persistent", True))


def instruction_content_key(text: str) -> str:
    return " ".join(str(text or "").split())


def instruction_display_rule(instruction: dict[str, Any]) -> str:
    """Prefer normalized summary for downstream LLM prompts."""
    normalized = str(instruction.get("normalized_rule") or "").strip()
    if normalized:
        return normalized
    return str(instruction.get("rule") or "")


def compact_instruction_for_llm(instruction: dict[str, Any]) -> dict[str, Any]:
    """Build a compact dict for Policy / FF / StateMachine prompts."""
    inst_id = str(instruction.get("id", "") or "")
    item: dict[str, Any] = {
        "id": inst_id,
        "preference_type": instruction.get("preference_type"),
        "category": instruction.get("category"),
        "hardness": instruction.get("hardness"),
        "persistent": instruction.get("persistent"),
        "rule": instruction_display_rule(instruction),
        "required_fields": instruction.get("required_fields"),
        "completion_check": instruction.get("completion_check"),
    }
    uncertainty = instruction.get("uncertainty")
    if uncertainty:
        item["uncertainty"] = uncertainty
    parse_status = instruction.get("parse_status")
    if parse_status:
        item["parse_status"] = parse_status
    source_rule = str(instruction.get("rule") or "").strip()
    display = str(item.get("rule") or "").strip()
    if source_rule and source_rule != display:
        item["source_rule"] = source_rule[:240]
    if instruction.get("schema_version"):
        item["schema_version"] = instruction.get("schema_version")
    scheme = instruction.get("scheme")
    if isinstance(scheme, dict):
        compact_scheme: dict[str, Any] = {
            "type": scheme.get("type"),
            "scope": scheme.get("scope"),
            "constraint": scheme.get("constraint"),
            "completion": scheme.get("completion"),
            "filter": scheme.get("filter"),
        }
        item["scheme"] = {key: value for key, value in compact_scheme.items() if value not in (None, "", [], {})}
    guard = instruction.get("guard_summary")
    if guard:
        item["guard_summary"] = str(guard)[:200]
    cycle = instruction.get("cycle") if isinstance(instruction.get("cycle"), dict) else {}
    if cycle:
        cycle_item: dict[str, Any] = {"length": cycle.get("length")}
        active_dates = cycle.get("active_dates")
        if isinstance(active_dates, list) and active_dates:
            cycle_item["active_dates"] = active_dates[:6]
        count = cycle.get("count")
        if isinstance(count, dict) and count.get("min") is not None:
            cycle_item["count_min"] = count.get("min")
        item["cycle"] = {k: v for k, v in cycle_item.items() if v not in (None, "", [])}
    checks = instruction.get("checks")
    if isinstance(checks, list) and checks:
        item["checks_count"] = len(checks)
        item["checks"] = checks[:6]
    steps = instruction.get("steps")
    if isinstance(steps, list):
        item["steps"] = [str(step)[:160] for step in steps[:4] if str(step).strip()]
    elif isinstance(steps, str) and steps.strip():
        item["steps"] = [steps[:160]]
    return {key: value for key, value in item.items() if value not in (None, "", [])}
