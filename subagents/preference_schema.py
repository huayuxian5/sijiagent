"""Preference parse v2 schema constants and validators."""
from __future__ import annotations

from typing import Any

SCHEMA_VERSION = "preference.scheme.v1"
PARSE_PROMPT_VERSION = "scheme.v1.3"

VALID_HARDNESS = frozenset({"hard", "soft", "unknown"})
VALID_CATEGORIES = frozenset({
    "时段禁行",
    "单日/指定日事件",
    "城市/路线回避",
    "货源类型/品类约束",
    "距离/空驶约束",
    "频次/次数约束",
    "连续驾驶/休息",
    "其他",
    "unknown",
})

VALID_PREFERENCE_TYPES = frozenset({
    "ACTION_FORBID",
    "NUMERIC_LIMIT",
    "TIME_WINDOW_STATIONARY",
    "DAILY_CONTINUOUS_REST",
    "OFF_DAY_QUOTA",
    "ORDER_QUOTA",
    "TARGET_CARGO_MUST_TAKE",
    "GEOFENCE_STAY_WITHIN",
    "GEOFENCE_FORBIDDEN_AREA",
    "MONTHLY_DEADHEAD_LIMIT",
    "DAILY_FIRST_ORDER_DEADLINE",
    "LOCATION_ARRIVAL_DEADLINE",
    "LOCATION_STAY_ON_DATE",
    "ROUTE_SEQUENCE_ON_DATE",
    "PREFERENCE_SCORE",
    "UNKNOWN",
})

CYCLE_LENGTHS = frozenset({"always", "day", "month", "once"})
CYCLE_RESETS = frozenset({"calendar_day", "calendar_month", "never"})
EVALUATE_AT = frozenset({"per_action", "end_of_day", "end_of_month", "event_end"})

SCOPE_ACTIONS = frozenset({"wait", "reposition", "take_order"})
SCOPE_PHASES = frozenset({"candidate", "executed", "history_aggregate"})
APPLIES_WHEN = frozenset({"always_active", "in_cycle_window"})

COMPLETION_MODES = frozenset({
    "never_expires",
    "per_cycle_satisfied",
    "window_expires",
    "month_quota_met",
})

ROUTING_CYCLE_KINDS = frozenset({"always", "day", "month", "once"})
ROUTING_CONSTRAINT_KINDS = frozenset({
    "time_window",
    "time_duration",
    "cargo_filter",
    "target_cargo",
    "geofence",
    "city_filter",
    "distance",
    "off_days",
    "route_sequence",
    "location_wait",
    "other",
})

CHECK_ACTIONS = frozenset({"wait", "reposition", "take_order"})
CHECK_PHASES = frozenset({
    "whole", "to_pickup", "haul", "at_pickup", "at_delivery", "cargo", "staying", "moving",
})
CHECK_MEASURES = frozenset({
    "distance", "duration", "clock_time", "city", "cargo_name", "location", "geofence", "still_day",
})
CHECK_COMPARES = frozenset({
    "max", "min", "equals", "not_equals", "contains", "not_contains",
    "in_window", "not_in_window", "near", "within",
})
ON_FAIL_EFFECTS = frozenset({"block", "warn"})

FORBIDDEN_CHECK_TOKENS = frozenset({
    "pickup_city", "delivery_city", "pickup_deadhead_km", "dist_to_pickup_km",
    "segment", "required_fields", "steps",
})


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def _as_str_list(value: Any) -> list[str]:
    return [str(item).strip() for item in _as_list(value) if str(item).strip()]


def validate_tier1(data: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not str(data.get("normalized_rule") or "").strip():
        errors.append("tier1: missing normalized_rule")
    preference_type = str(data.get("preference_type") or "").strip()
    if preference_type not in VALID_PREFERENCE_TYPES:
        errors.append(f"tier1: invalid preference_type {preference_type!r}")
    hardness = str(data.get("hardness") or "").strip().lower()
    if hardness not in VALID_HARDNESS:
        errors.append(f"tier1: invalid hardness {hardness!r}")
    routing = data.get("routing")
    if not isinstance(routing, dict):
        errors.append("tier1: missing routing object")
        return errors
    cycle_kind = str(routing.get("cycle_kind") or "").strip()
    if cycle_kind not in ROUTING_CYCLE_KINDS:
        errors.append(f"tier1: invalid routing.cycle_kind {cycle_kind!r}")
    scope_actions = _as_str_list(routing.get("scope_actions"))
    if scope_actions and any(item not in SCOPE_ACTIONS for item in scope_actions):
        errors.append("tier1: invalid routing.scope_actions")
    kinds = _as_str_list(routing.get("constraint_kinds"))
    if kinds and any(item not in ROUTING_CONSTRAINT_KINDS for item in kinds):
        errors.append("tier1: invalid routing.constraint_kinds")
    return errors


def validate_tier2a(data: dict[str, Any], routing: dict[str, Any] | None = None) -> list[str]:
    errors: list[str] = []
    cycle = data.get("cycle")
    scope = data.get("scope")
    completion = data.get("completion")
    if not isinstance(cycle, dict):
        return ["tier2a: missing cycle"]
    if not isinstance(scope, dict):
        return ["tier2a: missing scope"]
    if not isinstance(completion, dict):
        return ["tier2a: missing completion"]

    length = str(cycle.get("length") or "").strip()
    if length not in CYCLE_LENGTHS:
        errors.append(f"tier2a: invalid cycle.length {length!r}")

    if length == "once":
        active_dates = _as_str_list(cycle.get("active_dates"))
        window = cycle.get("window")
        if not active_dates and not isinstance(window, dict):
            errors.append("tier2a: once requires active_dates or window")

    evaluate_at = str(cycle.get("evaluate_at") or "").strip()
    if evaluate_at and evaluate_at not in EVALUATE_AT:
        errors.append(f"tier2a: invalid cycle.evaluate_at {evaluate_at!r}")

    reset = str(cycle.get("reset") or "").strip()
    if reset and reset not in CYCLE_RESETS:
        errors.append(f"tier2a: invalid cycle.reset {reset!r}")

    actions = _as_str_list(scope.get("actions"))
    if not actions:
        errors.append("tier2a: scope.actions empty")
    elif any(item not in SCOPE_ACTIONS for item in actions):
        errors.append("tier2a: invalid scope.actions")

    phase = str(scope.get("phase") or "").strip()
    if phase not in SCOPE_PHASES:
        errors.append(f"tier2a: invalid scope.phase {phase!r}")

    applies = str(scope.get("applies_when") or "always_active").strip()
    if applies not in APPLIES_WHEN:
        errors.append(f"tier2a: invalid scope.applies_when {applies!r}")

    mode = str(completion.get("mode") or "").strip()
    if mode not in COMPLETION_MODES:
        errors.append(f"tier2a: invalid completion.mode {mode!r}")

    if routing:
        rk = str(routing.get("cycle_kind") or "").strip()
        if rk and length and rk != length:
            errors.append(f"tier2a: cycle.length {length} != routing.cycle_kind {rk}")

    return errors


def validate_tier2b(data: dict[str, Any], routing: dict[str, Any] | None = None) -> list[str]:
    errors: list[str] = []
    checks = data.get("checks")
    route_plan = data.get("route_plan")
    on_fail = data.get("on_fail")

    if route_plan is None and not isinstance(checks, list):
        return ["tier2b: missing checks"]
    if route_plan is None and not checks:
        errors.append("tier2b: checks empty without route_plan")
    if not isinstance(on_fail, dict):
        errors.append("tier2b: missing on_fail")

    if isinstance(checks, list):
        for index, check in enumerate(checks):
            if not isinstance(check, dict):
                errors.append(f"tier2b: checks[{index}] not object")
                continue
            prefix = f"tier2b: checks[{index}]"
            action = str(check.get("action") or "").strip()
            phase = str(check.get("phase") or "").strip()
            measure = str(check.get("measure") or "").strip()
            compare = str(check.get("compare") or "").strip()
            if action not in CHECK_ACTIONS:
                errors.append(f"{prefix}: invalid action {action!r}")
            if phase not in CHECK_PHASES:
                errors.append(f"{prefix}: invalid phase {phase!r}")
            if measure not in CHECK_MEASURES:
                errors.append(f"{prefix}: invalid measure {measure!r}")
            if compare not in CHECK_COMPARES:
                errors.append(f"{prefix}: invalid compare {compare!r}")
            if check.get("value") is None:
                errors.append(f"{prefix}: missing value")
            blob = str(check).lower()
            for token in FORBIDDEN_CHECK_TOKENS:
                if token in blob:
                    errors.append(f"{prefix}: forbidden token {token}")

    if isinstance(on_fail, dict):
        effect = str(on_fail.get("effect") or "").strip()
        if effect not in ON_FAIL_EFFECTS:
            errors.append(f"tier2b: invalid on_fail.effect {effect!r}")
        block_actions = _as_str_list(on_fail.get("block_actions"))
        if any(item not in SCOPE_ACTIONS for item in block_actions):
            errors.append("tier2b: invalid on_fail.block_actions")

    if routing and str(routing.get("needs_sequence") or "").lower() in {"true", "1", "yes"}:
        if route_plan is None:
            errors.append("tier2b: routing.needs_sequence but route_plan is null")

    return errors


def validate_assembled_v2(inst: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if inst.get("schema_version") != SCHEMA_VERSION:
        errors.append("assembled: wrong schema_version")
    routing = inst.get("routing") if isinstance(inst.get("routing"), dict) else {}
    errors.extend(validate_tier2a({
        "cycle": inst.get("cycle"),
        "scope": inst.get("scope"),
        "completion": inst.get("completion"),
    }, routing))
    if str(inst.get("preference_type") or "") != "TARGET_CARGO_MUST_TAKE":
        errors.extend(validate_tier2b({
            "checks": inst.get("checks"),
            "on_fail": inst.get("on_fail"),
            "route_plan": inst.get("route_plan"),
        }, routing))
    return errors
