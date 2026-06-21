"""Merge, normalize, and derive fields for preference v2."""
from __future__ import annotations

import json
import re
from typing import Any

from .preference_schema import (
    APPLIES_WHEN,
    CHECK_ACTIONS,
    CHECK_COMPARES,
    CHECK_MEASURES,
    CHECK_PHASES,
    COMPLETION_MODES,
    CYCLE_LENGTHS,
    CYCLE_RESETS,
    EVALUATE_AT,
    ON_FAIL_EFFECTS,
    PARSE_PROMPT_VERSION,
    SCHEMA_VERSION,
    SCOPE_ACTIONS,
    SCOPE_PHASES,
    VALID_CATEGORIES,
    VALID_HARDNESS,
    VALID_PREFERENCE_TYPES,
    validate_assembled_v2,
    validate_tier1,
    validate_tier2a,
    validate_tier2b,
)

SCHEDULE_PREFERENCE_TYPES = {
    "LOCATION_ARRIVAL_DEADLINE",
    "LOCATION_STAY_ON_DATE",
    "ROUTE_SEQUENCE_ON_DATE",
}


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def _as_str_list(value: Any) -> list[str]:
    return [str(item).strip() for item in _as_list(value) if str(item).strip()]


def normalize_tier1(raw: dict[str, Any]) -> dict[str, Any]:
    routing = raw.get("routing") if isinstance(raw.get("routing"), dict) else {}
    active_dates = routing.get("active_dates")
    if active_dates is not None and not isinstance(active_dates, list):
        active_dates = _as_str_list(active_dates)
    elif isinstance(active_dates, list):
        active_dates = _as_str_list(active_dates) or None

    category = str(raw.get("category") or "unknown").strip()
    if category not in VALID_CATEGORIES:
        category = "unknown"
    hardness = str(raw.get("hardness") or "unknown").strip().lower()
    if hardness not in VALID_HARDNESS:
        hardness = "unknown"
    preference_type = str(raw.get("preference_type") or "UNKNOWN").strip()
    if preference_type not in VALID_PREFERENCE_TYPES:
        preference_type = "UNKNOWN"

    return {
        "normalized_rule": str(raw.get("normalized_rule") or "").strip(),
        "preference_type": preference_type,
        "category": category,
        "hardness": hardness,
        "uncertainty": str(raw.get("uncertainty") or "").strip(),
        "routing": {
            "preference_type": preference_type,
            "cycle_kind": str(routing.get("cycle_kind") or "always").strip(),
            "active_dates": active_dates,
            "scope_actions": _as_str_list(routing.get("scope_actions")) or ["take_order"],
            "blocked_actions": _as_str_list(routing.get("blocked_actions")),
            "constraint_kinds": _as_str_list(routing.get("constraint_kinds")) or ["other"],
            "needs_sequence": _coerce_bool(routing.get("needs_sequence", False)),
            "needs_history_aggregate": _coerce_bool(routing.get("needs_history_aggregate", False)),
            "count_min": _coerce_int(routing.get("count_min")),
            "distinct_by": routing.get("distinct_by"),
        },
    }


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return False


def normalize_tier2a(raw: dict[str, Any], routing: dict[str, Any]) -> dict[str, Any]:
    cycle = dict(raw.get("cycle") or {})
    scope = dict(raw.get("scope") or {})
    completion = dict(raw.get("completion") or {})

    length = str(cycle.get("length") or routing.get("cycle_kind") or "always").strip()
    rk = str(routing.get("cycle_kind") or "").strip()
    if rk in CYCLE_LENGTHS and length != rk:
        length = rk
    if length not in CYCLE_LENGTHS:
        length = rk or "always"

    active_dates = cycle.get("active_dates")
    if active_dates is not None:
        active_dates = _as_str_list(active_dates) or None
    elif routing.get("active_dates"):
        active_dates = _as_str_list(routing.get("active_dates")) or None

    window = cycle.get("window")
    if not isinstance(window, dict):
        window = None
    if length == "once" and active_dates and not window:
        window = {
            "start": f"{active_dates[0]}T00:00:00",
            "end": _day_after(active_dates[-1]),
        }

    count = cycle.get("count")
    if count is not None and not isinstance(count, dict):
        count = None
    count_min = _coerce_int(routing.get("count_min"))
    if count is None and count_min is not None and length == "month":
        distinct = str(routing.get("distinct_by") or "calendar_day").strip()
        if distinct not in {"calendar_day", "none"}:
            distinct = "calendar_day"
        count = {"min": count_min, "max": None, "distinct_by": distinct}

    reset = str(cycle.get("reset") or _default_reset(length)).strip()
    if reset not in CYCLE_RESETS:
        reset = _default_reset(length)

    evaluate_at = str(cycle.get("evaluate_at") or _default_evaluate_at(length)).strip()
    if evaluate_at not in EVALUATE_AT:
        evaluate_at = _default_evaluate_at(length)

    actions = _as_str_list(scope.get("actions")) or _as_str_list(routing.get("scope_actions")) or ["take_order"]
    actions = [a for a in actions if a in SCOPE_ACTIONS] or ["take_order"]

    phase = str(scope.get("phase") or _default_phase(routing)).strip()
    if phase not in SCOPE_PHASES:
        phase = _default_phase(routing)

    applies = str(scope.get("applies_when") or ("in_cycle_window" if length == "once" else "always_active")).strip()
    if applies not in APPLIES_WHEN:
        applies = "in_cycle_window" if length == "once" else "always_active"

    mode = str(completion.get("mode") or _default_completion_mode(length)).strip()
    if mode not in COMPLETION_MODES:
        mode = _default_completion_mode(length)

    expires_at = completion.get("expires_at")
    if expires_at is not None:
        expires_at = str(expires_at).strip() or None
    if length == "once" and window and not expires_at:
        expires_at = str(window.get("end") or "").strip() or None

    return {
        "cycle": {
            "length": length,
            "window": window,
            "active_dates": active_dates,
            "count": count,
            "reset": reset,
            "evaluate_at": evaluate_at,
        },
        "scope": {
            "actions": actions,
            "phase": phase,
            "applies_when": applies,
        },
        "completion": {
            "mode": mode,
            "track_progress": bool(completion.get("track_progress", mode == "month_quota_met")),
            "progress_key": completion.get("progress_key"),
            "expires_at": expires_at,
        },
    }


def normalize_tier2b(raw: dict[str, Any]) -> dict[str, Any]:
    checks_in = raw.get("checks")
    normalized_checks: list[dict[str, Any]] = []
    if isinstance(checks_in, list):
        for item in checks_in:
            if not isinstance(item, dict):
                continue
            action = str(item.get("action") or "").strip()
            phase = str(item.get("phase") or "whole").strip()
            measure = str(item.get("measure") or "").strip()
            compare = str(item.get("compare") or "").strip()
            if action not in CHECK_ACTIONS or phase not in CHECK_PHASES:
                continue
            if measure not in CHECK_MEASURES or compare not in CHECK_COMPARES:
                continue
            if item.get("value") is None:
                continue
            entry: dict[str, Any] = {
                "action": action,
                "phase": phase,
                "measure": measure,
                "compare": compare,
                "value": item.get("value"),
            }
            unit = item.get("unit")
            if unit is not None and str(unit).strip():
                entry["unit"] = str(unit).strip()
            normalized_checks.append(entry)

    on_fail = raw.get("on_fail") if isinstance(raw.get("on_fail"), dict) else {}
    effect = str(on_fail.get("effect") or "block").strip()
    if effect not in ON_FAIL_EFFECTS:
        effect = "block"
    block_actions = [a for a in _as_str_list(on_fail.get("block_actions")) if a in CHECK_ACTIONS]

    route_plan = raw.get("route_plan")
    if route_plan is not None and not isinstance(route_plan, dict):
        route_plan = None

    return {
        "checks": normalized_checks,
        "on_fail": {"effect": effect, "block_actions": block_actions},
        "route_plan": route_plan,
    }


def merge_v2(
    source_rule: str,
    tier1: dict[str, Any],
    tier2a: dict[str, Any],
    tier2b: dict[str, Any],
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    merged: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "parse_prompt_version": PARSE_PROMPT_VERSION,
        "source_rule": source_rule,
        "normalized_rule": tier1.get("normalized_rule", ""),
        "preference_type": tier1.get("preference_type", "UNKNOWN"),
        "category": tier1.get("category", "unknown"),
        "hardness": tier1.get("hardness", "unknown"),
        "uncertainty": tier1.get("uncertainty", ""),
        "routing": tier1.get("routing", {}),
        "cycle": tier2a.get("cycle", {}),
        "scope": tier2a.get("scope", {}),
        "completion": tier2a.get("completion", {}),
        "checks": tier2b.get("checks", []),
        "on_fail": tier2b.get("on_fail", {}),
        "route_plan": tier2b.get("route_plan"),
        "meta": dict(meta or {}),
        "persistent": _derive_persistent(tier2a),
    }
    merged["parse_status"] = resolve_parse_status(merged)
    merged["scheme"] = build_fixed_scheme(merged)
    merged["schedule_task"] = build_schedule_task(merged)
    merged["guard_summary"] = build_guard_summary(merged)
    merged["steps"] = checks_to_steps(merged)
    merged["completion_check"] = build_completion_check(merged)
    merged["required_fields"] = derive_required_fields(merged)
    return merged


def build_fixed_scheme(inst: dict[str, Any]) -> dict[str, Any]:
    """Build the canonical fixed template consumed by deterministic rule filters."""
    pref_type = str(inst.get("preference_type") or "UNKNOWN").strip()
    if pref_type not in VALID_PREFERENCE_TYPES:
        pref_type = "UNKNOWN"
    cycle = inst.get("cycle") if isinstance(inst.get("cycle"), dict) else {}
    scope = inst.get("scope") if isinstance(inst.get("scope"), dict) else {}
    completion = inst.get("completion") if isinstance(inst.get("completion"), dict) else {}
    checks = [dict(item) for item in inst.get("checks") or [] if isinstance(item, dict)]
    route_plan = inst.get("route_plan") if isinstance(inst.get("route_plan"), dict) else None
    on_fail = inst.get("on_fail") if isinstance(inst.get("on_fail"), dict) else {}
    block_actions = _as_str_list(on_fail.get("block_actions"))
    active_dates = _as_str_list(cycle.get("active_dates"))
    window = cycle.get("window") if isinstance(cycle.get("window"), dict) else None

    scheme: dict[str, Any] = {
        "scheme_version": "fixed_preference_scheme.v1",
        "type": pref_type,
        "hardness": inst.get("hardness", "unknown"),
        "source_rule": inst.get("source_rule"),
        "normalized_rule": inst.get("normalized_rule"),
        "scope": {
            "period": cycle.get("length"),
            "start_date": "2026-03-01",
            "end_date": "2026-03-31",
            "active_dates": active_dates or None,
            "window": window,
            "actions": _as_str_list(scope.get("actions")),
        },
        "filter": {
            "deterministic": pref_type not in {"PREFERENCE_SCORE", "UNKNOWN"},
            "candidate_actions": _as_str_list(scope.get("actions")),
            "blocked_actions": block_actions,
            "effect": on_fail.get("effect", "block"),
        },
        "facts": {
            "progress_key": completion.get("progress_key"),
            "track_progress": bool(completion.get("track_progress")),
            "required_fields": derive_required_fields(inst),
        },
    }
    if pref_type in SCHEDULE_PREFERENCE_TYPES:
        scheme["control_channel"] = "policy_schedule"
        scheme["exclude_from_future"] = True
        scheme["exclude_from_rule_filter"] = True
    if pref_type == "TARGET_CARGO_MUST_TAKE":
        scheme["control_channel"] = "target_cargo"
    if pref_type == "GEOFENCE_STAY_WITHIN":
        scheme["control_channel"] = "geofence"
    if pref_type == "GEOFENCE_FORBIDDEN_AREA":
        scheme["control_channel"] = "forbidden_geofence"
    if pref_type == "MONTHLY_DEADHEAD_LIMIT":
        scheme["control_channel"] = "monthly_deadhead"
    if pref_type == "DAILY_FIRST_ORDER_DEADLINE":
        scheme["control_channel"] = "daily_first_order_deadline"

    constraint = _build_constraint_for_type(pref_type, inst, checks, route_plan)
    if constraint:
        scheme["constraint"] = constraint
    trigger = _build_trigger_for_type(pref_type, cycle, checks, route_plan)
    if trigger:
        scheme["trigger"] = trigger
    completion_spec = _build_completion_for_type(pref_type, inst, completion, cycle, route_plan)
    if completion_spec:
        scheme["completion"] = completion_spec
    return _drop_empty(scheme)


def build_schedule_task(inst: dict[str, Any]) -> dict[str, Any] | None:
    pref_type = str(inst.get("preference_type") or "").strip()
    if pref_type not in SCHEDULE_PREFERENCE_TYPES:
        return None
    cycle = inst.get("cycle") if isinstance(inst.get("cycle"), dict) else {}
    routing = inst.get("routing") if isinstance(inst.get("routing"), dict) else {}
    completion = inst.get("completion") if isinstance(inst.get("completion"), dict) else {}
    scope = inst.get("scope") if isinstance(inst.get("scope"), dict) else {}
    checks = [dict(item) for item in inst.get("checks") or [] if isinstance(item, dict)]
    route_plan = inst.get("route_plan") if isinstance(inst.get("route_plan"), dict) else None
    active_dates = _as_str_list(cycle.get("active_dates"))
    if not active_dates:
        active_dates = _as_str_list(routing.get("active_dates"))
    scope_actions = _as_str_list(scope.get("actions")) or _as_str_list(routing.get("scope_actions"))
    constraint_kinds = _as_str_list(routing.get("constraint_kinds"))
    date = active_dates[0] if active_dates else None
    if route_plan and route_plan.get("date"):
        date = str(route_plan.get("date"))
    if not date and completion.get("expires_at"):
        date = str(completion.get("expires_at"))[:10]

    steps: list[dict[str, Any]] = []
    if pref_type == "ROUTE_SEQUENCE_ON_DATE":
        for step in _route_steps(route_plan):
            target = step.get("target_location") if isinstance(step.get("target_location"), dict) else None
            if not target:
                continue
            finish_before = step.get("finish_before")
            explicit_complete_before = step.get("complete_before")
            explicit_hold_until = step.get("hold_until")
            complete_before = explicit_complete_before
            hold_until = explicit_hold_until
            if finish_before and not complete_before and not hold_until:
                if _has_full_datetime(finish_before):
                    hold_until = finish_before
                else:
                    complete_before = finish_before
            action = "wait" if step.get("kind") == "stay" and (step.get("min_stay_minutes") or complete_before or hold_until) else "reposition"
            item = {
                "step_id": step.get("step_id"),
                "sequence_index": step.get("sequence_index"),
                "kind": step.get("kind") or action,
                "required_action": action,
                "target": target,
                "active_dates": _step_active_dates(step, active_dates, date),
                "scope_actions": scope_actions,
                "constraint_kinds": step.get("constraint_kinds") or constraint_kinds,
                "arrive_before": _wall_time_on_date(date, step.get("arrive_before")),
                "min_stay_minutes": step.get("min_stay_minutes"),
                "complete_before": _wall_time_on_date(date, complete_before),
                "hold_until": _wall_time_on_date(date, hold_until),
                "finish_before": _wall_time_on_date(date, finish_before),
            }
            if action == "wait":
                item["location"] = target
            steps.append(_drop_empty(item))
    else:
        target = _target_from_route_or_checks(route_plan, checks)
        if target:
            arrive_before = _deadline_from_route_or_checks(route_plan, checks)
            steps.append(_drop_empty({
                "step_id": "s1",
                "kind": "arrival",
                "required_action": "reposition",
                "target": target,
                "arrive_before": _wall_time_on_date(date, arrive_before),
            }))
            if pref_type == "LOCATION_STAY_ON_DATE":
                minutes = _duration_minutes_from_checks(checks) or 120
                steps.append(_drop_empty({
                    "step_id": "s2",
                    "kind": "stay",
                    "required_action": "wait",
                    "location": target,
                    "target": target,
                    "min_stay_minutes": minutes,
                    "hold_until": _wall_time_on_date(date, None),
                }))

    if not steps:
        return None
    return _drop_empty({
        "task_version": "schedule_task.v1",
        "preference_id": inst.get("id"),
        "type": pref_type,
        "hardness": inst.get("hardness", "unknown"),
        "active_date": date,
        "active_dates": active_dates,
        "scope_actions": scope_actions,
        "constraint_kinds": constraint_kinds,
        "window": cycle.get("window") if isinstance(cycle.get("window"), dict) else None,
        "exclude_from_future": True,
        "exclude_from_rule_filter": True,
        "steps": steps,
    })


def _wall_time_on_date(date: str | None, value: Any) -> str | None:
    if value in (None, ""):
        return None
    text = str(value)
    if "T" in text or (len(text) >= 10 and text[4:5] == "-" and text[7:8] == "-"):
        return text.replace("T", " ")[:16]
    if date and ":" in text:
        return f"{date} {text[:5]}"
    return text


def _step_active_dates(step: dict[str, Any], active_dates: list[str], fallback_date: str | None) -> list[str] | None:
    dates = _as_str_list(step.get("active_dates"))
    if dates:
        return dates
    step_date = step.get("date")
    if step_date:
        return [str(step_date)]
    if active_dates:
        return active_dates
    if fallback_date:
        return [fallback_date]
    return None


def _build_constraint_for_type(
    pref_type: str,
    inst: dict[str, Any],
    checks: list[dict[str, Any]],
    route_plan: dict[str, Any] | None,
) -> dict[str, Any]:
    if pref_type == "ACTION_FORBID":
        return {"conditions": _conditions_from_checks(checks), "checks": checks}
    if pref_type == "NUMERIC_LIMIT":
        check = _first_check(checks, measure="distance") or _first_check(checks)
        return {"metric": _metric_from_check(check), "operator": _operator_from_compare(check.get("compare") if check else None), "value": check.get("value") if check else None, "unit": check.get("unit") if check else None, "checks": checks}
    if pref_type == "MONTHLY_DEADHEAD_LIMIT":
        check = _first_check(checks, measure="distance") or _first_check(checks)
        return {
            "metric": "deadhead_km",
            "period": "month",
            "operator": _operator_from_compare(check.get("compare") if check else None) or "<=",
            "value": check.get("value") if check else _number_from_text(inst),
            "unit": check.get("unit") if check else "km",
            "count_actions": ["take_order", "reposition"],
            "checks": checks,
        }
    if pref_type == "DAILY_FIRST_ORDER_DEADLINE":
        deadline = _deadline_minute_from_checks(checks)
        if deadline is None:
            deadline = _first_order_deadline_from_text(inst)
        if deadline is None:
            deadline = 12 * 60
        return {
            "target_action": "take_order",
            "metric": "first_order_action_start_minute_of_day",
            "operator": "<",
            "deadline_minute_of_day": deadline,
            "deadline_time": _minute_to_clock(deadline),
            "period": "day",
            "checks": checks,
        }
    if pref_type == "TIME_WINDOW_STATIONARY":
        return {"required_action": "wait", "forbidden_actions": ["take_order", "reposition"], "windows": _time_windows_from_checks(checks), "required_full_coverage": True}
    if pref_type == "DAILY_CONTINUOUS_REST":
        minutes = _duration_minutes_from_checks(checks) or _duration_minutes_from_text(inst) or 480
        return {"required_action": "wait", "min_continuous_minutes": minutes, "counts_query_scan_before_wait": True, "forbidden_break_actions": ["take_order", "reposition"], "deadline": "end_of_day"}
    if pref_type == "OFF_DAY_QUOTA":
        count = _count_min(inst)
        return {"min_days": count, "day_definition": {"start_time": "00:00", "end_time": "24:00", "active_actions": ["take_order", "reposition"], "allowed_actions": ["wait"]}}
    if pref_type == "ORDER_QUOTA":
        return {"target_action": "take_order", "count_unit": _distinct_by(inst), "operator": ">=", "value": _count_min(inst), "conditions": _conditions_from_checks(checks)}
    if pref_type == "TARGET_CARGO_MUST_TAKE":
        target = _target_cargo_from_inst(inst)
        return {
            "target_action": "take_order",
            "target_cargo_id": target.get("cargo_id"),
            "expected_cargo_name": target.get("cargo_name"),
            "pickup_location": target.get("pickup_location"),
            "available_after": target.get("available_after"),
            "available_until": target.get("available_until"),
            "completion_event": "take_order_cargo_id",
        }
    if pref_type == "GEOFENCE_STAY_WITHIN":
        return _geofence_constraint_from_inst(inst, checks)
    if pref_type == "GEOFENCE_FORBIDDEN_AREA":
        return _forbidden_geofence_constraint_from_inst(inst, checks)
    if pref_type == "LOCATION_ARRIVAL_DEADLINE":
        target = _target_from_route_or_checks(route_plan, checks)
        return {"target_location": target, "arrive_before": _deadline_from_route_or_checks(route_plan, checks), "completion_event": "position_within_radius_before_deadline"}
    if pref_type == "LOCATION_STAY_ON_DATE":
        target = _target_from_route_or_checks(route_plan, checks)
        return {"target_location": target, "required_action": "wait", "min_stay_minutes": _duration_minutes_from_checks(checks) or 120, "stay_window": _date_window_from_inst(inst)}
    if pref_type == "ROUTE_SEQUENCE_ON_DATE":
        routing = inst.get("routing") if isinstance(inst.get("routing"), dict) else {}
        cycle = inst.get("cycle") if isinstance(inst.get("cycle"), dict) else {}
        scope = inst.get("scope") if isinstance(inst.get("scope"), dict) else {}
        return _drop_empty({
            "steps": _route_steps(route_plan),
            "ordered": True,
            "active_dates": _as_str_list(cycle.get("active_dates")) or _as_str_list(routing.get("active_dates")),
            "scope_actions": _as_str_list(scope.get("actions")) or _as_str_list(routing.get("scope_actions")),
            "constraint_kinds": _as_str_list(routing.get("constraint_kinds")),
        })
    if pref_type == "PREFERENCE_SCORE":
        return {"checks": checks, "direction": "prefer", "weight": 0.5}
    return {"checks": checks}


def _geofence_constraint_from_inst(inst: dict[str, Any], checks: list[dict[str, Any]]) -> dict[str, Any]:
    boundary = _geofence_boundary_from_text(str(inst.get("source_rule") or inst.get("normalized_rule") or inst.get("rule") or ""))
    if boundary is None:
        boundary = _geofence_boundary_from_checks(checks)
    anchor = _geofence_anchor_from_text(str(inst.get("source_rule") or inst.get("normalized_rule") or inst.get("rule") or ""))
    return _drop_empty({
        "boundary": boundary,
        "anchor_point": anchor,
        "applies_to": ["current_position", "wait", "reposition", "pickup", "dropoff"],
        "checks": checks,
    })


def _forbidden_geofence_constraint_from_inst(inst: dict[str, Any], checks: list[dict[str, Any]]) -> dict[str, Any]:
    text = str(inst.get("source_rule") or inst.get("normalized_rule") or inst.get("rule") or "")
    boundary = _forbidden_circle_from_text(text)
    if boundary is None:
        boundary = _forbidden_circle_from_checks(checks)
    return _drop_empty({
        "boundary": boundary,
        "applies_to": ["current_position", "wait", "reposition", "pickup", "dropoff"],
        "checks": checks,
    })


def _forbidden_circle_from_text(text: str) -> dict[str, Any] | None:
    match = re.search(
        r"[（(]\s*(-?\d+(?:\.\d+)?)\s*[,，]\s*(-?\d+(?:\.\d+)?)\s*[）)].{0,24}?半径\s*(\d+(?:\.\d+)?)\s*公里",
        text or "",
    )
    if not match:
        return None
    try:
        lat = float(match.group(1))
        lng = float(match.group(2))
        radius = float(match.group(3))
    except (TypeError, ValueError):
        return None
    if not (-90 <= lat <= 90 and -180 <= lng <= 180) or radius <= 0:
        return None
    return {"kind": "circle", "lat": round(lat, 6), "lng": round(lng, 6), "radius_km": round(radius, 3)}


def _forbidden_circle_from_checks(checks: list[dict[str, Any]]) -> dict[str, Any] | None:
    for check in checks:
        if not isinstance(check, dict):
            continue
        value = check.get("value")
        if not isinstance(value, dict):
            continue
        if not {"lat", "lng", "radius_km"}.issubset(value.keys()):
            continue
        try:
            return {
                "kind": "circle",
                "lat": float(value["lat"]),
                "lng": float(value["lng"]),
                "radius_km": float(value["radius_km"]),
            }
        except (TypeError, ValueError):
            continue
    return None


def _geofence_boundary_from_text(text: str) -> dict[str, Any] | None:
    numbers = [float(item) for item in re.findall(r"-?\d+(?:\.\d+)?", text or "")]
    if len(numbers) < 4:
        return None
    lats = [value for value in numbers if -90 <= value <= 90]
    lngs = [value for value in numbers if 90 < value <= 180]
    if len(lats) < 2 or len(lngs) < 2:
        return None
    min_lat = min(lats[-2:])
    max_lat = max(lats[-2:])
    min_lng = min(lngs[-2:])
    max_lng = max(lngs[-2:])
    return {
        "kind": "bbox",
        "min_lat": round(min_lat, 6),
        "max_lat": round(max_lat, 6),
        "min_lng": round(min_lng, 6),
        "max_lng": round(max_lng, 6),
    }


def _geofence_anchor_from_text(text: str) -> dict[str, Any] | None:
    match = re.search(r"(-?\d+(?:\.\d+)?)\s*[,，]\s*(-?\d+(?:\.\d+)?)", text or "")
    if not match:
        return None
    try:
        lat = float(match.group(1))
        lng = float(match.group(2))
    except (TypeError, ValueError):
        return None
    if not (-90 <= lat <= 90 and -180 <= lng <= 180):
        return None
    return {"lat": round(lat, 6), "lng": round(lng, 6), "source": "explicit_text"}


def _geofence_boundary_from_checks(checks: list[dict[str, Any]]) -> dict[str, Any] | None:
    for check in checks:
        if not isinstance(check, dict):
            continue
        value = check.get("value")
        if not isinstance(value, dict):
            continue
        keys = {"min_lat", "max_lat", "min_lng", "max_lng"}
        if not keys.issubset(value.keys()):
            continue
        try:
            return {
                "kind": "bbox",
                "min_lat": float(value["min_lat"]),
                "max_lat": float(value["max_lat"]),
                "min_lng": float(value["min_lng"]),
                "max_lng": float(value["max_lng"]),
            }
        except (TypeError, ValueError):
            continue
    return None


def _build_trigger_for_type(
    pref_type: str,
    cycle: dict[str, Any],
    checks: list[dict[str, Any]],
    route_plan: dict[str, Any] | None,
) -> dict[str, Any]:
    active_dates = _as_str_list(cycle.get("active_dates"))
    trigger: dict[str, Any] = {}
    if active_dates:
        trigger["dates"] = active_dates
    if pref_type == "TIME_WINDOW_STATIONARY":
        trigger["time_windows"] = _time_windows_from_checks(checks)
    if pref_type == "DAILY_FIRST_ORDER_DEADLINE":
        deadline = _deadline_minute_from_checks(checks)
        if deadline is not None:
            trigger["deadline_minute_of_day"] = deadline
            trigger["deadline_time"] = _minute_to_clock(deadline)
    if route_plan and route_plan.get("date"):
        trigger["date"] = route_plan.get("date")
    return trigger


def _build_completion_for_type(
    pref_type: str,
    inst: dict[str, Any],
    completion: dict[str, Any],
    cycle: dict[str, Any],
    route_plan: dict[str, Any] | None,
) -> dict[str, Any]:
    out: dict[str, Any] = {"mode": completion.get("mode"), "expires_at": completion.get("expires_at")}
    if pref_type in {"OFF_DAY_QUOTA", "ORDER_QUOTA"}:
        out["count_min"] = _count_min(inst)
        out["distinct_by"] = _distinct_by(inst)
    if pref_type == "TARGET_CARGO_MUST_TAKE":
        target = _target_cargo_from_inst(inst)
        out["target_cargo_id"] = target.get("cargo_id")
        out["mode"] = completion.get("mode") or "window_expires"
    if pref_type in {"LOCATION_STAY_ON_DATE", "LOCATION_ARRIVAL_DEADLINE", "ROUTE_SEQUENCE_ON_DATE"}:
        dates = _as_str_list(cycle.get("active_dates"))
        if dates:
            out["date"] = dates[0]
        if route_plan and route_plan.get("date"):
            out["date"] = route_plan.get("date")
    return out


def _first_check(checks: list[dict[str, Any]], *, measure: str | None = None) -> dict[str, Any] | None:
    for check in checks:
        if measure is None or check.get("measure") == measure:
            return check
    return None


def _conditions_from_checks(checks: list[dict[str, Any]]) -> dict[str, Any]:
    conditions: dict[str, Any] = {
        "cargo_name": [],
        "city": [],
        "location": [],
        "time_windows": [],
        "raw_checks": checks,
    }
    for check in checks:
        measure = check.get("measure")
        value = check.get("value")
        if measure == "cargo_name":
            conditions["cargo_name"].append({"compare": check.get("compare"), "value": value, "phase": check.get("phase")})
        elif measure == "city":
            conditions["city"].append({"compare": check.get("compare"), "value": value, "phase": check.get("phase")})
        elif measure == "location":
            conditions["location"].append({"compare": check.get("compare"), "value": value, "phase": check.get("phase")})
        elif measure == "clock_time":
            conditions["time_windows"].append(value)
    return _drop_empty(conditions)


def _metric_from_check(check: dict[str, Any] | None) -> str | None:
    if not check:
        return None
    if check.get("action") == "take_order" and check.get("phase") == "to_pickup" and check.get("measure") == "distance":
        return "pickup_deadhead_km"
    if check.get("action") == "take_order" and check.get("phase") == "haul" and check.get("measure") == "distance":
        return "haul_distance_km"
    return str(check.get("measure") or "") or None


def _number_from_text(inst: dict[str, Any]) -> float | None:
    text = str(inst.get("source_rule") or inst.get("normalized_rule") or inst.get("rule") or "")
    match = re.search(r"(\d+(?:\.\d+)?)\s*(?:km|公里)", text, flags=re.IGNORECASE)
    if not match:
        return None
    try:
        return float(match.group(1))
    except (TypeError, ValueError):
        return None


def _operator_from_compare(compare: Any) -> str | None:
    mapping = {"max": "<=", "min": ">=", "equals": "==", "not_equals": "!=", "contains": "contains", "not_contains": "not_contains", "near": "near"}
    return mapping.get(str(compare or ""))


def _time_windows_from_checks(checks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    windows: list[dict[str, Any]] = []
    for check in checks:
        if check.get("measure") != "clock_time":
            continue
        value = check.get("value")
        if isinstance(value, dict):
            windows.append(_compile_time_window(value.get("from"), value.get("to"), check.get("compare")))
    return windows


def _deadline_minute_from_checks(checks: list[dict[str, Any]]) -> int | None:
    for check in checks:
        if check.get("measure") != "clock_time":
            continue
        value = check.get("value")
        if isinstance(value, dict):
            for key in ("deadline", "to", "before", "by"):
                minute = _clock_minute(value.get(key))
                if minute is not None:
                    return minute
            minute = _clock_minute(value.get("from"))
            if minute is not None and str(check.get("compare") or "") == "max":
                return minute
        else:
            minute = _clock_minute(value)
            if minute is not None:
                return minute
    return None


def _first_order_deadline_from_text(inst: dict[str, Any]) -> int | None:
    text = str(inst.get("source_rule") or inst.get("normalized_rule") or inst.get("rule") or "")
    if not any(token in text for token in ("首单", "第一单", "第一笔")):
        return None
    for match in re.finditer(r"(\d{1,2}):(\d{1,2})|(\d{1,2})\s*点", text):
        try:
            hour = int(match.group(1) or match.group(3))
            minute = int(match.group(2) or 0)
        except (TypeError, ValueError):
            continue
        if 0 <= hour <= 23 and 0 <= minute <= 59:
            return hour * 60 + minute
    if "中午" in text:
        return 12 * 60
    return None


def _minute_to_clock(minute: int | None) -> str | None:
    if minute is None:
        return None
    minute = int(minute) % 1440
    return f"{minute // 60:02d}:{minute % 60:02d}"


def _compile_time_window(start_time: Any, end_time: Any, compare: Any = None) -> dict[str, Any]:
    window = {"start_time": start_time, "end_time": end_time, "compare": compare}
    start_minute = _clock_minute(start_time)
    end_minute = _clock_minute(end_time)
    if start_minute is None or end_minute is None:
        return window
    crosses_midnight = start_minute > end_minute
    window["crosses_midnight"] = crosses_midnight
    window["duration_minutes"] = (
        (1440 - start_minute + end_minute) if crosses_midnight else max(0, end_minute - start_minute)
    )
    window["window_semantics"] = (
        "daily_start_to_next_day_end" if crosses_midnight else "same_day_start_to_end"
    )
    return window


def _clock_minute(value: Any) -> int | None:
    if value in (None, ""):
        return None
    text = str(value).strip()
    if "T" in text:
        text = text.split("T", 1)[1]
    if " " in text:
        text = text.rsplit(" ", 1)[-1]
    text = text[:5]
    try:
        hour, minute = text.split(":", 1)
        return int(hour) * 60 + int(minute)
    except (TypeError, ValueError):
        return None


def _duration_minutes_from_checks(checks: list[dict[str, Any]]) -> int | None:
    for check in checks:
        if check.get("measure") != "duration":
            continue
        try:
            value = int(check.get("value"))
        except (TypeError, ValueError):
            continue
        unit = str(check.get("unit") or "minutes")
        if unit in {"hour", "hours"}:
            return value * 60
        return value
    return None


def _duration_minutes_from_text(inst: dict[str, Any]) -> int | None:
    text = str(inst.get("normalized_rule") or inst.get("source_rule") or "")
    import re

    match = re.search(r"(\d+(?:\.\d+)?)\s*(?:个)?小时", text)
    if match:
        return int(float(match.group(1)) * 60)
    match = re.search(r"(\d+)\s*分钟", text)
    if match:
        return int(match.group(1))
    return None


def _count_min(inst: dict[str, Any]) -> int | None:
    cycle = inst.get("cycle") if isinstance(inst.get("cycle"), dict) else {}
    count = cycle.get("count") if isinstance(cycle.get("count"), dict) else {}
    try:
        return int(count.get("min"))
    except (TypeError, ValueError):
        return None


def _distinct_by(inst: dict[str, Any]) -> str:
    cycle = inst.get("cycle") if isinstance(inst.get("cycle"), dict) else {}
    count = cycle.get("count") if isinstance(cycle.get("count"), dict) else {}
    return str(count.get("distinct_by") or "calendar_day")


def _target_from_route_or_checks(route_plan: dict[str, Any] | None, checks: list[dict[str, Any]]) -> dict[str, Any] | None:
    steps = _route_steps(route_plan)
    if steps:
        target = steps[0].get("target_location")
        if isinstance(target, dict):
            return target
    for check in checks:
        if check.get("measure") != "location":
            continue
        value = check.get("value")
        if isinstance(value, dict):
            return _location_from_value(value)
    return None


def _target_cargo_from_inst(inst: dict[str, Any]) -> dict[str, Any]:
    meta = inst.get("meta") if isinstance(inst.get("meta"), dict) else {}
    target = meta.get("target_cargo") if isinstance(meta.get("target_cargo"), dict) else {}
    out = {
        "cargo_id": target.get("cargo_id") or meta.get("target_cargo_id"),
        "cargo_name": target.get("cargo_name") or meta.get("target_cargo_name"),
        "pickup_location": target.get("pickup_location") if isinstance(target.get("pickup_location"), dict) else None,
        "available_after": target.get("available_after") or meta.get("target_cargo_available_after"),
        "available_until": target.get("available_until") or meta.get("target_cargo_available_until") or meta.get("visible_until"),
    }
    return _drop_empty(out)


def _deadline_from_route_or_checks(route_plan: dict[str, Any] | None, checks: list[dict[str, Any]]) -> str | None:
    steps = _route_steps(route_plan)
    for step in steps:
        if step.get("arrive_before"):
            return str(step.get("arrive_before"))
    for check in checks:
        if check.get("measure") == "clock_time" and isinstance(check.get("value"), dict):
            return check["value"].get("to")
    return None


def _date_window_from_inst(inst: dict[str, Any]) -> dict[str, Any]:
    cycle = inst.get("cycle") if isinstance(inst.get("cycle"), dict) else {}
    window = cycle.get("window") if isinstance(cycle.get("window"), dict) else {}
    return {"start_time": "00:00", "end_time": "24:00", "window": window}


def _route_steps(route_plan: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(route_plan, dict):
        return []
    stops = route_plan.get("steps")
    if not isinstance(stops, list):
        stops = route_plan.get("stops")
    if not isinstance(stops, list):
        return []
    steps: list[dict[str, Any]] = []
    for index, stop in enumerate(stops, start=1):
        if not isinstance(stop, dict):
            continue
        location = _location_from_value(stop)
        try:
            stay_minutes = int(stop.get("stay_minutes", stop.get("min_stay_minutes", 0)) or 0)
        except (TypeError, ValueError):
            stay_minutes = 0
        finish_before = stop.get("finish_before")
        complete_before = stop.get("complete_before")
        hold_until = stop.get("hold_until")
        step = {
            "step_id": str(stop.get("step_id") or f"s{index}"),
            "sequence_index": index,
            "kind": stop.get("kind") or ("stay" if stay_minutes > 0 or finish_before or complete_before or hold_until else "arrival"),
            "date": stop.get("date") or route_plan.get("date"),
            "active_dates": _as_str_list(stop.get("active_dates")),
            "scope_actions": _as_str_list(stop.get("scope_actions")),
            "constraint_kinds": _as_str_list(stop.get("constraint_kinds")),
            "target_location": location,
            "arrive_before": stop.get("arrive_before"),
            "min_stay_minutes": stay_minutes or None,
            "complete_before": complete_before,
            "hold_until": hold_until,
            "finish_before": finish_before,
        }
        steps.append(_drop_empty(step))
    return steps


def _location_from_value(value: dict[str, Any]) -> dict[str, Any]:
    has_coordinates = value.get("lat") is not None and value.get("lng") is not None
    out = {
        "name": value.get("name"),
        "lat": value.get("lat"),
        "lng": value.get("lng"),
        "radius_km": value.get("radius_km", 1 if has_coordinates else 5),
    }
    return _drop_empty(out)


def _has_full_datetime(value: Any) -> bool:
    text = str(value or "").strip().replace("T", " ")
    return len(text) >= 16 and text[4:5] == "-" and text[7:8] == "-" and ":" in text[11:16]


def _drop_empty(value: Any) -> Any:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            cleaned = _drop_empty(item)
            if cleaned not in (None, "", [], {}):
                out[key] = cleaned
        return out
    if isinstance(value, list):
        return [_drop_empty(item) for item in value if _drop_empty(item) not in (None, "", [], {})]
    return value


def attach_visibility_meta(inst: dict[str, Any], pref_record: dict[str, Any] | None) -> dict[str, Any]:
    if not pref_record or not isinstance(pref_record, dict):
        return inst
    meta = dict(inst.get("meta") or {})
    start_time = pref_record.get("start_time")
    end_time = pref_record.get("end_time")
    if start_time:
        meta["visible_from"] = str(start_time).strip()
    if end_time:
        meta["visible_until"] = str(end_time).strip()
    for key in ("penalty_amount", "penalty_cap"):
        if pref_record.get(key) is not None:
            meta[key] = pref_record.get(key)
    if meta:
        inst = dict(inst)
        inst["meta"] = meta
    return inst


def lookup_pref_record(preferences_raw: list[Any], content_key: str) -> dict[str, Any] | None:
    from .preference_utils import instruction_content_key

    target = instruction_content_key(content_key)
    for item in preferences_raw or []:
        if not isinstance(item, dict):
            continue
        text = instruction_content_key(str(item.get("content") or item.get("text") or ""))
        if text == target:
            return item
    return None


def lookup_visibility_record(preference_visibility: list[dict[str, Any]], content_key: str) -> dict[str, Any] | None:
    from .preference_utils import instruction_content_key

    target = instruction_content_key(content_key)
    for item in preference_visibility or []:
        if not isinstance(item, dict):
            continue
        text = instruction_content_key(str(item.get("content") or ""))
        if text == target:
            return item
    return None


def resolve_parse_status(inst: dict[str, Any]) -> str:
    errors = validate_assembled_v2(inst)
    if errors:
        if not inst.get("checks") and not inst.get("route_plan"):
            return "fallback"
        return "degraded"
    if inst.get("hardness") == "unknown" or inst.get("category") == "unknown":
        return "degraded"
    uncertainty = str(inst.get("uncertainty") or "").strip()
    if uncertainty:
        cycle = inst.get("cycle") if isinstance(inst.get("cycle"), dict) else {}
        if not (cycle.get("length") == "month" and isinstance(cycle.get("count"), dict)):
            return "degraded"
    if not str(inst.get("normalized_rule") or "").strip():
        return "degraded"
    return "parsed"


def build_guard_summary(inst: dict[str, Any]) -> str:
    parts = [str(inst.get("normalized_rule") or "").strip()]
    cycle = inst.get("cycle") if isinstance(inst.get("cycle"), dict) else {}
    active = cycle.get("active_dates")
    if isinstance(active, list) and active:
        parts.append(f"生效日:{','.join(str(d) for d in active)}")
    checks = inst.get("checks")
    if isinstance(checks, list) and checks:
        parts.append(f"{len(checks)}项检查")
    return "；".join(part for part in parts if part)[:320]


def checks_to_steps(inst: dict[str, Any]) -> list[str]:
    steps: list[str] = []
    checks = inst.get("checks")
    if not isinstance(checks, list):
        return steps
    for index, check in enumerate(checks, start=1):
        if not isinstance(check, dict):
            continue
        unit = f" {check['unit']}" if check.get("unit") else ""
        steps.append(
            f"步骤{index}: 对 {check.get('action')}({check.get('phase')}) "
            f"检查 {check.get('measure')} {check.get('compare')} {check.get('value')}{unit}"
        )
    route_plan = inst.get("route_plan")
    if isinstance(route_plan, dict):
        steps.append(f"步骤{len(steps)+1}: 按 route_plan 顺序完成途经点")
    return steps or [f"遵守: {inst.get('normalized_rule', '')}"]


def build_completion_check(inst: dict[str, Any]) -> str:
    completion = inst.get("completion") if isinstance(inst.get("completion"), dict) else {}
    mode = str(completion.get("mode") or "")
    expires = completion.get("expires_at")
    if expires:
        return f"completion.mode={mode}; expires_at={expires}"
    cycle = inst.get("cycle") if isinstance(inst.get("cycle"), dict) else {}
    count = cycle.get("count")
    if isinstance(count, dict) and count.get("min") is not None:
        return f"completion.mode={mode}; count.min={count.get('min')}"
    return f"completion.mode={mode or 'unknown'}"


def derive_required_fields(inst: dict[str, Any]) -> list[str]:
    fields: set[str] = {"current_wall_clock_time"}
    if str(inst.get("preference_type") or "") == "TARGET_CARGO_MUST_TAKE":
        fields.update({"cargo_id", "cargo", "preference_progress"})
    for check in inst.get("checks") or []:
        if not isinstance(check, dict):
            continue
        measure = str(check.get("measure") or "")
        if measure in {"city", "cargo_name"}:
            fields.add("cargo")
        if measure == "distance":
            fields.add("plan")
        if measure in {"duration", "clock_time", "still_day", "location"}:
            fields.add("recent_actions")
    if inst.get("route_plan"):
        fields.add("location")
    completion = inst.get("completion") if isinstance(inst.get("completion"), dict) else {}
    if completion.get("track_progress"):
        fields.add("preference_progress")
    return sorted(fields)


def _derive_persistent(tier2a: dict[str, Any]) -> bool:
    cycle = tier2a.get("cycle") if isinstance(tier2a.get("cycle"), dict) else {}
    length = str(cycle.get("length") or "")
    return length in {"always", "day", "month"}


def _default_reset(length: str) -> str:
    if length == "day":
        return "calendar_day"
    if length == "month":
        return "never"
    return "never"


def _default_evaluate_at(length: str) -> str:
    if length == "day":
        return "end_of_day"
    if length == "month":
        return "end_of_month"
    if length == "once":
        return "per_action"
    return "per_action"


def _default_phase(routing: dict[str, Any]) -> str:
    if routing.get("needs_history_aggregate"):
        return "history_aggregate"
    if routing.get("needs_sequence"):
        return "executed"
    return "candidate"


def _default_completion_mode(length: str) -> str:
    if length == "always":
        return "never_expires"
    if length == "day":
        return "per_cycle_satisfied"
    if length == "month":
        return "month_quota_met"
    return "window_expires"


def _day_after(date_str: str) -> str:
    from datetime import datetime, timedelta

    try:
        day = datetime.strptime(date_str[:10], "%Y-%m-%d")
        return (day + timedelta(days=1)).strftime("%Y-%m-%dT00:00:00")
    except ValueError:
        return f"{date_str}T00:00:00"


def dumps_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, default=str, separators=(",", ":"))
