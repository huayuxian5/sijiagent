"""Evaluate preference.v2 checks against candidate/final actions (generic, no per-driver rules)."""
from __future__ import annotations

from datetime import timedelta
from typing import Any

from ..domain.models import ActionPlan, DriverContext
from ..domain.rules import SIMULATION_EPOCH, is_in_daily_window, minute_of_day, wall_time_to_minute


def _as_str_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if value is None:
        return []
    return [str(value).strip()] if str(value).strip() else []


def _simulation_date(simulation_minute: int) -> str:
    dt = SIMULATION_EPOCH + timedelta(minutes=int(simulation_minute))
    return dt.strftime("%Y-%m-%d")


def _visible_window(inst: dict[str, Any]) -> tuple[int | None, int | None]:
    meta = inst.get("meta") if isinstance(inst.get("meta"), dict) else {}
    start = meta.get("visible_from")
    end = meta.get("visible_until")
    start_min = wall_time_to_minute(str(start)) if start else None
    end_min = wall_time_to_minute(str(end)) if end else None
    return start_min, end_min


def instruction_visible(inst: dict[str, Any], simulation_minute: int) -> bool:
    start_min, end_min = _visible_window(inst)
    if start_min is not None and simulation_minute < start_min:
        return False
    if end_min is not None and simulation_minute > end_min:
        return False
    return True


def instruction_in_cycle_window(inst: dict[str, Any], simulation_minute: int) -> bool:
    cycle = inst.get("cycle") if isinstance(inst.get("cycle"), dict) else {}
    scope = inst.get("scope") if isinstance(inst.get("scope"), dict) else {}
    length = str(cycle.get("length") or "always")
    applies = str(scope.get("applies_when") or "always_active")

    if length == "once" or applies == "in_cycle_window":
        active_dates = _as_str_list(cycle.get("active_dates"))
        if active_dates:
            return _simulation_date(simulation_minute) in active_dates
        window = cycle.get("window")
        if isinstance(window, dict):
            start_raw = str(window.get("start") or "").strip()
            end_raw = str(window.get("end") or "").strip()
            if start_raw and end_raw:
                try:
                    start_min = wall_time_to_minute(start_raw.replace("T", " ").replace("Z", "")[:19])
                    end_min = wall_time_to_minute(end_raw.replace("T", " ").replace("Z", "")[:19])
                    return start_min <= simulation_minute < end_min
                except ValueError:
                    pass
    return True


def _instruction_blocks_action(inst: dict[str, Any], plan_action: str) -> bool:
    on_fail = inst.get("on_fail") if isinstance(inst.get("on_fail"), dict) else {}
    if str(on_fail.get("effect") or "block") != "block":
        return False
    if str(inst.get("hardness") or "").lower() != "hard":
        return False
    blocked = _as_str_list(on_fail.get("block_actions"))
    return not blocked or plan_action in blocked


def _scope_applies(inst: dict[str, Any], plan_action: str) -> bool:
    scope = inst.get("scope") if isinstance(inst.get("scope"), dict) else {}
    actions = _as_str_list(scope.get("actions"))
    if actions and plan_action not in actions:
        return False
    phase = str(scope.get("phase") or "candidate")
    # Hard filter at plan selection only applies to candidate-scope rules.
    if phase != "candidate":
        return False
    return True


def _city_values_for_plan(plan: ActionPlan) -> list[str]:
    meta = plan.meta if isinstance(plan.meta, dict) else {}
    cities: list[str] = []
    for key in ("start_point", "end_point"):
        point = meta.get(key)
        if isinstance(point, dict):
            city = str(point.get("city") or "").strip()
            if city:
                cities.append(city)
    return cities


def _window_bounds(value: Any) -> tuple[str, str] | None:
    if not isinstance(value, dict):
        return None
    start = str(value.get("from") or value.get("start") or "").strip()
    end = str(value.get("to") or value.get("end") or "").strip()
    if start and end:
        return start, end
    return None


def _compare_check(
    check: dict[str, Any],
    plan: ActionPlan,
    ctx: DriverContext | None,
    simulation_minute: int,
) -> bool | None:
    """Return True if check passes, False if fails, None if not applicable."""
    action = str(check.get("action") or "")
    if action and action != plan.action:
        return None

    measure = str(check.get("measure") or "")
    compare = str(check.get("compare") or "")
    expected = check.get("value")
    phase = str(check.get("phase") or "whole")
    meta = plan.meta if isinstance(plan.meta, dict) else {}

    if measure == "distance" and phase in {"to_pickup", "whole", "haul"}:
        actual = float(meta.get("haul_km", 0) or 0) if phase == "haul" else float(meta.get("pickup_km", 0) or 0)
        limit = float(expected)
        if compare == "max":
            return actual <= limit
        if compare == "min":
            return actual >= limit
        return None

    if measure == "cargo_name" and plan.action == "take_order":
        actual = str(meta.get("cargo_name") or "")
        needle = str(expected or "")
        if compare == "not_contains":
            return needle not in actual
        if compare == "contains":
            return needle in actual
        if compare == "equals":
            return actual == needle
        if compare == "not_equals":
            return actual != needle
        return None

    if measure == "city":
        cities = _city_values_for_plan(plan)
        city_name = str(expected or "")
        if compare == "not_contains":
            if cities:
                return all(city_name not in city for city in cities)
            return None
        if compare == "contains":
            if cities:
                return any(city_name in city for city in cities)
            return None
        return None

    if measure == "clock_time":
        bounds = _window_bounds(expected)
        if not bounds:
            return None
        start_hhmm, end_hhmm = bounds
        in_window = is_in_daily_window(simulation_minute, start_hhmm, end_hhmm)
        if compare == "in_window":
            return in_window
        if compare == "not_in_window":
            return not in_window
        return None

    return None


def evaluate_plan_violations(
    plan: ActionPlan,
    instructions: list[dict[str, Any]],
    ctx: DriverContext | None,
) -> list[dict[str, str]]:
    if not plan.valid or plan.action not in {"take_order", "reposition", "wait"}:
        return []
    simulation_minute = int(ctx.simulation_minute) if ctx is not None else 0
    violations: list[dict[str, str]] = []

    for inst in instructions:
        if not isinstance(inst, dict):
            continue
        if str(inst.get("schema_version") or "") != "preference.v2":
            continue
        if not instruction_visible(inst, simulation_minute):
            continue
        if not instruction_in_cycle_window(inst, simulation_minute):
            continue
        if not _scope_applies(inst, plan.action):
            continue
        if not _instruction_blocks_action(inst, plan.action):
            continue

        checks = inst.get("checks")
        if not isinstance(checks, list) or not checks:
            continue

        inst_id = str(inst.get("id") or "")
        for check in checks:
            if not isinstance(check, dict):
                continue
            result = _compare_check(check, plan, ctx, simulation_minute)
            if result is False:
                violations.append({
                    "pref_id": inst_id,
                    "check": f"{check.get('action')}/{check.get('phase')}/{check.get('measure')}",
                    "reason": str(inst.get("normalized_rule") or inst.get("rule") or "")[:160],
                })
                break
    return violations


def apply_hard_filter_to_state(state) -> int:
    """Mark plans invalid when hard v2 checks fail. Returns blocked count."""
    ctx = state.driver_context
    instructions = state.preference_instructions.get("instructions", [])
    if not isinstance(instructions, list) or not instructions:
        return 0

    blocked = 0
    for plans in (state.simulated_plans, state.ranked_plans):
        if not isinstance(plans, list):
            continue
        for plan in plans:
            if not isinstance(plan, ActionPlan) or not plan.valid:
                continue
            violations = evaluate_plan_violations(plan, instructions, ctx)
            if not violations:
                continue
            blocked += 1
            first = violations[0]
            plan.valid = False
            plan.score = -1_000_000.0
            plan.reason = f"偏好硬过滤 {first.get('pref_id')}: {first.get('reason') or first.get('check')}"
            pref_eval = dict(plan.meta.get("preference_evaluation") or {})
            pref_eval.update({
                "source": "v2_hard_filter",
                "blocked": True,
                "violations": violations,
            })
            plan.meta["preference_evaluation"] = pref_eval
            plan.meta["future_feasibility"] = {
                "source": "v2_hard_filter",
                "feasible": False,
                "blocked": True,
                "preferred": False,
                "reason": plan.reason,
            }
    return blocked


def validate_final_plan(plan: ActionPlan, instructions: list[dict[str, Any]], ctx: DriverContext | None) -> str | None:
    violations = evaluate_plan_violations(plan, instructions, ctx)
    if violations:
        first = violations[0]
        return f"{first.get('pref_id')}: {first.get('reason') or first.get('check')}"
    return None
