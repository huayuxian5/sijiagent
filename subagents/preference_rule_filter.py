from __future__ import annotations

from datetime import timedelta
from typing import Any

from ..domain.models import ActionPlan
from ..domain.rules import SIMULATION_EPOCH, distance_to_minutes, haversine_km, parse_hhmm
from ..messages import TraceContext
from ..state_store import DecisionState, StateStore
from ..telemetry import Telemetry
from .scheme_handlers import (
    DAILY_FIRST_ORDER_DEADLINE_HANDLER,
    GEOFENCE_FORBIDDEN_AREA_HANDLER,
    GEOFENCE_STAY_WITHIN_HANDLER,
    MONTHLY_DEADHEAD_LIMIT_HANDLER,
    TIME_WINDOW_STATIONARY_HANDLER,
)

_DAY_MINUTES = 1440
_MONTH_DAYS = 31
_DEFAULT_RADIUS_KM = 5.0
_SCHEDULE_PREFERENCE_TYPES = {
    "LOCATION_ARRIVAL_DEADLINE",
    "LOCATION_STAY_ON_DATE",
    "ROUTE_SEQUENCE_ON_DATE",
}


class PreferenceRuleFilter:
    """Deterministic hard filter over fixed preference schemes."""

    phase = "FILTER_PREFERENCE_RULES"

    def __init__(self, store: StateStore, telemetry: Telemetry) -> None:
        self._store = store
        self._telemetry = telemetry

    def run(self, state: DecisionState, trace: TraceContext) -> DecisionState:
        self._telemetry.emit(trace, event="AGENT_STARTED", source="PreferenceRuleFilter", phase=self.phase)
        blocked = apply_scheme_filter_to_state(state)
        state.phase = self.phase
        self._store.checkpoint(state, "CKPT_RULE_FILTER_READY")
        self._telemetry.emit(
            trace,
            event="PREFERENCE_RULE_FILTERED",
            source="PreferenceRuleFilter",
            phase=self.phase,
            simulation_minute=state.driver_context.simulation_minute if state.driver_context else None,
            checkpoint_id="CKPT_RULE_FILTER_READY",
            payload={"blocked": blocked},
        )
        return state


def apply_scheme_filter_to_state(state: DecisionState) -> int:
    ctx = state.driver_context
    if ctx is None:
        return 0
    schemes = _schemes_from_state(state)
    if not schemes:
        return 0
    blocked_count = GEOFENCE_STAY_WITHIN_HANDLER.apply_filter_to_state(state)
    blocked_count += GEOFENCE_FORBIDDEN_AREA_HANDLER.apply_filter_to_state(state)
    blocked_count += MONTHLY_DEADHEAD_LIMIT_HANDLER.apply_filter_to_state(state)
    blocked_count += DAILY_FIRST_ORDER_DEADLINE_HANDLER.apply_filter_to_state(state)
    blocked_count += TIME_WINDOW_STATIONARY_HANDLER.apply_filter_to_state(state)
    facts = _ProgressFacts.from_state(state)
    blocked_keys: set[tuple[str, str | None]] = set()
    for plans in (state.simulated_plans, state.ranked_plans):
        if not isinstance(plans, list):
            continue
        for plan in plans:
            if not isinstance(plan, ActionPlan) or not plan.valid:
                continue
            if plan.action != "take_order":
                continue
            reasons = evaluate_plan_against_schemes(plan, schemes, ctx.simulation_minute, ctx.lat, ctx.lng, facts)
            if not reasons:
                continue
            key = (plan.action, plan.cargo_id)
            if key not in blocked_keys:
                blocked_count += 1
                blocked_keys.add(key)
            _block_plan(plan, reasons)
    return blocked_count


def evaluate_plan_against_schemes(
    plan: ActionPlan,
    schemes: list[dict[str, Any]],
    now: int,
    current_lat: float,
    current_lng: float,
    facts: "_ProgressFacts",
) -> list[dict[str, Any]]:
    reasons: list[dict[str, Any]] = []
    target_cargo_ids = _target_cargo_ids_from_schemes(schemes)
    for scheme in schemes:
        if not isinstance(scheme, dict):
            continue
        if not _scheme_is_hard_blocking(scheme, plan.action):
            continue
        pref_type = str(scheme.get("type") or "")
        if pref_type == "ACTION_FORBID":
            reason = _eval_action_forbid(plan, scheme)
        elif pref_type == "NUMERIC_LIMIT":
            reason = _eval_numeric_limit(plan, scheme)
        elif pref_type == "TIME_WINDOW_STATIONARY":
            reason = _eval_time_window_stationary(plan, scheme, now)
        elif pref_type == "DAILY_CONTINUOUS_REST":
            reason = _eval_daily_continuous_rest(plan, scheme, now, facts)
        elif pref_type == "OFF_DAY_QUOTA":
            reason = _eval_off_day_quota(plan, scheme, now, facts)
        elif pref_type == "LOCATION_STAY_ON_DATE":
            reason = _eval_location_stay_on_date(plan, scheme, now, current_lat, current_lng, facts)
        elif pref_type == "LOCATION_ARRIVAL_DEADLINE":
            if str(plan.cargo_id or "") in target_cargo_ids:
                reason = None
            else:
                reason = _eval_location_arrival_deadline(plan, scheme, now)
        elif pref_type == "ROUTE_SEQUENCE_ON_DATE":
            reason = _eval_route_sequence_on_date(plan, scheme, now, current_lat, current_lng)
        elif pref_type == "TARGET_CARGO_MUST_TAKE":
            reason = _eval_target_cargo_must_take(plan, scheme, now)
        else:
            reason = None
        if reason:
            reason.setdefault("preference_id", scheme.get("preference_id"))
            reason.setdefault("type", pref_type)
            reasons.append(reason)
    return reasons


def _target_cargo_ids_from_schemes(schemes: list[dict[str, Any]]) -> set[str]:
    out: set[str] = set()
    for scheme in schemes:
        if not isinstance(scheme, dict) or str(scheme.get("type") or "") != "TARGET_CARGO_MUST_TAKE":
            continue
        constraint = scheme.get("constraint") if isinstance(scheme.get("constraint"), dict) else {}
        cargo_id = str(constraint.get("target_cargo_id") or "").strip()
        if cargo_id:
            out.add(cargo_id)
    return out


def validate_plan_against_schemes(plan: ActionPlan, state: DecisionState) -> str | None:
    ctx = state.driver_context
    if ctx is None:
        return None
    schemes = _schemes_from_state(state)
    if not schemes:
        return None
    facts = _ProgressFacts.from_state(state)
    reasons = evaluate_plan_against_schemes(plan, schemes, ctx.simulation_minute, ctx.lat, ctx.lng, facts)
    if not reasons:
        return None
    first = reasons[0]
    return f"{first.get('preference_id') or first.get('type')}: {first.get('message')}"


def _schemes_from_state(state: DecisionState) -> list[dict[str, Any]]:
    hidden_ids = _hidden_preference_ids(state.preference_progress)
    raw = state.preference_instructions.get("schemes")
    if isinstance(raw, list) and raw:
        return [
            dict(item)
            for item in raw
            if isinstance(item, dict)
            and not _scheme_is_schedule(item)
            and not _scheme_is_hidden(item, hidden_ids)
        ]
    instructions = state.preference_instructions.get("instructions", [])
    schemes: list[dict[str, Any]] = []
    if isinstance(instructions, list):
        for inst in instructions:
            if not isinstance(inst, dict):
                continue
            inst_id = str(inst.get("id") or "")
            if inst_id and inst_id in hidden_ids:
                continue
            scheme = inst.get("scheme")
            if isinstance(scheme, dict):
                item = dict(scheme)
                item.setdefault("preference_id", inst.get("id"))
                item.setdefault("content_key", inst.get("content_key"))
                if _scheme_is_schedule(item):
                    continue
                if _scheme_is_hidden(item, hidden_ids):
                    continue
                schemes.append(item)
    return schemes


def _hidden_preference_ids(progress: dict[str, Any]) -> set[str]:
    if not isinstance(progress, dict):
        return set()
    hidden = {str(item) for item in progress.get("hidden_completed_ids", []) if str(item)}
    statuses = progress.get("preference_statuses")
    if isinstance(statuses, list):
        for item in statuses:
            if not isinstance(item, dict):
                continue
            if str(item.get("status") or "") != "satisfied_hide":
                continue
            pref_id = str(item.get("id") or "")
            if pref_id:
                hidden.add(pref_id)
    return hidden


def _scheme_is_hidden(scheme: dict[str, Any], hidden_ids: set[str]) -> bool:
    if not hidden_ids:
        return False
    if _must_keep_monitoring_type(str(scheme.get("type") or "")):
        return False
    pref_id = str(scheme.get("preference_id") or "")
    return bool(pref_id and pref_id in hidden_ids)


def _scheme_is_schedule(scheme: dict[str, Any]) -> bool:
    if str(scheme.get("type") or "") == "LOCATION_ARRIVAL_DEADLINE":
        return False
    if scheme.get("exclude_from_rule_filter") is True:
        return True
    return str(scheme.get("type") or "") in _SCHEDULE_PREFERENCE_TYPES


def _must_keep_monitoring_type(pref_type: str) -> bool:
    return pref_type in {
        "ACTION_FORBID",
        "NUMERIC_LIMIT",
        "TIME_WINDOW_STATIONARY",
        "DAILY_CONTINUOUS_REST",
        "LOCATION_ARRIVAL_DEADLINE",
        "GEOFENCE_STAY_WITHIN",
        "GEOFENCE_FORBIDDEN_AREA",
        "MONTHLY_DEADHEAD_LIMIT",
        "DAILY_FIRST_ORDER_DEADLINE",
    }


def _scheme_is_hard_blocking(scheme: dict[str, Any], action: str) -> bool:
    if str(scheme.get("hardness") or "").lower() != "hard":
        return False
    pref_type = str(scheme.get("type") or "")
    if (pref_type in _SCHEDULE_PREFERENCE_TYPES or scheme.get("exclude_from_rule_filter") is True) and pref_type != "LOCATION_ARRIVAL_DEADLINE":
        return False
    if action == "wait" and _wait_satisfies_type(pref_type):
        return False
    filter_spec = scheme.get("filter") if isinstance(scheme.get("filter"), dict) else {}
    if filter_spec.get("deterministic") is False:
        return False
    if str(filter_spec.get("effect") or "block") != "block":
        return False
    explicit_blocked = _as_str_list(filter_spec.get("blocked_actions"))
    if _wait_satisfies_type(pref_type):
        explicit_blocked = [item for item in explicit_blocked if item != "wait"]
    actions = (
        explicit_blocked
        or _default_blocked_actions(pref_type)
        or _as_str_list(filter_spec.get("candidate_actions"))
    )
    return not actions or action in actions


def _wait_satisfies_type(pref_type: str) -> bool:
    return pref_type in {
        "TIME_WINDOW_STATIONARY",
        "DAILY_CONTINUOUS_REST",
        "OFF_DAY_QUOTA",
        "LOCATION_STAY_ON_DATE",
    }


def _default_blocked_actions(pref_type: str) -> list[str]:
    if pref_type in {
        "ACTION_FORBID",
        "NUMERIC_LIMIT",
        "TIME_WINDOW_STATIONARY",
        "DAILY_CONTINUOUS_REST",
        "OFF_DAY_QUOTA",
        "TARGET_CARGO_MUST_TAKE",
        "GEOFENCE_STAY_WITHIN",
        "GEOFENCE_FORBIDDEN_AREA",
        "MONTHLY_DEADHEAD_LIMIT",
        "DAILY_FIRST_ORDER_DEADLINE",
        "LOCATION_ARRIVAL_DEADLINE",
    }:
        return ["take_order", "reposition"]
    return []


def _block_plan(plan: ActionPlan, reasons: list[dict[str, Any]]) -> None:
    first = reasons[0]
    message = str(first.get("message") or first.get("type") or "偏好硬过滤阻止")
    plan.valid = False
    plan.score = -1_000_000.0
    plan.reason = f"确定性偏好过滤阻止：{message}"
    plan.meta["rule_filter"] = {
        "blocked": True,
        "source": "fixed_scheme",
        "reasons": reasons,
    }
    pref_eval = dict(plan.meta.get("preference_evaluation") or {})
    pref_eval.update({"source": "fixed_scheme_filter", "blocked": True, "violations": reasons})
    plan.meta["preference_evaluation"] = pref_eval
    plan.meta["future_feasibility"] = {
        "source": "fixed_scheme_filter",
        "feasible": False,
        "blocked": True,
        "preferred": False,
        "reason": message,
    }


def _eval_action_forbid(plan: ActionPlan, scheme: dict[str, Any]) -> dict[str, Any] | None:
    constraint = scheme.get("constraint") if isinstance(scheme.get("constraint"), dict) else {}
    conditions = constraint.get("conditions") if isinstance(constraint.get("conditions"), dict) else {}
    checks = conditions.get("raw_checks") if isinstance(conditions.get("raw_checks"), list) else constraint.get("checks")
    if not isinstance(checks, list):
        return None
    for check in checks:
        if not isinstance(check, dict):
            continue
        if str(check.get("action") or plan.action) != plan.action:
            continue
        if _forbidden_condition_matches(check, plan):
            return {"message": _message(scheme, "候选触发禁止动作/货物/城市条件"), "check": check}
    return None


def _eval_numeric_limit(plan: ActionPlan, scheme: dict[str, Any]) -> dict[str, Any] | None:
    constraint = scheme.get("constraint") if isinstance(scheme.get("constraint"), dict) else {}
    metric = str(constraint.get("metric") or "")
    if metric in {"pickup_deadhead_km", "distance"}:
        actual = _float(plan.meta.get("pickup_km"))
    elif metric == "haul_distance_km":
        actual = _float(plan.meta.get("haul_km"))
    else:
        return None
    limit = _float(constraint.get("value"))
    if limit is None or actual is None:
        return None
    op = str(constraint.get("operator") or "<=")
    failed = actual > limit if op == "<=" else actual < limit if op == ">=" else False
    if failed:
        return {"message": _message(scheme, f"{metric}={actual:.1f} 超出限制 {op}{limit:g}"), "metric": metric, "actual": actual, "limit": limit}
    return None


def _eval_time_window_stationary(plan: ActionPlan, scheme: dict[str, Any], now: int) -> dict[str, Any] | None:
    constraint = scheme.get("constraint") if isinstance(scheme.get("constraint"), dict) else {}
    windows = constraint.get("windows")
    if not isinstance(windows, list) or not windows:
        windows = [{"start_time": "00:00", "end_time": "06:00"}]
    start = now
    end = int(plan.finish_minute or now + plan.duration_minutes)
    for window in windows:
        if not isinstance(window, dict):
            continue
        start_hhmm = str(window.get("start_time") or "00:00")
        end_hhmm = str(window.get("end_time") or "06:00")
        if _interval_overlaps_daily_window(start, end, start_hhmm, end_hhmm):
            return {"message": _message(scheme, f"订单区间 {_wall(start)}~{_wall(end)} 占用必须静止时段 {start_hhmm}-{end_hhmm}")}
    return None


def _eval_target_cargo_must_take(plan: ActionPlan, scheme: dict[str, Any], now: int) -> dict[str, Any] | None:
    constraint = scheme.get("constraint") if isinstance(scheme.get("constraint"), dict) else {}
    target_id = str(constraint.get("target_cargo_id") or "").strip()
    if not target_id:
        return None
    if str(plan.cargo_id or "").strip() == target_id:
        return None

    window_start = _wall_time_to_minute(constraint.get("available_after"))
    window_end = _wall_time_to_minute(constraint.get("available_until"))
    if window_start is None:
        return None

    start = int(now)
    end = int(plan.finish_minute or now + plan.duration_minutes)
    if end <= start:
        return None

    if window_end is not None:
        blocks_target_window = start < window_end and end > window_start
    else:
        blocks_target_window = end > window_start
    if not blocks_target_window:
        return None

    window_text = _wall(window_start)
    if window_end is not None:
        window_text = f"{window_text}~{_wall(window_end)}"
    return {
        "message": _message(
            scheme,
            f"非目标订单 {plan.cargo_id} 执行区间 {_wall(start)}~{_wall(end)} 覆盖指定熟货 {target_id} 可接窗口 {window_text}",
        ),
        "target_cargo_id": target_id,
        "candidate_cargo_id": plan.cargo_id,
        "window_start": window_start,
        "window_end": window_end,
    }


def _eval_daily_continuous_rest(plan: ActionPlan, scheme: dict[str, Any], now: int, facts: "_ProgressFacts") -> dict[str, Any] | None:
    constraint = scheme.get("constraint") if isinstance(scheme.get("constraint"), dict) else {}
    required = _int(constraint.get("min_continuous_minutes")) or 480
    finish = int(plan.finish_minute or now + plan.duration_minutes)
    start_day = max(0, now // _DAY_MINUTES)
    end_day = min(_MONTH_DAYS - 1, max(start_day, (max(now, finish) - 1) // _DAY_MINUTES))
    for day in range(start_day, end_day + 1):
        day_start = day * _DAY_MINUTES
        day_end = day_start + _DAY_MINUTES
        intervals = list(facts.wait_intervals_by_day.get(day, []))
        if facts.longest_wait(day) >= required:
            continue
        if finish < day_end and finish // _DAY_MINUTES == day:
            intervals.append((max(finish, day_start), day_end))
        longest_possible = _longest_merged_span(intervals)
        if longest_possible < required:
            remaining = max(0, day_end - max(finish, day_start)) if finish // _DAY_MINUTES == day else 0
            return {
                "message": _message(scheme, f"{_date(day)} 接单后最长可能连续休息 {longest_possible} 分钟，不足 {required} 分钟"),
                "day": _date(day),
                "longest_possible_minutes": longest_possible,
                "remaining_after_finish_minutes": remaining,
            }
    return None


def _eval_off_day_quota(plan: ActionPlan, scheme: dict[str, Any], now: int, facts: "_ProgressFacts") -> dict[str, Any] | None:
    constraint = scheme.get("constraint") if isinstance(scheme.get("constraint"), dict) else {}
    required = _int(constraint.get("min_days"))
    if not required:
        return None
    finish = int(plan.finish_minute or now + plan.duration_minutes)
    touched = set(range(now // _DAY_MINUTES, min(_MONTH_DAYS - 1, max(now, finish - 1) // _DAY_MINUTES) + 1))
    off_days = facts.completed_off_days - touched
    future_days = {day for day in range((max(now, finish) // _DAY_MINUTES) + 1, _MONTH_DAYS)}
    if len(off_days | future_days) < required:
        return {"message": _message(scheme, f"接单会使本月最多仅 {len(off_days | future_days)} 个全天静止日，少于 {required} 个")}
    return None


def _eval_location_stay_on_date(
    plan: ActionPlan,
    scheme: dict[str, Any],
    now: int,
    current_lat: float,
    current_lng: float,
    facts: "_ProgressFacts",
) -> dict[str, Any] | None:
    date_min = _scheme_date_minute(scheme)
    if date_min is None:
        return None
    day = date_min // _DAY_MINUTES
    target = _target_location(scheme)
    if not target:
        return None
    required = _int(_nested(scheme, "constraint", "min_stay_minutes")) or 120
    if facts.waited_near(day, target) >= required:
        return None
    # Start protecting the task on its date and late previous day.
    if now < date_min - 12 * 60:
        return None
    finish = int(plan.finish_minute or now + plan.duration_minutes)
    end_point = _plan_end_point(plan, current_lat, current_lng)
    travel = _travel_minutes(end_point, target)
    latest_finish = (day + 1) * _DAY_MINUTES
    if finish + travel + required > latest_finish:
        return {
            "message": _message(scheme, f"订单结束后无法在 {_date(day)} 到目标地点停留 {required} 分钟"),
            "target": target,
            "finish_time": _wall(finish),
        }
    return None


def _eval_location_arrival_deadline(plan: ActionPlan, scheme: dict[str, Any], now: int) -> dict[str, Any] | None:
    date_min = _scheme_date_minute(scheme)
    target = _target_location(scheme)
    if date_min is None and str(_nested(scheme, "scope", "period") or "").lower() in {"day", "daily"}:
        date_min = (int(now) // _DAY_MINUTES) * _DAY_MINUTES
    if date_min is None or not target:
        return None
    deadline = _deadline_minute(scheme, date_min)
    if deadline is None or now < date_min - 12 * 60:
        return None
    finish = int(plan.finish_minute or now + plan.duration_minutes)
    travel = _travel_minutes(_plan_end_point(plan, target.get("lat"), target.get("lng")), target)
    if finish + travel > deadline:
        return {"message": _message(scheme, f"订单后无法在 {_wall(deadline)} 前到达目标地点"), "target": target}
    return None


def _eval_route_sequence_on_date(plan: ActionPlan, scheme: dict[str, Any], now: int, current_lat: float, current_lng: float) -> dict[str, Any] | None:
    date_min = _scheme_date_minute(scheme)
    if date_min is None:
        return None
    if now < date_min - 12 * 60:
        return None
    steps = _route_steps_from_scheme(scheme)
    if not steps:
        steps = _fallback_route_steps(scheme)
    if not steps:
        return None
    cursor = int(plan.finish_minute or now + plan.duration_minutes)
    point = _plan_end_point(plan, current_lat, current_lng)
    for step in steps:
        target = step.get("target_location") if isinstance(step.get("target_location"), dict) else None
        if not target:
            continue
        cursor += _travel_minutes(point, target)
        deadline = _step_deadline_minute(step, date_min)
        if deadline is not None and cursor > deadline:
            return {"message": _message(scheme, f"订单后无法按时完成路线步骤 {target.get('name') or target}"), "deadline": _wall(deadline), "arrival": _wall(cursor)}
        stay = _int(step.get("min_stay_minutes")) or 0
        cursor += stay
        complete_before = _wall_time_to_minute(step.get("complete_before"))
        if complete_before is None:
            complete_before = _hhmm_on_date(step.get("complete_before") or step.get("finish_before"), date_min)
        if complete_before is not None and cursor > complete_before:
            return {"message": _message(scheme, f"订单后无法在 {_wall(complete_before)} 前完成停留任务"), "arrival": _wall(cursor)}
        hold_until = _wall_time_to_minute(step.get("hold_until"))
        if hold_until is not None and cursor > hold_until:
            return {"message": _message(scheme, f"订单后无法在 {_wall(hold_until)} 前开始驻留任务"), "arrival": _wall(cursor)}
        point = target
    return None


def _forbidden_condition_matches(check: dict[str, Any], plan: ActionPlan) -> bool:
    measure = str(check.get("measure") or "")
    compare = str(check.get("compare") or "")
    expected = check.get("value")
    if measure == "cargo_name":
        actual = str(plan.meta.get("cargo_name") or "")
        return _matches_forbidden_text(actual, compare, expected)
    if measure == "city":
        cities = _plan_cities(plan)
        if not cities:
            return False
        return any(_matches_forbidden_text(city, compare, expected) for city in cities)
    if measure == "distance":
        phase = str(check.get("phase") or "")
        actual = _float(plan.meta.get("haul_km")) if phase == "haul" else _float(plan.meta.get("pickup_km"))
        limit = _float(expected)
        if actual is None or limit is None:
            return False
        if compare == "max":
            return actual > limit
        if compare == "min":
            return actual < limit
    return False


class _ProgressFacts:
    def __init__(self) -> None:
        self.wait_intervals_by_day: dict[int, list[tuple[int, int]]] = {}
        self.active_minutes_by_day: dict[int, int] = {day: 0 for day in range(_MONTH_DAYS)}
        self.covered_minutes_by_day: dict[int, int] = {day: 0 for day in range(_MONTH_DAYS)}
        self.completed_off_days: set[int] = set()
        self._positioned_waits: list[dict[str, Any]] = []

    @classmethod
    def from_state(cls, state: DecisionState) -> "_ProgressFacts":
        facts = cls()
        spans = cls._action_spans(state)
        for span in spans:
            action = str(span.get("action") or "")
            start = _int(span.get("start"))
            end = _int(span.get("end"))
            if start is None or end is None or end <= start:
                continue
            if action == "wait":
                rest_start = cls._rest_start(span, start)
                facts._add_wait(rest_start, end, span)
            elif action in {"take_order", "reposition"}:
                facts._add_active(start, end)
        for day in range(_MONTH_DAYS):
            if facts.covered_minutes_by_day.get(day, 0) >= _DAY_MINUTES and facts.active_minutes_by_day.get(day, 0) == 0:
                facts.completed_off_days.add(day)
        return facts

    @staticmethod
    def _action_spans(state: DecisionState) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        progress = state.preference_progress if isinstance(state.preference_progress, dict) else {}
        raw_spans = progress.get("action_spans")
        if isinstance(raw_spans, list):
            out.extend(dict(item) for item in raw_spans if isinstance(item, dict))
        return out

    @staticmethod
    def _rest_start(span: dict[str, Any], start: int) -> int:
        starts = [start]
        events = span.get("query_scan_events")
        if isinstance(events, list):
            for event in events:
                if not isinstance(event, dict):
                    continue
                value = _int(event.get("start"))
                if value is not None:
                    starts.append(value)
        return min(starts)

    def _add_wait(self, start: int, end: int, span: dict[str, Any]) -> None:
        cur = start
        while cur < end:
            day = cur // _DAY_MINUTES
            chunk_end = min(end, (day + 1) * _DAY_MINUTES)
            if 0 <= day < _MONTH_DAYS:
                self.wait_intervals_by_day.setdefault(day, []).append((cur, chunk_end))
                self.covered_minutes_by_day[day] = self.covered_minutes_by_day.get(day, 0) + chunk_end - cur
            cur = chunk_end
        self._positioned_waits.append({"start": start, "end": end, "position": span.get("end_position") or span.get("start_position")})

    def _add_active(self, start: int, end: int) -> None:
        cur = start
        while cur < end:
            day = cur // _DAY_MINUTES
            chunk_end = min(end, (day + 1) * _DAY_MINUTES)
            if 0 <= day < _MONTH_DAYS:
                self.active_minutes_by_day[day] = self.active_minutes_by_day.get(day, 0) + chunk_end - cur
                self.covered_minutes_by_day[day] = self.covered_minutes_by_day.get(day, 0) + chunk_end - cur
            cur = chunk_end

    def longest_wait(self, day: int) -> int:
        return _longest_merged_span(self.wait_intervals_by_day.get(day, []))

    def waited_near(self, day: int, target: dict[str, Any]) -> int:
        total = 0
        day_start = day * _DAY_MINUTES
        day_end = day_start + _DAY_MINUTES
        for item in self._positioned_waits:
            position = item.get("position")
            if not isinstance(position, dict):
                continue
            if not _near(position, target):
                continue
            start = max(day_start, int(item.get("start", 0)))
            end = min(day_end, int(item.get("end", 0)))
            if end > start:
                total += end - start
        return total


def _as_str_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if value is None:
        return []
    return [str(value).strip()] if str(value).strip() else []


def _plan_cities(plan: ActionPlan) -> list[str]:
    cities: list[str] = []
    for key in ("start_point", "end_point"):
        point = plan.meta.get(key)
        if isinstance(point, dict) and point.get("city"):
            cities.append(str(point.get("city")))
    return cities


def _matches_forbidden_text(actual: str, compare: str, expected: Any) -> bool:
    needle = str(expected or "")
    if not needle:
        return False
    if compare == "not_contains":
        return needle in actual
    if compare == "contains":
        return needle in actual
    if compare == "equals":
        return actual == needle
    if compare == "not_equals":
        return actual == needle
    return needle in actual


def _interval_overlaps_daily_window(start: int, end: int, start_hhmm: str, end_hhmm: str) -> bool:
    if end <= start:
        return False
    w_start = parse_hhmm(start_hhmm)
    w_end = parse_hhmm(end_hhmm)
    first_day = max(0, start // _DAY_MINUTES - 1)
    last_day = min(_MONTH_DAYS, end // _DAY_MINUTES + 1)
    for day in range(first_day, last_day + 1):
        if w_start <= w_end:
            intervals = [(day * _DAY_MINUTES + w_start, day * _DAY_MINUTES + w_end)]
        else:
            intervals = [
                (day * _DAY_MINUTES + w_start, (day + 1) * _DAY_MINUTES),
                ((day + 1) * _DAY_MINUTES, (day + 1) * _DAY_MINUTES + w_end),
            ]
        for ws, we in intervals:
            if max(start, ws) < min(end, we):
                return True
    return False


def _longest_merged_span(intervals: list[tuple[int, int]]) -> int:
    cleaned = sorted((int(s), int(e)) for s, e in intervals if e > s)
    if not cleaned:
        return 0
    merged: list[list[int]] = []
    for start, end in cleaned:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return max(end - start for start, end in merged)


def _target_location(scheme: dict[str, Any]) -> dict[str, Any] | None:
    target = _nested(scheme, "constraint", "target_location")
    if isinstance(target, dict) and target.get("lat") is not None and target.get("lng") is not None:
        out = dict(target)
        out.setdefault("radius_km", _DEFAULT_RADIUS_KM)
        return out
    return None


def _route_steps_from_scheme(scheme: dict[str, Any]) -> list[dict[str, Any]]:
    steps = _nested(scheme, "constraint", "steps")
    if not isinstance(steps, list):
        return []
    resolved: list[dict[str, Any]] = []
    for item in steps:
        if not isinstance(item, dict):
            continue
        target = item.get("target_location")
        if not isinstance(target, dict) or target.get("lat") is None or target.get("lng") is None:
            continue
        resolved.append(dict(item))
    return resolved


def _fallback_route_steps(scheme: dict[str, Any]) -> list[dict[str, Any]]:
    return []


def _scheme_date_minute(scheme: dict[str, Any]) -> int | None:
    for value in (
        _nested(scheme, "completion", "date"),
        _nested(scheme, "trigger", "date"),
    ):
        minute = _date_to_minute(value)
        if minute is not None:
            return minute
    dates = _nested(scheme, "scope", "active_dates")
    if isinstance(dates, list) and dates:
        return _date_to_minute(dates[0])
    return None


def _date_to_minute(value: Any) -> int | None:
    if not value:
        return None
    text = str(value)[:10]
    try:
        from datetime import datetime

        day = datetime.strptime(text, "%Y-%m-%d")
    except ValueError:
        return None
    return int((day - SIMULATION_EPOCH).total_seconds() // 60)


def _wall_time_to_minute(value: Any) -> int | None:
    if not value:
        return None
    text = str(value).strip().replace("T", " ")
    if not text:
        return None
    if len(text) == 10 and text[4:5] == "-" and text[7:8] == "-":
        text = f"{text} 00:00:00"
    if len(text) == 16:
        text = f"{text}:00"
    if len(text) > 19:
        text = text[:19]
    try:
        from datetime import datetime

        dt = datetime.strptime(text, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None
    return int((dt - SIMULATION_EPOCH).total_seconds() // 60)


def _deadline_minute(scheme: dict[str, Any], date_min: int) -> int | None:
    return _hhmm_on_date(_nested(scheme, "constraint", "arrive_before"), date_min)


def _step_deadline_minute(step: dict[str, Any], date_min: int) -> int | None:
    return _hhmm_on_date(step.get("arrive_before"), date_min)


def _hhmm_on_date(value: Any, date_min: int) -> int | None:
    if not value:
        return None
    try:
        return date_min + parse_hhmm(str(value))
    except Exception:
        return None


def _plan_end_point(plan: ActionPlan, fallback_lat: Any, fallback_lng: Any) -> dict[str, Any]:
    point = plan.meta.get("end_point") or plan.meta.get("target_point")
    if isinstance(point, dict) and point.get("lat") is not None and point.get("lng") is not None:
        return point
    return {"lat": fallback_lat, "lng": fallback_lng}


def _travel_minutes(point: dict[str, Any], target: dict[str, Any]) -> int:
    try:
        return distance_to_minutes(haversine_km(float(point["lat"]), float(point["lng"]), float(target["lat"]), float(target["lng"])))
    except (KeyError, TypeError, ValueError):
        return 0


def _near(point: dict[str, Any], target: dict[str, Any]) -> bool:
    try:
        distance = haversine_km(float(point["lat"]), float(point["lng"]), float(target["lat"]), float(target["lng"]))
        return distance <= float(target.get("radius_km", _DEFAULT_RADIUS_KM) or _DEFAULT_RADIUS_KM)
    except (KeyError, TypeError, ValueError):
        return False


def _nested(data: dict[str, Any], *keys: str) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _message(scheme: dict[str, Any], fallback: str) -> str:
    rule = str(scheme.get("normalized_rule") or "").strip()
    return f"{rule}：{fallback}" if rule else fallback


def _wall(minute: int | None) -> str:
    if minute is None:
        return ""
    return (SIMULATION_EPOCH + timedelta(minutes=int(minute))).strftime("%Y-%m-%d %H:%M")


def _date(day: int) -> str:
    return (SIMULATION_EPOCH + timedelta(days=int(day))).strftime("%Y-%m-%d")


def _float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
