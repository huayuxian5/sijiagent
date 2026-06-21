from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from ..domain.models import ActionPlan
from ..domain.rules import SIMULATION_EPOCH, distance_to_minutes, haversine_km
from ..state_store import DecisionState

SCHEDULE_PREFERENCE_TYPES = {
    "LOCATION_ARRIVAL_DEADLINE",
    "LOCATION_STAY_ON_DATE",
    "ROUTE_SEQUENCE_ON_DATE",
}

TARGET_CARGO_PREP_HORIZON_MINUTES = 24 * 60
TARGET_CARGO_DEADLINE_BUFFER_MINUTES = 10
TARGET_CARGO_PICKUP_PREFERRED_RADIUS_KM = 30.0
ARRIVAL_DEADLINE_BUFFER_MINUTES = 10
SCHEDULE_RADIUS_KM = 5.0
DAY_MINUTES = 1440
MONTH_DAYS = 31
KEEP_MONITORING_TYPES = {
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


def hidden_preference_ids(state: DecisionState) -> set[str]:
    progress = state.preference_progress if isinstance(state.preference_progress, dict) else {}
    hidden = {str(item) for item in progress.get("hidden_completed_ids", []) if str(item)}
    statuses = progress.get("preference_statuses")
    if isinstance(statuses, list):
        for item in statuses:
            if not isinstance(item, dict):
                continue
            if str(item.get("status") or "") == "satisfied_hide" and str(item.get("id") or ""):
                hidden.add(str(item.get("id")))
    return hidden


def instruction_type(inst: dict[str, Any]) -> str:
    scheme = inst.get("scheme") if isinstance(inst.get("scheme"), dict) else {}
    return str(inst.get("preference_type") or scheme.get("type") or "")


def scheme_scope_period(inst: dict[str, Any]) -> str:
    cycle = inst.get("cycle") if isinstance(inst.get("cycle"), dict) else {}
    scheme = inst.get("scheme") if isinstance(inst.get("scheme"), dict) else {}
    scope = scheme.get("scope") if isinstance(scheme.get("scope"), dict) else {}
    return str(cycle.get("length") or scope.get("period") or "").lower()


def is_hard(inst: dict[str, Any]) -> bool:
    scheme = inst.get("scheme") if isinstance(inst.get("scheme"), dict) else {}
    return str(inst.get("hardness") or scheme.get("hardness") or "").lower() == "hard"


def iter_active_instructions(state: DecisionState, pref_type: str) -> list[dict[str, Any]]:
    hidden = hidden_preference_ids(state)
    instructions = state.preference_instructions.get("instructions", [])
    if not isinstance(instructions, list):
        return []
    out: list[dict[str, Any]] = []
    for inst in instructions:
        if not isinstance(inst, dict):
            continue
        inst_id = str(inst.get("id") or "")
        if instruction_type(inst) == pref_type:
            if (inst.get("completed") or inst_id in hidden) and pref_type not in KEEP_MONITORING_TYPES:
                continue
            out.append(inst)
    return out


def sim_to_wall(sim_minute: int | None) -> str | None:
    if sim_minute is None or sim_minute < 0:
        return None
    return (SIMULATION_EPOCH + timedelta(minutes=int(sim_minute))).strftime("%Y-%m-%d %H:%M")


def wall_to_sim_minute(value: Any) -> int | None:
    if value in (None, ""):
        return None
    text = str(value).strip().replace("T", " ")
    if len(text) == 10 and text[4:5] == "-" and text[7:8] == "-":
        text = f"{text} 00:00:00"
    for fmt, size in (("%Y-%m-%d %H:%M:%S", 19), ("%Y-%m-%d %H:%M", 16)):
        try:
            dt = datetime.strptime(text[:size], fmt)
            return int((dt - SIMULATION_EPOCH).total_seconds() // 60)
        except ValueError:
            continue
    return None


def clock_minute(value: Any) -> int | None:
    if value in (None, ""):
        return None
    text = str(value)
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


def is_near_target(lat: float, lng: float, target: Any, default_radius_km: float = SCHEDULE_RADIUS_KM) -> bool:
    if not isinstance(target, dict):
        return False
    try:
        target_lat = float(target.get("lat"))
        target_lng = float(target.get("lng"))
        radius = float(target.get("radius_km") or default_radius_km)
    except (TypeError, ValueError):
        return False
    return haversine_km(lat, lng, target_lat, target_lng) <= radius


def travel_minutes_to_target(lat: float, lng: float, target: dict[str, Any]) -> int | None:
    try:
        target_lat = float(target.get("lat"))
        target_lng = float(target.get("lng"))
    except (TypeError, ValueError):
        return None
    return max(1, distance_to_minutes(haversine_km(lat, lng, target_lat, target_lng)))


def plan_finish_minute(plan: ActionPlan, now: int) -> int | None:
    if plan.finish_minute is not None:
        try:
            return int(plan.finish_minute)
        except (TypeError, ValueError):
            pass
    try:
        return int(now) + int(plan.duration_minutes or 0)
    except (TypeError, ValueError):
        return None


def plan_end_point(plan: ActionPlan) -> dict[str, Any] | None:
    meta = plan.meta if isinstance(plan.meta, dict) else {}
    for key in ("end_point", "target_point"):
        point = meta.get(key)
        if isinstance(point, dict) and point.get("lat") is not None and point.get("lng") is not None:
            return point
    if plan.target_lat is not None and plan.target_lng is not None:
        return {"lat": plan.target_lat, "lng": plan.target_lng}
    return None


def point_distance_km(a: dict[str, Any], b: dict[str, Any]) -> float | None:
    try:
        return haversine_km(float(a.get("lat")), float(a.get("lng")), float(b.get("lat")), float(b.get("lng")))
    except (TypeError, ValueError):
        return None


class TargetCargoMustTakeHandler:
    scheme_type = "TARGET_CARGO_MUST_TAKE"

    def active_ids(self, state: DecisionState) -> set[str]:
        out: set[str] = set()
        for inst in iter_active_instructions(state, self.scheme_type):
            constraint = self._constraint(inst)
            cargo_id = str(constraint.get("target_cargo_id") or "").strip()
            if cargo_id:
                out.add(cargo_id)
        return out

    def active_targets(self, state: DecisionState) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for inst in iter_active_instructions(state, self.scheme_type):
            constraint = self._constraint(inst)
            cargo_id = str(constraint.get("target_cargo_id") or "").strip()
            pickup = constraint.get("pickup_location") if isinstance(constraint.get("pickup_location"), dict) else None
            if not cargo_id or not isinstance(pickup, dict):
                continue
            out.append({
                "preference_id": inst.get("id"),
                "target_cargo_id": cargo_id,
                "pickup_location": pickup,
                "expected_cargo_name": constraint.get("expected_cargo_name"),
                "available_after": constraint.get("available_after"),
                "available_until": constraint.get("available_until"),
            })
        return out

    def preference_for_plan(self, state: DecisionState, plan: ActionPlan) -> dict[str, Any] | None:
        cargo_id = str(plan.cargo_id or "").strip()
        if not cargo_id:
            return None
        for target in self.active_targets(state):
            if str(target.get("target_cargo_id") or "") != cargo_id:
                continue
            return {
                "preference_id": target.get("preference_id"),
                "target_cargo_id": cargo_id,
                "expected_cargo_name": target.get("expected_cargo_name"),
                "available_after": target.get("available_after"),
                "priority": "must_take_if_compliant",
            }
        return None

    def instruction_for_plan(self, state: DecisionState, plan: ActionPlan) -> tuple[dict[str, Any], dict[str, Any]] | None:
        cargo_id = str(plan.cargo_id or "").strip()
        if not cargo_id:
            return None
        for inst in iter_active_instructions(state, self.scheme_type):
            scheme = inst.get("scheme") if isinstance(inst.get("scheme"), dict) else {}
            constraint = scheme.get("constraint") if isinstance(scheme.get("constraint"), dict) else {}
            if str(constraint.get("target_cargo_id") or "").strip() == cargo_id:
                return inst, scheme
        return None

    def force_take_plan(self, state: DecisionState) -> ActionPlan | None:
        target_ids = self.active_ids(state)
        if not target_ids:
            return None
        for source in (state.ranked_plans, state.simulated_plans):
            for plan in source:
                if (
                    isinstance(plan, ActionPlan)
                    and plan.action == "take_order"
                    and str(plan.cargo_id or "") in target_ids
                ):
                    plan.valid = True
                    plan.score = max(float(plan.score or 0.0), 1_000_000.0)
                    plan.reason = f"[Scheme:{self.scheme_type}] force target cargo {plan.cargo_id}"
                    plan.meta.pop("rule_filter", None)
                    pref_eval = plan.meta.get("preference_evaluation")
                    if isinstance(pref_eval, dict) and pref_eval.get("blocked"):
                        plan.meta["preference_evaluation"] = {
                            "source": self.scheme_type,
                            "blocked": False,
                            "overrode_block_for_target_cargo": True,
                        }
                    future = plan.meta.get("future_feasibility")
                    if isinstance(future, dict) and future.get("blocked"):
                        plan.meta["future_feasibility"] = {
                            "source": self.scheme_type,
                            "feasible": True,
                            "blocked": False,
                            "preferred": True,
                            "reason": "target cargo must be taken when visible",
                        }
                    plan.meta["policy_forced_target_cargo"] = True
                    if source is state.simulated_plans and not any(
                        isinstance(existing, ActionPlan)
                        and existing.action == "take_order"
                        and str(existing.cargo_id or "") == str(plan.cargo_id or "")
                        for existing in state.ranked_plans
                    ):
                        state.ranked_plans.insert(0, plan)
                    return plan
        return None

    def plan_for_waiting(self, state: DecisionState) -> ActionPlan | None:
        ctx = state.driver_context
        if ctx is None:
            return None
        now = int(ctx.simulation_minute)
        target_ids = self.active_ids(state)
        if not target_ids:
            return None
        visible_ids = {str(c.cargo_id) for c in state.cargo_snapshot}
        if target_ids & visible_ids:
            return None
        wake = self.next_wakeup_minute(state, now)
        if wake is None or wake <= now:
            return None
        duration = max(1, min(1440, int(wake) - now))
        return ActionPlan(
            "wait",
            {"duration_minutes": duration},
            score=90_000.0,
            reason=f"[Scheme:{self.scheme_type}] wait for target cargo visibility",
            valid=True,
            finish_minute=now + duration,
            duration_minutes=duration,
            meta={
                "kind": "policy_target_cargo_wait",
                "policy_generated": True,
                "preference_generated": True,
                "priority": 1.0,
                "scheme_handler": self.scheme_type,
                "target_cargo_ids": sorted(target_ids),
            },
        )

    def plan_for_positioning(self, state: DecisionState) -> ActionPlan | None:
        ctx = state.driver_context
        if ctx is None:
            return None
        now = int(ctx.simulation_minute)
        for target in self.active_targets(state):
            available_after = wall_to_sim_minute(target.get("available_after"))
            if available_after is not None and now < available_after:
                continue
            pickup = target.get("pickup_location") if isinstance(target.get("pickup_location"), dict) else None
            deadline = wall_to_sim_minute(target.get("available_until"))
            if not isinstance(pickup, dict) or deadline is None or now >= deadline:
                continue
            if is_near_target(ctx.lat, ctx.lng, pickup):
                continue
            travel = travel_minutes_to_target(ctx.lat, ctx.lng, pickup)
            if travel is None:
                continue
            if now + travel + TARGET_CARGO_DEADLINE_BUFFER_MINUTES < deadline:
                continue
            try:
                lat = round(float(pickup.get("lat")), 5)
                lng = round(float(pickup.get("lng")), 5)
            except (TypeError, ValueError):
                continue
            return ActionPlan(
                "reposition",
                {"latitude": lat, "longitude": lng},
                score=80_000.0,
                reason=f"[Scheme:{self.scheme_type}] reposition to target pickup {target.get('target_cargo_id')}",
                valid=True,
                finish_minute=now + travel,
                duration_minutes=travel,
                target_lat=lat,
                target_lng=lng,
                meta={
                    "kind": "policy_target_cargo_positioning",
                    "policy_generated": True,
                    "target_cargo_id": target.get("target_cargo_id"),
                    "deadline_minute": deadline,
                    "target_point": {"lat": lat, "lng": lng},
                    "priority": 1.0,
                    "scheme_handler": self.scheme_type,
                },
            )
        return None

    def next_wakeup_minute(self, state: DecisionState, now: int) -> int | None:
        wakeups: list[int] = []
        for target in self.active_targets(state):
            start = wall_to_sim_minute(target.get("available_after"))
            end = wall_to_sim_minute(target.get("available_until"))
            if end is not None and now >= end:
                continue
            if start is None:
                wakeups.append(now + 1)
            elif now < start:
                wakeups.append(start)
            else:
                wakeups.append(now + 1)
        return min(wakeups) if wakeups else None

    def next_pickup_departure_minute(self, state: DecisionState, now: int) -> int | None:
        ctx = state.driver_context
        if ctx is None:
            return None
        departures: list[int] = []
        for target in self.active_targets(state):
            available_after = wall_to_sim_minute(target.get("available_after"))
            if available_after is not None and now < available_after:
                continue
            pickup = target.get("pickup_location") if isinstance(target.get("pickup_location"), dict) else None
            deadline = wall_to_sim_minute(target.get("available_until"))
            if not isinstance(pickup, dict) or deadline is None or now >= deadline:
                continue
            if is_near_target(ctx.lat, ctx.lng, pickup):
                continue
            travel = travel_minutes_to_target(ctx.lat, ctx.lng, pickup)
            if travel is None:
                continue
            departures.append(max(now + 1, deadline - travel - TARGET_CARGO_DEADLINE_BUFFER_MINUTES))
        return min(departures) if departures else None

    def future_filter(self, state: DecisionState) -> dict[str, int]:
        ctx = state.driver_context
        if ctx is None:
            return {"blocked": 0, "preferred": 0}
        targets = self.active_targets(state)
        if not targets:
            return {"blocked": 0, "preferred": 0}
        now = int(ctx.simulation_minute)
        blocked_count = 0
        preferred_count = 0
        for plan in state.simulated_plans:
            if not isinstance(plan, ActionPlan) or plan.action != "take_order" or not plan.valid:
                continue
            finish = plan_finish_minute(plan, now)
            end_point = plan_end_point(plan)
            if finish is None or not isinstance(end_point, dict):
                continue
            plan_cargo_id = str(plan.cargo_id or "")
            for target in targets:
                cargo_id = str(target.get("target_cargo_id") or "")
                if plan_cargo_id == cargo_id:
                    continue
                pickup = target.get("pickup_location") if isinstance(target.get("pickup_location"), dict) else None
                available_after = wall_to_sim_minute(target.get("available_after"))
                available_until = wall_to_sim_minute(target.get("available_until"))
                if not isinstance(pickup, dict) or available_after is None or available_until is None:
                    continue
                if now < available_after - TARGET_CARGO_PREP_HORIZON_MINUTES or now >= available_until:
                    continue
                distance = point_distance_km(end_point, pickup)
                if distance is None:
                    continue
                travel_after = distance_to_minutes(distance)
                if finish + travel_after + TARGET_CARGO_DEADLINE_BUFFER_MINUTES > available_until:
                    if self._mark_future_blocked(plan, target, distance, finish, available_until):
                        blocked_count += 1
                    break
                if distance <= TARGET_CARGO_PICKUP_PREFERRED_RADIUS_KM and finish <= available_until:
                    if self._mark_future_preferred(plan, target, distance, finish):
                        preferred_count += 1
                    break
        return {"blocked": blocked_count, "preferred": preferred_count}

    @staticmethod
    def _constraint(inst: dict[str, Any]) -> dict[str, Any]:
        scheme = inst.get("scheme") if isinstance(inst.get("scheme"), dict) else {}
        constraint = scheme.get("constraint") if isinstance(scheme.get("constraint"), dict) else {}
        return constraint

    def _mark_future_blocked(
        self,
        plan: ActionPlan,
        target: dict[str, Any],
        distance_km: float,
        finish: int,
        available_until: int,
    ) -> bool:
        future = plan.meta.get("future_feasibility") if isinstance(plan.meta, dict) else None
        if isinstance(future, dict) and future.get("source") == "target_cargo_scheme" and future.get("blocked"):
            return False
        reason = f"after plan dist_to_target_pickup={distance_km:.1f}km; miss target cargo window"
        plan.valid = False
        plan.score = -1_000_000.0
        plan.reason = f"{self.scheme_type} future block: {reason}"
        plan.meta["future_feasibility"] = {
            "source": "target_cargo_scheme",
            "feasible": False,
            "blocked": True,
            "preferred": False,
            "reason": reason,
            "target_cargo_id": target.get("target_cargo_id"),
            "finish_time": sim_to_wall(finish),
            "available_until": sim_to_wall(available_until),
            "distance_to_target_pickup_km": round(distance_km, 1),
            "scheme_handler": self.scheme_type,
        }
        return True

    def _mark_future_preferred(
        self,
        plan: ActionPlan,
        target: dict[str, Any],
        distance_km: float,
        finish: int,
    ) -> bool:
        future = plan.meta.get("future_feasibility") if isinstance(plan.meta, dict) else None
        if isinstance(future, dict) and future.get("blocked"):
            return False
        already = isinstance(future, dict) and future.get("source") == "target_cargo_scheme" and future.get("preferred")
        reason = f"end near target pickup ({distance_km:.1f}km); can connect target cargo"
        plan.meta["future_feasibility"] = {
            "source": "target_cargo_scheme",
            "feasible": True,
            "blocked": False,
            "preferred": True,
            "reason": reason,
            "target_cargo_id": target.get("target_cargo_id"),
            "finish_time": sim_to_wall(finish),
            "distance_to_target_pickup_km": round(distance_km, 1),
            "scheme_handler": self.scheme_type,
        }
        plan.meta["preference_generated"] = True
        plan.meta["priority"] = max(float(plan.meta.get("priority", 0.0) or 0.0), 0.9)
        if not str(plan.reason or "").startswith(f"{self.scheme_type} future prefer"):
            plan.reason = f"{self.scheme_type} future prefer: {reason}"
        return not already


class GeofenceStayWithinHandler:
    scheme_type = "GEOFENCE_STAY_WITHIN"

    def active_boundaries(self, state: DecisionState) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for inst in iter_active_instructions(state, self.scheme_type):
            if not is_hard(inst):
                continue
            scheme = inst.get("scheme") if isinstance(inst.get("scheme"), dict) else {}
            constraint = scheme.get("constraint") if isinstance(scheme.get("constraint"), dict) else {}
            boundary = constraint.get("boundary") if isinstance(constraint.get("boundary"), dict) else None
            if not isinstance(boundary, dict) or str(boundary.get("kind") or "bbox") != "bbox":
                continue
            normalized = self._normalize_bbox(boundary)
            if normalized is None:
                continue
            anchor = constraint.get("anchor_point") if isinstance(constraint.get("anchor_point"), dict) else None
            out.append({
                "preference_id": inst.get("id"),
                "boundary": normalized,
                "anchor_point": anchor,
            })
        return out

    def violations_for_plan(self, state: DecisionState, plan: ActionPlan) -> list[dict[str, Any]]:
        ctx = state.driver_context
        if ctx is None:
            return []
        violations: list[dict[str, Any]] = []
        for item in self.active_boundaries(state):
            boundary = item["boundary"]
            points = self._points_for_plan(ctx.lat, ctx.lng, plan)
            for label, point in points:
                if self.point_inside(point, boundary):
                    continue
                violations.append({
                    "preference_id": item.get("preference_id"),
                    "type": self.scheme_type,
                    "message": f"{label} outside geofence",
                    "point": point,
                    "boundary": boundary,
                    "scheme_handler": self.scheme_type,
                })
                break
        return violations

    def apply_filter_to_state(self, state: DecisionState) -> int:
        blocked = 0
        seen: set[tuple[str, str | None]] = set()
        for plans in (state.simulated_plans, state.ranked_plans):
            if not isinstance(plans, list):
                continue
            for plan in plans:
                if not isinstance(plan, ActionPlan) or not plan.valid:
                    continue
                if plan.action not in {"take_order", "reposition"}:
                    continue
                violations = self.violations_for_plan(state, plan)
                if not violations:
                    continue
                key = (plan.action, str(plan.cargo_id or plan.params.get("cargo_id") or ""))
                if key not in seen:
                    blocked += 1
                    seen.add(key)
                self._block_plan(plan, violations)
        return blocked

    def recovery_plan(self, state: DecisionState) -> ActionPlan | None:
        ctx = state.driver_context
        if ctx is None:
            return None
        current = {"lat": ctx.lat, "lng": ctx.lng}
        for item in self.active_boundaries(state):
            boundary = item["boundary"]
            if self.point_inside(current, boundary):
                continue
            target = self._recovery_target(item, current)
            if target is None:
                continue
            travel = travel_minutes_to_target(ctx.lat, ctx.lng, target)
            if travel is None:
                continue
            lat = round(float(target["lat"]), 5)
            lng = round(float(target["lng"]), 5)
            return ActionPlan(
                "reposition",
                {"latitude": lat, "longitude": lng},
                score=95_000.0,
                reason=f"[Scheme:{self.scheme_type}] recover inside geofence",
                valid=True,
                finish_minute=int(ctx.simulation_minute) + travel,
                duration_minutes=travel,
                target_lat=lat,
                target_lng=lng,
                meta={
                    "kind": "policy_geofence_recovery",
                    "policy_generated": True,
                    "preference_generated": True,
                    "priority": 1.0,
                    "scheme_handler": self.scheme_type,
                    "preference_id": item.get("preference_id"),
                    "target_point": {"lat": lat, "lng": lng},
                    "boundary": boundary,
                },
            )
        return None

    def point_inside(self, point: Any, boundary: dict[str, Any]) -> bool:
        if not isinstance(point, dict):
            return False
        try:
            lat = float(point.get("lat"))
            lng = float(point.get("lng"))
            return (
                float(boundary["min_lat"]) <= lat <= float(boundary["max_lat"])
                and float(boundary["min_lng"]) <= lng <= float(boundary["max_lng"])
            )
        except (KeyError, TypeError, ValueError):
            return False

    @staticmethod
    def _normalize_bbox(boundary: dict[str, Any]) -> dict[str, Any] | None:
        try:
            min_lat = float(boundary.get("min_lat"))
            max_lat = float(boundary.get("max_lat"))
            min_lng = float(boundary.get("min_lng"))
            max_lng = float(boundary.get("max_lng"))
        except (TypeError, ValueError):
            return None
        if min_lat > max_lat:
            min_lat, max_lat = max_lat, min_lat
        if min_lng > max_lng:
            min_lng, max_lng = max_lng, min_lng
        return {"kind": "bbox", "min_lat": min_lat, "max_lat": max_lat, "min_lng": min_lng, "max_lng": max_lng}

    @staticmethod
    def _points_for_plan(current_lat: float, current_lng: float, plan: ActionPlan) -> list[tuple[str, dict[str, Any]]]:
        points: list[tuple[str, dict[str, Any]]] = [("current_position", {"lat": current_lat, "lng": current_lng})]
        meta = plan.meta if isinstance(plan.meta, dict) else {}
        if plan.action == "take_order":
            for label, key in (("pickup", "start_point"), ("dropoff", "end_point")):
                point = meta.get(key)
                if isinstance(point, dict):
                    points.append((label, point))
        elif plan.action == "reposition":
            if plan.target_lat is not None and plan.target_lng is not None:
                points.append(("reposition_target", {"lat": plan.target_lat, "lng": plan.target_lng}))
            else:
                points.append(("reposition_target", {
                    "lat": plan.params.get("latitude", plan.params.get("lat")),
                    "lng": plan.params.get("longitude", plan.params.get("lng")),
                }))
        elif plan.action == "wait":
            points.append(("wait_position", {"lat": current_lat, "lng": current_lng}))
        return points

    def _recovery_target(self, item: dict[str, Any], current: dict[str, Any]) -> dict[str, Any] | None:
        boundary = item.get("boundary") if isinstance(item.get("boundary"), dict) else None
        if not isinstance(boundary, dict):
            return None
        anchor = item.get("anchor_point") if isinstance(item.get("anchor_point"), dict) else None
        if anchor is not None and self.point_inside(anchor, boundary):
            return {"lat": anchor.get("lat"), "lng": anchor.get("lng")}
        try:
            lat = min(max(float(current.get("lat")), float(boundary["min_lat"])), float(boundary["max_lat"]))
            lng = min(max(float(current.get("lng")), float(boundary["min_lng"])), float(boundary["max_lng"]))
        except (KeyError, TypeError, ValueError):
            return None
        return {"lat": lat, "lng": lng}

    @staticmethod
    def _block_plan(plan: ActionPlan, violations: list[dict[str, Any]]) -> None:
        first = violations[0]
        message = str(first.get("message") or "outside geofence")
        plan.valid = False
        plan.score = -1_000_000.0
        plan.reason = f"GEOFENCE_STAY_WITHIN block: {message}"
        plan.meta["geofence_filter"] = {
            "blocked": True,
            "source": "geofence_scheme",
            "violations": violations,
        }
        pref_eval = dict(plan.meta.get("preference_evaluation") or {})
        pref_eval.update({"source": "geofence_scheme", "blocked": True, "violations": violations})
        plan.meta["preference_evaluation"] = pref_eval
        plan.meta["future_feasibility"] = {
            "source": "geofence_scheme",
            "feasible": False,
            "blocked": True,
            "preferred": False,
            "reason": message,
        }


class TimeWindowStationaryHandler:
    scheme_type = "TIME_WINDOW_STATIONARY"

    def active_windows(self, state: DecisionState) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for inst in iter_active_instructions(state, self.scheme_type):
            if not is_hard(inst):
                continue
            scheme = inst.get("scheme") if isinstance(inst.get("scheme"), dict) else {}
            constraint = scheme.get("constraint") if isinstance(scheme.get("constraint"), dict) else {}
            windows = constraint.get("windows") if isinstance(constraint.get("windows"), list) else []
            if not windows:
                windows = [{"start_time": "00:00", "end_time": "06:00"}]
            for window in windows:
                if not isinstance(window, dict):
                    continue
                start = clock_minute(window.get("start_time"))
                end = clock_minute(window.get("end_time"))
                if start is None or end is None or start == end:
                    continue
                out.append({
                    "preference_id": inst.get("id"),
                    "start_minute_of_day": start,
                    "end_minute_of_day": end,
                    "start_time": self._format_clock(start),
                    "end_time": self._format_clock(end),
                    "raw": window,
                })
        return out

    def current_window_end_minute(self, state: DecisionState, now: int) -> int | None:
        ends: list[int] = []
        minute_of_day = int(now) % DAY_MINUTES
        for window in self.active_windows(state):
            start = int(window["start_minute_of_day"])
            end = int(window["end_minute_of_day"])
            in_window = start <= minute_of_day < end if start <= end else minute_of_day >= start or minute_of_day < end
            if not in_window:
                continue
            wait_minutes = (end - minute_of_day) % DAY_MINUTES
            if wait_minutes > 0:
                ends.append(int(now) + wait_minutes)
        return min(ends) if ends else None

    def plan_for_current_window(self, state: DecisionState) -> ActionPlan | None:
        ctx = state.driver_context
        if ctx is None:
            return None
        now = int(ctx.simulation_minute)
        end = self.current_window_end_minute(state, now)
        if end is None or end <= now:
            return None
        duration = end - now
        if duration <= 0:
            return None
        return ActionPlan(
            "wait",
            {"duration_minutes": duration},
            score=95_000.0,
            reason=f"[Scheme:{self.scheme_type}] wait until stationary window ends",
            valid=True,
            finish_minute=now + duration,
            duration_minutes=duration,
            meta={
                "kind": "policy_stationary_window_wait",
                "policy_generated": True,
                "preference_generated": True,
                "priority": 1.0,
                "scheme_handler": self.scheme_type,
            },
        )

    def violations_for_plan(self, state: DecisionState, plan: ActionPlan) -> list[dict[str, Any]]:
        if plan.action not in {"take_order", "reposition"}:
            return []
        ctx = state.driver_context
        if ctx is None:
            return []
        start = int(ctx.simulation_minute)
        finish = plan_finish_minute(plan, start)
        if finish is None:
            finish = start + int(plan.duration_minutes or 0)
        if finish <= start:
            return []
        violations: list[dict[str, Any]] = []
        for window in self.active_windows(state):
            if self._interval_overlaps_window(start, finish, int(window["start_minute_of_day"]), int(window["end_minute_of_day"])):
                violations.append({
                    "preference_id": window.get("preference_id"),
                    "type": self.scheme_type,
                    "message": f"{plan.action} interval {sim_to_wall(start)}~{sim_to_wall(finish)} overlaps stationary window {window['start_time']}-{window['end_time']}",
                    "start_minute": start,
                    "finish_minute": finish,
                    "window": window,
                    "scheme_handler": self.scheme_type,
                })
        return violations

    def apply_filter_to_state(self, state: DecisionState) -> int:
        blocked = 0
        seen: set[tuple[str, str | None, int | None]] = set()
        for plans in (state.simulated_plans, state.ranked_plans):
            if not isinstance(plans, list):
                continue
            for plan in plans:
                if not isinstance(plan, ActionPlan) or not plan.valid:
                    continue
                violations = self.violations_for_plan(state, plan)
                if not violations:
                    continue
                key = (plan.action, str(plan.cargo_id or plan.params.get("cargo_id") or ""), plan.finish_minute)
                if key not in seen:
                    blocked += 1
                    seen.add(key)
                self._block_plan(plan, violations)
        return blocked

    @staticmethod
    def _interval_overlaps_window(start: int, end: int, window_start: int, window_end: int) -> bool:
        if end <= start:
            return False
        first_day = max(0, int(start) // DAY_MINUTES - 1)
        last_day = min(MONTH_DAYS, int(end) // DAY_MINUTES + 1)
        for day in range(first_day, last_day + 1):
            if window_start <= window_end:
                intervals = [(day * DAY_MINUTES + window_start, day * DAY_MINUTES + window_end)]
            else:
                intervals = [
                    (day * DAY_MINUTES + window_start, (day + 1) * DAY_MINUTES),
                    ((day + 1) * DAY_MINUTES, (day + 1) * DAY_MINUTES + window_end),
                ]
            for ws, we in intervals:
                if max(start, ws) < min(end, we):
                    return True
        return False

    @staticmethod
    def _format_clock(value: int) -> str:
        return f"{int(value) // 60:02d}:{int(value) % 60:02d}"

    @staticmethod
    def _block_plan(plan: ActionPlan, violations: list[dict[str, Any]]) -> None:
        first = violations[0]
        message = str(first.get("message") or "stationary window conflict")
        plan.valid = False
        plan.score = -1_000_000.0
        plan.reason = f"TIME_WINDOW_STATIONARY block: {message}"
        plan.meta["stationary_window_filter"] = {
            "blocked": True,
            "source": "time_window_stationary_scheme",
            "violations": violations,
        }
        pref_eval = dict(plan.meta.get("preference_evaluation") or {})
        pref_eval.update({"source": "time_window_stationary_scheme", "blocked": True, "violations": violations})
        plan.meta["preference_evaluation"] = pref_eval
        plan.meta["future_feasibility"] = {
            "source": "time_window_stationary_scheme",
            "feasible": False,
            "blocked": True,
            "preferred": False,
            "reason": message,
        }


class DailyFirstOrderDeadlineHandler:
    scheme_type = "DAILY_FIRST_ORDER_DEADLINE"

    def active_deadlines(self, state: DecisionState) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for inst in iter_active_instructions(state, self.scheme_type):
            scheme = inst.get("scheme") if isinstance(inst.get("scheme"), dict) else {}
            constraint = scheme.get("constraint") if isinstance(scheme.get("constraint"), dict) else {}
            deadline = self._deadline_minute(constraint)
            if deadline is None:
                deadline = 12 * 60
            out.append({
                "preference_id": inst.get("id"),
                "deadline_minute_of_day": deadline,
                "deadline_time": constraint.get("deadline_time") or self._format_clock(deadline),
            })
        return out

    def violations_for_plan(self, state: DecisionState, plan: ActionPlan) -> list[dict[str, Any]]:
        if plan.action != "take_order":
            return []
        ctx = state.driver_context
        if ctx is None:
            return []
        action_start = int(ctx.simulation_minute)
        day = action_start // DAY_MINUTES
        if self._has_take_order_on_day(state, day):
            return []
        minute_of_day = action_start % DAY_MINUTES
        violations: list[dict[str, Any]] = []
        for item in self.active_deadlines(state):
            deadline = int(item["deadline_minute_of_day"])
            if minute_of_day < deadline:
                continue
            violations.append({
                "preference_id": item.get("preference_id"),
                "type": self.scheme_type,
                "message": f"first order starts at {self._format_clock(minute_of_day)}, not before {item['deadline_time']}",
                "action_start_minute": action_start,
                "action_start_time": sim_to_wall(action_start),
                "deadline_minute_of_day": deadline,
                "deadline_time": item["deadline_time"],
                "scheme_handler": self.scheme_type,
            })
        return violations

    def apply_filter_to_state(self, state: DecisionState) -> int:
        blocked = 0
        seen: set[tuple[str, str | None, int | None]] = set()
        for plans in (state.simulated_plans, state.ranked_plans):
            if not isinstance(plans, list):
                continue
            for plan in plans:
                if not isinstance(plan, ActionPlan) or not plan.valid:
                    continue
                violations = self.violations_for_plan(state, plan)
                if not violations:
                    continue
                key = (plan.action, str(plan.cargo_id or plan.params.get("cargo_id") or ""), plan.finish_minute)
                if key not in seen:
                    blocked += 1
                    seen.add(key)
                self._block_plan(plan, violations)
        return blocked

    @staticmethod
    def _has_take_order_on_day(state: DecisionState, day: int) -> bool:
        progress = state.preference_progress if isinstance(state.preference_progress, dict) else {}
        spans = progress.get("action_spans") if isinstance(progress.get("action_spans"), list) else []
        for span in spans:
            if not isinstance(span, dict) or str(span.get("action") or "") != "take_order":
                continue
            try:
                start = int(span.get("start"))
            except (TypeError, ValueError):
                continue
            if start // DAY_MINUTES == day:
                return True
        return False

    @staticmethod
    def _deadline_minute(constraint: dict[str, Any]) -> int | None:
        for key in ("deadline_minute_of_day", "deadline_minute"):
            try:
                value = int(constraint.get(key))
            except (TypeError, ValueError):
                continue
            if 0 <= value < DAY_MINUTES:
                return value
        return clock_minute(constraint.get("deadline_time"))

    @staticmethod
    def _format_clock(value: int) -> str:
        return f"{int(value) // 60:02d}:{int(value) % 60:02d}"

    @staticmethod
    def _block_plan(plan: ActionPlan, violations: list[dict[str, Any]]) -> None:
        first = violations[0]
        message = str(first.get("message") or "daily first order deadline exceeded")
        plan.valid = False
        plan.score = -1_000_000.0
        plan.reason = f"DAILY_FIRST_ORDER_DEADLINE block: {message}"
        plan.meta["daily_first_order_deadline_filter"] = {
            "blocked": True,
            "source": "daily_first_order_deadline_scheme",
            "violations": violations,
        }
        pref_eval = dict(plan.meta.get("preference_evaluation") or {})
        pref_eval.update({"source": "daily_first_order_deadline_scheme", "blocked": True, "violations": violations})
        plan.meta["preference_evaluation"] = pref_eval
        plan.meta["future_feasibility"] = {
            "source": "daily_first_order_deadline_scheme",
            "feasible": False,
            "blocked": True,
            "preferred": False,
            "reason": message,
        }


class MonthlyDeadheadLimitHandler:
    scheme_type = "MONTHLY_DEADHEAD_LIMIT"

    def active_limits(self, state: DecisionState) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for inst in iter_active_instructions(state, self.scheme_type):
            scheme = inst.get("scheme") if isinstance(inst.get("scheme"), dict) else {}
            constraint = scheme.get("constraint") if isinstance(scheme.get("constraint"), dict) else {}
            try:
                limit = float(constraint.get("value"))
            except (TypeError, ValueError):
                continue
            if limit < 0:
                continue
            out.append({
                "preference_id": inst.get("id"),
                "limit_km": limit,
                "metric": constraint.get("metric") or "deadhead_km",
            })
        return out

    def current_deadhead_km(self, state: DecisionState) -> float:
        total = 0.0
        progress = state.preference_progress if isinstance(state.preference_progress, dict) else {}
        spans = progress.get("action_spans") if isinstance(progress.get("action_spans"), list) else []
        for span in spans:
            if not isinstance(span, dict):
                continue
            action = str(span.get("action") or "")
            if action == "take_order":
                total += self._pickup_deadhead_from_span(span) or 0.0
            elif action == "reposition":
                total += self._float(span.get("distance_km")) or self._distance_from_positions(span) or 0.0
        return total

    def deadhead_for_plan(self, state: DecisionState, plan: ActionPlan) -> float:
        meta = plan.meta if isinstance(plan.meta, dict) else {}
        if plan.action == "take_order":
            value = self._float(meta.get("pickup_km"))
            if value is None:
                value = self._float(meta.get("pickup_deadhead_km"))
            if value is not None:
                return value
            ctx = state.driver_context
            start_point = meta.get("start_point") if isinstance(meta.get("start_point"), dict) else None
            if ctx is None or start_point is None:
                return 0.0
            try:
                return haversine_km(ctx.lat, ctx.lng, float(start_point["lat"]), float(start_point["lng"]))
            except (KeyError, TypeError, ValueError):
                return 0.0
        if plan.action == "reposition":
            value = self._float(meta.get("distance_km"))
            if value is not None:
                return value
            ctx = state.driver_context
            if ctx is None:
                return 0.0
            try:
                lat = float(plan.target_lat if plan.target_lat is not None else plan.params.get("latitude", plan.params.get("lat")))
                lng = float(plan.target_lng if plan.target_lng is not None else plan.params.get("longitude", plan.params.get("lng")))
            except (TypeError, ValueError):
                return 0.0
            return haversine_km(ctx.lat, ctx.lng, lat, lng)
        return 0.0

    def violations_for_plan(self, state: DecisionState, plan: ActionPlan) -> list[dict[str, Any]]:
        if plan.action not in {"take_order", "reposition"}:
            return []
        added = self.deadhead_for_plan(state, plan)
        if added <= 0:
            return []
        current = self.current_deadhead_km(state)
        violations: list[dict[str, Any]] = []
        for item in self.active_limits(state):
            limit = float(item["limit_km"])
            projected = current + added
            if projected <= limit:
                continue
            violations.append({
                "preference_id": item.get("preference_id"),
                "type": self.scheme_type,
                "message": f"monthly deadhead {projected:.1f}km exceeds limit {limit:.1f}km",
                "current_deadhead_km": round(current, 2),
                "candidate_deadhead_km": round(added, 2),
                "projected_deadhead_km": round(projected, 2),
                "limit_km": round(limit, 2),
                "scheme_handler": self.scheme_type,
            })
        return violations

    def apply_filter_to_state(self, state: DecisionState) -> int:
        blocked = 0
        seen: set[tuple[str, str | None, int | None]] = set()
        for plans in (state.simulated_plans, state.ranked_plans):
            if not isinstance(plans, list):
                continue
            for plan in plans:
                if not isinstance(plan, ActionPlan) or not plan.valid:
                    continue
                violations = self.violations_for_plan(state, plan)
                if not violations:
                    continue
                key = (plan.action, str(plan.cargo_id or plan.params.get("cargo_id") or ""), plan.finish_minute)
                if key not in seen:
                    blocked += 1
                    seen.add(key)
                self._block_plan(plan, violations)
        return blocked

    @staticmethod
    def _float(value: Any) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _distance_from_positions(span: dict[str, Any]) -> float | None:
        start = span.get("start_position")
        end = span.get("end_position")
        if not isinstance(start, dict) or not isinstance(end, dict):
            return None
        try:
            return haversine_km(float(start["lat"]), float(start["lng"]), float(end["lat"]), float(end["lng"]))
        except (KeyError, TypeError, ValueError):
            return None

    def _pickup_deadhead_from_span(self, span: dict[str, Any]) -> float | None:
        value = self._float(span.get("pickup_deadhead_km"))
        if value is not None:
            return value
        start = span.get("start_position")
        pickup = span.get("start_point")
        if not isinstance(start, dict) or not isinstance(pickup, dict):
            return None
        try:
            return haversine_km(float(start["lat"]), float(start["lng"]), float(pickup["lat"]), float(pickup["lng"]))
        except (KeyError, TypeError, ValueError):
            return None

    @staticmethod
    def _block_plan(plan: ActionPlan, violations: list[dict[str, Any]]) -> None:
        first = violations[0]
        message = str(first.get("message") or "monthly deadhead limit exceeded")
        plan.valid = False
        plan.score = -1_000_000.0
        plan.reason = f"MONTHLY_DEADHEAD_LIMIT block: {message}"
        plan.meta["monthly_deadhead_filter"] = {
            "blocked": True,
            "source": "monthly_deadhead_scheme",
            "violations": violations,
        }
        pref_eval = dict(plan.meta.get("preference_evaluation") or {})
        pref_eval.update({"source": "monthly_deadhead_scheme", "blocked": True, "violations": violations})
        plan.meta["preference_evaluation"] = pref_eval
        plan.meta["future_feasibility"] = {
            "source": "monthly_deadhead_scheme",
            "feasible": False,
            "blocked": True,
            "preferred": False,
            "reason": message,
        }


class GeofenceForbiddenAreaHandler:
    scheme_type = "GEOFENCE_FORBIDDEN_AREA"
    RECOVERY_MARGIN_KM = 2.0

    def active_areas(self, state: DecisionState) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for inst in iter_active_instructions(state, self.scheme_type):
            if not is_hard(inst):
                continue
            scheme = inst.get("scheme") if isinstance(inst.get("scheme"), dict) else {}
            constraint = scheme.get("constraint") if isinstance(scheme.get("constraint"), dict) else {}
            boundary = constraint.get("boundary") if isinstance(constraint.get("boundary"), dict) else None
            normalized = self._normalize_circle(boundary)
            if normalized is None:
                continue
            out.append({"preference_id": inst.get("id"), "boundary": normalized})
        return out

    def violations_for_plan(self, state: DecisionState, plan: ActionPlan) -> list[dict[str, Any]]:
        ctx = state.driver_context
        if ctx is None:
            return []
        violations: list[dict[str, Any]] = []
        for item in self.active_areas(state):
            boundary = item["boundary"]
            for label, point in self._points_for_plan(ctx.lat, ctx.lng, plan):
                distance = self._distance_to_center(point, boundary)
                if distance is None or distance > float(boundary["radius_km"]):
                    continue
                violations.append({
                    "preference_id": item.get("preference_id"),
                    "type": self.scheme_type,
                    "message": f"{label} enters forbidden circle",
                    "point": point,
                    "distance_to_center_km": round(distance, 2),
                    "boundary": boundary,
                    "scheme_handler": self.scheme_type,
                })
                break
        return violations

    def apply_filter_to_state(self, state: DecisionState) -> int:
        blocked = 0
        seen: set[tuple[str, str | None, int | None]] = set()
        for plans in (state.simulated_plans, state.ranked_plans):
            if not isinstance(plans, list):
                continue
            for plan in plans:
                if not isinstance(plan, ActionPlan) or not plan.valid:
                    continue
                violations = self.violations_for_plan(state, plan)
                if not violations:
                    continue
                key = (plan.action, str(plan.cargo_id or plan.params.get("cargo_id") or ""), plan.finish_minute)
                if key not in seen:
                    blocked += 1
                    seen.add(key)
                self._block_plan(plan, violations)
        return blocked

    def recovery_plan(self, state: DecisionState) -> ActionPlan | None:
        ctx = state.driver_context
        if ctx is None:
            return None
        current = {"lat": ctx.lat, "lng": ctx.lng}
        for item in self.active_areas(state):
            boundary = item["boundary"]
            distance = self._distance_to_center(current, boundary)
            if distance is None or distance > float(boundary["radius_km"]):
                continue
            target = self._outside_target(current, boundary)
            if target is None:
                continue
            travel = travel_minutes_to_target(ctx.lat, ctx.lng, target)
            if travel is None:
                continue
            lat = round(float(target["lat"]), 5)
            lng = round(float(target["lng"]), 5)
            return ActionPlan(
                "reposition",
                {"latitude": lat, "longitude": lng},
                score=96_000.0,
                reason=f"[Scheme:{self.scheme_type}] leave forbidden area",
                valid=True,
                finish_minute=int(ctx.simulation_minute) + travel,
                duration_minutes=travel,
                target_lat=lat,
                target_lng=lng,
                meta={
                    "kind": "policy_forbidden_geofence_recovery",
                    "policy_generated": True,
                    "preference_generated": True,
                    "priority": 1.0,
                    "scheme_handler": self.scheme_type,
                    "preference_id": item.get("preference_id"),
                    "target_point": {"lat": lat, "lng": lng},
                    "boundary": boundary,
                },
            )
        return None

    @staticmethod
    def _normalize_circle(boundary: Any) -> dict[str, Any] | None:
        if not isinstance(boundary, dict) or str(boundary.get("kind") or "circle") != "circle":
            return None
        try:
            lat = float(boundary.get("lat"))
            lng = float(boundary.get("lng"))
            radius = float(boundary.get("radius_km"))
        except (TypeError, ValueError):
            return None
        if not (-90 <= lat <= 90 and -180 <= lng <= 180) or radius <= 0:
            return None
        return {"kind": "circle", "lat": lat, "lng": lng, "radius_km": radius}

    @staticmethod
    def _points_for_plan(current_lat: float, current_lng: float, plan: ActionPlan) -> list[tuple[str, dict[str, Any]]]:
        points: list[tuple[str, dict[str, Any]]] = [("current_position", {"lat": current_lat, "lng": current_lng})]
        meta = plan.meta if isinstance(plan.meta, dict) else {}
        if plan.action == "take_order":
            for label, key in (("pickup", "start_point"), ("dropoff", "end_point")):
                point = meta.get(key)
                if isinstance(point, dict):
                    points.append((label, point))
        elif plan.action == "reposition":
            if plan.target_lat is not None and plan.target_lng is not None:
                points.append(("reposition_target", {"lat": plan.target_lat, "lng": plan.target_lng}))
            else:
                points.append(("reposition_target", {
                    "lat": plan.params.get("latitude", plan.params.get("lat")),
                    "lng": plan.params.get("longitude", plan.params.get("lng")),
                }))
        elif plan.action == "wait":
            points.append(("wait_position", {"lat": current_lat, "lng": current_lng}))
        return points

    @staticmethod
    def _distance_to_center(point: Any, boundary: dict[str, Any]) -> float | None:
        if not isinstance(point, dict):
            return None
        try:
            return haversine_km(float(point.get("lat")), float(point.get("lng")), float(boundary["lat"]), float(boundary["lng"]))
        except (KeyError, TypeError, ValueError):
            return None

    def _outside_target(self, current: dict[str, Any], boundary: dict[str, Any]) -> dict[str, Any] | None:
        try:
            center_lat = float(boundary["lat"])
            center_lng = float(boundary["lng"])
            cur_lat = float(current["lat"])
            cur_lng = float(current["lng"])
            radius = float(boundary["radius_km"]) + self.RECOVERY_MARGIN_KM
        except (KeyError, TypeError, ValueError):
            return None
        d_lat = cur_lat - center_lat
        d_lng = cur_lng - center_lng
        if abs(d_lat) < 1e-6 and abs(d_lng) < 1e-6:
            return {"lat": center_lat + radius / 111.0, "lng": center_lng}
        scale = radius / max(0.001, haversine_km(center_lat, center_lng, cur_lat, cur_lng))
        return {"lat": center_lat + d_lat * scale, "lng": center_lng + d_lng * scale}

    @staticmethod
    def _block_plan(plan: ActionPlan, violations: list[dict[str, Any]]) -> None:
        first = violations[0]
        message = str(first.get("message") or "forbidden geofence conflict")
        plan.valid = False
        plan.score = -1_000_000.0
        plan.reason = f"GEOFENCE_FORBIDDEN_AREA block: {message}"
        plan.meta["forbidden_geofence_filter"] = {
            "blocked": True,
            "source": "forbidden_geofence_scheme",
            "violations": violations,
        }
        pref_eval = dict(plan.meta.get("preference_evaluation") or {})
        pref_eval.update({"source": "forbidden_geofence_scheme", "blocked": True, "violations": violations})
        plan.meta["preference_evaluation"] = pref_eval
        plan.meta["future_feasibility"] = {
            "source": "forbidden_geofence_scheme",
            "feasible": False,
            "blocked": True,
            "preferred": False,
            "reason": message,
        }


class LocationArrivalDeadlineHandler:
    scheme_type = "LOCATION_ARRIVAL_DEADLINE"

    def active_daily_targets(self, state: DecisionState, now: int) -> list[tuple[dict[str, Any], int]]:
        today = (sim_to_wall(now) or "")[:10]
        if not today:
            return []
        out: list[tuple[dict[str, Any], int]] = []
        for inst in iter_active_instructions(state, self.scheme_type):
            if scheme_scope_period(inst) not in {"day", "daily"}:
                continue
            if not is_hard(inst):
                continue
            scheme = inst.get("scheme") if isinstance(inst.get("scheme"), dict) else {}
            constraint = scheme.get("constraint") if isinstance(scheme.get("constraint"), dict) else {}
            target = constraint.get("target_location") if isinstance(constraint.get("target_location"), dict) else None
            deadline_text = constraint.get("arrive_before")
            if not isinstance(target, dict) or not deadline_text:
                continue
            base = wall_to_sim_minute(f"{today} 00:00")
            clock = clock_minute(deadline_text)
            if base is not None and clock is not None:
                out.append((target, base + clock))
        return out

    def next_departure_minute(self, state: DecisionState, now: int) -> int | None:
        ctx = state.driver_context
        if ctx is None:
            return None
        departures: list[int] = []
        for target, deadline in self.active_daily_targets(state, now):
            if is_near_target(ctx.lat, ctx.lng, target):
                continue
            travel = travel_minutes_to_target(ctx.lat, ctx.lng, target)
            if travel is None:
                continue
            latest = int(deadline) - int(travel) - ARRIVAL_DEADLINE_BUFFER_MINUTES
            if now < int(deadline):
                departures.append(max(now + 1, latest))
        return min(departures) if departures else None

    def plan_for_departure_guard(self, state: DecisionState) -> ActionPlan | None:
        ctx = state.driver_context
        if ctx is None:
            return None
        now = int(ctx.simulation_minute)
        best: tuple[int, dict[str, Any], int] | None = None
        for target, deadline in self.active_daily_targets(state, now):
            if is_near_target(ctx.lat, ctx.lng, target):
                continue
            travel = travel_minutes_to_target(ctx.lat, ctx.lng, target)
            if travel is None or now >= int(deadline):
                continue
            latest = int(deadline) - int(travel) - ARRIVAL_DEADLINE_BUFFER_MINUTES
            candidate = (latest, target, travel)
            if best is None or candidate[0] < best[0]:
                best = candidate
        if best is None:
            return None
        latest, target, travel = best
        try:
            lat = round(float(target.get("lat")), 5)
            lng = round(float(target.get("lng")), 5)
        except (TypeError, ValueError):
            return None
        if now < latest:
            duration = max(1, min(1440, latest - now))
            return ActionPlan(
                "wait",
                {"duration_minutes": duration},
                score=70_000.0,
                reason=f"[Scheme:{self.scheme_type}] wait until departure deadline",
                valid=True,
                finish_minute=now + duration,
                duration_minutes=duration,
                meta={
                    "kind": "policy_arrival_deadline_wait",
                    "policy_generated": True,
                    "preference_generated": True,
                    "priority": 1.0,
                    "scheme_handler": self.scheme_type,
                    "deadline_minute": int(best[0] + travel + ARRIVAL_DEADLINE_BUFFER_MINUTES),
                    "departure_minute": latest,
                    "target_point": {"lat": lat, "lng": lng},
                },
            )
        return ActionPlan(
            "reposition",
            {"latitude": lat, "longitude": lng},
            score=90_000.0,
            reason=f"[Scheme:{self.scheme_type}] reposition before arrival deadline",
            valid=True,
            finish_minute=now + travel,
            duration_minutes=travel,
            target_lat=lat,
            target_lng=lng,
            meta={
                "kind": "policy_arrival_deadline_positioning",
                "policy_generated": True,
                "preference_generated": True,
                "priority": 1.0,
                "scheme_handler": self.scheme_type,
                "deadline_minute": int(best[0] + travel + ARRIVAL_DEADLINE_BUFFER_MINUTES),
                "departure_minute": latest,
                "target_point": {"lat": lat, "lng": lng},
            },
        )


TARGET_CARGO_HANDLER = TargetCargoMustTakeHandler()
GEOFENCE_STAY_WITHIN_HANDLER = GeofenceStayWithinHandler()
GEOFENCE_FORBIDDEN_AREA_HANDLER = GeofenceForbiddenAreaHandler()
MONTHLY_DEADHEAD_LIMIT_HANDLER = MonthlyDeadheadLimitHandler()
DAILY_FIRST_ORDER_DEADLINE_HANDLER = DailyFirstOrderDeadlineHandler()
TIME_WINDOW_STATIONARY_HANDLER = TimeWindowStationaryHandler()
LOCATION_ARRIVAL_DEADLINE_HANDLER = LocationArrivalDeadlineHandler()

SCHEME_HANDLERS = {
    TARGET_CARGO_HANDLER.scheme_type: TARGET_CARGO_HANDLER,
    GEOFENCE_STAY_WITHIN_HANDLER.scheme_type: GEOFENCE_STAY_WITHIN_HANDLER,
    GEOFENCE_FORBIDDEN_AREA_HANDLER.scheme_type: GEOFENCE_FORBIDDEN_AREA_HANDLER,
    MONTHLY_DEADHEAD_LIMIT_HANDLER.scheme_type: MONTHLY_DEADHEAD_LIMIT_HANDLER,
    DAILY_FIRST_ORDER_DEADLINE_HANDLER.scheme_type: DAILY_FIRST_ORDER_DEADLINE_HANDLER,
    TIME_WINDOW_STATIONARY_HANDLER.scheme_type: TIME_WINDOW_STATIONARY_HANDLER,
    LOCATION_ARRIVAL_DEADLINE_HANDLER.scheme_type: LOCATION_ARRIVAL_DEADLINE_HANDLER,
}


def exclude_from_future(inst: dict[str, Any]) -> bool:
    pref_type = instruction_type(inst)
    if pref_type == TARGET_CARGO_HANDLER.scheme_type:
        return False
    if inst.get("exclude_from_future") is True:
        return True
    if isinstance(inst.get("schedule_task"), dict):
        return True
    if pref_type in SCHEDULE_PREFERENCE_TYPES:
        return True
    scheme = inst.get("scheme") if isinstance(inst.get("scheme"), dict) else {}
    return bool(isinstance(scheme, dict) and scheme.get("exclude_from_future") is True)
