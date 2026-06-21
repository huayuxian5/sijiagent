from __future__ import annotations

import json
import logging
from datetime import timedelta
from typing import Any

from ..domain.rules import DEFAULT_MONTH_HORIZON_MINUTES, day_index, haversine_km, wall_time_to_minute
from ..gateway import GatewayLayer
from ..messages import TraceContext
from ..state_store import DecisionState, StateStore
from ..telemetry import Telemetry

logger = logging.getLogger("agent.query_agent")

QUERY_SYSTEM = (
    "你是市场侦察决策专家。决定本轮是否需要额外查询货源。"
    "你只输出合法 JSON，不要输出 markdown 代码块或任何解释文字。"
)

QUERY_TEMPLATE = """决定是否需要额外查询货源。

当前状态：
{input_json}

决策规则（优先级从高到低）：
1. 历史热点：如果当前无高质量货源（hourly_rate < 60），查询历史收益高的区域
2. 当前线路机会：如果可见货源终点附近有历史高收益区域，查询该区域

跳过条件（不查）：
- 当前有 hourly_rate > 80 的货源
- 预算为 0
- 当前货源充足（> 5 个有效候选）

输出格式：
{{"thought":"分析原因","should_query_extra":true/false,"query_targets":[{{"lat":30.27,"lng":120.15,"reason":"reason_code","priority":0.92,"budget_cost":1}}],"stop_reason":"reason"}}

reason_code 可选值：historical_hotspot, route_opportunity

stop_reason 可选值：current_market_good_enough, budget_exhausted, no_useful_target

示例1（需要查）：
{{"thought":"当前货源时均收益42元偏低，历史热点有更好货源","should_query_extra":true,"query_targets":[{{"lat":23.13,"lng":113.26,"reason":"historical_hotspot","priority":0.9,"budget_cost":1}}],"stop_reason":"historical_hotspot"}}

示例2（不查）：
{{"thought":"当前有65元时均收益的好货源","should_query_extra":false,"query_targets":[],"stop_reason":"current_market_good_enough"}}
"""

_DAILY_BUDGET = 20
_URGENT_BONUS = 3
_MAX_EXTRA_PER_TURN = 3
_MAX_DISTANCE_KM = 500
_MERGE_DISTANCE_KM = 20


class QueryAgent:
    """有预算的市场侦察员，决定是否额外查货、查哪里、为什么。"""

    phase = "QUERY_CARGO"

    def __init__(self, store: StateStore, telemetry: Telemetry, gateway: GatewayLayer) -> None:
        self._store = store
        self._telemetry = telemetry
        self._gateway = gateway

    def run(self, state: DecisionState, trace: TraceContext) -> DecisionState:
        self._telemetry.emit(trace, event="AGENT_STARTED", source="QueryAgent", phase=self.phase)
        ctx = state.driver_context
        if ctx is None:
            return state

        self._refresh_budget(state)

        skip_reason = self._should_skip(state)
        if skip_reason:
            state.extra_query_targets = []
            state.phase = self.phase
            self._telemetry.emit(
                trace, event="QUERY_SKIPPED", source="QueryAgent",
                phase=self.phase, simulation_minute=ctx.simulation_minute,
                payload={"reason": skip_reason},
            )
            return state

        targets = self._decide_targets(state, trace)
        state.extra_query_targets = targets
        state.phase = self.phase
        self._store.checkpoint(state, "CKPT_QUERY_READY")
        self._telemetry.emit(
            trace, event="QUERY_DECIDED", source="QueryAgent",
            phase=self.phase, simulation_minute=ctx.simulation_minute,
            checkpoint_id="CKPT_QUERY_READY",
            payload={"target_count": len(targets), "budget_remaining": state.query_budget_remaining},
        )
        return state

    def _refresh_budget(self, state: DecisionState) -> None:
        ctx = state.driver_context
        if ctx is None:
            return
        today = day_index(ctx.simulation_minute)
        if state.query_budget_day != today:
            state.query_budget_remaining = _DAILY_BUDGET
            state.query_budget_day = today

    def _should_skip(self, state: DecisionState) -> str | None:
        if self._target_cargo_query_targets(state):
            return None

        if state.query_budget_remaining <= 0:
            return "budget_exhausted"

        if not self._has_query_targets(state):
            return "no_useful_target"

        current = self._summarize_current_cargos(state)
        best_rate = float(current.get("best_hourly_rate", 0) or 0)
        valid_count = int(current.get("valid_count", 0) or 0)
        if valid_count > 0:
            if best_rate > 80:
                return "current_market_good_enough"
            if valid_count > 5 and best_rate > 60:
                return "current_market_good_enough"

        return None

    def _has_hotspots(self, state: DecisionState) -> bool:
        long = self._store.long_memory(state.driver_id)
        return bool(long.hotspots or long.pickup_hotspots)

    def _has_query_targets(self, state: DecisionState) -> bool:
        return bool(self._target_cargo_query_targets(state)) or self._has_hotspots(state)

    def _decide_targets(self, state: DecisionState, trace: TraceContext) -> list[dict[str, Any]]:
        ctx = state.driver_context
        if ctx is None:
            return []
        forced_targets = self._target_cargo_query_targets(state)

        input_data = self._build_input(state)
        payload = {
            "messages": [
                {"role": "system", "content": QUERY_SYSTEM},
                {"role": "user", "content": QUERY_TEMPLATE.format(
                    input_json=json.dumps(input_data, ensure_ascii=False, default=str, separators=(",", ":")),
                )},
            ],
            "temperature": 0.0,
            "max_tokens": 800,
            "enable_thinking": False,
        }

        try:
            result = self._gateway.llm_chat_json(payload, trace, "QueryAgent")
        except Exception as exc:
            logger.warning("QueryAgent LLM call failed: %s", exc)
            return self._merge_query_targets(forced_targets, self._fallback_targets(state))

        if not isinstance(result, dict):
            return self._merge_query_targets(forced_targets, self._fallback_targets(state))

        if not result.get("should_query_extra"):
            return forced_targets

        raw_targets = result.get("query_targets", [])
        if not isinstance(raw_targets, list):
            return forced_targets

        return self._merge_query_targets(forced_targets, self._post_process_targets(state, raw_targets))

    def _target_cargo_query_targets(self, state: DecisionState) -> list[dict[str, Any]]:
        hidden_ids = self._hidden_preference_ids(state)
        visible_ids = {str(c.cargo_id) for c in state.cargo_snapshot}
        targets: list[dict[str, Any]] = []
        instructions = state.preference_instructions.get("instructions", [])
        if not isinstance(instructions, list):
            return targets
        for inst in instructions:
            if not isinstance(inst, dict):
                continue
            inst_id = str(inst.get("id") or "")
            if inst.get("completed") or inst_id in hidden_ids:
                continue
            scheme = inst.get("scheme") if isinstance(inst.get("scheme"), dict) else {}
            pref_type = str(inst.get("preference_type") or scheme.get("type") or "")
            if pref_type != "TARGET_CARGO_MUST_TAKE":
                continue
            constraint = scheme.get("constraint") if isinstance(scheme.get("constraint"), dict) else {}
            cargo_id = str(constraint.get("target_cargo_id") or "").strip()
            if not cargo_id or cargo_id in visible_ids:
                continue
            if not self._target_cargo_is_available(state, constraint):
                continue
            target: dict[str, Any] = {
                "cargo_id": cargo_id,
                "reason": "target_cargo",
                "priority": 1.0,
                "budget_cost": 0,
                "preference_id": inst_id,
            }
            pickup = constraint.get("pickup_location")
            if isinstance(pickup, dict):
                if pickup.get("lat") is not None:
                    target["lat"] = pickup.get("lat")
                if pickup.get("lng") is not None:
                    target["lng"] = pickup.get("lng")
                if pickup.get("name"):
                    target["name"] = pickup.get("name")
            targets.append(target)
        return targets

    @staticmethod
    def _target_cargo_is_available(state: DecisionState, constraint: dict[str, Any]) -> bool:
        ctx = state.driver_context
        available_after = str(constraint.get("available_after") or "").strip()
        available_until = str(constraint.get("available_until") or "").strip()
        if ctx is None:
            return True
        now = int(ctx.simulation_minute)
        if not available_after:
            if not available_until:
                return True
            try:
                return now < wall_time_to_minute(available_until)
            except ValueError:
                return True
        try:
            if now < wall_time_to_minute(available_after):
                return False
            if available_until and now >= wall_time_to_minute(available_until):
                return False
            return True
        except ValueError:
            return True

    @staticmethod
    def _hidden_preference_ids(state: DecisionState) -> set[str]:
        progress = state.preference_progress if isinstance(state.preference_progress, dict) else {}
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

    @staticmethod
    def _merge_query_targets(forced: list[dict[str, Any]], extra: list[dict[str, Any]]) -> list[dict[str, Any]]:
        merged: list[dict[str, Any]] = []
        seen_cargo_ids: set[str] = set()
        seen_points: set[tuple[float, float]] = set()
        for item in list(forced or []) + list(extra or []):
            if not isinstance(item, dict):
                continue
            cargo_id = str(item.get("cargo_id") or "").strip()
            if cargo_id:
                if cargo_id in seen_cargo_ids:
                    continue
                seen_cargo_ids.add(cargo_id)
                merged.append(item)
                continue
            try:
                key = (round(float(item.get("lat")), 3), round(float(item.get("lng")), 3))
            except (TypeError, ValueError):
                continue
            if key in seen_points:
                continue
            seen_points.add(key)
            merged.append(item)
        return merged

    def _build_input(self, state: DecisionState) -> dict[str, Any]:
        ctx = state.driver_context
        if ctx is None:
            return {}

        from ..domain.rules import SIMULATION_EPOCH
        now_dt = SIMULATION_EPOCH + timedelta(minutes=ctx.simulation_minute)
        remaining = DEFAULT_MONTH_HORIZON_MINUTES - ctx.simulation_minute
        remaining_days = remaining // 1440

        current_cargos = self._summarize_current_cargos(state)
        market_memory = self._build_market_memory(state)

        short = self._store.short_memory(state.driver_id)
        recent_actions = []
        recent = short.get_actions_within(ctx.simulation_minute, 720) if ctx else list(short.recent_actions)
        for item in recent[-5:]:
            act = item.get("action", {})
            recent_actions.append({
                "action": act.get("action", ""),
                "minute": item.get("minute"),
            })

        return {
            "context": {
                "position": {"lat": round(ctx.lat, 5), "lng": round(ctx.lng, 5)},
                "time": now_dt.strftime("%Y-%m-%d %H:%M"),
                "simulation_minute": ctx.simulation_minute,
                "remaining": f"{remaining_days}天{(remaining % 1440) // 60}小时",
                "completed_orders": ctx.completed_order_count,
                "recent_actions": recent_actions,
            },
            "current_cargos": current_cargos,
            "market_memory": market_memory,
            "query_budget": {
                "remaining_today": state.query_budget_remaining,
                "cost_per_query": 1,
            },
        }

    def _summarize_current_cargos(self, state: DecisionState) -> dict[str, Any]:
        plans = [p for p in state.ranked_plans if p.valid and p.action == "take_order"]
        if not plans:
            return self._summarize_snapshot_cargos(state)

        rates = [float(p.meta.get("net_income_per_hour", 0) or 0) for p in plans]
        top3 = []
        for p in plans[:3]:
            top3.append({
                "cargo_id": p.cargo_id,
                "hourly_rate": round(float(p.meta.get("net_income_per_hour", 0) or 0), 1),
                "net_income": round(p.net_income, 1),
                "duration_minutes": p.duration_minutes,
            })

        has_pref_match = any(
            (p.meta.get("preference_evaluation", {}).get("preference_score", 0) or 0) > 0
            for p in plans
        )

        return {
            "count": len(state.ranked_plans),
            "valid_count": len(plans),
            "best_hourly_rate": round(max(rates), 1) if rates else 0,
            "avg_hourly_rate": round(sum(rates) / len(rates), 1) if rates else 0,
            "has_preference_match": has_pref_match,
            "top_3": top3,
        }

    def _summarize_snapshot_cargos(self, state: DecisionState) -> dict[str, Any]:
        ctx = state.driver_context
        if ctx is None or not state.cargo_snapshot:
            return {"count": len(state.cargo_snapshot), "valid_count": 0, "best_hourly_rate": 0, "has_preference_match": False, "top_3": []}

        items: list[dict[str, Any]] = []
        for cand in state.cargo_snapshot:
            cargo = cand.cargo if isinstance(cand.cargo, dict) else {}
            start = cargo.get("start", {}) if isinstance(cargo.get("start"), dict) else {}
            end = cargo.get("end", {}) if isinstance(cargo.get("end"), dict) else {}
            try:
                start_lat = float(start["lat"])
                start_lng = float(start["lng"])
                end_lat = float(end["lat"])
                end_lng = float(end["lng"])
            except (KeyError, TypeError, ValueError):
                continue
            haul_km = haversine_km(start_lat, start_lng, end_lat, end_lng)
            price = float(cargo.get("price", 0) or 0)
            if price > 10_000:
                price /= 100.0
            line_minutes = int(cargo.get("cost_time_minutes", 0) or 0)
            pickup_minutes = max(0, int(round(cand.pickup_distance_km)))
            duration_minutes = max(1, pickup_minutes + line_minutes)
            cost = ctx.cost_per_km * (cand.pickup_distance_km + haul_km)
            net_income = price - cost
            hourly_rate = net_income / max(duration_minutes / 60.0, 0.25)
            items.append({
                "cargo_id": cand.cargo_id,
                "cargo_name": cargo.get("cargo_name", ""),
                "hourly_rate": round(hourly_rate, 1),
                "net_income": round(net_income, 1),
                "duration_minutes": duration_minutes,
                "pickup_km": round(cand.pickup_distance_km, 1),
                "haul_km": round(haul_km, 1),
                "route": f"{self._city_from_point(start)}->{self._city_from_point(end)}",
            })

        items.sort(key=lambda item: float(item.get("hourly_rate", 0) or 0), reverse=True)
        rates = [float(item.get("hourly_rate", 0) or 0) for item in items]
        return {
            "count": len(state.cargo_snapshot),
            "valid_count": len(items),
            "best_hourly_rate": round(max(rates), 1) if rates else 0,
            "avg_hourly_rate": round(sum(rates) / len(rates), 1) if rates else 0,
            "has_preference_match": False,
            "top_3": items[:3],
        }

    @staticmethod
    def _city_from_point(point: Any) -> str:
        if isinstance(point, dict):
            city = point.get("city")
            if city not in (None, ""):
                return str(city)
            lat = point.get("lat", "?")
            lng = point.get("lng", "?")
            return f"({lat},{lng})"
        return "?"

    def _build_market_memory(self, state: DecisionState) -> dict[str, Any]:
        long = self._store.long_memory(state.driver_id)
        ctx = state.driver_context

        hotspots = []
        for (lat, lng), weight in sorted(long.hotspots.items(), key=lambda x: x[1], reverse=True)[:5]:
            entry = {"lat": lat, "lng": lng, "weight": round(weight, 1), "type": "destination"}
            if ctx:
                entry["distance_km"] = round(haversine_km(ctx.lat, ctx.lng, lat, lng), 1)
            hotspots.append(entry)

        pickup_hotspots = []
        for (lat, lng), weight in sorted(long.pickup_hotspots.items(), key=lambda x: x[1], reverse=True)[:5]:
            entry = {"lat": lat, "lng": lng, "weight": round(weight, 1), "type": "pickup"}
            failure_count = long.reposition_failures.get((lat, lng), 0)
            if failure_count > 0:
                entry["failure_count"] = failure_count
            if ctx:
                entry["distance_km"] = round(haversine_km(ctx.lat, ctx.lng, lat, lng), 1)
            pickup_hotspots.append(entry)

        return {
            "hotspots": hotspots,
            "pickup_hotspots": pickup_hotspots,
            "total_hotspots": len(long.hotspots),
            "total_pickup_hotspots": len(long.pickup_hotspots),
        }

    def _post_process_targets(self, state: DecisionState, raw_targets: list) -> list[dict[str, Any]]:
        ctx = state.driver_context
        if ctx is None:
            return []

        valid_targets = []
        for t in raw_targets:
            if not isinstance(t, dict):
                continue
            lat = t.get("lat")
            lng = t.get("lng")
            if lat is None or lng is None:
                continue
            try:
                lat = float(lat)
                lng = float(lng)
            except (TypeError, ValueError):
                continue
            if not (-90 <= lat <= 90 and -180 <= lng <= 180):
                continue
            dist = haversine_km(ctx.lat, ctx.lng, lat, lng)
            if dist > _MAX_DISTANCE_KM:
                continue
            if dist < 5:
                continue
            valid_targets.append({
                "lat": round(lat, 5),
                "lng": round(lng, 5),
                "reason": str(t.get("reason", "unknown")),
                "priority": float(t.get("priority", 0.5) or 0.5),
                "budget_cost": 1,
                "distance_km": round(dist, 1),
            })

        merged = self._merge_nearby(valid_targets)
        merged.sort(key=lambda t: t.get("priority", 0), reverse=True)

        budget = state.query_budget_remaining
        result = merged[:min(_MAX_EXTRA_PER_TURN, budget)]

        state.query_budget_remaining -= len(result)
        return result

    @staticmethod
    def _merge_nearby(targets: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if len(targets) <= 1:
            return targets
        merged = []
        used = set()
        for i, t in enumerate(targets):
            if i in used:
                continue
            group = [t]
            for j in range(i + 1, len(targets)):
                if j in used:
                    continue
                dist = haversine_km(t["lat"], t["lng"], targets[j]["lat"], targets[j]["lng"])
                if dist < _MERGE_DISTANCE_KM:
                    group.append(targets[j])
                    used.add(j)
            best = max(group, key=lambda g: g.get("priority", 0))
            merged.append(best)
            used.add(i)
        return merged

    def _fallback_targets(self, state: DecisionState) -> list[dict[str, Any]]:
        long = self._store.long_memory(state.driver_id)
        ctx = state.driver_context
        if ctx is None:
            return []
        if not long.hotspots:
            return []

        candidates = []
        for (lat, lng), weight in sorted(long.hotspots.items(), key=lambda x: x[1], reverse=True)[:3]:
            dist = haversine_km(ctx.lat, ctx.lng, lat, lng)
            if 10 < dist < _MAX_DISTANCE_KM:
                candidates.append({
                    "lat": lat, "lng": lng,
                    "reason": "historical_hotspot",
                    "priority": min(weight / 10.0, 1.0),
                    "budget_cost": 1,
                    "distance_km": round(dist, 1),
                })

        budget = state.query_budget_remaining
        result = candidates[:min(_MAX_EXTRA_PER_TURN, budget)]
        state.query_budget_remaining -= len(result)
        return result
