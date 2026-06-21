from __future__ import annotations

import json
import logging
from datetime import timedelta
from typing import Any

from ..domain.models import ActionPlan
from ..domain.rules import SIMULATION_EPOCH
from ..gateway import GatewayLayer
from ..messages import TraceContext
from ..state_store import DecisionState, StateStore
from ..telemetry import Telemetry
from .preference_rule_filter import _ProgressFacts, _date
from .scheme_handlers import (
    GEOFENCE_FORBIDDEN_AREA_HANDLER,
    GEOFENCE_STAY_WITHIN_HANDLER,
    MONTHLY_DEADHEAD_LIMIT_HANDLER,
    TARGET_CARGO_HANDLER,
    TIME_WINDOW_STATIONARY_HANDLER,
    exclude_from_future as scheme_exclude_from_future,
)

logger = logging.getLogger("agent.future_feasibility_agent")

_SCHEDULE_PREFERENCE_TYPES = {
    "LOCATION_ARRIVAL_DEADLINE",
    "LOCATION_STAY_ON_DATE",
    "ROUTE_SEQUENCE_ON_DATE",
}

FUTURE_DECISION_SYSTEM = (
    "你是偏好可行性评估器。你根据当前状态、偏好进度和候选动作，判断哪些候选会让 active 偏好难以满足。"
    "只输出合法 JSON，不要输出 markdown 或解释性文字。"
)

FUTURE_DECISION_TEMPLATE = """请对候选动作做未来可行性判断。

当前状态：
{context_json}

active 结构化偏好指令：
{instructions_json}

当前偏好进度：
{progress_json}

最近动作：
{recent_actions_json}

候选动作：
{candidates_json}

要求：
1. 只判断给定的 candidate_id，不要编造订单或改写候选参数。
2. 如果某个候选会让 active 偏好不可恢复或明显难以继续满足，把它放入 blocked_candidate_ids。
3. 如果某个候选明显有助于满足 active 偏好，把它放入 preferred_candidate_ids。
4. 不要输出或建议新的动作；wait/reposition 由最终策略模块在没有合规订单时决定。
5. reasons 只写被 blocked 或 preferred 的简短原因，避免输出过长。
6. 必须检查时间资源：每日连续休息、固定时段休息、整天静止、指定日期到达/停留，都要看候选的完整执行区间、结束时刻、结束后的当日剩余时间和相关周期剩余时间。
7. 对每日连续休息类偏好，若候选结束后同一自然日已无足够连续 wait 窗口完成剩余休息，通常应 block；不能用“后续补休/明天补”解释当日 hard 偏好。
8. 对固定时段休息类偏好，只要候选完整执行区间会占用禁休窗口，通常应 block；不要只看接单开始时刻或装卸时刻。
9. 对整天静止类偏好，只要候选会让一个仍可能作为静止日的自然日发生 take_order/reposition，就要视为消耗该日资格；当月剩余可用整天不足时应 block。
10. 对指定日期地点任务，若候选会错过到达/停留窗口，或占用必须 reposition/wait 的时间，应 block。

输出格式：
{
  "blocked_candidate_ids": ["ff_0"],
  "preferred_candidate_ids": ["ff_3"],
  "reasons": {
    "ff_0": "简短原因",
    "ff_3": "简短原因"
  }
}
"""


FUTURE_DECISION_SYSTEM += (
    " Output compact JSON only. Do not include reasoning, self-correction, or analysis."
    " Return exactly one JSON object and stop. Never write Wait, re-evaluate, Final decision, or a second JSON object."
    " Keep each reason under 40 Chinese chars."
    " 休息证据按评测 step 口径：本步最终动作为 wait 时，本步 wait 前置 query_scan 与 wait 执行时间合并计入连续休息/静止；本步最终动作为 take_order 或 reposition 时，query_scan 不计入休息；接单内部等待不计入休息。"
)

FUTURE_DECISION_GUARDRAILS = """

补充判定原则：
1. DAILY_CONTINUOUS_REST 若当前自然日已经有足够长的连续 wait/静止事实，后续接单不会“打断已经完成的休息成果”；不要仅因会结束当前等待状态而 block。
2. TIME_WINDOW_STATIONARY 只保护固定时段本身；候选完整执行区间不占用该时段时，不要因为窗口刚结束或曾经处于窗口而 block。
3. OFF_DAY_QUOTA 是月度资源：只有在接单后本月剩余可用完整静止日不足以达标时才 block；配额已达成时不要再因该配额 block。
4. ORDER_QUOTA 是进度目标：装/卸货地点命中目标地点的订单是在推进配额，应 preferred 或至少保留，不要说它“消耗名额”而 block。
5. 已由确定性规则过滤掉的候选不会出现在这里；这里不要扩大 block，只处理确定会让 active hard 偏好不可恢复的候选。
"""


class FutureFeasibilityAgent:
    """Use the LLM to keep one-step choices aligned with active preferences."""

    phase = "CHECK_FUTURE_FEASIBILITY"
    _BASE_CANDIDATE_LIMIT = 10
    _MAX_CANDIDATE_LIMIT = 16
    _RECENT_ACTION_LIMIT = 12
    _QUERY_EVENT_LIMIT = 3

    def __init__(self, store: StateStore, telemetry: Telemetry, gateway: GatewayLayer | None = None) -> None:
        self._store = store
        self._telemetry = telemetry
        self._gateway = gateway

    def run(self, state: DecisionState, trace: TraceContext) -> DecisionState:
        ctx = state.driver_context
        if ctx is None:
            return state
        if not ctx.preferences_text and not state.preference_instructions.get("instructions"):
            state.phase = self.phase
            return state

        self._telemetry.emit(trace, event="AGENT_STARTED", source="FutureFeasibilityAgent", phase=self.phase)
        from .preference_evaluator import apply_hard_filter_to_state

        hard_blocked = apply_hard_filter_to_state(state)
        geofence_blocked = GEOFENCE_STAY_WITHIN_HANDLER.apply_filter_to_state(state)
        forbidden_geofence_blocked = GEOFENCE_FORBIDDEN_AREA_HANDLER.apply_filter_to_state(state)
        monthly_deadhead_blocked = MONTHLY_DEADHEAD_LIMIT_HANDLER.apply_filter_to_state(state)
        stationary_blocked = TIME_WINDOW_STATIONARY_HANDLER.apply_filter_to_state(state)
        target_prefilter = self._apply_target_cargo_future_filter(state)
        candidates = self._candidate_plans_for_llm(state)
        result = self._llm_decide_candidates(state, candidates, trace)
        blocked, preferred = self._apply_llm_decision(state, candidates, result)
        target_prefilter_after_llm = self._apply_target_cargo_future_filter(state)
        target_blocked = max(target_prefilter.get("blocked", 0), target_prefilter_after_llm.get("blocked", 0))
        target_preferred = max(target_prefilter.get("preferred", 0), target_prefilter_after_llm.get("preferred", 0))
        self._write_llm_progress_note(state, result, blocked, len(candidates), preferred, target_prefilter={
            "blocked": target_blocked,
            "preferred": target_preferred,
        })
        state.phase = self.phase
        self._store.checkpoint(state, "CKPT_FUTURE_FEASIBILITY_READY")
        self._telemetry.emit(
            trace,
            event="FUTURE_FEASIBILITY_CHECKED",
            source="FutureFeasibilityAgent",
            phase=self.phase,
            simulation_minute=ctx.simulation_minute,
            checkpoint_id="CKPT_FUTURE_FEASIBILITY_READY",
            payload={
                "source": "llm",
                "evaluated": len(candidates),
                "rejected": blocked,
                "preferred": preferred,
                "target_cargo_blocked": target_blocked,
                "target_cargo_preferred": target_preferred,
                "hard_blocked": hard_blocked + geofence_blocked + forbidden_geofence_blocked + monthly_deadhead_blocked + stationary_blocked,
                "geofence_blocked": geofence_blocked,
                "forbidden_geofence_blocked": forbidden_geofence_blocked,
                "monthly_deadhead_blocked": monthly_deadhead_blocked,
                "stationary_blocked": stationary_blocked,
                "llm_ok": isinstance(result, dict),
            },
        )
        return state

    def _candidate_plans_for_llm(self, state: DecisionState) -> list[ActionPlan]:
        source = state.ranked_plans if state.ranked_plans else state.simulated_plans
        take_orders = [p for p in source if p.valid and p.action == "take_order"]
        if not state.ranked_plans:
            take_orders.sort(key=self._candidate_value, reverse=True)

        chosen = self._select_candidates_for_llm(take_orders, state)

        selected: list[ActionPlan] = []
        seen: set[int] = set()
        for plan in chosen:
            marker = id(plan)
            if marker in seen:
                continue
            seen.add(marker)
            plan.meta["future_candidate_id"] = f"ff_{len(selected)}"
            selected.append(plan)
        return selected

    def _select_candidates_for_llm(self, plans: list[ActionPlan], state: DecisionState) -> list[ActionPlan]:
        target_ids = self._active_target_cargo_ids(state)
        if len(plans) <= self._BASE_CANDIDATE_LIMIT:
            return plans
        selected = list(plans[: self._BASE_CANDIDATE_LIMIT])
        baseline = self._candidate_value(selected[-1]) if selected else 0.0
        for plan in plans[self._BASE_CANDIDATE_LIMIT : self._MAX_CANDIDATE_LIMIT]:
            if self._has_preference_signal(plan) or self._is_close_value(self._candidate_value(plan), baseline):
                selected.append(plan)
        selected = selected[: self._MAX_CANDIDATE_LIMIT]
        seen = {id(plan) for plan in selected}
        for plan in plans[self._MAX_CANDIDATE_LIMIT:]:
            if str(plan.cargo_id or "") in target_ids and id(plan) not in seen:
                selected.append(plan)
                seen.add(id(plan))
        return selected

    @staticmethod
    def _active_target_cargo_ids(state: DecisionState) -> set[str]:
        return TARGET_CARGO_HANDLER.active_ids(state)

    @staticmethod
    def _active_target_cargo_targets(state: DecisionState) -> list[dict[str, Any]]:
        return TARGET_CARGO_HANDLER.active_targets(state)

    @staticmethod
    def _is_close_value(value: float, baseline: float) -> bool:
        tolerance = max(120.0, abs(baseline) * 0.12)
        return value >= baseline - tolerance

    @staticmethod
    def _has_preference_signal(plan: ActionPlan) -> bool:
        meta = plan.meta if isinstance(plan.meta, dict) else {}
        pref_eval = meta.get("preference_evaluation")
        if isinstance(pref_eval, dict):
            try:
                if float(pref_eval.get("preference_score", 0) or 0) > 0:
                    return True
            except (TypeError, ValueError):
                pass
        if meta.get("preference_generated"):
            return True
        future = meta.get("future_feasibility")
        return isinstance(future, dict) and bool(future.get("preferred"))

    @staticmethod
    def _candidate_value(plan: ActionPlan) -> float:
        hours = max(float(plan.duration_minutes or 0) / 60.0, 0.25)
        return float(plan.net_income or 0.0) + (float(plan.net_income or 0.0) / hours) * 6.0

    def _llm_decide_candidates(
        self,
        state: DecisionState,
        candidates: list[ActionPlan],
        trace: TraceContext,
    ) -> dict[str, Any] | None:
        if self._gateway is None:
            return None
        ctx = state.driver_context
        if ctx is None:
            return None
        active_instructions = self._active_instructions(state)
        content = self._render_template(FUTURE_DECISION_TEMPLATE, {
            "context_json": json.dumps(self._context_summary(state, ctx), ensure_ascii=False, default=str, separators=(",", ":")),
            "preferences_json": json.dumps(self._active_preferences(state, active_instructions), ensure_ascii=False, default=str, separators=(",", ":")),
            "instructions_json": json.dumps(self._compact_instructions(active_instructions), ensure_ascii=False, default=str, separators=(",", ":")),
            "progress_json": json.dumps(self._progress_for_llm(state), ensure_ascii=False, default=str, separators=(",", ":")),
            "recent_actions_json": json.dumps(self._recent_actions_summary(state), ensure_ascii=False, default=str, separators=(",", ":")),
            "candidates_json": json.dumps([self._plan_summary(state, p) for p in candidates], ensure_ascii=False, default=str, separators=(",", ":")),
        }) + FUTURE_DECISION_GUARDRAILS
        payload = {
            "messages": [
                {"role": "system", "content": FUTURE_DECISION_SYSTEM},
                {"role": "user", "content": content},
            ],
            "temperature": 0.0,
            "max_tokens": 700,
            "enable_thinking": False,
        }
        try:
            result = self._gateway.llm_chat_json(payload, trace, "FutureFeasibilityAgent")
        except Exception as exc:
            logger.warning("FutureFeasibilityAgent LLM failed: %s", exc)
            return None
        return result if isinstance(result, dict) else None

    def _apply_llm_decision(self, state: DecisionState, candidates: list[ActionPlan], result: dict[str, Any] | None) -> tuple[int, int]:
        by_id = {str(plan.meta.get("future_candidate_id")): plan for plan in candidates}
        if not isinstance(result, dict):
            for plan in candidates:
                plan.meta["future_feasibility"] = {
                    "source": "llm_unavailable",
                    "feasible": True,
                    "blocked": False,
                    "preferred": False,
                    "reason": "LLM unavailable; skipped future-feasibility add-on check",
                }
            return 0, 0
            for plan in candidates:
                plan.meta["future_feasibility"] = {
                    "source": "llm_unavailable",
                    "feasible": plan.action in {"wait", "reposition"},
                    "blocked": plan.action == "take_order",
                    "preferred": False,
                    "reason": "LLM 不可用，仅保留 wait/reposition 支持候选。",
                }
                if plan.action == "take_order":
                    plan.valid = False
                    plan.score = -1_000_000.0
                    plan.reason = "未来可行性 LLM 不可用，避免执行不可逆动作。"
            for plan in state.simulated_plans:
                if plan.action == "take_order":
                    plan.valid = False
                    plan.score = -1_000_000.0
                    plan.reason = "未来可行性 LLM 不可用，避免执行不可逆动作。"
            return 0, 0

        blocked_ids = self._as_str_set(result.get("blocked_candidate_ids"))
        preferred_ids = self._as_str_set(result.get("preferred_candidate_ids"))
        reasons = result.get("reasons", {})
        if not isinstance(reasons, dict):
            reasons = {}

        blocked_count = 0
        preferred_count = 0
        for cid, plan in by_id.items():
            blocked = cid in blocked_ids
            preferred = cid in preferred_ids
            reason = str(reasons.get(cid, "") or "").strip()
            plan.meta["future_feasibility"] = {
                "source": "llm",
                "feasible": not blocked,
                "blocked": blocked,
                "preferred": preferred,
                "reason": reason,
            }
            if blocked:
                blocked_count += 1
                plan.valid = False
                plan.score = -1_000_000.0
                plan.reason = f"未来可行性 LLM 阻止：{reason or '可能破坏偏好'}"
            elif preferred:
                preferred_count += 1
                plan.meta["preference_generated"] = True
                plan.meta["priority"] = max(float(plan.meta.get("priority", 0.0) or 0.0), 0.8)
                if reason:
                    plan.reason = f"未来可行性 LLM 推荐：{reason}"
        return blocked_count, preferred_count

    def _apply_target_cargo_future_filter(self, state: DecisionState) -> dict[str, int]:
        return TARGET_CARGO_HANDLER.future_filter(state)

    def _active_instructions(self, state: DecisionState) -> list[dict[str, Any]]:
        instructions = state.preference_instructions.get("instructions", [])
        if not isinstance(instructions, list):
            return []
        completed_ids = self._str_set(state.preference_progress.get("completed_ids", []))
        hidden_ids = self._str_set(state.preference_progress.get("hidden_completed_ids", []))
        keep_ids = self._keep_monitoring_preference_ids(state)
        active: list[dict[str, Any]] = []
        for inst in instructions:
            if not isinstance(inst, dict):
                continue
            if self._exclude_from_future(inst):
                continue
            inst_id = str(inst.get("id", "") or "")
            must_keep = self._must_keep_monitoring(inst)
            if (inst.get("completed") or inst_id in hidden_ids) and not must_keep:
                continue
            if inst_id in completed_ids and inst_id not in keep_ids and not must_keep:
                continue
            active.append(inst)
        return active

    @staticmethod
    def _exclude_from_future(instruction: dict[str, Any]) -> bool:
        return scheme_exclude_from_future(instruction)

    @staticmethod
    def _must_keep_monitoring(instruction: dict[str, Any]) -> bool:
        pref_type = str(instruction.get("preference_type") or "")
        scheme = instruction.get("scheme") if isinstance(instruction.get("scheme"), dict) else {}
        pref_type = pref_type or str(scheme.get("type") or "")
        return pref_type in {
            "ACTION_FORBID",
            "NUMERIC_LIMIT",
            "TIME_WINDOW_STATIONARY",
            "DAILY_CONTINUOUS_REST",
            "GEOFENCE_STAY_WITHIN",
            "GEOFENCE_FORBIDDEN_AREA",
            "MONTHLY_DEADHEAD_LIMIT",
        }

    def _active_preferences(self, state: DecisionState, active_instructions: list[dict[str, Any]] | None = None) -> list[Any]:
        ctx = state.driver_context
        if ctx is None:
            return []
        raw_items = list(ctx.preferences_raw or ctx.preferences_text or [])
        instructions = state.preference_instructions.get("instructions", [])
        if not isinstance(instructions, list) or not raw_items:
            return raw_items
        active_instructions = active_instructions if active_instructions is not None else self._active_instructions(state)
        if not active_instructions:
            return []
        active_ids = {id(inst) for inst in active_instructions}
        active_items: list[Any] = []
        for idx, inst in enumerate(instructions):
            if id(inst) in active_ids and idx < len(raw_items):
                active_items.append(raw_items[idx])
        return active_items

    @staticmethod
    def _progress_for_llm(state: DecisionState) -> dict[str, Any]:
        progress = dict(state.preference_progress or {})
        progress.pop("failed_ids", None)
        statuses = progress.get("preference_statuses")
        if isinstance(statuses, list):
            progress["preference_statuses"] = [
                {
                    "id": item.get("id"),
                    "status": "active" if str(item.get("status", "")) == "failed" else item.get("status"),
                }
                for item in statuses
                if isinstance(item, dict)
            ]
        spans = progress.pop("action_spans", [])
        if isinstance(spans, list):
            progress["action_span_count"] = len(spans)
        if isinstance(state.preference_progress.get("target_cargo_progress"), dict):
            progress["target_cargo_progress"] = state.preference_progress.get("target_cargo_progress")
        progress.pop("text", None)
        return progress

    @staticmethod
    def _compact_instructions(instructions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        from .preference_utils import compact_instruction_for_llm

        compact: list[dict[str, Any]] = []
        for inst in instructions:
            if not isinstance(inst, dict):
                continue
            compact.append(compact_instruction_for_llm(inst))
        return compact

    @staticmethod
    def _keep_monitoring_preference_ids(state: DecisionState) -> set[str]:
        value = state.preference_progress.get("preference_statuses", [])
        if not isinstance(value, list):
            return set()
        out: set[str] = set()
        for item in value:
            if not isinstance(item, dict):
                continue
            if str(item.get("status", "")) != "satisfied_keep_monitoring":
                continue
            inst_id = str(item.get("id", "") or "").strip()
            if inst_id:
                out.add(inst_id)
        return out

    @staticmethod
    def _str_set(value: Any) -> set[str]:
        if isinstance(value, str):
            return {value} if value.strip() else set()
        if isinstance(value, list):
            return {str(item) for item in value if str(item).strip()}
        return set()

    @staticmethod
    def _as_str_set(value: Any) -> set[str]:
        if isinstance(value, str):
            return {value} if value.strip() else set()
        if isinstance(value, list):
            return {str(item).strip() for item in value if str(item).strip()}
        return set()

    def _plan_summary(self, state: DecisionState, plan: ActionPlan) -> dict[str, Any]:
        ctx = state.driver_context
        cargo = self._cargo_by_id(state, plan.cargo_id)
        out: dict[str, Any] = {
            "candidate_id": plan.meta.get("future_candidate_id"),
            "action": plan.action,
            "params": plan.params,
            "score": round(float(plan.score or 0.0), 2),
            "duration_minutes": int(plan.duration_minutes or 0),
            "finish_time": self._sim_to_wall(plan.finish_minute),
            "finish_day_remaining": self._day_remaining_resource(plan.finish_minute),
        }
        if plan.action == "take_order":
            out.update({
                "cargo_name": plan.meta.get("cargo_name"),
                "net_income": self._round_or_none(plan.net_income),
                "pickup_km": self._round_or_none(plan.meta.get("pickup_km")),
                "haul_km": self._round_or_none(plan.meta.get("haul_km")),
                "pickup_arrival": self._sim_to_wall(
                    int(ctx.simulation_minute) + int(plan.meta.get("pickup_minutes", 0) or 0)
                ) if ctx else None,
                "delivery_done": self._sim_to_wall(plan.finish_minute),
            })
            if isinstance(cargo, dict):
                if cargo.get("price") is not None:
                    out["price"] = cargo.get("price")
                start = cargo.get("start", {})
                end = cargo.get("end", {})
                start_city = self._city_from_point(start)
                end_city = self._city_from_point(end)
                if start_city and end_city:
                    out["route"] = f"{start_city}->{end_city}"
                if isinstance(start, dict):
                    out["from"] = f"({start.get('lat','?')},{start.get('lng','?')})"
                if isinstance(end, dict):
                    out["to"] = f"({end.get('lat','?')},{end.get('lng','?')})"
                load_time = cargo.get("load_time")
                if isinstance(load_time, list) and len(load_time) == 2:
                    out["load_window"] = f"{load_time[0]}~{load_time[1]}"
        if plan.action == "reposition":
            out["target"] = {"lat": plan.target_lat, "lng": plan.target_lng}
            out["distance_km"] = self._round_or_none(plan.meta.get("distance_km"))
        return {k: v for k, v in out.items() if v is not None}

    def _context_summary(self, state: DecisionState, ctx: Any) -> dict[str, Any]:
        facts = _ProgressFacts.from_state(state)
        current_day = int(ctx.simulation_minute) // 1440
        return {
            "driver_id": ctx.driver_id,
            "simulation_minute": ctx.simulation_minute,
            "wall_time": ctx.simulation_wall_time,
            "position": {"lat": round(ctx.lat, 5), "lng": round(ctx.lng, 5)},
            "current_day_remaining": self._day_remaining_resource(ctx.simulation_minute),
            "deterministic_progress_facts": {
                "current_day": self._sim_to_wall(current_day * 1440)[:10] if self._sim_to_wall(current_day * 1440) else None,
                "current_day_longest_wait_minutes": facts.longest_wait(current_day),
                "current_day_active_minutes": facts.active_minutes_by_day.get(current_day, 0),
                "completed_off_day_count": len(facts.completed_off_days),
                "completed_off_days": [_date(day) for day in sorted(facts.completed_off_days)],
            },
            "completed_order_count": ctx.completed_order_count,
            "query_scan": self._query_scan_summary(state),
        }

    @staticmethod
    def _query_scan_timeline(state: DecisionState) -> list[dict[str, Any]]:
        timeline: list[dict[str, Any]] = []
        for event in state.query_scan_events[-FutureFeasibilityAgent._QUERY_EVENT_LIMIT:]:
            if not isinstance(event, dict):
                continue
            try:
                start = int(event.get("start"))
                end = int(event.get("end"))
            except (TypeError, ValueError):
                continue
            if end <= start:
                continue
            timeline.append(
                {
                    "kind": "query_scan",
                    "start": start,
                    "end": end,
                    "duration_minutes": end - start,
                    "items_count": event.get("items_count"),
                    "location": {"lat": event.get("lat"), "lng": event.get("lng")},
                    "source": event.get("source"),
                }
            )
        return timeline

    @staticmethod
    def _query_scan_summary(state: DecisionState) -> dict[str, Any]:
        return {
            "total_minutes": int(state.query_scan_minutes),
            "event_count": len(state.query_scan_events),
            "recent_events": FutureFeasibilityAgent._query_scan_timeline(state),
        }

    def _recent_actions_summary(self, state: DecisionState) -> list[dict[str, Any]]:
        ctx = state.driver_context
        short = self._store.short_memory(state.driver_id)
        recent = short.get_actions_within(ctx.simulation_minute, 2880) if ctx else list(short.recent_actions)
        out: list[dict[str, Any]] = []
        for item in recent[-self._RECENT_ACTION_LIMIT:]:
            action = item.get("action", {})
            minute = item.get("minute")
            out.append({
                "minute": minute,
                "time": self._sim_to_wall(minute),
                "action": action.get("action") if isinstance(action, dict) else None,
                "params": action.get("params") if isinstance(action, dict) else None,
            })
        return out

    @staticmethod
    def _cargo_by_id(state: DecisionState, cargo_id: Any) -> dict[str, Any] | None:
        target = str(cargo_id or "").strip()
        if not target:
            return None
        for cand in state.cargo_snapshot:
            if cand.cargo_id == target and isinstance(cand.cargo, dict):
                return cand.cargo
        return None

    @staticmethod
    def _city_from_point(point: Any) -> str | None:
        if isinstance(point, dict):
            city = point.get("city")
            return str(city) if city not in (None, "") else None
        return None

    def _write_llm_progress_note(
        self,
        state: DecisionState,
        result: dict[str, Any] | None,
        rejected: int,
        evaluated: int,
        preferred: int = 0,
        target_prefilter: dict[str, int] | None = None,
    ) -> None:
        note = ""
        if isinstance(result, dict):
            note = str(result.get("notes", "") or "").strip()
        state.preference_progress["future_feasibility"] = {
            "source": "llm",
            "evaluated": evaluated,
            "rejected": rejected,
            "preferred": preferred,
            "target_cargo_prefilter": target_prefilter or {"blocked": 0, "preferred": 0},
            "llm_ok": isinstance(result, dict),
            "note_present": bool(note),
        }
        return
    @staticmethod
    def _render_template(template: str, values: dict[str, str]) -> str:
        rendered = template
        for key, value in values.items():
            rendered = rendered.replace("{" + key + "}", value)
        return rendered

    @staticmethod
    def _sim_to_wall(sim_minute: Any) -> str | None:
        if sim_minute is None:
            return None
        try:
            minute = int(sim_minute)
        except (TypeError, ValueError):
            return None
        return (SIMULATION_EPOCH + timedelta(minutes=minute)).strftime("%Y-%m-%d %H:%M")

    @staticmethod
    def _day_remaining_resource(sim_minute: Any) -> dict[str, Any] | None:
        if sim_minute is None:
            return None
        try:
            minute = int(sim_minute)
        except (TypeError, ValueError):
            return None
        wall = SIMULATION_EPOCH + timedelta(minutes=minute)
        minutes_since_midnight = wall.hour * 60 + wall.minute
        remaining = max(0, 1440 - minutes_since_midnight)
        return {
            "date": wall.strftime("%Y-%m-%d"),
            "time": wall.strftime("%H:%M"),
            "remaining_minutes": remaining,
            "remaining_hours": round(remaining / 60.0, 2),
        }

    @staticmethod
    def _round_or_none(value: Any, digits: int = 2) -> float | None:
        try:
            return round(float(value), digits)
        except (TypeError, ValueError):
            return None
