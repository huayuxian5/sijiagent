from __future__ import annotations

import json
import logging
from typing import Any

from ..domain.models import ActionPlan
from ..domain.rules import DEFAULT_MONTH_HORIZON_MINUTES, distance_to_minutes, haversine_km
from ..gateway import GatewayLayer
from ..messages import TraceContext
from ..state_store import DecisionState, StateStore
from ..telemetry import Telemetry
from .scheme_handlers import (
    GEOFENCE_FORBIDDEN_AREA_HANDLER,
    GEOFENCE_STAY_WITHIN_HANDLER,
    LOCATION_ARRIVAL_DEADLINE_HANDLER,
    MONTHLY_DEADHEAD_LIMIT_HANDLER,
    TARGET_CARGO_HANDLER,
    TIME_WINDOW_STATIONARY_HANDLER,
)

logger = logging.getLogger("agent.policy_agent")

_POLICY_TOP_LIMIT = 10
_OBSERVED_REPOSITION_RADIUS_KM = 50.0
_SCHEDULE_RADIUS_KM = 5.0
_DEADLINE_BUFFER_MINUTES = 10
_SCHEDULE_PREFERENCE_TYPES = {
    "LOCATION_ARRIVAL_DEADLINE",
    "LOCATION_STAY_ON_DATE",
    "ROUTE_SEQUENCE_ON_DATE",
}

POLICY_SYSTEM = (
    "你是一个货运平台的卡车司机找货决策 AI。"
    "你只能输出合法的 JSON，不要输出 markdown 代码块标记，不要输出任何解释性文字。"
    "必须输出 reason 字段，但 reason 必须在同一个 JSON 对象内部，且 JSON 闭合后立即停止。"
    "hard 偏好是动作可行性的先决条件，不是收益权衡项；不能为了收益推迟、牺牲或赌未来补齐 hard 偏好。"
    "你的首要目标是让所有 hard 偏好在各自周期内按时满足；只有在偏好安全后，才允许考虑收益。"
    "如果没有接单候选,不要选择chosen_index;选择动作wait or reposition."
)

POLICY_DECISION_GUARDRAILS = """

补充决策原则：
1. 每日连续休息类偏好只要求当天存在一段达标的连续 wait/静止事实；一旦当天已经达标，后续接单不会打断已经完成的休息成果。此时只需避免影响下一次未完成的日周期。
2. 固定时段静止类偏好只保护固定窗口本身；候选完整执行区间不占用该窗口时，不要因为窗口刚结束或之前在休息而继续等待。
3. 整天静止/月度休息日配额达成后，不要再为了该配额整天等待；未达成时也只在剩余完整自然日不足时强制保留当天。
4. 月度接单/地点天数配额中，命中目标地点的订单是在推进配额，不是消耗配额；应优先考虑这些订单的合规性和收益。
5. wait/reposition 是为了推进明确的 hard 偏好，不是默认保守动作；当候选订单不违反 hard 偏好且能保持周期目标可完成时，应在合规候选中选择收益更好的订单。
"""

POLICY_DECIDE_TEMPLATE = """选择一个候选动作。

位置({lat},{lng})；时间 {now}；本月剩余{remaining}；当日剩余时间：{today_remaining_json}；已{completed_orders}单。

近期动作（墙钟时间）：
{recent_actions}

历史订单摘要：
{episodic_summary}

偏好指令（必须遵守）：
{preference_instructions}

偏好进度：
{preference_progress}

接单候选（hourly_rate=时均收益，arrival=到达时间）：
{plans_json}

偏好进度事实：
{preference_facts_json}

排程任务（只给策略决策使用，不属于候选订单过滤）：
{schedule_tasks_json}

reposition 可用的可观察位置（包含地名/来源和经纬度）：
{coordinate_evidence_json}

规则：
1. 先判 hard 偏好，再看收益。收益不能参与“是否合规”的判断，只能在所有 hard 偏好都安全的候选之间排序。
2. 当前若存在未满足、临近截止、或仍需推进的 hard 偏好，必须优先选择最能推进该偏好的动作；接单只有在不会延误该偏好时才可选。
3. 每个 hard 偏好都必须在自己的周期内完成：自然日偏好看当天，月度偏好看本月，指定日期/截止时间偏好看那个日期和截止点；不能用其他周期的完成抵消当前周期失败。
4. 本月剩余时间和当日剩余时间都是 hard 偏好可行性的资源；判断每日类偏好时，必须检查当前当日剩余时间和候选执行后的当日剩余时间。
5. 选择 take_order 前必须能明确证明：候选完整执行后，同一周期内仍有足够连续时间/正确位置/截止余量完成所有未满足 hard 偏好；证明不了就不要接单。
6. 禁止使用“先接单、后续再补”“还有时间以后安排”“本次不直接违反所以可以接”“唯一有收益的选择”“接完立刻 wait”作为选择接单的理由。
7. 若接单会让当天剩余连续休息窗口不足、会占用固定休息/禁行窗口、会错过指定日期地点任务、或会消耗仍需保留的整天静止资格，必须拒绝该接单并直接 wait 或 reposition。
8. 判断接单候选必须检查完整占用区间：从当前查询后的动作开始，到空驶到装货地、等装、运输、卸货完成为止；不要只看当前时刻、装货窗口或货物名称。
9. 如果候选完整占用区间与强制休息/禁行/到达截止/地点停留窗口冲突，或会压缩到无法完成，必须拒绝该候选。
10. 对每日连续休息、固定时段休息、整天静止这类休息偏好，按评测 step 口径判断：若本步最终动作是 wait，本步 wait 前置 query_scan 与 wait 执行时间合并视为连续静止/休息；若本步最终动作是 take_order 或 reposition，query_scan 不算休息，接单内部等待、装卸等待、运输和 reposition 也不是休息。
11. 对整天静止类偏好，只要某个自然日发生过 take_order 或 reposition，该日就不能再作为整天静止日；若当前动作会破坏一个仍需保留的整天静止日，优先 wait。
12. 对指定地点停留/到达类偏好，若当前日期接近任务日或已经进入任务日，必须优先保留到达、停留和转场时间；reposition 目标必须来自可观察位置，且要优先选择与偏好地点文字和坐标最匹配的位置。
13. 单次约束每次动作都必须遵守；即使历史已经失败，也不能因此放开后续同类违规。
14. 可以选择接单候选，或直接生成 wait/reposition；当所有接单候选都不安全，必须直接生成 wait 或 reposition。
15. 直接生成 wait 时，duration_minutes 必须是正数，且不能超过剩余仿真时间；为了满足休息窗口，等待时长要覆盖所需的完整窗口，不要少等几分钟。
16. 直接生成 reposition 时，latitude/longitude 必须来自可观察位置；不要编造 cargo_id 或坐标。
17. 如果信息不足以判断某个接单候选是否符合偏好，优先直接 wait；如果偏好需要到达某地且已有可观察坐标，优先 reposition。
18. 必须输出 reason 字段，reason 用一句短中文说明本动作如何保护或推进 hard 偏好；不要在 JSON 外写任何解释，JSON 闭合后立即停止。

输出格式只能是以下三选一：
{{"chosen_index":2,"reason":"候选执行区间不破坏当前 hard 偏好"}}
{{"action":"wait","params":{{"duration_minutes":60}},"reason":"等待以满足当前周期 hard 偏好窗口"}}
{{"action":"reposition","params":{{"latitude":12.34567,"longitude":98.76543}},"reason":"移动到可观察位置以推进地点类 hard 偏好"}}
"""


class PolicyAgent:
    phase = "SELECT_POLICY"

    def __init__(self, store: StateStore, telemetry: Telemetry, gateway: GatewayLayer) -> None:
        self._store = store
        self._telemetry = telemetry
        self._gateway = gateway

    def run(self, state: DecisionState, trace: TraceContext) -> DecisionState:
        self._telemetry.emit(trace, event="AGENT_STARTED", source="PolicyAgent", phase=self.phase)
        chosen = self._choose(state, trace)
        state.selected_intent = chosen
        state.phase = self.phase
        self._store.checkpoint(state, "CKPT_POLICY_READY")
        self._telemetry.emit(
            trace, event="DECISION_SELECTED", source="PolicyAgent", phase=self.phase,
            simulation_minute=state.driver_context.simulation_minute if state.driver_context else None,
            checkpoint_id="CKPT_POLICY_READY",
            payload={"action": chosen.action, "params": chosen.params, "score": round(chosen.score, 2), "reason": chosen.reason},
        )
        return state

    def _choose(self, state: DecisionState, trace: TraceContext) -> ActionPlan:
        ctx = state.driver_context
        if ctx is None:
            raise ValueError("missing driver_context")

        return self._llm_decide(state, trace)

    def _llm_decide(self, state: DecisionState, trace: TraceContext) -> ActionPlan:
        ctx = state.driver_context
        if ctx is None:
            raise ValueError("missing driver_context")

        plans_data = self._build_candidates(state)
        pref_instructions = self._format_instructions(state)
        pref_progress = self._format_progress(state)
        preference_facts = self._build_preference_progress_facts(state)
        schedule_tasks = self._active_schedule_tasks(state)
        coordinate_evidence = self._coordinate_evidence(state, schedule_tasks=schedule_tasks)
        target_plan = self._forced_target_cargo_plan(state)
        if target_plan is not None:
            return target_plan
        stationary_wait_plan = self._plan_for_stationary_window(state)
        if stationary_wait_plan is not None:
            return stationary_wait_plan
        forbidden_recovery_plan = self._plan_for_forbidden_geofence_recovery(state)
        if forbidden_recovery_plan is not None:
            return forbidden_recovery_plan
        geofence_recovery_plan = self._plan_for_geofence_recovery(state)
        if geofence_recovery_plan is not None:
            return geofence_recovery_plan
        deadline_guard_plan = self._plan_for_arrival_deadline_guard(state)
        if deadline_guard_plan is not None:
            return deadline_guard_plan
        target_wait_plan = self._plan_for_target_cargo_waiting(state)
        if target_wait_plan is not None:
            return target_wait_plan
        target_position_plan = self._plan_for_target_cargo_positioning(state)
        if target_position_plan is not None:
            return target_position_plan
        schedule_plan = self._plan_for_active_schedule(state, schedule_tasks)
        if schedule_plan is not None:
            return schedule_plan

        from datetime import timedelta
        from ..domain.rules import SIMULATION_EPOCH
        now_str = (SIMULATION_EPOCH + timedelta(minutes=ctx.simulation_minute)).strftime("%Y-%m-%d %H:%M")

        plans_str = json.dumps(plans_data, ensure_ascii=False, default=str, separators=(",", ":"))
        if not plans_data:
            plans_str = "（无可接订单，请选择 wait 或 reposition）"

        recent_actions = self._build_recent_actions(state)
        episodic_summary = self._build_episodic_summary(state)
        user_content = POLICY_DECIDE_TEMPLATE.format(
            lat=round(ctx.lat, 5), lng=round(ctx.lng, 5),
            now=now_str,
            remaining=self._format_remaining(ctx.simulation_minute),
            today_remaining_json=json.dumps(
                self._day_remaining_resource(ctx.simulation_minute, label="当前当日剩余时间"),
                ensure_ascii=False,
                default=str,
                separators=(",", ":"),
            ),
            completed_orders=ctx.completed_order_count,
            recent_actions=recent_actions,
            episodic_summary=episodic_summary,
            preference_instructions=pref_instructions,
            preference_progress=pref_progress,
            plans_json=plans_str,
            preference_facts_json=json.dumps(preference_facts, ensure_ascii=False, default=str, separators=(",", ":")),
            schedule_tasks_json=json.dumps(schedule_tasks, ensure_ascii=False, default=str, separators=(",", ":")),
            coordinate_evidence_json=json.dumps(coordinate_evidence, ensure_ascii=False, default=str, separators=(",", ":")),
        ) + POLICY_DECISION_GUARDRAILS

        payload = {
            "messages": [
                {"role": "system", "content": POLICY_SYSTEM},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0.0,
            "max_tokens": 220,
            "enable_thinking": False,
        }

        result = self._gateway.llm_chat_json(payload, trace, "PolicyAgent")
        if not isinstance(result, dict):
            raise RuntimeError("LLM returned non-dict")

        return self._interpret_decision(state, trace, result)

    def _plan_from_generated_action(
        self,
        state: DecisionState,
        decision: dict[str, Any],
        coordinate_evidence: list[dict[str, Any]],
    ) -> ActionPlan | None:
        ctx = state.driver_context
        if ctx is None:
            return None
        action = str(decision.get("action", "") or "").strip()
        params = decision.get("params", {})
        if not isinstance(params, dict):
            return None
        llm_reason = self._clean_llm_reason(decision.get("reason"))
        now = int(ctx.simulation_minute)
        remaining = max(0, DEFAULT_MONTH_HORIZON_MINUTES - now)
        if action == "wait":
            try:
                duration = int(params.get("duration_minutes", 0) or 0)
            except (TypeError, ValueError):
                return None
            if duration <= 0 or duration > 1440 or duration > remaining:
                return None
            duration = self._cap_wait_for_stationary_window(state, now, duration)
            duration = self._cap_wait_for_target_cargo(state, now, duration)
            duration = self._cap_wait_for_deadline_departure(state, now, duration)
            finish = now + duration
            timeline = self._query_scan_timeline(state)
            timeline.append({"kind": "idle", "start": now, "end": finish})
            return ActionPlan(
                "wait",
                {"duration_minutes": duration},
                score=-5.0,
                reason=f"[Policy direct action] {llm_reason or 'wait'}",
                valid=True,
                finish_minute=finish,
                duration_minutes=duration,
                meta={
                    "kind": "policy_generated_direct_action",
                    "policy_generated": True,
                    "preference_generated": True,
                    "priority": 0.9,
                    "query_scan_minutes": int(state.query_scan_minutes),
                    "timeline": timeline,
                },
            )

        if action == "reposition":
            try:
                lat = round(float(params.get("latitude", params.get("lat"))), 5)
                lng = round(float(params.get("longitude", params.get("lng"))), 5)
            except (TypeError, ValueError):
                return None
            if not (-90 <= lat <= 90 and -180 <= lng <= 180):
                return None
            if not self._coordinate_is_observed(lat, lng, coordinate_evidence):
                return None
            dist = haversine_km(ctx.lat, ctx.lng, lat, lng)
            duration = distance_to_minutes(dist)
            if duration <= 0 or duration > remaining:
                return None
            finish = now + duration
            timeline = self._query_scan_timeline(state)
            timeline.append({"kind": "reposition", "start": now, "end": finish})
            return ActionPlan(
                "reposition",
                {"latitude": lat, "longitude": lng},
                score=-10.0,
                reason=f"[Policy direct action] {llm_reason or 'reposition'}",
                valid=True,
                finish_minute=finish,
                duration_minutes=duration,
                target_lat=lat,
                target_lng=lng,
                meta={
                    "kind": "policy_generated_direct_action",
                    "policy_generated": True,
                    "preference_generated": True,
                    "priority": 0.7,
                    "distance_km": dist,
                    "target_point": {"lat": lat, "lng": lng},
                    "query_scan_minutes": int(state.query_scan_minutes),
                    "timeline": timeline,
                },
            )
        return None

    def _build_candidates(self, state: DecisionState) -> list[dict[str, Any]]:
        ctx = state.driver_context
        if ctx is None:
            return []
        cargo_by_id = {cand.cargo_id: cand.cargo for cand in state.cargo_snapshot}

        plans_data = []
        for idx, plan in enumerate(self._policy_candidate_plans(state)):
            pickup_min = int(plan.meta.get("pickup_minutes", 0) or 0)
            wait_load = int(plan.meta.get("wait_for_load", 0) or 0)
            pickup_km = float(plan.meta.get("pickup_km", 0) or 0)
            haul_km = float(plan.meta.get("haul_km", 0) or 0)

            pickup_arrival = ctx.simulation_minute + pickup_min
            loading_done = pickup_arrival + wait_load
            action_start = int(ctx.simulation_minute)

            d: dict[str, Any] = {
                "index": idx,
                "score": round(plan.score, 2),
                "cargo_id": plan.cargo_id,
                "net_income": round(plan.net_income, 1),
                "hourly_rate": round(float(plan.meta.get("net_income_per_hour", 0) or 0), 1),
                # 时间
                "pickup_arrival": self._sim_to_wall(pickup_arrival),
                "loading_done": self._sim_to_wall(loading_done),
                "delivery_done": self._sim_to_wall(plan.finish_minute),
                "action_dates": self._action_dates_resource(action_start, plan.finish_minute),
                "finish_day_remaining": self._day_remaining_resource(plan.finish_minute, label="订单结束后当日剩余时间"),
                "total_minutes": plan.duration_minutes,
                # 距离
                "dist_to_pickup_km": round(pickup_km, 1),
                "haul_km": round(haul_km, 1),
                "total_km": round(pickup_km + haul_km, 1),
            }

            cargo = cargo_by_id.get(plan.cargo_id or "")
            if cargo:
                d["cargo_name"] = cargo.get("cargo_name", "")
                if cargo.get("price") is not None:
                    d["price"] = cargo.get("price")
                start = cargo.get("start", {})
                end = cargo.get("end", {})
                start_city = self._city_from_point(start)
                end_city = self._city_from_point(end)
                if start_city and end_city:
                    d["route"] = f"{start_city}->{end_city}"
                d["from"] = f"({start.get('lat','?')},{start.get('lng','?')})"
                d["to"] = f"({end.get('lat','?')},{end.get('lng','?')})"

                # 装货时间窗
                load_time = cargo.get("load_time")
                if isinstance(load_time, list) and len(load_time) == 2:
                    d["load_window"] = f"{load_time[0]}~{load_time[1]}"

            # 额外查询标记
            if cargo:
                for cand in state.cargo_snapshot:
                    if cand.cargo_id == plan.cargo_id and cand.extra:
                        d["extra_query"] = True
                        if cand.query_origin:
                            d["query_from"] = cand.query_origin.get("reason", "")
                        break

            target_pref = self._target_cargo_preference_for_plan(state, plan)
            if target_pref:
                d["target_cargo_preference"] = target_pref

            plans_data.append(d)
        return plans_data

    def _build_preference_progress_facts(self, state: DecisionState) -> dict[str, Any]:
        ctx = state.driver_context
        now = int(ctx.simulation_minute) if ctx is not None else 0
        day_idx = now // 1440
        facts = self._policy_action_facts(state)
        by_day: dict[int, dict[str, Any]] = {}
        for day in range(0, min(31, max(day_idx + 1, 0))):
            by_day[day] = {
                "active_minutes": 0,
                "wait_minutes": 0,
                "wait_intervals": [],
            }
        for fact in facts:
            action = str(fact.get("action") or "")
            try:
                start = int(fact.get("start_minute"))
                end = int(fact.get("end_minute"))
            except (TypeError, ValueError):
                continue
            if end <= start:
                continue
            rest_start = start
            rest_end = end
            if action == "wait":
                try:
                    rest_start = int(fact.get("rest_effective_start_minute", start))
                    rest_end = int(fact.get("rest_effective_end_minute", end))
                except (TypeError, ValueError):
                    rest_start, rest_end = start, end
                rest_start = min(rest_start, start)
                rest_end = max(rest_start, rest_end)

            cur = rest_start if action == "wait" else start
            interval_end = rest_end if action == "wait" else end
            while cur < interval_end:
                day = cur // 1440
                day_end = (day + 1) * 1440
                chunk_end = min(day_end, interval_end)
                item = by_day.setdefault(day, {"active_minutes": 0, "wait_minutes": 0, "wait_intervals": []})
                if action in {"take_order", "reposition"}:
                    item["active_minutes"] += chunk_end - cur
                elif action == "wait":
                    item["wait_minutes"] += chunk_end - cur
                    item["wait_intervals"].append((cur, chunk_end))
                cur = chunk_end

        day_summaries: list[dict[str, Any]] = []
        for day in sorted(k for k in by_day if 0 <= k < 31):
            item = by_day[day]
            longest_wait = self._longest_merged_span_minutes(item.get("wait_intervals", []))
            if day >= max(0, day_idx - 6):
                day_summaries.append({
                    "day_index": day,
                    "date": self._sim_to_wall(day * 1440)[:10] if self._sim_to_wall(day * 1440) else None,
                    "active_minutes": int(item.get("active_minutes", 0) or 0),
                    "wait_minutes": int(item.get("wait_minutes", 0) or 0),
                    "longest_wait_minutes": longest_wait,
                })

        progress = state.preference_progress if isinstance(state.preference_progress, dict) else {}
        current_day = by_day.get(day_idx, {"active_minutes": 0, "wait_minutes": 0, "wait_intervals": []})
        statuses = progress.get("preference_statuses")
        return {
            "now_minute": now,
            "now_time": self._sim_to_wall(now),
            "day_index": day_idx,
            "minute_of_day": now % 1440,
            "minutes_until_midnight": 1440 - (now % 1440) if now % 1440 else 1440,
            "month_remaining_minutes": max(0, DEFAULT_MONTH_HORIZON_MINUTES - now),
            "current_day": {
                "active_minutes": int(current_day.get("active_minutes", 0) or 0),
                "wait_minutes": int(current_day.get("wait_minutes", 0) or 0),
                "longest_wait_minutes": self._longest_merged_span_minutes(current_day.get("wait_intervals", [])),
            },
            "rest_counting_rule": "评测按 step 口径计算休息：本步最终动作为 wait 时，本步 wait 前置 query_scan 与 wait 执行时间合并计入连续休息/静止；本步最终动作为 take_order 或 reposition 时，query_scan 不计入休息；接单内部等待、装卸等待、运输和 reposition 都不计入休息。",
            "recent_days": day_summaries[-8:],
            "recent_action_facts": facts[-12:],
            "order_quota_progress": progress.get("order_quota_progress") if isinstance(progress.get("order_quota_progress"), dict) else {},
            "target_cargo_progress": progress.get("target_cargo_progress") if isinstance(progress.get("target_cargo_progress"), dict) else {},
            "schedule_progress": progress.get("schedule_progress") if isinstance(progress.get("schedule_progress"), dict) else {},
            "progress_statuses": [
                {
                    "id": item.get("id"),
                    "status": "active" if str(item.get("status", "")) == "failed" else item.get("status"),
                }
                for item in statuses[-10:]
                if isinstance(item, dict)
            ] if isinstance(statuses, list) else [],
        }

    def _policy_action_facts(self, state: DecisionState) -> list[dict[str, Any]]:
        facts: list[dict[str, Any]] = []
        progress = state.preference_progress if isinstance(state.preference_progress, dict) else {}
        spans = progress.get("action_spans", [])
        if isinstance(spans, list):
            for raw in spans:
                if isinstance(raw, dict):
                    facts.extend(self._query_scan_facts_from_events(raw.get("query_scan_events")))
                    fact = self._policy_fact_from_span(raw)
                    if fact:
                        facts.append(fact)
        short = self._store.short_memory(state.driver_id)
        for item in short.recent_actions:
            action = item.get("action", {})
            if not isinstance(action, dict):
                continue
            facts.extend(self._query_scan_facts_from_events(action.get("_query_scan_events")))
            raw = {
                "action": action.get("action"),
                "start": action.get("_start_minute", item.get("minute")),
                "end": action.get("_end_minute"),
                "duration_minutes": action.get("_duration_minutes", action.get("params", {}).get("duration_minutes", 0)),
                "cargo_id": action.get("params", {}).get("cargo_id") if isinstance(action.get("params"), dict) else None,
                "start_position": action.get("_start_position"),
                "end_position": action.get("_end_position"),
                "query_scan_events": action.get("_query_scan_events"),
                "query_scan_minutes": action.get("_query_scan_minutes"),
            }
            fact = self._policy_fact_from_span(raw)
            if fact:
                facts.append(fact)
        facts.extend(self._query_scan_facts_from_events(state.query_scan_events))
        dedup: dict[tuple[Any, Any, Any], dict[str, Any]] = {}
        for fact in facts:
            key = (fact.get("action"), fact.get("start_minute"), fact.get("end_minute"))
            dedup[key] = fact
        return sorted(dedup.values(), key=lambda x: int(x.get("start_minute", 0) or 0))

    @classmethod
    def _policy_fact_from_span(cls, raw: dict[str, Any]) -> dict[str, Any] | None:
        try:
            start = int(raw.get("start"))
        except (TypeError, ValueError):
            return None
        end_raw = raw.get("end")
        if end_raw is None:
            try:
                duration = int(raw.get("duration_minutes", 0) or 0)
            except (TypeError, ValueError):
                duration = 0
            end = start + max(0, duration)
        else:
            try:
                end = int(end_raw)
            except (TypeError, ValueError):
                end = start
        end = max(start, end)
        action = str(raw.get("action") or "")
        fact: dict[str, Any] = {
            "action": action,
            "start_minute": start,
            "end_minute": end,
            "duration_minutes": max(0, end - start),
            "start_time": cls._sim_to_wall(start),
            "end_time": cls._sim_to_wall(end),
        }
        for key in ("cargo_id", "cargo_name", "source", "items_count", "query_scan_minutes"):
            if raw.get(key) is not None:
                fact[key] = raw.get(key)
        if action == "wait":
            rest_start = cls._rest_effective_start_from_query_scan(raw, start)
            if rest_start < end:
                fact["rest_effective_start_minute"] = rest_start
                fact["rest_effective_end_minute"] = end
                fact["rest_effective_start_time"] = cls._sim_to_wall(rest_start)
                fact["rest_effective_end_time"] = cls._sim_to_wall(end)
                fact["rest_credit_minutes"] = end - rest_start
        for key in ("start_position", "end_position", "start_point", "end_point", "target_point"):
            point = raw.get(key)
            if isinstance(point, dict):
                fact[key] = dict(point)
        return fact

    @staticmethod
    def _rest_effective_start_from_query_scan(raw: dict[str, Any], action_start: int) -> int:
        starts: list[int] = []
        events = raw.get("query_scan_events") or raw.get("_query_scan_events")
        if isinstance(events, list):
            for event in events:
                if not isinstance(event, dict):
                    continue
                try:
                    starts.append(int(event.get("start")))
                except (TypeError, ValueError):
                    continue
        if starts:
            return min([action_start] + starts)
        try:
            query_minutes = int(raw.get("query_scan_minutes", 0) or 0)
        except (TypeError, ValueError):
            query_minutes = 0
        return action_start - max(0, query_minutes)

    @classmethod
    def _query_scan_facts_from_events(cls, events: Any) -> list[dict[str, Any]]:
        facts: list[dict[str, Any]] = []
        if not isinstance(events, list):
            return facts
        for raw in events:
            if not isinstance(raw, dict):
                continue
            event = dict(raw)
            event["action"] = "query_scan"
            fact = cls._policy_fact_from_span(event)
            if fact:
                facts.append(fact)
        return facts

    @staticmethod
    def _longest_merged_span_minutes(intervals: Any) -> int:
        if not isinstance(intervals, list):
            return 0
        normalized: list[tuple[int, int]] = []
        for item in intervals:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                continue
            try:
                start = int(item[0])
                end = int(item[1])
            except (TypeError, ValueError):
                continue
            if end > start:
                normalized.append((start, end))
        if not normalized:
            return 0
        normalized.sort()
        merged: list[list[int]] = []
        for start, end in normalized:
            if not merged or start > merged[-1][1]:
                merged.append([start, end])
            else:
                merged[-1][1] = max(merged[-1][1], end)
        return max(end - start for start, end in merged)

    def _future_rejection_summary(self, state: DecisionState) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        seen: set[str] = set()
        for plan in self._all_take_order_plans(state):
            future = plan.meta.get("future_feasibility")
            if not isinstance(future, dict) or not future.get("blocked"):
                continue
            key = str(plan.cargo_id or plan.meta.get("future_candidate_id") or id(plan))
            if key in seen:
                continue
            seen.add(key)
            item: dict[str, Any] = {
                "candidate_id": plan.meta.get("future_candidate_id"),
                "cargo_id": plan.cargo_id,
                "score": round(float(plan.score or 0.0), 2),
                "net_income": round(float(plan.net_income or 0.0), 1),
                "duration_minutes": int(plan.duration_minutes or 0),
                "reason": future.get("reason") or plan.reason,
                "start": self._point_summary(plan.meta.get("start_point")),
                "end": self._point_summary(plan.meta.get("end_point")),
                "target": self._point_summary(plan.meta.get("target_point")),
            }
            out.append({k: v for k, v in item.items() if v not in (None, "", [])})
            if len(out) >= 12:
                break
        return out

    def _active_schedule_tasks(self, state: DecisionState) -> list[dict[str, Any]]:
        ctx = state.driver_context
        if ctx is None:
            return []
        now_wall = self._sim_to_wall(ctx.simulation_minute) or ""
        today = now_wall[:10]
        instructions = state.preference_instructions.get("instructions", [])
        if not isinstance(instructions, list):
            return []
        progress = state.preference_progress.get("schedule_progress", {}) if isinstance(state.preference_progress, dict) else {}
        hidden_ids = set(self._hidden_preference_ids(state))
        schedule_facts = self._policy_action_facts(state)
        out: list[dict[str, Any]] = []
        for inst in instructions:
            if not isinstance(inst, dict):
                continue
            pref_id = str(inst.get("id") or "")
            task = inst.get("schedule_task")
            scheme = inst.get("scheme") if isinstance(inst.get("scheme"), dict) else {}
            cycle = inst.get("cycle") if isinstance(inst.get("cycle"), dict) else {}
            pref_type = str(inst.get("preference_type") or scheme.get("type") or "")
            scheme_scope = scheme.get("scope") if isinstance(scheme.get("scope"), dict) else {}
            cycle_period = str(cycle.get("length") or scheme_scope.get("period") or "").lower()
            is_daily_schedule = pref_type in _SCHEDULE_PREFERENCE_TYPES and cycle_period in {"day", "daily"}
            if pref_id and pref_id in hidden_ids and not is_daily_schedule:
                continue
            if not isinstance(task, dict) and pref_type in _SCHEDULE_PREFERENCE_TYPES:
                task = self._schedule_task_from_scheme(inst)
            if not isinstance(task, dict):
                continue
            task = self._normalize_schedule_task(task)
            active_date = str(task.get("active_date") or "")
            if not active_date and today and cycle_period in {"day", "daily"}:
                task = dict(task)
                task["active_date"] = today
                active_date = today
            active_dates = PolicyAgent._schedule_str_list(task.get("active_dates"))
            if active_dates and today and today not in active_dates and not PolicyAgent._schedule_task_covers_date(task, today):
                continue
            if not active_dates and active_date and today and active_date != today and not PolicyAgent._schedule_task_covers_date(task, today):
                continue
            entry = progress.get(pref_id) if isinstance(progress, dict) else None
            entry = entry if isinstance(entry, dict) else {}
            deterministic_entry = self._schedule_progress_from_facts(task, schedule_facts)
            if deterministic_entry:
                entry = deterministic_entry
            current_step = self._next_schedule_step(task, entry, ctx.lat, ctx.lng, ctx.simulation_minute)
            item = {
                "preference_id": pref_id,
                "type": task.get("type") or pref_type,
                "hardness": task.get("hardness") or inst.get("hardness"),
                "active_date": active_date or None,
                "active_dates": active_dates,
                "status": entry.get("status") or ("completed" if current_step is None else "active"),
                "completed_step_ids": entry.get("completed_step_ids", []),
                "current_step": current_step,
                "steps": task.get("steps", []),
            }
            out.append({k: v for k, v in item.items() if v not in (None, "", [])})
        return out

    @staticmethod
    def _normalize_schedule_task(task: dict[str, Any]) -> dict[str, Any]:
        if str(task.get("type") or "") != "ROUTE_SEQUENCE_ON_DATE":
            return task
        out = dict(task)
        steps: list[dict[str, Any]] = []
        for raw_step in out.get("steps") or []:
            if not isinstance(raw_step, dict):
                continue
            step = dict(raw_step)
            for key in ("target", "location"):
                point = step.get(key)
                if not isinstance(point, dict):
                    continue
                if point.get("lat") is None or point.get("lng") is None:
                    continue
                normalized = dict(point)
                radius = PolicyAgent._safe_float(normalized.get("radius_km"), 1.0)
                normalized["radius_km"] = min(radius, 1.0) if radius > 0 else 1.0
                step[key] = normalized
            if step.get("finish_before") and not step.get("complete_before") and not step.get("hold_until"):
                if PolicyAgent._is_full_datetime_value(step.get("finish_before")):
                    step["hold_until"] = step.get("finish_before")
                else:
                    step["complete_before"] = step.get("finish_before")
            steps.append(step)
        out["steps"] = steps
        return out

    @classmethod
    def _schedule_progress_from_facts(cls, task: dict[str, Any], facts: list[dict[str, Any]]) -> dict[str, Any]:
        steps = task.get("steps")
        if not isinstance(steps, list):
            return {}
        active_dates = cls._schedule_str_list(task.get("active_dates"))
        window_start, _ = cls._schedule_task_window(task)
        active_date = "" if window_start is not None or len(active_dates) > 1 else str(task.get("active_date") or (active_dates[0] if active_dates else ""))
        completed: list[str] = []
        wait_by_step: dict[str, int] = {}
        current_step: str | None = None
        valid_step_count = 0
        cursor_minute = int(window_start or 0)
        for step in steps:
            if not isinstance(step, dict):
                continue
            step_id = str(step.get("step_id") or "")
            target = step.get("target") if isinstance(step.get("target"), dict) else step.get("location")
            if not step_id or not isinstance(target, dict):
                continue
            valid_step_count += 1
            kind = str(step.get("kind") or "")
            reached, reached_minute = cls._schedule_target_reached_after(facts, target, active_date, cursor_minute)
            waited, wait_complete_minute = cls._schedule_wait_minutes_after(
                facts,
                target,
                active_date,
                cursor_minute,
                cls._safe_int(step.get("min_stay_minutes"), 0),
            )
            wait_by_step[step_id] = waited
            min_stay = cls._safe_int(step.get("min_stay_minutes"), 0)
            hold_until = step.get("hold_until")
            requires_hold = cls._schedule_step_requires_hold_until(step)
            finish_satisfied, finish_minute = (
                (True, None)
                if not requires_hold
                else cls._schedule_wait_covers_finish_after(facts, target, hold_until, active_date, cursor_minute)
            )
            if kind == "stay":
                if reached and finish_satisfied and (not min_stay or waited >= min_stay):
                    completed.append(step_id)
                    cursor_candidates = [item for item in (wait_complete_minute, finish_minute, reached_minute) if item is not None]
                    if cursor_candidates:
                        cursor_minute = max(cursor_minute, max(cursor_candidates))
                    continue
            elif kind == "arrival" and str(task.get("type") or "") == "LOCATION_ARRIVAL_DEADLINE":
                if current_step is None:
                    current_step = step_id
                continue
            elif reached:
                completed.append(step_id)
                if reached_minute is not None:
                    cursor_minute = max(cursor_minute, reached_minute)
                continue
            if current_step is None:
                current_step = step_id
        if valid_step_count <= 0:
            return {}
        entry: dict[str, Any] = {
            "status": "completed" if len(completed) >= valid_step_count else "active",
            "completed_step_ids": completed,
            "wait_minutes_by_step": wait_by_step,
        }
        if current_step:
            entry["current_step_id"] = current_step
            entry["current_step_wait_minutes"] = int(wait_by_step.get(current_step, 0) or 0)
        return entry

    @classmethod
    def _schedule_target_reached(cls, facts: list[dict[str, Any]], target: dict[str, Any], active_date: str = "") -> bool:
        for fact in facts:
            if active_date and not cls._fact_touches_date(fact, active_date):
                continue
            for key in ("end_position", "start_position", "end_point", "start_point", "target_point"):
                point = fact.get(key)
                if cls._point_near_schedule_target(point, target):
                    return True
        return False

    @classmethod
    def _schedule_target_reached_after(
        cls,
        facts: list[dict[str, Any]],
        target: dict[str, Any],
        active_date: str = "",
        earliest_minute: int = 0,
    ) -> tuple[bool, int | None]:
        for fact in facts:
            if active_date and not cls._fact_touches_date(fact, active_date):
                continue
            start = cls._safe_int(fact.get("start_minute"), 0)
            end = cls._safe_int(fact.get("end_minute"), start)
            if end < earliest_minute:
                continue
            for key, minute in (
                ("start_position", start),
                ("start_point", start),
                ("target_point", end),
                ("end_position", end),
                ("end_point", end),
            ):
                point = fact.get(key)
                if cls._point_near_schedule_target(point, target):
                    return True, max(minute, earliest_minute)
        return False, None

    @classmethod
    def _schedule_wait_minutes(cls, facts: list[dict[str, Any]], target: dict[str, Any], active_date: str = "") -> int:
        total = 0
        for fact in facts:
            if active_date and not cls._fact_touches_date(fact, active_date):
                continue
            if str(fact.get("action") or "") != "wait":
                continue
            point = fact.get("end_position") or fact.get("start_position")
            if not cls._point_near_schedule_target(point, target):
                continue
            total += cls._safe_int(fact.get("rest_credit_minutes"), cls._safe_int(fact.get("duration_minutes"), 0))
        return total

    @classmethod
    def _schedule_wait_minutes_after(
        cls,
        facts: list[dict[str, Any]],
        target: dict[str, Any],
        active_date: str = "",
        earliest_minute: int = 0,
        required_minutes: int = 0,
    ) -> tuple[int, int | None]:
        total = 0
        completed_at: int | None = None
        for fact in facts:
            if active_date and not cls._fact_touches_date(fact, active_date):
                continue
            if str(fact.get("action") or "") != "wait":
                continue
            start = cls._safe_int(fact.get("start_minute"), 0)
            end = cls._safe_int(fact.get("end_minute"), start)
            if end <= earliest_minute:
                continue
            point = fact.get("end_position") or fact.get("start_position")
            if not cls._point_near_schedule_target(point, target):
                continue
            credit = max(0, end - max(start, earliest_minute))
            if credit <= 0:
                credit = cls._safe_int(fact.get("rest_credit_minutes"), cls._safe_int(fact.get("duration_minutes"), 0))
            before_total = total
            total += credit
            if required_minutes > 0 and completed_at is None and total >= required_minutes:
                completed_at = max(start, earliest_minute) + max(0, required_minutes - before_total)
        return total, completed_at

    @classmethod
    def _schedule_wait_covers_finish(cls, facts: list[dict[str, Any]], target: dict[str, Any], finish_before: Any, active_date: str = "") -> bool:
        finish_minute = cls._wall_to_sim_minute(finish_before)
        if finish_minute is None:
            return False
        for fact in facts:
            if active_date and not cls._fact_touches_date(fact, active_date):
                continue
            if str(fact.get("action") or "") != "wait":
                continue
            point = fact.get("end_position") or fact.get("start_position")
            if not cls._point_near_schedule_target(point, target):
                continue
            start = cls._safe_int(fact.get("start_minute"), -1)
            end = cls._safe_int(fact.get("end_minute"), -1)
            if start <= finish_minute <= end:
                return True
        return False

    @classmethod
    def _schedule_wait_covers_finish_after(
        cls,
        facts: list[dict[str, Any]],
        target: dict[str, Any],
        finish_before: Any,
        active_date: str = "",
        earliest_minute: int = 0,
    ) -> tuple[bool, int | None]:
        finish_minute = cls._wall_to_sim_minute(finish_before)
        if finish_minute is None or finish_minute < earliest_minute:
            return False, None
        for fact in facts:
            if active_date and not cls._fact_touches_date(fact, active_date):
                continue
            if str(fact.get("action") or "") != "wait":
                continue
            point = fact.get("end_position") or fact.get("start_position")
            if not cls._point_near_schedule_target(point, target):
                continue
            start = cls._safe_int(fact.get("start_minute"), -1)
            end = cls._safe_int(fact.get("end_minute"), -1)
            if start <= finish_minute <= end:
                return True, finish_minute
        return False, None

    @staticmethod
    def _fact_touches_date(fact: dict[str, Any], active_date: str) -> bool:
        start = str(fact.get("start_time") or "")
        end = str(fact.get("end_time") or "")
        return start.startswith(active_date) or end.startswith(active_date)

    @classmethod
    def _schedule_task_covers_date(cls, task: dict[str, Any], date: str) -> bool:
        if not date:
            return False
        start, end = cls._schedule_task_window(task)
        if start is None or end is None:
            return False
        day_start = cls._wall_to_sim_minute(f"{date[:10]} 00:00")
        if day_start is None:
            return False
        day_end = day_start + 1440
        return start < day_end and end >= day_start

    @classmethod
    def _schedule_task_window(cls, task: dict[str, Any]) -> tuple[int | None, int | None]:
        starts: list[int] = []
        for value in cls._schedule_str_list(task.get("active_dates")):
            minute = cls._wall_to_sim_minute(f"{value[:10]} 00:00")
            if minute is not None:
                starts.append(minute)
        active_date = str(task.get("active_date") or "").strip()
        if active_date:
            minute = cls._wall_to_sim_minute(f"{active_date[:10]} 00:00")
            if minute is not None:
                starts.append(minute)
        ends: list[int] = []
        steps = task.get("steps") if isinstance(task.get("steps"), list) else []
        for step in steps:
            if not isinstance(step, dict):
                continue
            for key in ("hold_until", "complete_before", "arrive_before"):
                minute = cls._wall_to_sim_minute(step.get(key))
                if minute is not None:
                    ends.append(minute)
        if not starts:
            return None, max(ends) if ends else None
        return min(starts), max(ends) if ends else max(starts) + 1440

    @staticmethod
    def _point_near_schedule_target(point: Any, target: dict[str, Any]) -> bool:
        if not isinstance(point, dict):
            return False
        try:
            lat = float(point.get("lat"))
            lng = float(point.get("lng"))
            target_lat = float(target.get("lat"))
            target_lng = float(target.get("lng"))
            radius = float(target.get("radius_km") or _SCHEDULE_RADIUS_KM)
        except (TypeError, ValueError):
            return False
        return haversine_km(lat, lng, target_lat, target_lng) <= radius

    def _plan_for_active_schedule(self, state: DecisionState, schedule_tasks: list[dict[str, Any]]) -> ActionPlan | None:
        ctx = state.driver_context
        if ctx is None:
            return None
        for task in schedule_tasks:
            if not isinstance(task, dict) or str(task.get("hardness") or "").lower() != "hard":
                continue
            step = task.get("current_step")
            if not isinstance(step, dict):
                continue
            if step.get("required_action") == "reposition":
                wait_for_window = self._wait_until_stationary_window_end(state)
                if wait_for_window is not None:
                    return wait_for_window
            action = str(step.get("required_action") or "")
            if action == "wait":
                duration = self._schedule_wait_duration(step)
                remaining = max(0, DEFAULT_MONTH_HORIZON_MINUTES - int(ctx.simulation_minute))
                duration = min(duration, remaining)
                duration = self._cap_wait_for_stationary_window(state, int(ctx.simulation_minute), duration)
                duration = self._cap_wait_for_target_cargo(state, int(ctx.simulation_minute), duration)
                if str(task.get("type") or "") != "ROUTE_SEQUENCE_ON_DATE":
                    duration = self._cap_wait_for_deadline_departure(state, int(ctx.simulation_minute), duration)
                if duration <= 0:
                    return None
                finish = int(ctx.simulation_minute) + duration
                return ActionPlan(
                    "wait",
                    {"duration_minutes": duration},
                    score=50_000.0,
                    reason=f"[Policy schedule] {task.get('preference_id')} {step.get('step_id')} wait",
                    valid=True,
                    finish_minute=finish,
                    duration_minutes=duration,
                    meta={"kind": "policy_schedule", "schedule_task": task, "schedule_step": step, "policy_generated": True, "priority": 1.0},
                )
            if action == "reposition":
                target = step.get("target") if isinstance(step.get("target"), dict) else step.get("location")
                if not isinstance(target, dict):
                    continue
                try:
                    lat = float(target.get("lat"))
                    lng = float(target.get("lng"))
                except (TypeError, ValueError):
                    continue
                minutes = max(1, distance_to_minutes(haversine_km(ctx.lat, ctx.lng, lat, lng)))
                finish = int(ctx.simulation_minute) + minutes
                return ActionPlan(
                    "reposition",
                    {"latitude": round(lat, 5), "longitude": round(lng, 5)},
                    score=50_000.0,
                    reason=f"[Policy schedule] {task.get('preference_id')} {step.get('step_id')} reposition",
                    valid=True,
                    finish_minute=finish,
                    duration_minutes=minutes,
                    meta={"kind": "policy_schedule", "schedule_task": task, "schedule_step": step, "policy_generated": True, "priority": 1.0},
                )
        return None

    def _coordinate_evidence(self, state: DecisionState, *, schedule_tasks: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
        ctx = state.driver_context
        points: list[dict[str, Any]] = []
        if ctx is not None:
            self._add_evidence_point(points, "current_position", ctx.lat, ctx.lng, name="当前位置")
        for task in schedule_tasks or []:
            steps = task.get("steps") if isinstance(task, dict) else None
            if not isinstance(steps, list):
                continue
            for step in steps:
                if not isinstance(step, dict):
                    continue
                target = step.get("target") if isinstance(step.get("target"), dict) else step.get("location")
                if not isinstance(target, dict):
                    continue
                self._add_evidence_point(
                    points,
                    "schedule_task",
                    target.get("lat"),
                    target.get("lng"),
                    point=target,
                    name=str(target.get("name") or f"schedule:{task.get('preference_id')}/{step.get('step_id')}"),
                )
        for event in state.query_scan_events[-8:]:
            if not isinstance(event, dict):
                continue
            self._add_evidence_point(points, "query_scan", event.get("lat"), event.get("lng"), name="查询位置")
        for plan in self._all_take_order_plans(state):
            future = plan.meta.get("future_feasibility")
            if isinstance(future, dict) and not future.get("blocked"):
                continue
            for key in ("start_point", "end_point", "target_point"):
                point = plan.meta.get(key)
                if isinstance(point, dict):
                    self._add_evidence_point(
                        points,
                        f"candidate_{key}",
                        point.get("lat"),
                        point.get("lng"),
                        point=point,
                        name=self._evidence_point_name(key, point),
                    )
            if len(points) >= 30:
                break
        return points[:30]

    @staticmethod
    def _schedule_task_from_scheme(inst: dict[str, Any]) -> dict[str, Any] | None:
        scheme = inst.get("scheme") if isinstance(inst.get("scheme"), dict) else {}
        pref_type = str(inst.get("preference_type") or scheme.get("type") or "")
        if pref_type not in _SCHEDULE_PREFERENCE_TYPES:
            return None
        completion = scheme.get("completion") if isinstance(scheme.get("completion"), dict) else {}
        constraint = scheme.get("constraint") if isinstance(scheme.get("constraint"), dict) else {}
        date = completion.get("date")
        active_dates = PolicyAgent._schedule_str_list(constraint.get("active_dates"))
        if not active_dates and date:
            active_dates = [str(date)]
        scope_actions = PolicyAgent._schedule_str_list(constraint.get("scope_actions"))
        constraint_kinds = PolicyAgent._schedule_str_list(constraint.get("constraint_kinds"))
        steps: list[dict[str, Any]] = []
        if pref_type == "ROUTE_SEQUENCE_ON_DATE":
            raw_steps = constraint.get("steps") if isinstance(constraint.get("steps"), list) else []
            for item in raw_steps:
                if not isinstance(item, dict):
                    continue
                target = item.get("target_location") if isinstance(item.get("target_location"), dict) else None
                if not target:
                    continue
                action = "wait" if item.get("kind") == "stay" and (item.get("min_stay_minutes") or item.get("complete_before") or item.get("hold_until") or item.get("finish_before")) else "reposition"
                step = {
                    "step_id": item.get("step_id"),
                    "sequence_index": item.get("sequence_index"),
                    "kind": item.get("kind") or action,
                    "required_action": action,
                    "target": target,
                    "location": target if action == "wait" else None,
                    "active_dates": PolicyAgent._schedule_str_list(item.get("active_dates")) or active_dates,
                    "scope_actions": PolicyAgent._schedule_str_list(item.get("scope_actions")) or scope_actions,
                    "constraint_kinds": PolicyAgent._schedule_str_list(item.get("constraint_kinds")) or constraint_kinds,
                    "arrive_before": item.get("arrive_before"),
                    "min_stay_minutes": item.get("min_stay_minutes"),
                    "complete_before": item.get("complete_before"),
                    "hold_until": item.get("hold_until"),
                    "finish_before": item.get("finish_before"),
                }
                steps.append({k: v for k, v in step.items() if v not in (None, "", [])})
        else:
            target = constraint.get("target_location") if isinstance(constraint.get("target_location"), dict) else None
            if target:
                steps.append({"step_id": "s1", "kind": "arrival", "required_action": "reposition", "target": target, "arrive_before": constraint.get("arrive_before")})
                if pref_type == "LOCATION_STAY_ON_DATE":
                    steps.append({
                        "step_id": "s2",
                        "kind": "stay",
                        "required_action": "wait",
                        "target": target,
                        "location": target,
                        "min_stay_minutes": constraint.get("min_stay_minutes") or 120,
                    })
        if not steps:
            return None
        return {
            "task_version": "schedule_task.v1",
            "preference_id": inst.get("id"),
            "type": pref_type,
            "hardness": scheme.get("hardness") or inst.get("hardness"),
            "active_date": date,
            "active_dates": active_dates,
            "scope_actions": scope_actions,
            "constraint_kinds": constraint_kinds,
            "exclude_from_future": True,
            "exclude_from_rule_filter": True,
            "steps": steps,
        }

    @staticmethod
    def _schedule_str_list(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, (list, tuple, set)):
            return [str(item).strip() for item in value if str(item).strip()]
        text = str(value).strip()
        return [text] if text else []

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _next_schedule_step(task: dict[str, Any], progress: dict[str, Any], lat: float, lng: float, simulation_minute: int | None = None) -> dict[str, Any] | None:
        steps = task.get("steps")
        if not isinstance(steps, list):
            return None
        completed = {str(item) for item in progress.get("completed_step_ids", []) if str(item)}
        wait_by_step = progress.get("wait_minutes_by_step") if isinstance(progress.get("wait_minutes_by_step"), dict) else {}
        current_step_id = str(progress.get("current_step_id") or "")
        current_wait = int(progress.get("current_step_wait_minutes", 0) or 0)
        for raw_step in steps:
            if not isinstance(raw_step, dict):
                continue
            step_id = str(raw_step.get("step_id") or "")
            if step_id and step_id in completed:
                continue
            target = raw_step.get("target") if isinstance(raw_step.get("target"), dict) else raw_step.get("location")
            near = PolicyAgent._is_near_schedule_target(lat, lng, target)
            kind = str(raw_step.get("kind") or "")
            min_stay = PolicyAgent._safe_int(raw_step.get("min_stay_minutes"), 0)
            waited = PolicyAgent._safe_int(wait_by_step.get(step_id), 0)
            requires_hold = PolicyAgent._schedule_step_requires_hold_until(raw_step)
            finish_remaining = PolicyAgent._schedule_finish_remaining(raw_step, simulation_minute) if requires_hold else 0
            if current_step_id == step_id:
                waited = max(waited, current_wait)
            if kind == "arrival" and near:
                continue
            step = dict(raw_step)
            if kind == "arrival":
                deadline = PolicyAgent._schedule_step_deadline(raw_step, task.get("active_date"), simulation_minute)
                if deadline is not None and simulation_minute is not None and isinstance(target, dict):
                    travel = PolicyAgent._travel_minutes_to_target(lat, lng, target)
                    if travel is not None and int(simulation_minute) + travel + _DEADLINE_BUFFER_MINUTES < deadline:
                        continue
                step["required_action"] = "reposition"
                return step
            if kind == "stay":
                if near:
                    if min_stay and waited >= min_stay and finish_remaining <= 0:
                        continue
                    if not min_stay and raw_step.get("hold_until") and finish_remaining <= 0:
                        continue
                    step["required_action"] = "wait"
                    remaining_by_duration = max(0, min_stay - waited) if min_stay else 0
                    if finish_remaining > 0:
                        step["remaining_wait_minutes"] = max(1, finish_remaining, remaining_by_duration)
                    else:
                        step["remaining_wait_minutes"] = max(1, remaining_by_duration) if remaining_by_duration else 60
                else:
                    step["required_action"] = "reposition"
            else:
                step["required_action"] = "reposition"
            return step
        return None

    @staticmethod
    def _is_near_schedule_target(lat: float, lng: float, target: Any) -> bool:
        if not isinstance(target, dict):
            return False
        try:
            target_lat = float(target.get("lat"))
            target_lng = float(target.get("lng"))
            radius = float(target.get("radius_km") or _SCHEDULE_RADIUS_KM)
        except (TypeError, ValueError):
            return False
        return haversine_km(lat, lng, target_lat, target_lng) <= radius

    @staticmethod
    def _schedule_wait_duration(step: dict[str, Any]) -> int:
        return max(1, min(1440, PolicyAgent._safe_int(step.get("remaining_wait_minutes"), PolicyAgent._safe_int(step.get("min_stay_minutes"), 60))))

    @staticmethod
    def _schedule_finish_remaining(step: dict[str, Any], simulation_minute: int | None) -> int:
        hold_until = step.get("hold_until")
        if not hold_until or simulation_minute is None:
            return 0
        target_minute = PolicyAgent._wall_to_sim_minute(hold_until)
        if target_minute is None:
            return 0
        return max(0, target_minute - int(simulation_minute))

    @staticmethod
    def _schedule_step_requires_hold_until(step: dict[str, Any]) -> bool:
        return PolicyAgent._wall_to_sim_minute(step.get("hold_until")) is not None

    @staticmethod
    def _schedule_step_deadline(step: dict[str, Any], active_date: Any, simulation_minute: int | None) -> int | None:
        value = step.get("arrive_before") or step.get("complete_before")
        if value in (None, ""):
            return None
        minute = PolicyAgent._wall_to_sim_minute(value)
        if minute is not None:
            return minute
        text = str(value).strip().replace("T", " ")
        if ":" not in text:
            return None
        date = str(active_date or "").strip()[:10]
        if not date and simulation_minute is not None:
            wall = PolicyAgent._sim_to_wall(simulation_minute)
            date = wall[:10] if wall else ""
        if not date:
            return None
        clock = PolicyAgent._clock_minute(text)
        base = PolicyAgent._wall_to_sim_minute(f"{date} 00:00")
        if clock is None or base is None:
            return None
        return base + clock

    @staticmethod
    def _is_full_datetime_value(value: Any) -> bool:
        text = str(value or "").strip().replace("T", " ")
        return len(text) >= 16 and text[4:5] == "-" and text[7:8] == "-" and ":" in text[11:16]

    @staticmethod
    def _travel_minutes_to_target(lat: float, lng: float, target: dict[str, Any]) -> int | None:
        try:
            target_lat = float(target.get("lat"))
            target_lng = float(target.get("lng"))
        except (TypeError, ValueError):
            return None
        return max(1, distance_to_minutes(haversine_km(lat, lng, target_lat, target_lng)))

    @staticmethod
    def _wall_to_sim_minute(value: Any) -> int | None:
        if value in (None, ""):
            return None
        from datetime import datetime
        from ..domain.rules import SIMULATION_EPOCH

        text = str(value).strip().replace("T", " ")
        if len(text) == 10 and text[4:5] == "-" and text[7:8] == "-":
            text = f"{text} 00:00"
        if len(text) >= 16:
            text = text[:16]
        try:
            dt = datetime.strptime(text, "%Y-%m-%d %H:%M")
        except ValueError:
            return None
        return int((dt - SIMULATION_EPOCH).total_seconds() // 60)

    def _wait_until_stationary_window_end(self, state: DecisionState) -> ActionPlan | None:
        ctx = state.driver_context
        if ctx is None:
            return None
        minute_of_day = int(ctx.simulation_minute) % 1440
        instructions = state.preference_instructions.get("instructions", [])
        if not isinstance(instructions, list):
            return None
        for inst in instructions:
            if not isinstance(inst, dict):
                continue
            scheme = inst.get("scheme") if isinstance(inst.get("scheme"), dict) else {}
            if str(inst.get("preference_type") or scheme.get("type") or "") != "TIME_WINDOW_STATIONARY":
                continue
            if str(inst.get("hardness") or scheme.get("hardness") or "").lower() != "hard":
                continue
            constraint = scheme.get("constraint") if isinstance(scheme.get("constraint"), dict) else {}
            windows = constraint.get("windows") if isinstance(constraint.get("windows"), list) else []
            for window in windows:
                if not isinstance(window, dict):
                    continue
                start = self._clock_minute(window.get("start_time"))
                end = self._clock_minute(window.get("end_time"))
                if start is None or end is None:
                    continue
                in_window = start <= minute_of_day < end if start <= end else minute_of_day >= start or minute_of_day < end
                if not in_window:
                    continue
                wait_minutes = (end - minute_of_day) % 1440
                if wait_minutes <= 0:
                    continue
                wait_minutes = self._cap_wait_for_target_cargo(state, int(ctx.simulation_minute), wait_minutes)
                finish = int(ctx.simulation_minute) + wait_minutes
                return ActionPlan(
                    "wait",
                    {"duration_minutes": wait_minutes},
                    score=50_000.0,
                    reason="[Policy schedule] wait until hard stationary window ends before schedule movement",
                    valid=True,
                    finish_minute=finish,
                    duration_minutes=wait_minutes,
                    meta={"kind": "policy_schedule_window_guard", "policy_generated": True, "priority": 1.0},
                )
        return None

    @staticmethod
    def _clock_minute(value: Any) -> int | None:
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

    @staticmethod
    def _safe_int(value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _all_take_order_plans(state: DecisionState) -> list[ActionPlan]:
        out: list[ActionPlan] = []
        seen: set[int] = set()
        for source in (state.ranked_plans, state.simulated_plans):
            for plan in source:
                if plan.action != "take_order":
                    continue
                marker = id(plan)
                if marker in seen:
                    continue
                seen.add(marker)
                out.append(plan)
        return out

    @staticmethod
    def _add_evidence_point(
        points: list[dict[str, Any]],
        source: str,
        lat: Any,
        lng: Any,
        *,
        point: dict[str, Any] | None = None,
        name: str | None = None,
    ) -> None:
        try:
            lat_f = round(float(lat), 5)
            lng_f = round(float(lng), 5)
        except (TypeError, ValueError):
            return
        if not (-90 <= lat_f <= 90 and -180 <= lng_f <= 180):
            return
        for existing in points:
            try:
                if abs(float(existing.get("lat")) - lat_f) < 0.00001 and abs(float(existing.get("lng")) - lng_f) < 0.00001:
                    return
            except (TypeError, ValueError):
                continue
        item: dict[str, Any] = {"source": source, "name": name or source, "lat": lat_f, "lng": lng_f}
        if isinstance(point, dict):
            for key in ("city", "district", "county", "address"):
                value = point.get(key)
                if value not in (None, ""):
                    item[key] = value
        points.append(item)

    @staticmethod
    def _evidence_point_name(kind: str, point: dict[str, Any]) -> str:
        place = None
        for key in ("address", "city", "district", "county"):
            value = point.get(key)
            if value not in (None, ""):
                place = str(value)
                break
        role = {
            "start_point": "候选装货地",
            "end_point": "候选卸货地",
            "target_point": "候选目标点",
        }.get(kind, "候选位置")
        return f"{role}：{place}" if place else role

    @staticmethod
    def _coordinate_is_observed(lat: float, lng: float, evidence: list[dict[str, Any]]) -> bool:
        for point in evidence:
            try:
                point_lat = float(point.get("lat"))
                point_lng = float(point.get("lng"))
                if haversine_km(point_lat, point_lng, lat, lng) <= _OBSERVED_REPOSITION_RADIUS_KM:
                    return True
            except (TypeError, ValueError):
                continue
        return False

    @staticmethod
    def _point_summary(point: Any) -> dict[str, Any] | None:
        if not isinstance(point, dict):
            return None
        out: dict[str, Any] = {}
        if point.get("city") not in (None, ""):
            out["city"] = point.get("city")
        try:
            out["lat"] = round(float(point.get("lat")), 5)
            out["lng"] = round(float(point.get("lng")), 5)
        except (TypeError, ValueError):
            pass
        return out or None

    @staticmethod
    def _query_scan_timeline(state: DecisionState) -> list[dict[str, Any]]:
        timeline: list[dict[str, Any]] = []
        for event in state.query_scan_events:
            if not isinstance(event, dict):
                continue
            try:
                start = int(event.get("start"))
                end = int(event.get("end"))
            except (TypeError, ValueError):
                continue
            if end <= start:
                continue
            timeline.append({
                "kind": "query_scan",
                "start": start,
                "end": end,
                "duration_minutes": end - start,
                "location": {"lat": event.get("lat"), "lng": event.get("lng")},
                "source": event.get("source"),
                "items_count": event.get("items_count"),
            })
        return timeline

    def _interpret_decision(self, state: DecisionState, trace: TraceContext, decision: dict[str, Any]) -> ActionPlan:
        ctx = state.driver_context
        if ctx is None:
            raise ValueError("missing driver_context")

        chosen_index = decision.get("chosen_index")
        llm_reason = self._clean_llm_reason(decision.get("reason"))
        candidate_plans = self._policy_candidate_plans(state)
        if isinstance(chosen_index, int) and 0 <= chosen_index < len(candidate_plans):
            chosen = candidate_plans[chosen_index]
            if not chosen.valid:
                raise RuntimeError(f"chosen_index {chosen_index} is invalid (valid=False)")
            if self._stationary_violations_for_plan(state, chosen):
                fallback = self._fallback_wait_plan(state, f"LLM chose plan blocked by TIME_WINDOW_STATIONARY: {chosen.cargo_id}")
                if fallback is not None:
                    return fallback
            if self._forbidden_geofence_violations_for_plan(state, chosen) or self._monthly_deadhead_violations_for_plan(state, chosen):
                fallback = self._fallback_wait_plan(state, f"LLM chose plan blocked by scheme handler: {chosen.cargo_id}")
                if fallback is not None:
                    return fallback
            if llm_reason:
                chosen.reason = f"[Policy LLM] {llm_reason}"
            self._telemetry.emit(
                trace, event="POLICY_LLM_SELECTED", source="PolicyAgent", phase=self.phase,
                simulation_minute=ctx.simulation_minute,
                payload={"chosen_index": chosen_index, "action": chosen.action, "llm_reason": llm_reason},
            )
            return chosen
        if isinstance(chosen_index, int):
            fallback = self._fallback_wait_plan(
                state,
                f"LLM chose invalid chosen_index {chosen_index} with {len(candidate_plans)} take_order candidates",
            )
            if fallback is not None:
                self._telemetry.emit(
                    trace,
                    event="POLICY_INVALID_CHOSEN_INDEX_FALLBACK",
                    source="PolicyAgent",
                    phase=self.phase,
                    simulation_minute=ctx.simulation_minute,
                    payload={
                        "chosen_index": chosen_index,
                        "candidate_count": len(candidate_plans),
                        "fallback_action": fallback.action,
                        "llm_reason": llm_reason,
                    },
                )
                return fallback

        direct_action = str(decision.get("action", "") or "").strip()
        if direct_action in {"wait", "reposition"}:
            coordinate_evidence = self._coordinate_evidence(state)
            generated = self._plan_from_generated_action(state, decision, coordinate_evidence)
            if generated is not None:
                if (
                    self._geofence_violations_for_plan(state, generated)
                    or self._forbidden_geofence_violations_for_plan(state, generated)
                    or self._monthly_deadhead_violations_for_plan(state, generated)
                    or self._stationary_violations_for_plan(state, generated)
                ):
                    generated = None
            if generated is not None:
                self._telemetry.emit(
                    trace,
                    event="POLICY_LLM_SELECTED",
                    source="PolicyAgent",
                    phase=self.phase,
                    simulation_minute=ctx.simulation_minute,
                    payload={"action": generated.action, "params": generated.params, "direct_action": True, "llm_reason": llm_reason},
                )
                return generated
            fallback = self._fallback_wait_plan(state, f"LLM direct {direct_action} did not match executable constraints")
            if fallback is not None:
                self._telemetry.emit(
                    trace,
                    event="POLICY_DIRECT_ACTION_FALLBACK",
                    source="PolicyAgent",
                    phase=self.phase,
                    simulation_minute=ctx.simulation_minute,
                    payload={"requested_action": direct_action, "fallback_action": fallback.action, "reason": fallback.reason},
                )
                return fallback

        raise RuntimeError(f"LLM decision must choose chosen_index or direct wait/reposition action: {decision}")

    @staticmethod
    def _clean_llm_reason(value: Any) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        text = " ".join(text.split())
        return text[:240]

    @staticmethod
    def _fallback_wait_plan(state: DecisionState, reason: str) -> ActionPlan | None:
        ctx = state.driver_context
        if ctx is None:
            return None
        now = int(ctx.simulation_minute)
        remaining = max(0, DEFAULT_MONTH_HORIZON_MINUTES - now)
        if remaining <= 0:
            return None
        duration = min(30, remaining)
        duration = PolicyAgent._cap_wait_for_stationary_window(state, now, duration)
        duration = PolicyAgent._cap_wait_for_target_cargo(state, now, duration)
        duration = PolicyAgent._cap_wait_for_deadline_departure(state, now, duration)
        finish = now + duration
        return ActionPlan(
            "wait",
            {"duration_minutes": duration},
            score=-50.0,
            reason=f"[Policy fallback] {reason}",
            valid=True,
            finish_minute=finish,
            duration_minutes=duration,
            meta={"kind": "policy_generated_fallback", "policy_generated": True, "priority": 0.1},
        )

    @staticmethod
    def _policy_candidate_plans(state: DecisionState) -> list[ActionPlan]:
        """Return exactly the take-order plans exposed to the policy LLM."""
        plans = [
            p for p in state.ranked_plans
            if p.action == "take_order" and p.valid
        ]
        target_ids = PolicyAgent._active_target_cargo_ids(state)
        if PolicyAgent._has_future_llm_filter(state):
            plans = [
                p for p in plans
                if (
                    isinstance(p.meta.get("future_feasibility"), dict)
                    and p.meta["future_feasibility"].get("source") == "llm"
                )
                or str(p.cargo_id or "") in target_ids
            ]
        head = plans[:_POLICY_TOP_LIMIT]
        seen = {id(plan) for plan in head}
        for plan in plans[_POLICY_TOP_LIMIT:]:
            if str(plan.cargo_id or "") in target_ids and id(plan) not in seen:
                head.append(plan)
                seen.add(id(plan))
        return head[: max(_POLICY_TOP_LIMIT, len(head))]

    @staticmethod
    def _forced_target_cargo_plan(state: DecisionState) -> ActionPlan | None:
        return TARGET_CARGO_HANDLER.force_take_plan(state)

    @staticmethod
    def _plan_for_geofence_recovery(state: DecisionState) -> ActionPlan | None:
        return GEOFENCE_STAY_WITHIN_HANDLER.recovery_plan(state)

    @staticmethod
    def _geofence_violations_for_plan(state: DecisionState, plan: ActionPlan) -> list[dict[str, Any]]:
        return GEOFENCE_STAY_WITHIN_HANDLER.violations_for_plan(state, plan)

    @staticmethod
    def _forbidden_geofence_violations_for_plan(state: DecisionState, plan: ActionPlan) -> list[dict[str, Any]]:
        return GEOFENCE_FORBIDDEN_AREA_HANDLER.violations_for_plan(state, plan)

    @staticmethod
    def _monthly_deadhead_violations_for_plan(state: DecisionState, plan: ActionPlan) -> list[dict[str, Any]]:
        return MONTHLY_DEADHEAD_LIMIT_HANDLER.violations_for_plan(state, plan)

    @staticmethod
    def _plan_for_forbidden_geofence_recovery(state: DecisionState) -> ActionPlan | None:
        return GEOFENCE_FORBIDDEN_AREA_HANDLER.recovery_plan(state)

    @staticmethod
    def _stationary_violations_for_plan(state: DecisionState, plan: ActionPlan) -> list[dict[str, Any]]:
        return TIME_WINDOW_STATIONARY_HANDLER.violations_for_plan(state, plan)

    @staticmethod
    def _plan_for_stationary_window(state: DecisionState) -> ActionPlan | None:
        return TIME_WINDOW_STATIONARY_HANDLER.plan_for_current_window(state)

    @staticmethod
    def _plan_for_arrival_deadline_guard(state: DecisionState) -> ActionPlan | None:
        return LOCATION_ARRIVAL_DEADLINE_HANDLER.plan_for_departure_guard(state)

    @staticmethod
    def _plan_for_target_cargo_waiting(state: DecisionState) -> ActionPlan | None:
        return TARGET_CARGO_HANDLER.plan_for_waiting(state)

    @staticmethod
    def _plan_for_target_cargo_positioning(state: DecisionState) -> ActionPlan | None:
        return TARGET_CARGO_HANDLER.plan_for_positioning(state)

    @staticmethod
    def _active_target_cargo_ids(state: DecisionState) -> set[str]:
        return TARGET_CARGO_HANDLER.active_ids(state)

    @staticmethod
    def _active_target_cargo_pickups(state: DecisionState) -> list[dict[str, Any]]:
        return TARGET_CARGO_HANDLER.active_targets(state)

    @staticmethod
    def _target_cargo_preference_for_plan(state: DecisionState, plan: ActionPlan) -> dict[str, Any] | None:
        return TARGET_CARGO_HANDLER.preference_for_plan(state, plan)

    @staticmethod
    def _cap_wait_for_stationary_window(state: DecisionState, now: int, duration: int) -> int:
        if duration <= 0:
            return duration
        window_end = PolicyAgent._current_stationary_window_end_minute(state, now)
        if window_end is None:
            return duration
        finish = now + duration
        if finish <= window_end:
            return duration
        return max(1, window_end - now)

    @staticmethod
    def _current_stationary_window_end_minute(state: DecisionState, now: int) -> int | None:
        return TIME_WINDOW_STATIONARY_HANDLER.current_window_end_minute(state, now)

    @staticmethod
    def _cap_wait_for_target_cargo(state: DecisionState, now: int, duration: int) -> int:
        if duration <= 0:
            return duration
        wake_minute = PolicyAgent._next_target_cargo_wakeup_minute(state, now)
        if wake_minute is None:
            return duration
        finish = now + duration
        if finish <= wake_minute:
            return duration
        return max(1, wake_minute - now)

    @staticmethod
    def _cap_wait_for_deadline_departure(state: DecisionState, now: int, duration: int) -> int:
        if duration <= 0:
            return duration
        departures = [
            minute
            for minute in (
                PolicyAgent._next_arrival_deadline_departure_minute(state, now),
                PolicyAgent._next_target_pickup_departure_minute(state, now),
            )
            if minute is not None and minute > now
        ]
        if not departures:
            return duration
        wake_minute = min(departures)
        finish = now + duration
        if finish <= wake_minute:
            return duration
        return max(1, wake_minute - now)

    @staticmethod
    def _next_arrival_deadline_departure_minute(state: DecisionState, now: int) -> int | None:
        return LOCATION_ARRIVAL_DEADLINE_HANDLER.next_departure_minute(state, now)

    @staticmethod
    def _active_arrival_deadline_targets(state: DecisionState, now: int) -> list[tuple[dict[str, Any], int]]:
        return LOCATION_ARRIVAL_DEADLINE_HANDLER.active_daily_targets(state, now)

    @staticmethod
    def _next_target_pickup_departure_minute(state: DecisionState, now: int) -> int | None:
        return TARGET_CARGO_HANDLER.next_pickup_departure_minute(state, now)

    @staticmethod
    def _next_target_cargo_wakeup_minute(state: DecisionState, now: int) -> int | None:
        return TARGET_CARGO_HANDLER.next_wakeup_minute(state, now)

    @staticmethod
    def _has_future_llm_filter(state: DecisionState) -> bool:
        return any(
            isinstance(p.meta.get("future_feasibility"), dict)
            and p.meta["future_feasibility"].get("source") == "llm"
            for p in state.simulated_plans
        )

    @staticmethod
    def _sim_to_wall(sim_minute: int | None) -> str | None:
        if sim_minute is None or sim_minute < 0:
            return None
        from datetime import timedelta
        from ..domain.rules import SIMULATION_EPOCH
        return (SIMULATION_EPOCH + timedelta(minutes=int(sim_minute))).strftime("%Y-%m-%d %H:%M")

    @staticmethod
    def _city_from_point(point: Any) -> str | None:
        if isinstance(point, dict):
            city = point.get("city")
            return str(city) if city not in (None, "") else None
        return None

    @staticmethod
    def _format_remaining(simulation_minute: int) -> str:
        remaining = DEFAULT_MONTH_HORIZON_MINUTES - simulation_minute
        if remaining <= 0:
            return "已结束"
        days = remaining // 1440
        hours = (remaining % 1440) // 60
        return f"{days}天{hours}小时" if days > 0 else f"{hours}小时"

    @staticmethod
    def _day_remaining_resource(sim_minute: int | None, *, label: str) -> dict[str, Any]:
        if sim_minute is None:
            return {"label": label, "date": None, "wall_time": None, "minutes": 0, "text": f"{label}未知"}
        minute = max(0, int(sim_minute))
        wall = PolicyAgent._sim_to_wall(minute)
        minute_of_day = minute % 1440
        remaining = 1440 - minute_of_day if minute_of_day else 1440
        date_text = PolicyAgent._wall_date_text(wall)
        return {
            "label": label,
            "date": wall[:10] if wall else None,
            "wall_time": wall,
            "minutes": remaining,
            "text": f"{date_text}{label}为{PolicyAgent._format_minutes_duration(remaining)}",
        }

    @staticmethod
    def _action_dates_resource(start_minute: int | None, finish_minute: int | None) -> dict[str, Any]:
        start_wall = PolicyAgent._sim_to_wall(start_minute) if start_minute is not None else None
        finish_wall = PolicyAgent._sim_to_wall(finish_minute) if finish_minute is not None else None
        start_date = start_wall[:10] if start_wall else None
        finish_date = finish_wall[:10] if finish_wall else None
        start_text = PolicyAgent._wall_date_time_text(start_wall)
        finish_text = PolicyAgent._wall_date_time_text(finish_wall)
        return {
            "start_date": start_date,
            "start_wall_time": start_wall,
            "finish_date": finish_date,
            "finish_wall_time": finish_wall,
            "crosses_date": bool(start_date and finish_date and start_date != finish_date),
            "text": f"{start_text}开始，{finish_text}结束" if start_wall and finish_wall else "",
        }

    @staticmethod
    def _format_minutes_duration(minutes: int) -> str:
        minutes = max(0, int(minutes))
        hours = minutes // 60
        mins = minutes % 60
        if hours and mins:
            return f"{hours}小时{mins}分钟"
        if hours:
            return f"{hours}小时"
        return f"{mins}分钟"

    @staticmethod
    def _wall_date_text(wall: str | None) -> str:
        if not wall:
            return ""
        try:
            month = int(wall[5:7])
            day = int(wall[8:10])
        except (TypeError, ValueError):
            return wall[:10]
        return f"{month}月{day}日"

    @staticmethod
    def _wall_date_time_text(wall: str | None) -> str:
        if not wall:
            return "未知时间"
        return f"{PolicyAgent._wall_date_text(wall)} {wall[11:16]}"

    @staticmethod
    def _format_instructions(state: DecisionState) -> str:
        """格式化步骤化指令给 PolicyAgent — 只包含未完成的。"""
        data = state.preference_instructions
        instructions = data.get("instructions", [])
        if not instructions:
            return "无特殊偏好"
        lines = []
        completed_ids = set(PolicyAgent._completed_preference_ids(state))
        hidden_ids = set(PolicyAgent._hidden_preference_ids(state))
        compact = PolicyAgent._compact_instruction_items(instructions, completed_ids, hidden_ids, state)
        return json.dumps(compact, ensure_ascii=False, default=str, separators=(",", ":")) if compact else "no active preference instructions"
        for inst in instructions:
            if not isinstance(inst, dict):
                continue
            if inst.get("completed"):
                continue
            inst_id = str(inst.get("id", ""))
            if inst_id in hidden_ids:
                continue
            if inst_id in completed_ids and inst_id not in PolicyAgent._keep_monitoring_preference_ids(state):
                continue

            steps = inst.get("steps", [])
            check = inst.get("completion_check", "")

            lines.append(f"[{inst_id}]")
            for step in steps:
                lines.append(f"  {step}")
            if check:
                lines.append(f"  判定: {check}")
            lines.append("")
        return "\n".join(lines).strip() if lines else "无未完成的偏好约束"

    @staticmethod
    def _compact_instruction_items(
        instructions: list[Any],
        completed_ids: set[str],
        hidden_ids: set[str],
        state: DecisionState,
    ) -> list[dict[str, Any]]:
        compact: list[dict[str, Any]] = []
        keep_monitoring = PolicyAgent._keep_monitoring_preference_ids(state)
        for inst in instructions:
            if not isinstance(inst, dict):
                continue
            if inst.get("completed"):
                continue
            inst_id = str(inst.get("id", "") or "")
            must_keep = PolicyAgent._must_keep_monitoring(inst)
            if inst_id in hidden_ids and not must_keep:
                continue
            if inst_id in completed_ids and inst_id not in keep_monitoring and not must_keep:
                continue
            from .preference_utils import compact_instruction_for_llm

            item = compact_instruction_for_llm(inst)
            if isinstance(inst.get("steps"), list):
                item["steps"] = [str(step)[:150] for step in inst.get("steps", [])[:3] if str(step).strip()]
            compact.append(item)
        return compact

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

    @staticmethod
    def _completed_preference_ids(state: DecisionState) -> list[str]:
        value = state.preference_progress.get("completed_ids", [])
        if isinstance(value, str):
            return [value] if value.strip() else []
        if isinstance(value, list):
            return [str(item) for item in value if str(item).strip()]
        return []

    @staticmethod
    def _hidden_preference_ids(state: DecisionState) -> list[str]:
        value = state.preference_progress.get("hidden_completed_ids", [])
        if isinstance(value, str):
            return [value] if value.strip() else []
        if isinstance(value, list):
            return [str(item) for item in value if str(item).strip()]
        return []

    @staticmethod
    def _keep_monitoring_preference_ids(state: DecisionState) -> set[str]:
        value = state.preference_progress.get("preference_statuses", [])
        if not isinstance(value, list):
            return set()
        out: set[str] = set()
        for item in value:
            if not isinstance(item, dict):
                continue
            if str(item.get("status", "")) == "satisfied_keep_monitoring":
                inst_id = str(item.get("id", "")).strip()
                if inst_id:
                    out.add(inst_id)
        return out

    @staticmethod
    def _format_progress(state: DecisionState) -> str:
        """格式化偏好进度（文字描述）。"""
        progress = state.preference_progress
        if not progress:
            return "暂无进度记录"
        compact: dict[str, Any] = {}
        for key in (
            "completed_ids",
            "active_ids",
            "hidden_completed_ids",
            "kept_active_ids",
            "unknown_ids",
            "missing_information",
            "future_feasibility",
            "order_quota_progress",
            "target_cargo_progress",
            "schedule_progress",
            "last_llm_update_minute",
            "last_local_update_minute",
        ):
            if key in progress:
                compact[key] = progress.get(key)
        statuses = progress.get("preference_statuses")
        if isinstance(statuses, list):
            compact["preference_statuses"] = [
                {"id": item.get("id"), "status": "active" if str(item.get("status", "")) == "failed" else item.get("status")}
                for item in statuses[-10:]
                if isinstance(item, dict)
            ]
        spans = progress.get("action_spans")
        if isinstance(spans, list):
            compact["action_span_count"] = len(spans)
        if compact:
            return json.dumps(compact, ensure_ascii=False, default=str, separators=(",", ":"))
        return "暂无进度记录"

    def _build_recent_actions(self, state: DecisionState) -> str:
        """构建近期动作摘要（墙钟时间，含路线信息）。"""
        short = self._store.short_memory(state.driver_id)
        ctx = state.driver_context
        recent = short.get_actions_within(ctx.simulation_minute, 1440) if ctx else list(short.recent_actions)
        if not recent:
            return "无近期动作"

        # 构建 cargo_id → 信息映射（当前可见 + 历史订单）
        cargo_map = {c.cargo_id: c.cargo for c in state.cargo_snapshot}
        long = self._store.long_memory(state.driver_id)
        ep_map: dict[str, dict[str, Any]] = {}
        for ep in long.episodic_memory:
            cid = ep.get("cargo_id")
            if cid:
                ep_map[cid] = ep

        lines = []
        for item in recent[-8:]:
            minute = item.get("minute")
            act = item.get("action", {})
            action_type = act.get("action", "")
            params = act.get("params", {})
            wall = self._sim_to_wall(minute) if minute else "?"

            if action_type == "take_order":
                cargo_id = params.get("cargo_id", "")
                cargo = cargo_map.get(cargo_id)
                if cargo:
                    start = cargo.get("start", {})
                    end = cargo.get("end", {})
                    start_city = self._city_from_point(start) or f"({start.get('lat','?')},{start.get('lng','?')})"
                    end_city = self._city_from_point(end) or f"({end.get('lat','?')},{end.get('lng','?')})"
                    price = cargo.get("price", 0)
                    if price and price > 10000:
                        price = price / 100.0
                    lines.append(f"  {wall} 接单 {start_city}->{end_city} 收入{price:.0f}元")
                elif cargo_id in ep_map:
                    ep = ep_map[cargo_id]
                    income = float(ep.get("net_income", 0) or 0)
                    tlat = ep.get("target_lat", "?")
                    tlng = ep.get("target_lng", "?")
                    lines.append(f"  {wall} 接单 ->({tlat},{tlng}) 收入{income:.0f}元")
                else:
                    lines.append(f"  {wall} 接单 {cargo_id}")
            elif action_type == "wait":
                dur = int(params.get("duration_minutes", 0) or 0)
                hours = dur // 60
                mins = dur % 60
                if hours > 0 and mins > 0:
                    time_str = f"{hours}小时{mins}分"
                elif hours > 0:
                    time_str = f"{hours}小时"
                else:
                    time_str = f"{mins}分钟"
                end_wall = self._sim_to_wall(minute + dur) if minute else "?"
                lines.append(f"  {wall} 等待{time_str} 至{end_wall}")
            elif action_type == "reposition":
                lat = params.get("latitude", params.get("lat", "?"))
                lng = params.get("longitude", params.get("lng", "?"))
                lines.append(f"  {wall} reposition to ({lat},{lng})")
            else:
                lines.append(f"  {wall} {action_type}")
        return "\n".join(lines) if lines else "无"

    def _build_episodic_summary(self, state: DecisionState) -> str:
        """构建历史订单摘要 + 今日聚合数据。"""
        long = self._store.long_memory(state.driver_id)
        ctx = state.driver_context
        lines = []

        # 今日聚合数据
        if ctx:
            summary = long.get_daily_summary(ctx.simulation_minute)
            lines.append(f"今日(第{summary.day_index}天): 接{summary.orders_taken}单 收入{summary.total_income:.0f}元 工作{summary.work_minutes}分 wait{summary.rest_minutes}分")

        # 历史总计
        episodes = long.episodic_memory
        if episodes:
            total_income = sum(float(ep.get("net_income", 0) or 0) for ep in episodes)
            count = len(episodes)
            lines.append(f"历史总计: {count}单 {total_income:.0f}元")

        # 最近 5 单
        recent_eps = episodes[-5:] if episodes else []
        if recent_eps:
            lines.append("最近订单:")
            for ep in recent_eps:
                wall = self._sim_to_wall(ep.get("simulation_minute"))
                income = float(ep.get("net_income", 0) or 0)
                lines.append(f"  {wall} 收入{income:.0f}元")

        # 目的地统计
        if episodes:
            dest_counts: dict[str, int] = {}
            for ep in episodes:
                lat = ep.get("target_lat")
                lng = ep.get("target_lng")
                if lat and lng:
                    key = f"({round(float(lat),1)},{round(float(lng),1)})"
                    dest_counts[key] = dest_counts.get(key, 0) + 1
            top_dests = sorted(dest_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            if top_dests:
                lines.append("常去目的地:")
                for d, c in top_dests:
                    lines.append(f"  {d}: {c}单")

        return "\n".join(lines) if lines else "暂无历史数据"
