from __future__ import annotations

import json
import logging
from typing import Any

from ..gateway import GatewayLayer
from ..domain.rules import haversine_km
from ..messages import TraceContext
from ..state_store import DecisionState, StateStore
from ..telemetry import Telemetry
from .preference_rule_filter import _ProgressFacts, _int, _scheme_date_minute, _target_location
from .preference_utils import is_persistent_preference_instruction

logger = logging.getLogger("agent.preference_state_machine")

_SCHEDULE_RADIUS_KM = 5.0


def _nested_dict(value: dict[str, Any], *keys: str) -> Any:
    cur: Any = value
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _order_quota_required(scheme: dict[str, Any]) -> int | None:
    return (
        _int(_nested_dict(scheme, "constraint", "value"))
        or _int(_nested_dict(scheme, "completion", "count_min"))
        or _int(_nested_dict(scheme, "constraint", "count_min"))
    )


def _order_quota_matching_days(spans: list[dict[str, Any]], scheme: dict[str, Any]) -> set[int]:
    out: set[int] = set()
    for span in spans:
        if not isinstance(span, dict) or str(span.get("action") or "") != "take_order":
            continue
        if not _span_matches_order_quota(span, scheme):
            continue
        start = _int(span.get("start"))
        end = _int(span.get("end")) or start
        if start is None:
            continue
        last = max(start, int(end or start) - 1)
        for day in range(start // 1440, last // 1440 + 1):
            out.add(day)
    return out


def _sim_day_to_date(day: int) -> str:
    from datetime import timedelta
    from ..domain.rules import SIMULATION_EPOCH

    return (SIMULATION_EPOCH + timedelta(days=int(day))).strftime("%Y-%m-%d")


def _span_matches_order_quota(span: dict[str, Any], scheme: dict[str, Any]) -> bool:
    conditions = _nested_dict(scheme, "constraint", "conditions")
    checks = None
    if isinstance(conditions, dict):
        checks = conditions.get("raw_checks") or conditions.get("checks")
    if not isinstance(checks, list) or not checks:
        return True
    relevant = 0
    for check in checks:
        if not isinstance(check, dict):
            continue
        action = str(check.get("action") or "take_order")
        if action and action != "take_order":
            continue
        measure = str(check.get("measure") or "")
        if measure not in {"city", "cargo_name"}:
            continue
        relevant += 1
        if not _span_check_passes(span, check):
            return False
    return relevant > 0


def _span_check_passes(span: dict[str, Any], check: dict[str, Any]) -> bool:
    measure = str(check.get("measure") or "")
    compare = str(check.get("compare") or "contains")
    expected = str(check.get("value") or "")
    if not expected:
        return True
    if measure == "cargo_name":
        values = [str(span.get("cargo_name") or "")]
    elif measure == "city":
        values = []
        for key in ("start_point", "end_point"):
            point = span.get(key)
            if not isinstance(point, dict):
                continue
            for field in ("city", "district", "county", "address", "name"):
                value = point.get(field)
                if value not in (None, ""):
                    values.append(str(value))
    else:
        return True
    if compare in {"contains", "equals"}:
        return any((expected in value) if compare == "contains" else (expected == value) for value in values)
    if compare in {"not_contains", "not_equals"}:
        return all((expected not in value) if compare == "not_contains" else (expected != value) for value in values)
    return any(expected in value for value in values)


PROGRESS_SYSTEM = (
    "你是偏好进度追踪器。根据司机的最新动作和当前状态，更新偏好指令的完成情况。"
    "只输出合法 JSON，不要输出 markdown 代码块或任何解释文字。"
    "近期动作事实可能包含 query_scan 和位置字段；query_scan 是环境查询耗时事实，不等同于接单或空驶，"
    "休息证据按评测 step 口径：如果本步最终动作是 wait，本步 wait 前置 query_scan 与 wait 执行时间合并计入连续静止/休息；"
    "如果本步最终动作是 take_order 或 reposition，query_scan 不计入休息；"
    "是否满足某条偏好只能依据偏好文本和证据判断。"
)

PROGRESS_TEMPLATE = """更新偏好指令的完成进度。

当前时间：{now}
当前位置：({lat},{lng})

偏好指令（可能包含 category/hardness/required_fields/steps/completion_check）：
{instructions}

上一轮进度描述：
{last_progress}

司机最新动作（JSON；take_order 会包含 cargo_name、起终点位置字段、距离与时间线）：
{latest_action}

近期动作事实（JSON；包含 action、start_time、end_time，用于判断动作时序状态）：
{recent_action_facts}

任务：
1. 逐条检查每个偏好指令的完成情况
2. 对照每条的 completion_check 判断是否已满足
3. 用一段话描述当前进度（中文，简洁明了）
4. 为每条偏好给出生命周期状态，不要只按是否 persistent 机械保留
5. 如果某条偏好已经满足但后续仍可能被破坏、仍需持续监控，使用 satisfied_keep_monitoring
6. 如果判断某条偏好仍缺字段，不要臆测；在 missing_information 中说明缺什么
7. 只记录进度状态，不判断历史是否违约；即使已有动作看似不满足偏好，也继续使用 active 或 unknown，不要输出 failed
8. 不要输出 completed_ids 或 active_ids；这两个集合由代码按 status 统一生成

输出格式：
{{
  "progress_text": "用一段话描述所有偏好的完成进度",
  "preferences": [
    {{
      "id": "pref_1",
      "status": "active|satisfied_keep_monitoring|satisfied_hide|unknown",
      "evidence": "依据哪些动作、位置、时间或进度"
    }}
  ],
  "missing_information": {{"pref_id": ["缺少字段1", "缺少字段2"]}}
}}

示例：
{{
  "progress_text": "pref_1 当前仍需跟踪；pref_2 已根据最新动作满足，需要关注下一个周期。",
  "preferences": [
    {{"id":"pref_1","status":"active","evidence":"现有事实不足以关闭该偏好"}},
    {{"id":"pref_2","status":"satisfied_hide","evidence":"最新动作事实与 completion_check 一致"}}
  ],
  "missing_information": {{}}
}}
"""


class PreferenceStateMachine:
    """决策后偏好状态维护：更新进度描述，标记已完成，过滤已完成的不喂给 PolicyAgent。"""

    phase = "UPDATE_PREFERENCE_STATE"
    _RECENT_FACT_LIMIT = 20
    _LLM_REFRESH_INTERVAL_MINUTES = 720

    def __init__(self, store: StateStore, telemetry: Telemetry, gateway: GatewayLayer) -> None:
        self._store = store
        self._telemetry = telemetry
        self._gateway = gateway

    def run(self, state: DecisionState, trace: TraceContext) -> DecisionState:
        self._telemetry.emit(trace, event="AGENT_STARTED", source="PreferenceStateMachine", phase=self.phase)
        ctx = state.driver_context
        if ctx is None:
            return state

        instructions = state.preference_instructions.get("instructions", [])
        if not instructions:
            state.phase = self.phase
            return state

        decision = state.selected_intent
        if decision is None:
            return state

        # 用 LLM 更新进度描述
        should_call, skip_reason = self._should_call_llm_update(state, instructions, decision)
        if not should_call:
            self._local_progress_update(state, skip_reason)
            state.phase = self.phase
            self._store.checkpoint(state, "CKPT_PREF_STATE_READY")
            self._telemetry.emit(
                trace,
                event="PREFERENCE_STATE_LLM_SKIPPED",
                source="PreferenceStateMachine",
                phase=self.phase,
                simulation_minute=ctx.simulation_minute,
                payload={"reason": skip_reason},
            )
            return state

        prior_action_spans = state.preference_progress.get("action_spans", [])
        result = self._update_progress(state, instructions, decision, trace)

        if result:
            progress_text = result.get("progress_text", "")
            lifecycle = self._normalize_lifecycle(result.get("preferences", []), instructions)
            lifecycle = self._apply_deterministic_lifecycle_overrides(state, lifecycle, instructions)
            missing_information = self._normalize_missing_information(result.get("missing_information", {}))
            reported_completed_ids: set[str] = set()
            active_ids: set[str] = set()
            hidden_completed_ids: set[str] = set()
            kept_active_ids: set[str] = set()
            unknown_ids: set[str] = set()

            # 更新进度记忆
            state.preference_progress = {
                "text": progress_text,
                "completed_ids": [],
                "active_ids": [],
            }
            if isinstance(prior_action_spans, list):
                state.preference_progress["action_spans"] = prior_action_spans
            if lifecycle:
                state.preference_progress["preference_statuses"] = lifecycle
            if isinstance(missing_information, dict) and missing_information:
                state.preference_progress["missing_information"] = missing_information

            lifecycle_by_id = {item["id"]: item for item in lifecycle}
            if lifecycle_by_id:
                for inst in instructions:
                    inst_id = str(inst.get("id", ""))
                    status = str(lifecycle_by_id.get(inst_id, {}).get("status", "unknown"))
                    if status == "satisfied_hide":
                        if self._must_keep_monitoring(inst):
                            inst.pop("completed", None)
                            kept_active_ids.add(inst_id)
                        else:
                            inst["completed"] = True
                            hidden_completed_ids.add(inst_id)
                        reported_completed_ids.add(inst_id)
                    elif status == "satisfied_keep_monitoring":
                        inst.pop("completed", None)
                        kept_active_ids.add(inst_id)
                        reported_completed_ids.add(inst_id)
                    elif status == "active":
                        inst.pop("completed", None)
                        active_ids.add(inst_id)
                    else:
                        inst.pop("completed", None)
                        active_ids.add(inst_id)
                        unknown_ids.add(inst_id)
            else:
                # 兼容旧格式：没有生命周期时只隐藏显式 completed 且非持续的老式任务。
                for inst in instructions:
                    inst_id = str(inst.get("id", ""))
                    if inst_id not in reported_completed_ids:
                        continue
                    if is_persistent_preference_instruction(inst):
                        inst.pop("completed", None)
                        kept_active_ids.add(inst_id)
                    else:
                        inst["completed"] = True
                        hidden_completed_ids.add(inst_id)

            if kept_active_ids:
                active_ids.update(kept_active_ids)
                state.preference_progress["kept_active_ids"] = sorted(kept_active_ids)
            if hidden_completed_ids:
                state.preference_progress["hidden_completed_ids"] = sorted(hidden_completed_ids)
            if unknown_ids:
                state.preference_progress["unknown_ids"] = sorted(unknown_ids)
            state.preference_progress["completed_ids"] = sorted(reported_completed_ids)
            state.preference_progress["active_ids"] = sorted(active_ids - hidden_completed_ids)
            self._sync_order_quota_progress(state, instructions)
            self._sync_target_cargo_progress(state, instructions)
            self._sync_progress_id_sets(state.preference_progress, instructions)
            self._sync_schedule_progress(state, instructions)
            self._strip_preference_status_reasons(state.preference_progress)
            state.preference_progress["last_llm_update_minute"] = ctx.simulation_minute
            state.preference_progress["last_llm_update_action"] = getattr(decision, "action", "")

            # 持久化
            long = self._store.long_memory(state.driver_id)
            long.preference_progress = state.preference_progress
            self._store.save_preference(state.driver_id, "preference_progress", state.preference_progress)

            self._telemetry.emit(
                trace, event="PREFERENCE_STATE_UPDATED", source="PreferenceStateMachine",
                phase=self.phase, simulation_minute=ctx.simulation_minute,
                payload={
                    "reported_completed_count": len(reported_completed_ids),
                    "hidden_completed_count": len(hidden_completed_ids),
                    "active_count": len(active_ids),
                    "lifecycle_count": len(lifecycle),
                },
            )
        else:
            # LLM 失败时的 fallback：用简单规则更新
            self._fallback_update(state, instructions, decision)

        state.phase = self.phase
        self._store.checkpoint(state, "CKPT_PREF_STATE_READY")
        return state

    def _should_call_llm_update(self, state: DecisionState, instructions: list[dict], decision: Any) -> tuple[bool, str]:
        ctx = state.driver_context
        if ctx is None:
            return False, "missing_context"
        progress = state.preference_progress if isinstance(state.preference_progress, dict) else {}
        if not progress:
            return True, "first_progress_update"
        action = str(getattr(decision, "action", "") or "")
        if action == "take_order":
            return True, "take_order"
        if self._decision_has_preference_support(decision):
            return True, "preference_support_action"
        if self._decision_crosses_day_or_month(ctx.simulation_minute, decision):
            return True, "crosses_calendar_boundary"
        if progress.get("missing_information"):
            return True, "missing_information_recheck"
        if progress.get("unknown_ids"):
            return True, "uncertain_preference"
        statuses = progress.get("preference_statuses")
        if isinstance(statuses, list):
            for item in statuses:
                if isinstance(item, dict) and str(item.get("status", "")) == "unknown":
                    return True, "uncertain_preference"
        try:
            last_llm = int(progress.get("last_llm_update_minute"))
        except (TypeError, ValueError):
            try:
                last_local = int(progress.get("last_local_update_minute"))
            except (TypeError, ValueError):
                return True, "missing_llm_update_marker"
            if int(ctx.simulation_minute) - last_local < self._LLM_REFRESH_INTERVAL_MINUTES:
                return False, "recent_local_update"
            return True, "missing_llm_update_marker"
        if int(ctx.simulation_minute) - last_llm >= self._LLM_REFRESH_INTERVAL_MINUTES:
            return True, "periodic_refresh"
        return False, "low_information_change"

    @staticmethod
    def _decision_has_preference_support(decision: Any) -> bool:
        meta = getattr(decision, "meta", {})
        if not isinstance(meta, dict):
            return False
        if meta.get("preference_generated"):
            return True
        future = meta.get("future_feasibility")
        return isinstance(future, dict) and bool(future.get("preferred"))

    def _decision_crosses_day_or_month(self, start_minute: int, decision: Any) -> bool:
        finish = getattr(decision, "finish_minute", None)
        if finish is None:
            try:
                finish = int(start_minute) + int(getattr(decision, "duration_minutes", 0) or 0)
            except (TypeError, ValueError):
                finish = start_minute
        try:
            start = int(start_minute)
            end = max(start, int(finish))
        except (TypeError, ValueError):
            return False
        if end == start:
            return False
        last_active_minute = max(start, end - 1)
        if start // 1440 != last_active_minute // 1440:
            return True
        start_wall = self._sim_to_wall(start)
        end_wall = self._sim_to_wall(last_active_minute)
        return bool(start_wall and end_wall and start_wall[:7] != end_wall[:7])

    def _local_progress_update(self, state: DecisionState, reason: str) -> None:
        ctx = state.driver_context
        progress = dict(state.preference_progress or {})
        progress.pop("failed_ids", None)
        statuses = progress.get("preference_statuses")
        if isinstance(statuses, list):
            progress["preference_statuses"] = [
                {
                    **{k: v for k, v in item.items() if k != "reason"},
                    "status": "active" if str(item.get("status", "")) == "failed" else item.get("status"),
                }
                for item in statuses
                if isinstance(item, dict)
            ]
        progress.setdefault("text", "Progress carried forward without LLM refresh.")
        progress["last_local_update_reason"] = reason
        if ctx is not None:
            progress["last_local_update_minute"] = ctx.simulation_minute
        progress["local_update_count"] = int(progress.get("local_update_count", 0) or 0) + 1
        state.preference_progress = progress
        self._sync_order_quota_progress(state, state.preference_instructions.get("instructions", []))
        self._sync_target_cargo_progress(state, state.preference_instructions.get("instructions", []))
        self._sync_progress_id_sets(state.preference_progress, state.preference_instructions.get("instructions", []))
        self._sync_schedule_progress(state, state.preference_instructions.get("instructions", []))
        self._strip_preference_status_reasons(state.preference_progress)
        progress = state.preference_progress
        state.preference_progress = progress
        long = self._store.long_memory(state.driver_id)
        long.preference_progress = progress
        self._store.save_preference(state.driver_id, "preference_progress", progress)

    @staticmethod
    def _as_str_list(value: Any) -> list[str]:
        if isinstance(value, str):
            return [value] if value.strip() else []
        if isinstance(value, list):
            return [str(item) for item in value if str(item).strip()]
        return []

    @classmethod
    def _normalize_missing_information(cls, value: Any) -> dict[str, list[str]]:
        if not isinstance(value, dict):
            return {}
        normalized: dict[str, list[str]] = {}
        for key, raw_items in value.items():
            items = cls._as_str_list(raw_items)
            if items:
                normalized[str(key)] = items
        return normalized

    @classmethod
    def _normalize_lifecycle(cls, value: Any, instructions: list[dict]) -> list[dict[str, str]]:
        if not isinstance(value, list):
            return []
        valid_ids = {str(inst.get("id", "")) for inst in instructions if isinstance(inst, dict)}
        instruction_by_id = {str(inst.get("id", "")): inst for inst in instructions if isinstance(inst, dict)}
        allowed = {"active", "satisfied_keep_monitoring", "satisfied_hide", "unknown"}
        out: list[dict[str, str]] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            inst_id = str(item.get("id", "")).strip()
            if not inst_id or inst_id not in valid_ids:
                continue
            status = str(item.get("status", "unknown") or "unknown").strip()
            if status == "failed":
                status = "active"
            if status not in allowed:
                status = "unknown"
            if status == "satisfied_hide" and cls._must_keep_monitoring(instruction_by_id.get(inst_id, {})):
                status = "satisfied_keep_monitoring"
            out.append({
                "id": inst_id,
                "status": status,
                "evidence": str(item.get("evidence", "") or ""),
            })
        return out

    @staticmethod
    def _strip_preference_status_reasons(progress: dict[str, Any]) -> None:
        statuses = progress.get("preference_statuses")
        if not isinstance(statuses, list):
            return
        cleaned: list[dict[str, Any]] = []
        for item in statuses:
            if not isinstance(item, dict):
                continue
            next_item = dict(item)
            next_item.pop("reason", None)
            cleaned.append(next_item)
        progress["preference_statuses"] = cleaned

    @classmethod
    def _sync_progress_id_sets(cls, progress: dict[str, Any], instructions: list[dict]) -> None:
        completed_ids: set[str] = set()
        active_ids: set[str] = set()
        hidden_completed_ids: set[str] = set()
        kept_active_ids: set[str] = set()
        unknown_ids: set[str] = set()

        statuses = progress.get("preference_statuses")
        status_by_id: dict[str, str] = {}
        if isinstance(statuses, list):
            for item in statuses:
                if not isinstance(item, dict):
                    continue
                inst_id = str(item.get("id", "") or "").strip()
                status = str(item.get("status", "unknown") or "unknown").strip()
                if inst_id:
                    status_by_id[inst_id] = "active" if status == "failed" else status

        if status_by_id:
            for inst in instructions:
                if not isinstance(inst, dict):
                    continue
                inst_id = str(inst.get("id", "") or "").strip()
                if not inst_id:
                    continue
                status = status_by_id.get(inst_id, "unknown")
                if status == "satisfied_hide":
                    if cls._must_keep_monitoring(inst):
                        inst.pop("completed", None)
                        completed_ids.add(inst_id)
                        active_ids.add(inst_id)
                        kept_active_ids.add(inst_id)
                    else:
                        inst["completed"] = True
                        completed_ids.add(inst_id)
                        hidden_completed_ids.add(inst_id)
                elif status == "satisfied_keep_monitoring":
                    inst.pop("completed", None)
                    completed_ids.add(inst_id)
                    active_ids.add(inst_id)
                    kept_active_ids.add(inst_id)
                elif status == "active":
                    inst.pop("completed", None)
                    active_ids.add(inst_id)
                else:
                    inst.pop("completed", None)
                    active_ids.add(inst_id)
                    unknown_ids.add(inst_id)
        else:
            for inst in instructions:
                if not isinstance(inst, dict):
                    continue
                inst_id = str(inst.get("id", "") or "").strip()
                if not inst_id:
                    continue
                if inst.get("completed") and not is_persistent_preference_instruction(inst):
                    completed_ids.add(inst_id)
                    hidden_completed_ids.add(inst_id)
                else:
                    inst.pop("completed", None)
                    active_ids.add(inst_id)

        progress["completed_ids"] = sorted(completed_ids)
        progress["active_ids"] = sorted(active_ids - hidden_completed_ids)
        for key, values in (
            ("hidden_completed_ids", hidden_completed_ids),
            ("kept_active_ids", kept_active_ids),
            ("unknown_ids", unknown_ids),
        ):
            if values:
                progress[key] = sorted(values)
            else:
                progress.pop(key, None)

    def _sync_schedule_progress(self, state: DecisionState, instructions: list[dict]) -> None:
        if not isinstance(state.preference_progress, dict):
            state.preference_progress = {}
        facts = self._known_action_spans(state)
        progress_by_id: dict[str, Any] = {}
        for inst in instructions:
            if not isinstance(inst, dict):
                continue
            pref_id = str(inst.get("id") or "")
            task = inst.get("schedule_task")
            if not isinstance(task, dict):
                continue
            task = self._normalize_schedule_task(task)
            steps = task.get("steps")
            if not isinstance(steps, list):
                continue
            active_dates = self._schedule_str_list(task.get("active_dates"))
            window_start, _ = self._schedule_task_window(task)
            active_date = "" if window_start is not None or len(active_dates) > 1 else str(task.get("active_date") or (active_dates[0] if active_dates else ""))
            completed: list[str] = []
            wait_by_step: dict[str, int] = {}
            current_step: str | None = None
            cursor_minute = int(window_start or 0)
            for step in steps:
                if not isinstance(step, dict):
                    continue
                step_id = str(step.get("step_id") or "")
                target = step.get("target") if isinstance(step.get("target"), dict) else step.get("location")
                if not step_id or not isinstance(target, dict):
                    continue
                kind = str(step.get("kind") or "")
                reached, reached_minute = self._schedule_target_reached_after(facts, target, active_date, cursor_minute)
                min_stay = self._safe_int(step.get("min_stay_minutes"), 0)
                wait_minutes, wait_complete_minute = self._schedule_wait_minutes_after(
                    facts,
                    target,
                    active_date,
                    cursor_minute,
                    min_stay,
                )
                wait_by_step[step_id] = wait_minutes
                hold_until = step.get("hold_until")
                requires_hold = self._schedule_step_requires_hold_until(step)
                finish_satisfied, finish_minute = (
                    (True, None)
                    if not requires_hold
                    else self._schedule_wait_covers_finish_after(facts, target, hold_until, active_date, cursor_minute)
                )
                if kind == "stay":
                    if reached and finish_satisfied and (not min_stay or wait_minutes >= min_stay):
                        completed.append(step_id)
                        cursor_candidates = [item for item in (wait_complete_minute, finish_minute, reached_minute) if item is not None]
                        if cursor_candidates:
                            cursor_minute = max(cursor_minute, max(cursor_candidates))
                        continue
                elif reached:
                    completed.append(step_id)
                    if reached_minute is not None:
                        cursor_minute = max(cursor_minute, reached_minute)
                    continue
                if current_step is None:
                    current_step = step_id
            status = "completed" if len(completed) >= len([s for s in steps if isinstance(s, dict)]) else "active"
            entry = {
                "status": status,
                "completed_step_ids": completed,
                "current_step_id": current_step,
                "wait_minutes_by_step": wait_by_step,
            }
            if current_step:
                entry["current_step_wait_minutes"] = int(wait_by_step.get(current_step, 0) or 0)
            progress_by_id[pref_id] = {k: v for k, v in entry.items() if v not in (None, "", [])}
        if progress_by_id:
            state.preference_progress["schedule_progress"] = progress_by_id
        else:
            state.preference_progress.pop("schedule_progress", None)

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
                try:
                    radius = float(normalized.get("radius_km"))
                except (TypeError, ValueError):
                    radius = 1.0
                normalized["radius_km"] = min(radius, 1.0) if radius > 0 else 1.0
                step[key] = normalized
            if step.get("finish_before") and not step.get("complete_before") and not step.get("hold_until"):
                if PreferenceStateMachine._is_full_datetime_value(step.get("finish_before")):
                    step["hold_until"] = step.get("finish_before")
                else:
                    step["complete_before"] = step.get("finish_before")
            steps.append(step)
        out["steps"] = steps
        return out

    def _sync_order_quota_progress(self, state: DecisionState, instructions: list[dict]) -> None:
        if not isinstance(state.preference_progress, dict):
            state.preference_progress = {}
        raw_spans = state.preference_progress.get("action_spans", [])
        spans = [item for item in raw_spans if isinstance(item, dict)] if isinstance(raw_spans, list) else []
        progress_by_id: dict[str, Any] = {}
        status_by_id: dict[str, dict[str, Any]] = {}
        existing_statuses = state.preference_progress.get("preference_statuses")
        if isinstance(existing_statuses, list):
            for item in existing_statuses:
                if isinstance(item, dict) and str(item.get("id") or "").strip():
                    status_by_id[str(item.get("id"))] = dict(item)

        for inst in instructions:
            if not isinstance(inst, dict):
                continue
            pref_id = str(inst.get("id") or "").strip()
            if not pref_id:
                continue
            scheme = inst.get("scheme") if isinstance(inst.get("scheme"), dict) else {}
            pref_type = str(inst.get("preference_type") or scheme.get("type") or "")
            if pref_type != "ORDER_QUOTA":
                continue
            required = _order_quota_required(scheme)
            if not required:
                continue
            matched_days = sorted(_order_quota_matching_days(spans, scheme))
            completed = len(matched_days)
            remaining = max(0, int(required) - completed)
            completed_dates = [_sim_day_to_date(day) for day in matched_days]
            status = "satisfied_hide" if remaining == 0 else "active"
            progress_by_id[pref_id] = {
                "status": "completed" if remaining == 0 else "active",
                "required_days": int(required),
                "completed_days": completed_dates,
                "completed_count": completed,
                "remaining_days": remaining,
            }
            status_item = status_by_id.get(pref_id, {"id": pref_id})
            status_item.update({
                "status": status,
                "reason": f"确定性统计：匹配订单自然日 {completed}/{int(required)}。",
                "evidence": "仅统计已执行且被接受的 take_order，并按 scheme 的订单条件命中后按自然日去重。",
            })
            status_by_id[pref_id] = status_item

        if progress_by_id:
            state.preference_progress["order_quota_progress"] = progress_by_id
            ordered_statuses: list[dict[str, Any]] = []
            seen: set[str] = set()
            if isinstance(existing_statuses, list):
                for item in existing_statuses:
                    if not isinstance(item, dict):
                        continue
                    inst_id = str(item.get("id") or "").strip()
                    if inst_id and inst_id in status_by_id:
                        ordered_statuses.append(status_by_id[inst_id])
                        seen.add(inst_id)
                    else:
                        ordered_statuses.append(item)
            for inst_id, item in status_by_id.items():
                if inst_id not in seen:
                    ordered_statuses.append(item)
            state.preference_progress["preference_statuses"] = ordered_statuses
        else:
            state.preference_progress.pop("order_quota_progress", None)

    def _sync_target_cargo_progress(self, state: DecisionState, instructions: list[dict]) -> None:
        if not isinstance(state.preference_progress, dict):
            state.preference_progress = {}
        facts = self._known_action_spans(state)
        progress_by_id: dict[str, Any] = {}
        status_by_id: dict[str, dict[str, Any]] = {}
        existing_statuses = state.preference_progress.get("preference_statuses")
        if isinstance(existing_statuses, list):
            for item in existing_statuses:
                if isinstance(item, dict) and str(item.get("id") or "").strip():
                    status_by_id[str(item.get("id"))] = dict(item)

        for inst in instructions:
            if not isinstance(inst, dict):
                continue
            pref_id = str(inst.get("id") or "").strip()
            if not pref_id:
                continue
            scheme = inst.get("scheme") if isinstance(inst.get("scheme"), dict) else {}
            pref_type = str(inst.get("preference_type") or scheme.get("type") or "")
            if pref_type != "TARGET_CARGO_MUST_TAKE":
                continue
            constraint = scheme.get("constraint") if isinstance(scheme.get("constraint"), dict) else {}
            cargo_id = str(constraint.get("target_cargo_id") or "").strip()
            if not cargo_id:
                continue
            completed_fact = next(
                (
                    fact for fact in facts
                    if isinstance(fact, dict)
                    and str(fact.get("action") or "") == "take_order"
                    and str(fact.get("cargo_id") or "").strip() == cargo_id
                ),
                None,
            )
            completed = completed_fact is not None
            status = "satisfied_hide" if completed else "active"
            progress_by_id[pref_id] = {
                "status": "completed" if completed else "active",
                "target_cargo_id": cargo_id,
                "completed": completed,
                "evidence_time": completed_fact.get("start_time") if completed_fact else None,
            }
            status_item = status_by_id.get(pref_id, {"id": pref_id})
            status_item.update({
                "status": status,
                "reason": (
                    f"确定性统计：已接指定熟货源 {cargo_id}。"
                    if completed else
                    f"确定性统计：尚未发现已执行 take_order {cargo_id}。"
                ),
                "evidence": "按历史动作 cargo_id 精确匹配，不由 LLM 判断。",
            })
            status_by_id[pref_id] = status_item

        if progress_by_id:
            state.preference_progress["target_cargo_progress"] = progress_by_id
            ordered_statuses: list[dict[str, Any]] = []
            seen: set[str] = set()
            if isinstance(existing_statuses, list):
                for item in existing_statuses:
                    if not isinstance(item, dict):
                        continue
                    inst_id = str(item.get("id") or "").strip()
                    if inst_id and inst_id in status_by_id:
                        ordered_statuses.append(status_by_id[inst_id])
                        seen.add(inst_id)
                    else:
                        ordered_statuses.append(item)
            for inst_id, item in status_by_id.items():
                if inst_id not in seen:
                    ordered_statuses.append(item)
            state.preference_progress["preference_statuses"] = ordered_statuses
        else:
            state.preference_progress.pop("target_cargo_progress", None)

    @classmethod
    def _schedule_target_reached(cls, facts: list[dict[str, Any]], target: dict[str, Any], active_date: str = "") -> bool:
        for fact in facts:
            if active_date and not cls._fact_touches_date(fact, active_date):
                continue
            for key in ("end_position", "start_position"):
                point = fact.get(key)
                if cls._point_near_target(point, target):
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
            for key, minute in (("start_position", start), ("end_position", end)):
                point = fact.get(key)
                if cls._point_near_target(point, target):
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
            if not cls._point_near_target(point, target):
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
            if not cls._point_near_target(point, target):
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
            if not cls._point_near_target(point, target):
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
            if not cls._point_near_target(point, target):
                continue
            start = cls._safe_int(fact.get("start_minute"), -1)
            end = cls._safe_int(fact.get("end_minute"), -1)
            if start <= finish_minute <= end:
                return True, finish_minute
        return False, None

    @staticmethod
    def _schedule_str_list(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, (list, tuple, set)):
            return [str(item).strip() for item in value if str(item).strip()]
        text = str(value).strip()
        return [text] if text else []

    @classmethod
    def _schedule_step_requires_hold_until(cls, step: dict[str, Any]) -> bool:
        return cls._wall_to_sim_minute(step.get("hold_until")) is not None

    @staticmethod
    def _is_full_datetime_value(value: Any) -> bool:
        text = str(value or "").strip().replace("T", " ")
        return len(text) >= 16 and text[4:5] == "-" and text[7:8] == "-" and ":" in text[11:16]

    @staticmethod
    def _fact_touches_date(fact: dict[str, Any], active_date: str) -> bool:
        start = str(fact.get("start_time") or "")
        end = str(fact.get("end_time") or "")
        return start.startswith(active_date) or end.startswith(active_date)

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
    def _point_near_target(point: Any, target: dict[str, Any]) -> bool:
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

    @staticmethod
    def _safe_int(value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

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

    @staticmethod
    def _must_keep_monitoring(instruction: dict[str, Any]) -> bool:
        pref_type = str(instruction.get("preference_type") or "")
        scheme = instruction.get("scheme") if isinstance(instruction.get("scheme"), dict) else {}
        pref_type = pref_type or str(scheme.get("type") or "")
        cycle = instruction.get("cycle") if isinstance(instruction.get("cycle"), dict) else {}
        scheme_scope = scheme.get("scope") if isinstance(scheme.get("scope"), dict) else {}
        if str(cycle.get("length") or scheme_scope.get("period") or "").lower() in {"day", "daily"}:
            return True
        return pref_type in {
            "ACTION_FORBID",
            "NUMERIC_LIMIT",
            "TIME_WINDOW_STATIONARY",
            "DAILY_CONTINUOUS_REST",
            "GEOFENCE_STAY_WITHIN",
            "GEOFENCE_FORBIDDEN_AREA",
            "MONTHLY_DEADHEAD_LIMIT",
            "DAILY_FIRST_ORDER_DEADLINE",
        }

    @staticmethod
    def _apply_deterministic_lifecycle_overrides(
        state: DecisionState,
        lifecycle: list[dict[str, str]],
        instructions: list[dict],
    ) -> list[dict[str, str]]:
        if not lifecycle:
            return lifecycle
        facts = _ProgressFacts.from_state(state)
        raw_spans = state.preference_progress.get("action_spans", []) if isinstance(state.preference_progress, dict) else []
        spans = [item for item in raw_spans if isinstance(item, dict)] if isinstance(raw_spans, list) else []
        by_id = {str(item.get("id") or ""): dict(item) for item in lifecycle if isinstance(item, dict)}
        for inst in instructions:
            if not isinstance(inst, dict):
                continue
            inst_id = str(inst.get("id") or "")
            if not inst_id or inst_id not in by_id:
                continue
            scheme = inst.get("scheme") if isinstance(inst.get("scheme"), dict) else {}
            pref_type = str(inst.get("preference_type") or scheme.get("type") or "")
            if pref_type == "OFF_DAY_QUOTA":
                required = _int(_nested_dict(scheme, "constraint", "min_days")) or _int(_nested_dict(scheme, "completion", "count_min"))
                if not required:
                    continue
                completed = len(facts.completed_off_days)
                if completed >= required:
                    by_id[inst_id].update({
                        "status": "satisfied_hide",
                        "reason": f"确定性统计：完整静止日 {completed}/{required} 已达标。",
                        "evidence": "action_spans 覆盖完整自然日且无 take_order/reposition。",
                    })
                elif by_id[inst_id].get("status") == "satisfied_hide":
                    by_id[inst_id].update({
                        "status": "active",
                        "reason": f"确定性统计：完整静止日仅 {completed}/{required}，不能隐藏。",
                        "evidence": "只有覆盖完整24小时且无 active 动作的自然日才计入。",
                    })
            elif pref_type == "LOCATION_STAY_ON_DATE":
                date_min = _scheme_date_minute(scheme)
                target = _target_location(scheme)
                required = _int(_nested_dict(scheme, "constraint", "min_stay_minutes")) or 120
                if date_min is None or not target:
                    continue
                day = date_min // 1440
                waited = facts.waited_near(day, target)
                if waited >= required:
                    by_id[inst_id].update({
                        "status": "satisfied_hide",
                        "reason": f"确定性统计：目标附近停留 {waited}/{required} 分钟，已达标。",
                        "evidence": "仅统计目标半径内的 wait 静止时间。",
                    })
                elif by_id[inst_id].get("status") == "satisfied_hide":
                    by_id[inst_id].update({
                        "status": "active",
                        "reason": f"确定性统计：目标附近停留仅 {waited}/{required} 分钟，不能隐藏。",
                        "evidence": "到达目标点不等于已完成停留；必须在目标附近 wait 满要求时长。",
                    })
            elif pref_type == "ORDER_QUOTA":
                required = _order_quota_required(scheme)
                if not required:
                    continue
                matched_days = _order_quota_matching_days(spans, scheme)
                completed = len(matched_days)
                if completed >= required:
                    by_id[inst_id].update({
                        "status": "satisfied_hide",
                        "reason": f"确定性统计：匹配订单自然日 {completed}/{required} 已达标。",
                        "evidence": "按已执行 take_order 的起终点/货物条件匹配并按自然日去重。",
                    })
                elif by_id[inst_id].get("status") == "satisfied_hide":
                    by_id[inst_id].update({
                        "status": "active",
                        "reason": f"确定性统计：匹配订单自然日仅 {completed}/{required}，不能隐藏。",
                        "evidence": "只有已执行且命中 scheme 条件的 take_order 才计入配额。",
                    })
        return [by_id[str(item.get("id") or "")] for item in lifecycle if str(item.get("id") or "") in by_id]

    def _known_action_spans(self, state: DecisionState) -> list[dict[str, Any]]:
        facts: list[dict[str, Any]] = []
        raw_spans = state.preference_progress.get("action_spans", [])
        if isinstance(raw_spans, list):
            for raw in raw_spans:
                if not isinstance(raw, dict):
                    continue
                facts.extend(self._query_scan_facts_from_events(raw.get("query_scan_events")))
                fact = self._action_fact_from_span(raw)
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
                "start_position": action.get("_start_position"),
                "end_position": action.get("_end_position"),
                "position_changed": action.get("_position_changed"),
                "query_scan_events": action.get("_query_scan_events"),
                "query_scan_minutes": action.get("_query_scan_minutes"),
                "pickup_deadhead_km": action.get("_pickup_deadhead_km"),
                "haul_distance_km": action.get("_haul_distance_km"),
                "start_point": action.get("_start_point"),
                "end_point": action.get("_end_point"),
            }
            params = action.get("params", {})
            if isinstance(params, dict) and params.get("cargo_id") is not None:
                raw["cargo_id"] = params.get("cargo_id")
            fact = self._action_fact_from_span(raw)
            if fact:
                facts.append(fact)
        return facts

    @classmethod
    def _action_fact_from_span(cls, raw: dict[str, Any]) -> dict[str, Any] | None:
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
        action = str(raw.get("action", "") or "")
        fact: dict[str, Any] = {
            "action": action,
            "start_minute": start,
            "end_minute": end,
            "duration_minutes": max(0, end - start),
            "start_time": cls._sim_to_wall(start),
            "end_time": cls._sim_to_wall(end),
        }
        for key in (
            "cargo_id",
            "cargo_name",
            "source",
            "items_count",
            "query_scan_minutes",
            "pickup_deadhead_km",
            "haul_distance_km",
        ):
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
        start_position = cls._normalize_point(raw.get("start_position") or raw.get("_start_position"))
        end_position = cls._normalize_point(raw.get("end_position") or raw.get("_end_position"))
        if start_position is None:
            start_position = cls._point_from_values(raw.get("lat"), raw.get("lng"))
        if end_position is None and action == "query_scan" and start_position is not None:
            end_position = dict(start_position)
        cls._attach_position_fields(fact, start_position, end_position, raw.get("position_changed"))
        for key in ("start_point", "end_point", "target_point"):
            point = cls._normalize_point(raw.get(key))
            if point is not None:
                fact[key] = point
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
            fact = cls._action_fact_from_span(event)
            if fact:
                facts.append(fact)
        return facts

    @classmethod
    def _attach_position_fields(
        cls,
        fact: dict[str, Any],
        start_position: dict[str, float] | None,
        end_position: dict[str, float] | None,
        position_changed: Any = None,
    ) -> None:
        if start_position is not None:
            fact["start_position"] = start_position
        if end_position is not None:
            fact["end_position"] = end_position
        changed: bool | None = None
        if isinstance(position_changed, bool):
            changed = position_changed
        elif start_position is not None and end_position is not None:
            changed = not cls._same_point(start_position, end_position)
        if changed is not None:
            fact["position_changed"] = changed
            fact["stationary"] = not changed

    @staticmethod
    def _normalize_point(value: Any) -> dict[str, float] | None:
        if not isinstance(value, dict):
            return None
        return PreferenceStateMachine._point_from_values(value.get("lat"), value.get("lng"))

    @staticmethod
    def _point_from_values(lat: Any, lng: Any) -> dict[str, float] | None:
        try:
            lat_f = round(float(lat), 5)
            lng_f = round(float(lng), 5)
        except (TypeError, ValueError):
            return None
        if not (-90 <= lat_f <= 90 and -180 <= lng_f <= 180):
            return None
        return {"lat": lat_f, "lng": lng_f}

    @staticmethod
    def _same_point(a: dict[str, Any], b: dict[str, Any]) -> bool:
        try:
            return (
                round(float(a.get("lat")), 5) == round(float(b.get("lat")), 5)
                and round(float(a.get("lng")), 5) == round(float(b.get("lng")), 5)
            )
        except (TypeError, ValueError):
            return False

    def _update_progress(
        self, state: DecisionState, instructions: list[dict],
        decision: Any, trace: TraceContext,
    ) -> dict[str, Any] | None:
        """调用 LLM 更新进度。"""
        ctx = state.driver_context
        if ctx is None:
            return None

        from datetime import timedelta
        from ..domain.rules import SIMULATION_EPOCH
        now_str = (SIMULATION_EPOCH + timedelta(minutes=ctx.simulation_minute)).strftime("%Y-%m-%d %H:%M")

        # 构建最新动作描述，包含货源品类、路线位置与距离，供偏好进度判断使用。
        action_desc = self._describe_action(state, decision, ctx.simulation_minute)

        # 上一轮进度
        last_progress = json.dumps(
            self._compact_progress_for_llm(state.preference_progress),
            ensure_ascii=False,
            default=str,
            separators=(",", ":"),
        )

        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": PROGRESS_SYSTEM
                    + " 休息证据按评测 step 口径：本步最终动作为 wait 时，本步 wait 前置 query_scan 与 wait 执行时间合并计入连续休息/静止；"
                    + "本步最终动作为 take_order 或 reposition 时，query_scan 不计入休息；接单内部等待不计入休息。",
                },
                {"role": "user", "content": PROGRESS_TEMPLATE.format(
                    now=now_str,
                    lat=round(ctx.lat, 5),
                    lng=round(ctx.lng, 5),
                    instructions=json.dumps(self._compact_instructions(instructions), ensure_ascii=False, default=str, separators=(",", ":")),
                    last_progress=last_progress,
                    latest_action=action_desc,
                    recent_action_facts=json.dumps(
                        self._recent_action_facts(state, decision),
                        ensure_ascii=False,
                        default=str,
                        separators=(",", ":"),
                    ),
                )},
            ],
            "temperature": 0.0,
            "max_tokens": 2000,
            "enable_thinking": False,
        }

        try:
            result = self._gateway.llm_chat_json(payload, trace, "PreferenceStateMachine")
            if isinstance(result, dict) and "progress_text" in result:
                return result
        except Exception as exc:
            logger.warning("PreferenceStateMachine LLM failed: %s", exc)

        return None

    @staticmethod
    def _compact_progress_for_llm(progress: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(progress, dict):
            return {}
        compact: dict[str, Any] = {}
        for key in (
            "hidden_completed_ids",
            "kept_active_ids",
            "unknown_ids",
            "missing_information",
            "target_cargo_progress",
            "last_llm_update_minute",
            "last_local_update_minute",
        ):
            if key in progress:
                compact[key] = progress.get(key)
        statuses = progress.get("preference_statuses")
        if isinstance(statuses, list):
            compact["preference_statuses"] = [
                {**item, "status": "active" if str(item.get("status", "")) == "failed" else item.get("status")}
                for item in statuses[-12:]
                if isinstance(item, dict)
            ]
        spans = progress.get("action_spans")
        if isinstance(spans, list):
            compact["action_span_count"] = len(spans)
        text = str(progress.get("text", "") or "")
        if text:
            compact["text"] = text[-600:]
        return compact

    @staticmethod
    def _compact_instructions(instructions: list[dict]) -> list[dict[str, Any]]:
        from .preference_utils import compact_instruction_for_llm

        compact: list[dict[str, Any]] = []
        for inst in instructions:
            if not isinstance(inst, dict):
                continue
            compact.append(compact_instruction_for_llm(inst))
        return compact

    def _recent_action_facts(self, state: DecisionState, decision: Any) -> list[dict[str, Any]]:
        ctx = state.driver_context
        if ctx is None:
            return []
        facts = self._known_action_spans(state)
        facts.extend(self._query_scan_facts_from_events(state.query_scan_events))

        now = int(ctx.simulation_minute)
        finish = getattr(decision, "finish_minute", None)
        if finish is None:
            try:
                finish = now + int(getattr(decision, "duration_minutes", 0) or 0)
            except (TypeError, ValueError):
                finish = now
        try:
            finish_int = max(now, int(finish))
        except (TypeError, ValueError):
            finish_int = now
        projected = self._action_fact_from_span({
            "action": str(getattr(decision, "action", "") or ""),
            "start": now,
            "end": finish_int,
            "duration_minutes": finish_int - now,
            "query_scan_events": state.query_scan_events,
            "query_scan_minutes": state.query_scan_minutes,
        }) or {
            "action": str(getattr(decision, "action", "") or ""),
            "start_minute": now,
            "end_minute": finish_int,
            "duration_minutes": finish_int - now,
            "start_time": self._sim_to_wall(now),
            "end_time": self._sim_to_wall(finish_int),
        }
        projected["projected"] = True
        start_position = self._point_from_values(ctx.lat, ctx.lng)
        end_position = self._projected_decision_end_position(state, decision, start_position)
        self._attach_position_fields(projected, start_position, end_position)
        facts.append(projected)
        facts = self._dedupe_action_facts(facts)
        facts.sort(key=lambda item: int(item.get("start_minute", 0) or 0))
        return self._summarize_action_facts(facts) + facts[-self._RECENT_FACT_LIMIT:]

    @staticmethod
    def _projected_decision_end_position(
        state: DecisionState,
        decision: Any,
        start_position: dict[str, float] | None,
    ) -> dict[str, float] | None:
        action = str(getattr(decision, "action", "") or "")
        if action == "wait":
            return dict(start_position) if start_position else None
        if action == "reposition":
            params = getattr(decision, "params", {})
            if isinstance(params, dict):
                point = PreferenceStateMachine._point_from_values(
                    params.get("latitude", params.get("lat")),
                    params.get("longitude", params.get("lng")),
                )
                if point:
                    return point
        point = PreferenceStateMachine._point_from_values(
            getattr(decision, "target_lat", None),
            getattr(decision, "target_lng", None),
        )
        if point:
            return point
        meta = getattr(decision, "meta", {})
        if isinstance(meta, dict):
            for key in ("target_point", "end_point"):
                point = PreferenceStateMachine._normalize_point(meta.get(key))
                if point:
                    return point
        return dict(start_position) if start_position else None

    @staticmethod
    def _dedupe_action_facts(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        deduped: list[dict[str, Any]] = []
        seen: set[tuple[Any, ...]] = set()
        for fact in facts:
            key = (
                fact.get("action"),
                fact.get("start_minute"),
                fact.get("end_minute"),
                fact.get("cargo_id"),
                fact.get("source"),
                bool(fact.get("projected")),
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(fact)
        return deduped

    @classmethod
    def _summarize_action_facts(cls, facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not facts:
            return []
        counts: dict[str, int] = {}
        duration_by_action: dict[str, int] = {}
        stationary_duration_by_action: dict[str, int] = {}
        calendar_days: dict[str, dict[str, Any]] = {}
        projected_count = 0
        for item in facts:
            action = str(item.get("action", "") or "unknown")
            counts[action] = counts.get(action, 0) + 1
            duration = cls._fact_duration(item)
            duration_by_action[action] = duration_by_action.get(action, 0) + duration
            if item.get("projected"):
                projected_count += 1
            if item.get("stationary") is True:
                stationary_duration_by_action[action] = stationary_duration_by_action.get(action, 0) + duration
            cls._add_calendar_segments(calendar_days, item, action, duration)
        return [{
            "kind": "summary",
            "known_fact_count": len(facts),
            "projected_fact_count": projected_count,
            "action_counts": counts,
            "duration_minutes_by_action": duration_by_action,
            "stationary_minutes_by_action": stationary_duration_by_action,
            "calendar_days": [calendar_days[key] for key in sorted(calendar_days)][-45:],
            "first_time": facts[0].get("start_time"),
            "last_time": facts[-1].get("end_time"),
        }]

    @staticmethod
    def _fact_duration(item: dict[str, Any]) -> int:
        try:
            start = int(item.get("start_minute"))
            end = int(item.get("end_minute"))
            return max(0, end - start)
        except (TypeError, ValueError):
            try:
                return max(0, int(item.get("duration_minutes", 0) or 0))
            except (TypeError, ValueError):
                return 0

    @classmethod
    def _add_calendar_segments(
        cls,
        calendar_days: dict[str, dict[str, Any]],
        item: dict[str, Any],
        action: str,
        duration: int,
    ) -> None:
        if duration <= 0:
            return
        try:
            start = int(item.get("start_minute"))
            end = int(item.get("end_minute"))
        except (TypeError, ValueError):
            return
        end = max(start, end)
        cur = start
        while cur < end:
            day_end = ((cur // 1440) + 1) * 1440
            seg_end = min(end, day_end)
            seg_duration = max(0, seg_end - cur)
            wall = cls._sim_to_wall(cur)
            date = wall[:10] if wall else str(cur // 1440)
            entry = calendar_days.setdefault(date, {
                "date": date,
                "duration_by_action": {},
                "stationary_minutes": 0,
                "moving_minutes": 0,
                "position_unknown_minutes": 0,
            })
            durations = entry["duration_by_action"]
            durations[action] = int(durations.get(action, 0) or 0) + seg_duration
            if item.get("stationary") is True:
                entry["stationary_minutes"] += seg_duration
            elif item.get("stationary") is False:
                entry["moving_minutes"] += seg_duration
            else:
                entry["position_unknown_minutes"] += seg_duration
            cur = seg_end

    def _describe_action(self, state: DecisionState, decision: Any, sim_minute: int) -> str:
        """描述最新决策动作。"""
        wall = self._sim_to_wall(sim_minute)

        if decision.action == "take_order":
            return self._describe_take_order(state, decision, wall)
        if decision.action == "wait":
            end_wall = self._sim_to_wall(sim_minute + int(decision.duration_minutes or 0))
            payload = {
                "wall_time": wall,
                "action": "wait",
                "duration_minutes": int(decision.duration_minutes or 0),
                "end_time": end_wall,
            }
            return json.dumps(payload, ensure_ascii=False, default=str, separators=(",", ":"))
        if decision.action == "reposition":
            lat = decision.params.get("latitude", decision.params.get("lat", "?"))
            lng = decision.params.get("longitude", decision.params.get("lng", "?"))
            payload = {
                "wall_time": wall,
                "action": "reposition",
                "target": {"lat": lat, "lng": lng},
                "distance_km": self._round_or_none(decision.meta.get("distance_km")),
                "duration_minutes": int(decision.duration_minutes or 0),
                "finish_time": self._sim_to_wall(decision.finish_minute),
            }
            return json.dumps(payload, ensure_ascii=False, default=str, separators=(",", ":"))
        return json.dumps({"wall_time": wall, "action": decision.action}, ensure_ascii=False, separators=(",", ":"))

    def _describe_take_order(self, state: DecisionState, decision: Any, wall: str) -> str:
        cargo = self._cargo_by_id(state, decision.cargo_id)
        meta = decision.meta if isinstance(getattr(decision, "meta", None), dict) else {}
        start = self._point_from_cargo_or_meta(cargo, meta, "start", "start_point")
        end = self._point_from_cargo_or_meta(cargo, meta, "end", "end_point")
        pickup_km = self._round_or_none(meta.get("pickup_km"))
        haul_km = self._round_or_none(meta.get("haul_km"))
        total_km = None
        if pickup_km is not None and haul_km is not None:
            total_km = round(pickup_km + haul_km, 2)
        cargo_name = None
        if isinstance(cargo, dict):
            cargo_name = cargo.get("cargo_name")
        cargo_name = cargo_name or meta.get("cargo_name")

        pickup_minutes = self._int_or_none(meta.get("pickup_minutes"))
        wait_for_load = self._int_or_none(meta.get("wait_for_load"))
        t0 = state.driver_context.simulation_minute if state.driver_context else None
        pickup_arrival = t0 + pickup_minutes if t0 is not None and pickup_minutes is not None else None
        loading_done = pickup_arrival + wait_for_load if pickup_arrival is not None and wait_for_load is not None else None

        payload = {
            "wall_time": wall,
            "action": "take_order",
            "cargo_id": decision.cargo_id,
            "cargo_name": cargo_name,
            "start": start,
            "end": end,
            "pickup_deadhead_km": pickup_km,
            "haul_distance_km": haul_km,
            "total_distance_km": total_km,
            "duration_minutes": int(decision.duration_minutes or 0),
            "pickup_arrival": self._sim_to_wall(pickup_arrival),
            "loading_done": self._sim_to_wall(loading_done),
            "delivery_done": self._sim_to_wall(decision.finish_minute),
            "load_window": cargo.get("load_time") if isinstance(cargo, dict) else meta.get("load_window"),
            "rest_credit_minutes": 0,
            "wait_for_load_counts_as_rest": False,
            "rest_counting_note": "take_order 动作不提供休息额度；接单内部等待和本步 query_scan 都不计入休息。",
            "net_income": self._round_or_none(getattr(decision, "net_income", None)),
        }
        return json.dumps(payload, ensure_ascii=False, default=str, separators=(",", ":"))

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
    def _point_from_cargo_or_meta(cargo: dict[str, Any] | None, meta: dict[str, Any], cargo_key: str, meta_key: str) -> dict[str, Any] | None:
        point = cargo.get(cargo_key) if isinstance(cargo, dict) else None
        if not isinstance(point, dict):
            point = meta.get(meta_key)
        if not isinstance(point, dict):
            return None
        out: dict[str, Any] = {}
        if point.get("city") not in (None, ""):
            out["city"] = point.get("city")
        if point.get("lat") is not None:
            out["lat"] = point.get("lat")
        if point.get("lng") is not None:
            out["lng"] = point.get("lng")
        return out or None

    @staticmethod
    def _round_or_none(value: Any, digits: int = 2) -> float | None:
        try:
            return round(float(value), digits)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _int_or_none(value: Any) -> int | None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _sim_to_wall(sim_minute: int | None) -> str | None:
        if sim_minute is None:
            return None
        from datetime import timedelta
        from ..domain.rules import SIMULATION_EPOCH
        return (SIMULATION_EPOCH + timedelta(minutes=int(sim_minute))).strftime("%Y-%m-%d %H:%M")

    def _fallback_update(self, state: DecisionState, instructions: list[dict], decision: Any) -> None:
        """LLM 失败时的简单规则更新。"""
        prior_action_spans = state.preference_progress.get("action_spans", [])
        lines = []

        for inst in instructions:
            inst_id = inst.get("id", "")
            rule = inst.get("rule", "")
            ongoing = inst.get("ongoing", True)

            if inst.get("completed"):
                lines.append(f"{inst_id}: 已完成")
                continue

            if ongoing:
                lines.append(f"{inst_id}: 持续检查中 - {rule}")
            else:
                lines.append(f"{inst_id}: 未完成 - {rule}")

        state.preference_progress = {
            "text": "\n".join(lines) if lines else "无偏好约束",
            "last_local_update_reason": "llm_update_failed",
        }
        if isinstance(prior_action_spans, list):
            state.preference_progress["action_spans"] = prior_action_spans
        if state.driver_context is not None:
            state.preference_progress["last_local_update_minute"] = state.driver_context.simulation_minute
        self._sync_order_quota_progress(state, instructions)
        self._sync_target_cargo_progress(state, instructions)
        self._sync_progress_id_sets(state.preference_progress, instructions)
        self._sync_schedule_progress(state, instructions)
        self._strip_preference_status_reasons(state.preference_progress)
        long = self._store.long_memory(state.driver_id)
        long.preference_progress = state.preference_progress
        self._store.save_preference(state.driver_id, "preference_progress", state.preference_progress)
