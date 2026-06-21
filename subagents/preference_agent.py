from __future__ import annotations

import json
import logging
import re
from typing import Any

from ..gateway import GatewayLayer
from ..messages import TraceContext
from ..state_store import DecisionState, StateStore
from ..telemetry import Telemetry
from .preference_compile import (
    attach_visibility_meta,
    build_completion_check,
    build_fixed_scheme,
    build_guard_summary,
    build_schedule_task,
    checks_to_steps,
    derive_required_fields,
    dumps_json,
    lookup_pref_record,
    lookup_visibility_record,
    merge_v2,
    normalize_tier1,
    normalize_tier2a,
    normalize_tier2b,
    resolve_parse_status,
)
from .preference_prompts import (
    TIER1_SYSTEM,
    TIER1_TEMPLATE,
    TIER2A_SYSTEM,
    TIER2A_TEMPLATE,
    TIER2B_SYSTEM,
    TIER2B_TEMPLATE,
)
from .preference_schema import PARSE_PROMPT_VERSION, SCHEMA_VERSION, validate_tier1, validate_tier2a, validate_tier2b
from .preference_utils import instruction_content_key

logger = logging.getLogger("agent.preference_agent")


class PreferenceAgent:
    """三级 LLM 偏好解析：语义(1) → 周期片(2a) → 约束片(2b)。按 content-key 增量缓存。"""

    phase = "PARSE_PREFERENCES"

    def __init__(self, store: StateStore, telemetry: Telemetry, gateway: GatewayLayer) -> None:
        self._store = store
        self._telemetry = telemetry
        self._gateway = gateway
        self.last_llm_call_count = 0

    def run(self, state: DecisionState, trace: TraceContext) -> DecisionState:
        ctx = state.driver_context
        if ctx is None:
            return state

        if not ctx.preferences_text:
            state.phase = self.phase
            return state

        long = self._store.long_memory(state.driver_id)
        current_key = self._canonical_preferences(ctx.preferences_text)
        old_key = long.preference_memory.get("preference_text_key")
        if old_key is None:
            old_raw = long.preference_memory.get("raw_preferences")
            old_key = self._canonical_preferences(old_raw if isinstance(old_raw, list) else [])

        cached_instructions = long.preference_memory.get("parsed_instructions")
        parsed_by_content = self._load_parsed_by_content(long.preference_memory, cached_instructions)
        old_assembled = list(cached_instructions) if isinstance(cached_instructions, list) else []

        if old_key == current_key and isinstance(cached_instructions, list) and cached_instructions:
            if all(self._is_v2_cached(parsed_by_content.get(key, {})) for key in current_key):
                for key in current_key:
                    item = parsed_by_content.get(key)
                    if isinstance(item, dict):
                        parsed_by_content[key] = self._finalize_instruction(
                            self._extract_assembled(item, key),
                            key,
                            0,
                        )
                instructions = self._assemble_instructions(current_key, parsed_by_content)
                instructions = self._postprocess_instructions(instructions, state, long.preference_progress)
                state.preference_instructions = {"instructions": instructions, "schemes": self._extract_schemes(instructions)}
                state.preference_progress = dict(long.preference_progress)
                state.phase = self.phase
                self.last_llm_call_count = 0
                self._telemetry.emit(
                    trace,
                    event="INSTRUCTIONS_CACHED",
                    source="PreferenceAgent",
                    phase=self.phase,
                    simulation_minute=ctx.simulation_minute,
                    payload={"instruction_count": len(instructions), "llm_call_count": 0},
                )
                return state

        self._telemetry.emit(trace, event="AGENT_STARTED", source="PreferenceAgent", phase=self.phase)
        self.last_llm_call_count = 0

        added_keys = [key for key in current_key if key not in parsed_by_content or not self._is_v2_cached(parsed_by_content.get(key, {}))]
        removed_keys = [key for key in parsed_by_content if key not in current_key]
        for key in removed_keys:
            parsed_by_content.pop(key, None)

        reused_count = 0
        for key in current_key:
            cached = parsed_by_content.get(key)
            if isinstance(cached, dict) and self._is_v2_cached(cached):
                parsed_by_content[key] = self._finalize_instruction(self._extract_assembled(cached, key), key, 0)
                reused_count += 1
                continue
            index = current_key.index(key) + 1
            pref_meta = lookup_visibility_record(ctx.preference_visibility, key) or lookup_pref_record(ctx.preferences_raw, key)
            inst = self._parse_single_preference(key, index, trace, pref_meta)
            if inst:
                parsed_by_content[key] = {
                    "tier1": inst.get("_tier1"),
                    "tier2a": {"cycle": inst.get("cycle"), "scope": inst.get("scope"), "completion": inst.get("completion")},
                    "tier2b": {"checks": inst.get("checks"), "on_fail": inst.get("on_fail"), "route_plan": inst.get("route_plan")},
                    "assembled": {k: v for k, v in inst.items() if not k.startswith("_")},
                }

        all_instructions = self._assemble_instructions(current_key, parsed_by_content)
        all_instructions = self._postprocess_instructions(all_instructions, state, long.preference_progress)
        if old_assembled and long.preference_progress:
            long.preference_progress = self._remap_preference_progress(
                old_assembled,
                all_instructions,
                dict(long.preference_progress),
                removed_keys,
            )
            self._store.save_preference(state.driver_id, "preference_progress", long.preference_progress)

        long.preference_memory["raw_preferences"] = current_key
        long.preference_memory["preference_text_key"] = current_key
        long.preference_memory["parsed_instructions"] = all_instructions
        long.preference_memory["parsed_by_content"] = parsed_by_content
        long.preference_memory["parse_prompt_version"] = PARSE_PROMPT_VERSION
        self._store.save_preference(state.driver_id, "raw_preferences", current_key)
        self._store.save_preference(state.driver_id, "preference_text_key", current_key)
        self._store.save_preference(state.driver_id, "parsed_instructions", all_instructions)
        self._store.save_preference(state.driver_id, "parsed_by_content", parsed_by_content)
        self._store.save_preference(state.driver_id, "parse_prompt_version", PARSE_PROMPT_VERSION)

        state.preference_instructions = {"instructions": all_instructions, "schemes": self._extract_schemes(all_instructions)}
        state.preference_progress = dict(long.preference_progress)
        state.phase = self.phase

        self._telemetry.emit(
            trace,
            event="INSTRUCTIONS_PARSED",
            source="PreferenceAgent",
            phase=self.phase,
            simulation_minute=ctx.simulation_minute,
            payload={
                "instruction_count": len(all_instructions),
                "preference_count": len(current_key),
                "llm_call_count": self.last_llm_call_count,
                "reused_count": reused_count,
                "added_count": len(added_keys),
                "removed_count": len(removed_keys),
                "schema_version": SCHEMA_VERSION,
            },
        )
        return state

    def _parse_single_preference(
        self,
        text: str,
        index: int,
        trace: TraceContext,
        pref_meta: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        tier1_raw = self._llm_tier1(text, index, trace)
        if tier1_raw is None:
            return self._fallback_single(text, index, pref_meta)

        tier1 = normalize_tier1(tier1_raw)
        tier1_errors = validate_tier1(tier1)
        if tier1_errors:
            logger.warning("PreferenceAgent tier1 validation pref %d: %s", index, tier1_errors)

        tier2a_raw = self._llm_tier2a(tier1, index, trace)
        if tier2a_raw is None:
            tier2a = normalize_tier2a({}, tier1.get("routing", {}))
        else:
            tier2a = normalize_tier2a(tier2a_raw, tier1.get("routing", {}))
        tier2a_errors = validate_tier2a(tier2a, tier1.get("routing"))
        if tier2a_errors:
            logger.warning("PreferenceAgent tier2a validation pref %d: %s", index, tier2a_errors)

        tier2b_raw = self._llm_tier2b(tier1, tier2a, index, trace)
        if tier2b_raw is None:
            tier2b = normalize_tier2b({})
        else:
            tier2b = normalize_tier2b(tier2b_raw)
        tier2b_errors = validate_tier2b(tier2b, tier1.get("routing"))
        if tier2b_errors:
            logger.warning("PreferenceAgent tier2b validation pref %d: %s", index, tier2b_errors)

        meta = {"parse_tier": "1+2a+2b"}
        merged = merge_v2(text, tier1, tier2a, tier2b, meta=meta)
        merged = attach_visibility_meta(merged, pref_meta)
        merged["_tier1"] = tier1
        merged["parse_status"] = resolve_parse_status(merged)
        return self._finalize_instruction(merged, text, index)

    def _llm_tier1(self, text: str, index: int, trace: TraceContext) -> dict[str, Any] | None:
        payload = {
            "messages": [
                {"role": "system", "content": TIER1_SYSTEM},
                {"role": "user", "content": TIER1_TEMPLATE.format(preference_text=text)},
            ],
            "temperature": 0.0,
            "max_tokens": 800,
            "enable_thinking": False,
        }
        return self._call_llm(payload, trace, index, "tier1", required_keys={"normalized_rule", "preference_type", "routing"})

    def _llm_tier2a(self, tier1: dict[str, Any], index: int, trace: TraceContext) -> dict[str, Any] | None:
        payload = {
            "messages": [
                {"role": "system", "content": TIER2A_SYSTEM},
                {"role": "user", "content": TIER2A_TEMPLATE.format(
                    normalized_rule=tier1.get("normalized_rule", ""),
                    routing_json=dumps_json(tier1.get("routing", {})),
                )},
            ],
            "temperature": 0.0,
            "max_tokens": 700,
            "enable_thinking": False,
        }
        return self._call_llm(payload, trace, index, "tier2a", required_keys={"cycle", "scope", "completion"})

    def _llm_tier2b(self, tier1: dict[str, Any], tier2a: dict[str, Any], index: int, trace: TraceContext) -> dict[str, Any] | None:
        payload = {
            "messages": [
                {"role": "system", "content": TIER2B_SYSTEM},
                {"role": "user", "content": TIER2B_TEMPLATE.format(
                    normalized_rule=tier1.get("normalized_rule", ""),
                    routing_json=dumps_json(tier1.get("routing", {})),
                    tier2a_json=dumps_json(tier2a),
                )},
            ],
            "temperature": 0.0,
            "max_tokens": 1000,
            "enable_thinking": False,
        }
        return self._call_llm(payload, trace, index, "tier2b", required_keys={"checks", "on_fail"})

    def _call_llm(
        self,
        payload: dict[str, Any],
        trace: TraceContext,
        index: int,
        tier: str,
        required_keys: set[str],
    ) -> dict[str, Any] | None:
        try:
            result = self._gateway.llm_chat_json(payload, trace, f"PreferenceAgent/{tier}")
        except Exception as exc:
            logger.warning("PreferenceAgent %s LLM failed for pref %d: %s", tier, index, exc)
            return None
        self.last_llm_call_count += 1
        if not isinstance(result, dict):
            logger.warning("PreferenceAgent %s unexpected type pref %d: %s", tier, index, type(result))
            return None
        if not required_keys.issubset(result.keys()):
            logger.warning("PreferenceAgent %s missing keys pref %d: %s", tier, index, required_keys - set(result.keys()))
            return None
        return result

    @staticmethod
    def _is_v2_cached(item: dict[str, Any]) -> bool:
        if not isinstance(item, dict):
            return False
        assembled = item.get("assembled") if isinstance(item.get("assembled"), dict) else item
        if not isinstance(assembled, dict):
            return False
        if assembled.get("schema_version") != SCHEMA_VERSION:
            return False
        if assembled.get("parse_prompt_version") != PARSE_PROMPT_VERSION:
            return False
        return bool(assembled.get("normalized_rule"))

    @staticmethod
    def _extract_assembled(item: dict[str, Any], source_text: str) -> dict[str, Any]:
        if isinstance(item.get("assembled"), dict):
            return dict(item["assembled"])
        return dict(item)

    @staticmethod
    def _assemble_instructions(current_key: list[str], parsed_by_content: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
        assembled: list[dict[str, Any]] = []
        for index, key in enumerate(current_key, start=1):
            raw = parsed_by_content.get(key)
            if not isinstance(raw, dict):
                continue
            item = dict(PreferenceAgent._extract_assembled(raw, key))
            item["id"] = f"pref_{index}"
            item["content_key"] = key
            item["rule"] = key
            assembled.append(PreferenceAgent._sanitize_instruction(item))
        return assembled

    def _postprocess_instructions(
        self,
        instructions: list[dict[str, Any]],
        state: DecisionState,
        preference_progress: dict[str, Any],
    ) -> list[dict[str, Any]]:
        ctx = state.driver_context
        address_book = self._build_location_address_book(state, preference_progress)
        self._add_preference_location_entries(address_book, instructions)
        out: list[dict[str, Any]] = []
        for item in instructions:
            if not isinstance(item, dict):
                continue
            inst = dict(item)
            content_key = str(inst.get("content_key") or inst.get("rule") or "")
            pref_meta = None
            if ctx is not None:
                pref_meta = lookup_visibility_record(ctx.preference_visibility, content_key) or lookup_pref_record(ctx.preferences_raw, content_key)
                inst = attach_visibility_meta(inst, pref_meta)
            self._force_penalty_hard(inst, pref_meta)
            self._postprocess_target_cargo_instruction(inst)
            self._postprocess_monthly_deadhead_instruction(inst)
            self._postprocess_daily_first_order_deadline_instruction(inst)
            self._postprocess_forbidden_geofence_instruction(inst)
            self._postprocess_geofence_instruction(inst)
            self._postprocess_numeric_limit_instruction(inst)
            self._postprocess_schedule_instruction(inst)
            self._resolve_instruction_locations(inst, address_book)
            self._refresh_derived_fields(inst)
            sanitized = self._sanitize_instruction(inst)
            out.append(sanitized)
            out.extend(self._split_composite_instruction(sanitized))
        return self._dedupe_instructions_by_id(out)

    @staticmethod
    def _dedupe_instructions_by_id(instructions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        seen: set[str] = set()
        for inst in instructions:
            if not isinstance(inst, dict):
                continue
            inst_id = str(inst.get("id") or "")
            if inst_id and inst_id in seen:
                continue
            if inst_id:
                seen.add(inst_id)
            out.append(inst)
        return out

    @staticmethod
    def _force_penalty_hard(inst: dict[str, Any], pref_meta: dict[str, Any] | None) -> None:
        meta = inst.get("meta") if isinstance(inst.get("meta"), dict) else {}
        penalty_values = [
            meta.get("penalty_amount"),
            meta.get("penalty_cap"),
        ]
        if isinstance(pref_meta, dict):
            penalty_values.extend([pref_meta.get("penalty_amount"), pref_meta.get("penalty_cap")])
        has_penalty = False
        for value in penalty_values:
            try:
                if float(value) > 0:
                    has_penalty = True
                    break
            except (TypeError, ValueError):
                continue
        if not has_penalty:
            return
        inst["hardness"] = "hard"
        routing = inst.get("routing") if isinstance(inst.get("routing"), dict) else {}
        routing["hardness_source"] = "penalty_metadata"
        inst["routing"] = routing

    @classmethod
    def _refresh_derived_fields(cls, inst: dict[str, Any]) -> None:
        inst["parse_status"] = resolve_parse_status(inst)
        inst["scheme"] = build_fixed_scheme(inst)
        inst["schedule_task"] = build_schedule_task(inst)
        inst["guard_summary"] = build_guard_summary(inst)
        inst["steps"] = checks_to_steps(inst)
        inst["completion_check"] = build_completion_check(inst)
        inst["required_fields"] = derive_required_fields(inst)

    @classmethod
    def _split_composite_instruction(cls, inst: dict[str, Any]) -> list[dict[str, Any]]:
        meta = inst.get("meta") if isinstance(inst.get("meta"), dict) else {}
        if meta.get("split_from_composite"):
            return []
        source_text = cls._instruction_source_text(inst)
        if not cls._looks_like_daily_home_before_night_rule(source_text):
            return []
        coords = cls._explicit_coordinates(source_text)
        if not coords:
            return []
        coord = coords[0]
        try:
            lat = float(coord["lat"])
            lng = float(coord["lng"])
        except (KeyError, TypeError, ValueError):
            return []
        parent_id = str(inst.get("id") or "pref")
        content_key = str(inst.get("content_key") or inst.get("rule") or source_text)
        target = {
            "name": "自家位置",
            "lat": lat,
            "lng": lng,
            "radius_km": 1,
        }
        arrival_child = {
            "schema_version": SCHEMA_VERSION,
            "parse_prompt_version": PARSE_PROMPT_VERSION,
            "id": f"{parent_id}_home_arrival",
            "content_key": f"{content_key} [拆分:每日23点前到家]",
            "source_rule": source_text,
            "rule": "每天23:00前车辆须到达自家位置1公里内。",
            "normalized_rule": "每天23:00前车辆须到达自家位置1公里内。",
            "preference_type": "LOCATION_ARRIVAL_DEADLINE",
            "category": "单日/指定日事件",
            "hardness": "hard",
            "uncertainty": "",
            "persistent": True,
            "routing": {
                "preference_type": "LOCATION_ARRIVAL_DEADLINE",
                "cycle_kind": "day",
                "active_dates": None,
                "scope_actions": ["reposition"],
                "blocked_actions": [],
                "constraint_kinds": ["location_wait"],
                "needs_sequence": False,
                "needs_history_aggregate": False,
                "count_min": None,
                "distinct_by": None,
            },
            "cycle": {
                "length": "day",
                "window": None,
                "active_dates": None,
                "count": None,
                "reset": "calendar_day",
                "evaluate_at": "event_end",
            },
            "scope": {"actions": ["reposition"], "phase": "candidate", "applies_when": "always_active"},
            "completion": {"mode": "per_cycle_satisfied", "track_progress": True, "progress_key": "schedule_progress", "expires_at": None},
            "checks": [
                {
                    "action": "reposition",
                    "phase": "whole",
                    "measure": "location",
                    "compare": "near",
                    "value": target,
                    "unit": "km",
                },
                {
                    "action": "reposition",
                    "phase": "whole",
                    "measure": "clock_time",
                    "compare": "max",
                    "value": "23:00",
                },
            ],
            "on_fail": {"effect": "block", "block_actions": ["take_order", "reposition"]},
            "route_plan": {
                "stops": [
                    {
                        "step_id": "home_before_23",
                        "kind": "arrival",
                        "name": "自家位置",
                        "lat": lat,
                        "lng": lng,
                        "radius_km": 1,
                        "arrive_before": "23:00",
                    }
                ]
            },
            "meta": {"parent_preference_id": parent_id, "split_from_composite": "daily_home_before_night"},
        }
        stationary_child = {
            "schema_version": SCHEMA_VERSION,
            "parse_prompt_version": PARSE_PROMPT_VERSION,
            "id": f"{parent_id}_night_stationary",
            "content_key": f"{content_key} [拆分:夜间不接单不空驶]",
            "source_rule": source_text,
            "rule": "每天23:00至次日08:00不接单、不空驶。",
            "normalized_rule": "每天23:00至次日08:00不接单、不空驶。",
            "preference_type": "TIME_WINDOW_STATIONARY",
            "category": "时段/休息约束",
            "hardness": "hard",
            "uncertainty": "",
            "persistent": True,
            "routing": {
                "preference_type": "TIME_WINDOW_STATIONARY",
                "cycle_kind": "day",
                "active_dates": None,
                "scope_actions": ["take_order", "reposition"],
                "blocked_actions": ["take_order", "reposition"],
                "constraint_kinds": ["time_window"],
                "needs_sequence": False,
                "needs_history_aggregate": False,
                "count_min": None,
                "distinct_by": None,
            },
            "cycle": {
                "length": "day",
                "window": {"start": "23:00", "end": "08:00"},
                "active_dates": None,
                "count": None,
                "reset": "calendar_day",
                "evaluate_at": "per_action",
            },
            "scope": {"actions": ["take_order", "reposition"], "phase": "candidate", "applies_when": "always_active"},
            "completion": {"mode": "never_expires", "track_progress": True, "progress_key": "time_window_progress", "expires_at": None},
            "checks": [
                {
                    "action": "take_order",
                    "phase": "whole",
                    "measure": "clock_time",
                    "compare": "not_in_window",
                    "value": {"from": "23:00", "to": "08:00"},
                },
                {
                    "action": "reposition",
                    "phase": "whole",
                    "measure": "clock_time",
                    "compare": "not_in_window",
                    "value": {"from": "23:00", "to": "08:00"},
                },
            ],
            "on_fail": {"effect": "block", "block_actions": ["take_order", "reposition"]},
            "meta": {"parent_preference_id": parent_id, "split_from_composite": "daily_home_before_night"},
        }
        children = [arrival_child, stationary_child]
        out: list[dict[str, Any]] = []
        for child in children:
            cls._refresh_derived_fields(child)
            out.append(cls._sanitize_instruction(child))
        return out

    @staticmethod
    def _looks_like_daily_home_before_night_rule(text: str) -> bool:
        if not text:
            return False
        compact = re.sub(r"\s+", "", str(text))
        has_home_deadline = ("23点前" in compact or "23:00前" in compact) and ("自家位置" in compact or "家" in compact)
        has_night_window = ("23点至次日8点" in compact or "23:00至次日8:00" in compact or "23点到次日8点" in compact)
        has_forbidden_movement = "不接单" in compact and ("不空跑" in compact or "不空驶" in compact)
        return has_home_deadline and has_night_window and has_forbidden_movement

    @classmethod
    def _postprocess_geofence_instruction(cls, inst: dict[str, Any]) -> None:
        if not cls._is_structured_geofence_instruction(inst):
            return
        inst["preference_type"] = "GEOFENCE_STAY_WITHIN"
        inst.setdefault("category", "unknown")
        if str(inst.get("hardness") or "").lower() == "unknown":
            inst["hardness"] = "hard"
        routing = dict(inst.get("routing") or {})
        routing.update({
            "preference_type": "GEOFENCE_STAY_WITHIN",
            "cycle_kind": "always",
            "active_dates": None,
            "scope_actions": ["wait", "reposition", "take_order"],
            "blocked_actions": ["wait", "reposition", "take_order"],
            "constraint_kinds": ["geofence"],
            "needs_sequence": False,
            "needs_history_aggregate": False,
            "count_min": None,
            "distinct_by": None,
        })
        inst["routing"] = routing
        inst["cycle"] = {
            "length": "always",
            "window": None,
            "active_dates": None,
            "count": None,
            "reset": "never",
            "evaluate_at": "per_action",
        }
        inst["scope"] = {"actions": ["wait", "reposition", "take_order"], "phase": "candidate", "applies_when": "always_active"}
        inst["completion"] = {"mode": "never_expires", "track_progress": True, "progress_key": "geofence_progress", "expires_at": None}
        inst["on_fail"] = {"effect": "block", "block_actions": ["wait", "reposition", "take_order"]}
        meta = dict(inst.get("meta") or {})
        meta["geofence_postprocess"] = "structured_only"
        inst["meta"] = meta

    @staticmethod
    def _is_structured_geofence_instruction(inst: dict[str, Any]) -> bool:
        pref_type = str(inst.get("preference_type") or "")
        if pref_type == "GEOFENCE_FORBIDDEN_AREA":
            return False
        if pref_type == "GEOFENCE_STAY_WITHIN":
            return True
        routing = inst.get("routing") if isinstance(inst.get("routing"), dict) else {}
        if str(routing.get("preference_type") or "") == "GEOFENCE_STAY_WITHIN":
            return True
        kinds = routing.get("constraint_kinds")
        if isinstance(kinds, list) and any(str(item) == "geofence" for item in kinds):
            return True
        checks = inst.get("checks")
        if isinstance(checks, list):
            return any(isinstance(item, dict) and str(item.get("measure") or "") == "geofence" for item in checks)
        return False

    @classmethod
    def _postprocess_monthly_deadhead_instruction(cls, inst: dict[str, Any]) -> None:
        source_text = cls._instruction_source_text(inst)
        if not cls._looks_like_monthly_deadhead_limit(source_text, inst):
            return
        limit = cls._extract_distance_limit_km(source_text)
        inst["preference_type"] = "MONTHLY_DEADHEAD_LIMIT"
        inst["category"] = "距离/空驶约束"
        inst["hardness"] = "hard"
        routing = dict(inst.get("routing") or {})
        routing.update({
            "preference_type": "MONTHLY_DEADHEAD_LIMIT",
            "cycle_kind": "month",
            "active_dates": None,
            "scope_actions": ["take_order", "reposition"],
            "blocked_actions": ["take_order", "reposition"],
            "constraint_kinds": ["distance"],
            "needs_sequence": False,
            "needs_history_aggregate": True,
            "count_min": None,
            "distinct_by": "none",
        })
        inst["routing"] = routing
        inst["cycle"] = {
            "length": "month",
            "window": {"start": "2026-03-01T00:00:00", "end": "2026-04-01T00:00:00"},
            "active_dates": None,
            "count": None,
            "reset": "calendar_month",
            "evaluate_at": "per_action",
        }
        inst["scope"] = {"actions": ["take_order", "reposition"], "phase": "candidate", "applies_when": "always_active"}
        inst["completion"] = {"mode": "never_expires", "track_progress": True, "progress_key": "monthly_deadhead_progress", "expires_at": None}
        inst["checks"] = [{
            "action": "reposition",
            "phase": "whole",
            "measure": "distance",
            "compare": "max",
            "value": limit,
            "unit": "km",
        }]
        inst["on_fail"] = {"effect": "block", "block_actions": ["take_order", "reposition"]}
        meta = dict(inst.get("meta") or {})
        meta["monthly_deadhead_limit_km"] = limit
        inst["meta"] = meta

    @staticmethod
    def _looks_like_monthly_deadhead_limit(text: str, inst: dict[str, Any]) -> bool:
        pref_type = str(inst.get("preference_type") or "")
        routing = inst.get("routing") if isinstance(inst.get("routing"), dict) else {}
        return (
            pref_type == "MONTHLY_DEADHEAD_LIMIT"
            or str(routing.get("preference_type") or "") == "MONTHLY_DEADHEAD_LIMIT"
        )

    @classmethod
    def _postprocess_daily_first_order_deadline_instruction(cls, inst: dict[str, Any]) -> None:
        source_text = cls._instruction_source_text(inst)
        if not cls._looks_like_daily_first_order_deadline(source_text):
            return
        deadline = cls._first_order_deadline_time(source_text) or "12:00"
        inst["preference_type"] = "DAILY_FIRST_ORDER_DEADLINE"
        inst["category"] = "单日/指定日事件"
        inst["hardness"] = "hard"
        routing = dict(inst.get("routing") or {})
        routing.update({
            "preference_type": "DAILY_FIRST_ORDER_DEADLINE",
            "cycle_kind": "day",
            "active_dates": None,
            "scope_actions": ["take_order"],
            "blocked_actions": ["take_order"],
            "constraint_kinds": ["time_window"],
            "needs_sequence": False,
            "needs_history_aggregate": False,
            "count_min": None,
            "distinct_by": None,
        })
        inst["routing"] = routing
        inst["cycle"] = {
            "length": "day",
            "window": None,
            "active_dates": None,
            "count": None,
            "reset": "calendar_day",
            "evaluate_at": "per_action",
        }
        inst["scope"] = {"actions": ["take_order"], "phase": "candidate", "applies_when": "always_active"}
        inst["completion"] = {
            "mode": "per_cycle_satisfied",
            "track_progress": True,
            "progress_key": "daily_first_order_deadline",
            "expires_at": None,
        }
        inst["checks"] = [{
            "action": "take_order",
            "phase": "whole",
            "measure": "clock_time",
            "compare": "max",
            "value": deadline,
            "unit": "HH:MM",
        }]
        inst["on_fail"] = {"effect": "block", "block_actions": ["take_order"]}
        inst["route_plan"] = None
        meta = dict(inst.get("meta") or {})
        meta["daily_first_order_deadline_postprocess"] = "text_pattern"
        inst["meta"] = meta

    @staticmethod
    def _looks_like_daily_first_order_deadline(text: str) -> bool:
        if not text:
            return False
        has_first_order = any(token in text for token in ("首单", "第一单", "第一笔"))
        has_start = any(token in text for token in ("开工", "接单", "开始"))
        has_deadline = any(token in text for token in ("不晚于", "前", "以前", "早于", "之前"))
        return has_first_order and has_start and has_deadline

    @staticmethod
    def _first_order_deadline_time(text: str) -> str | None:
        import re

        for match in re.finditer(r"(\d{1,2}):(\d{1,2})|(\d{1,2})\s*点", text):
            try:
                hour = int(match.group(1) or match.group(3))
                minute = int(match.group(2) or 0)
            except (TypeError, ValueError):
                continue
            if 0 <= hour <= 23 and 0 <= minute <= 59:
                return f"{hour:02d}:{minute:02d}"
        if "中午" in text:
            return "12:00"
        return None

    @classmethod
    def _postprocess_forbidden_geofence_instruction(cls, inst: dict[str, Any]) -> None:
        source_text = cls._instruction_source_text(inst)
        pref_type = str(inst.get("preference_type") or "")
        routing = inst.get("routing") if isinstance(inst.get("routing"), dict) else {}
        structured = pref_type == "GEOFENCE_FORBIDDEN_AREA" or str(routing.get("preference_type") or "") == "GEOFENCE_FORBIDDEN_AREA"
        if not structured:
            return
        circle = cls._extract_forbidden_circle(source_text)
        inst["preference_type"] = "GEOFENCE_FORBIDDEN_AREA"
        inst["category"] = "城市/路线回避"
        if str(inst.get("hardness") or "").lower() == "unknown":
            inst["hardness"] = "hard"
        routing = dict(inst.get("routing") or {})
        routing.update({
            "preference_type": "GEOFENCE_FORBIDDEN_AREA",
            "cycle_kind": "always",
            "active_dates": None,
            "scope_actions": ["wait", "reposition", "take_order"],
            "blocked_actions": ["wait", "reposition", "take_order"],
            "constraint_kinds": ["geofence"],
            "needs_sequence": False,
            "needs_history_aggregate": False,
            "count_min": None,
            "distinct_by": None,
        })
        inst["routing"] = routing
        inst["cycle"] = {
            "length": "always",
            "window": None,
            "active_dates": None,
            "count": None,
            "reset": "never",
            "evaluate_at": "per_action",
        }
        inst["scope"] = {"actions": ["wait", "reposition", "take_order"], "phase": "candidate", "applies_when": "always_active"}
        inst["completion"] = {"mode": "never_expires", "track_progress": True, "progress_key": "forbidden_geofence_progress", "expires_at": None}
        if circle is not None:
            inst["checks"] = [{
                "action": "reposition",
                "phase": "whole",
                "measure": "geofence",
                "compare": "not_contains",
                "value": circle,
                "unit": "km",
            }]
            meta = dict(inst.get("meta") or {})
            meta["forbidden_geofence_circle"] = circle
            inst["meta"] = meta
        inst["on_fail"] = {"effect": "block", "block_actions": ["wait", "reposition", "take_order"]}

    @staticmethod
    def _extract_distance_limit_km(text: str) -> float | None:
        match = re.search(r"(\d+(?:\.\d+)?)\s*(?:km|公里)", str(text or ""), flags=re.IGNORECASE)
        if not match:
            return None
        try:
            return float(match.group(1))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _extract_forbidden_circle(text: str) -> dict[str, Any] | None:
        source = str(text or "")
        if not any(token in source for token in ("不得进入", "禁入", "不要进入", "不能进入")):
            return None
        match = re.search(
            r"[（(]\s*(-?\d+(?:\.\d+)?)\s*[,，]\s*(-?\d+(?:\.\d+)?)\s*[）)].{0,24}?半径\s*(\d+(?:\.\d+)?)\s*公里",
            source,
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

    @classmethod
    def _postprocess_numeric_limit_instruction(cls, inst: dict[str, Any]) -> None:
        pref_type = str(inst.get("preference_type") or "")
        routing = inst.get("routing") if isinstance(inst.get("routing"), dict) else {}
        if pref_type != "NUMERIC_LIMIT" and str(routing.get("preference_type") or "") != "NUMERIC_LIMIT":
            return
        source_text = cls._instruction_source_text(inst)
        if not cls._looks_like_take_order_haul_distance_limit(source_text):
            return
        checks = inst.get("checks")
        if not isinstance(checks, list):
            return
        changed = False
        next_checks: list[Any] = []
        for check in checks:
            if (
                isinstance(check, dict)
                and str(check.get("action") or "") == "take_order"
                and str(check.get("measure") or "") == "distance"
            ):
                item = dict(check)
                item["phase"] = "haul"
                next_checks.append(item)
                changed = True
            else:
                next_checks.append(check)
        if changed:
            inst["checks"] = next_checks
            meta = dict(inst.get("meta") or {})
            meta["numeric_limit_phase_postprocess"] = "take_order_haul_distance"
            inst["meta"] = meta

    @staticmethod
    def _looks_like_take_order_haul_distance_limit(text: str) -> bool:
        compact = re.sub(r"\s+", "", str(text or ""))
        if not compact:
            return False
        if "空驶" in compact or "空跑" in compact or "赶路" in compact:
            return False
        return (
            "装卸距离" in compact
            or "运距" in compact
            or "运输距离" in compact
            or ("装货点" in compact and "卸货点" in compact and "距离" in compact)
            or ("装货地" in compact and "卸货地" in compact and "距离" in compact)
        )

    @classmethod
    def _postprocess_schedule_instruction(cls, inst: dict[str, Any]) -> None:
        pref_type = str(inst.get("preference_type") or "")
        if pref_type not in {"LOCATION_STAY_ON_DATE", "LOCATION_ARRIVAL_DEADLINE", "ROUTE_SEQUENCE_ON_DATE"}:
            return
        source_text = cls._instruction_source_text(inst)
        if not source_text:
            return
        if pref_type == "ROUTE_SEQUENCE_ON_DATE":
            cls._repair_route_plan_from_text(inst, source_text)

    @classmethod
    def _postprocess_target_cargo_instruction(cls, inst: dict[str, Any]) -> None:
        source_text = cls._instruction_source_text(inst)
        target = cls._extract_target_cargo(source_text)
        if not target:
            return
        inst["preference_type"] = "TARGET_CARGO_MUST_TAKE"
        inst["category"] = "\u8d27\u6e90\u7c7b\u578b/\u54c1\u7c7b\u7ea6\u675f"
        if str(inst.get("hardness") or "").lower() == "unknown":
            inst["hardness"] = "hard"
        inst["persistent"] = False
        inst["normalized_rule"] = (
            f"\u5fc5\u987b\u4f18\u5148\u4e89\u53d6\u6307\u5b9a\u719f\u8d27\u6e90 {target['cargo_id']}"
            f"\uff0c\u53ef\u89c1\u4e14\u4e0d\u8fdd\u53cd\u5176\u4ed6\u786c\u7ea6\u675f\u65f6\u5e94\u63a5\u5355\u3002"
        )
        routing = dict(inst.get("routing") or {})
        routing.update({
            "preference_type": "TARGET_CARGO_MUST_TAKE",
            "cycle_kind": "once",
            "scope_actions": ["take_order"],
            "blocked_actions": [],
            "constraint_kinds": ["target_cargo"],
            "needs_sequence": False,
            "needs_history_aggregate": False,
            "count_min": None,
            "distinct_by": None,
        })
        active_date = cls._target_cargo_active_date(target, source_text)
        if active_date:
            routing["active_dates"] = [active_date]
        inst["routing"] = routing
        inst["cycle"] = {
            "length": "once",
            "window": {
                "start": f"{active_date or '2026-03-01'}T00:00:00",
                "end": "2026-04-01T00:00:00",
            },
            "active_dates": [active_date] if active_date else None,
            "count": None,
            "reset": "never",
            "evaluate_at": "event_end",
        }
        inst["scope"] = {"actions": ["take_order"], "phase": "candidate", "applies_when": "in_cycle_window"}
        inst["completion"] = {
            "mode": "window_expires",
            "track_progress": True,
            "progress_key": "target_cargo_progress",
            "expires_at": "2026-04-01T00:00:00",
        }
        inst["checks"] = []
        inst["on_fail"] = {"effect": "block", "block_actions": ["take_order"]}
        inst["route_plan"] = None
        meta = dict(inst.get("meta") or {})
        if meta.get("visible_from") and not target.get("available_after"):
            target["available_after"] = meta.get("visible_from")
        if meta.get("visible_until") and not target.get("available_until"):
            target["available_until"] = meta.get("visible_until")
        meta["target_cargo"] = target
        meta["target_cargo_id"] = target["cargo_id"]
        if target.get("available_after"):
            meta["target_cargo_available_after"] = target.get("available_after")
        if target.get("available_until"):
            meta["target_cargo_available_until"] = target.get("available_until")
        inst["meta"] = meta

    @classmethod
    def _extract_target_cargo(cls, text: str) -> dict[str, Any] | None:
        source = str(text or "")
        id_match = re.search(
            r"(?:\u719f\u8d27\u6e90|\u719f\u8d27|\u8d27\u6e90|\u8d27\u6e90\u7f16\u53f7|\u7f16\u53f7|cargo_id)[^\d]{0,12}(\d{4,})",
            source,
            flags=re.IGNORECASE,
        )
        if not id_match:
            return None
        cargo_id = id_match.group(1)
        target: dict[str, Any] = {"cargo_id": cargo_id}
        cargo_name = cls._extract_target_cargo_name(source)
        if cargo_name:
            target["cargo_name"] = cargo_name
        coords = cls._explicit_coordinates(source)
        if coords:
            coord = coords[0]
            pickup: dict[str, Any] = {"lat": coord["lat"], "lng": coord["lng"], "radius_km": 5}
            aliases = [str(item).strip() for item in coord.get("aliases", []) if str(item).strip()]
            if aliases:
                pickup["name"] = aliases[0]
                pickup["aliases"] = aliases[:6]
            target["pickup_location"] = pickup
        available_after = cls._extract_target_cargo_available_after(source)
        if available_after:
            target["available_after"] = available_after
        return target

    @staticmethod
    def _extract_target_cargo_name(text: str) -> str | None:
        for pattern in (
            r"\u54c1\u7c7b[：:\s\u300c\u300e\u201c\"]+([^」』”\"\uff1b;，,\u3002\s]+)",
            r"\u7c7b\u522b[：:\s\u300c\u300e\u201c\"]+([^」』”\"\uff1b;，,\u3002\s]+)",
        ):
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()
        return None

    @staticmethod
    def _extract_target_cargo_available_after(text: str) -> str | None:
        match = re.search(r"(20\d{2}-\d{1,2}-\d{1,2})\s+(\d{1,2}:\d{2}(?::\d{2})?)", text)
        if not match:
            return None
        date = match.group(1)
        time = match.group(2)
        parts = date.split("-")
        try:
            normalized_date = f"{int(parts[0]):04d}-{int(parts[1]):02d}-{int(parts[2]):02d}"
            if len(time) == 5:
                time = f"{time}:00"
            return f"{normalized_date} {time}"
        except (ValueError, IndexError):
            return f"{date} {time}"

    @staticmethod
    def _target_cargo_active_date(target: dict[str, Any], source_text: str) -> str | None:
        available = str(target.get("available_after") or "")
        if len(available) >= 10:
            return available[:10]
        match = re.search(r"20\d{2}-\d{1,2}-\d{1,2}", source_text)
        if not match:
            return None
        parts = match.group(0).split("-")
        try:
            return f"{int(parts[0]):04d}-{int(parts[1]):02d}-{int(parts[2]):02d}"
        except (ValueError, IndexError):
            return match.group(0)

    @classmethod
    def _repair_route_plan_from_text(cls, inst: dict[str, Any], source_text: str) -> None:
        route_plan = inst.get("route_plan")
        if not isinstance(route_plan, dict):
            rebuilt = cls._route_plan_from_sequence_text(source_text)
            if rebuilt:
                inst["route_plan"] = rebuilt
                route_plan = rebuilt
                meta = dict(inst.get("meta") or {})
                repairs = list(meta.get("schedule_postprocess_repairs") or [])
                repairs.append({"field": "route_plan", "source": "source_text_sequence_fallback"})
                meta["schedule_postprocess_repairs"] = repairs[-8:]
                inst["meta"] = meta
        if not isinstance(route_plan, dict):
            return
        stops = route_plan.get("steps")
        field_prefix = "route_plan.steps"
        if not isinstance(stops, list):
            stops = route_plan.get("stops")
            field_prefix = "route_plan.stops"
        if not isinstance(stops, list) or not stops:
            return

        normalized_stops = cls._normalize_route_plan_steps(stops)
        if normalized_stops is not stops:
            route_plan["steps"] = normalized_stops
            stops = normalized_stops
            field_prefix = "route_plan.steps"

        cls._enrich_route_stop_names(stops, source_text)
        finish_time = cls._latest_schedule_finish_time(source_text)
        if finish_time:
            cls._apply_route_finish_time(stops, finish_time)

        meta = dict(inst.get("meta") or {})
        repairs = list(meta.get("schedule_postprocess_repairs") or [])
        if finish_time:
            repairs.append({"field": f"{field_prefix}[-1].hold_until", "value": finish_time, "source": "source_text_stay_until"})
        if repairs:
            meta["schedule_postprocess_repairs"] = repairs[-8:]
            inst["meta"] = meta

    @classmethod
    def _normalize_route_plan_steps(cls, stops: list[Any]) -> list[Any]:
        normalized: list[Any] = []
        changed = False
        for stop in stops:
            if not isinstance(stop, dict):
                normalized.append(stop)
                continue
            kind = str(stop.get("kind") or "").strip().lower()
            has_stay = cls._route_stop_has_stay_constraint(stop)
            if kind != "arrival" or not has_stay:
                normalized.append(stop)
                continue

            changed = True
            arrival = dict(stop)
            for key in (
                "stay_minutes",
                "min_stay_minutes",
                "complete_before",
                "hold_until",
                "finish_before",
            ):
                arrival.pop(key, None)
            arrival["kind"] = "arrival"
            arrival["scope_actions"] = ["reposition"]
            arrival["constraint_kinds"] = ["route_sequence"]
            normalized.append(arrival)

            stay = {
                key: stop.get(key)
                for key in (
                    "name",
                    "lat",
                    "lng",
                    "radius_km",
                    "date",
                    "active_dates",
                )
                if stop.get(key) not in (None, "", [])
            }
            stay["kind"] = "stay"
            stay["scope_actions"] = ["wait"]
            stay["constraint_kinds"] = ["location_wait"]
            for key in (
                "stay_minutes",
                "min_stay_minutes",
                "complete_before",
                "hold_until",
                "finish_before",
            ):
                if stop.get(key) not in (None, "", []):
                    stay[key] = stop.get(key)
            if (stay.get("stay_minutes") or stay.get("min_stay_minutes")) and not stay.get("complete_before") and stop.get("arrive_before"):
                stay["complete_before"] = stop.get("arrive_before")
            normalized.append(stay)

        if not changed:
            return stops
        for index, stop in enumerate(normalized, start=1):
            if isinstance(stop, dict):
                stop["step_id"] = f"s{index}"
        return normalized

    @staticmethod
    def _route_stop_has_stay_constraint(stop: dict[str, Any]) -> bool:
        for key in ("stay_minutes", "min_stay_minutes", "complete_before", "hold_until", "finish_before"):
            value = stop.get(key)
            if value not in (None, "", [], 0, "0"):
                return True
        return False

    @classmethod
    def _route_plan_from_sequence_text(cls, source_text: str) -> dict[str, Any] | None:
        coords = cls._explicit_coordinates(source_text)
        if len(coords) < 2:
            return None
        date = cls._first_schedule_date(source_text)
        if not date:
            return None
        finish_time = cls._latest_schedule_finish_time(source_text)
        deadline = cls._schedule_deadline_before_hold(source_text, finish_time)
        first = coords[0]
        last = coords[1]
        first_name = cls._coordinate_name(first, "step1")
        last_name = cls._coordinate_name(last, "step2")
        steps: list[dict[str, Any]] = [
            {
                "step_id": "s1",
                "kind": "arrival",
                "active_dates": [date],
                "scope_actions": ["reposition"],
                "constraint_kinds": ["route_sequence"],
                "name": first_name,
                "lat": first.get("lat"),
                "lng": first.get("lng"),
                "radius_km": 1,
            }
        ]
        if deadline:
            steps[0]["arrive_before"] = deadline
        stay_minutes = cls._first_stay_minutes(source_text)
        if stay_minutes:
            stay_step = {
                "step_id": "s2",
                "kind": "stay",
                "active_dates": [date],
                "scope_actions": ["wait"],
                "constraint_kinds": ["location_wait"],
                "name": first_name,
                "lat": first.get("lat"),
                "lng": first.get("lng"),
                "radius_km": 1,
                "stay_minutes": stay_minutes,
            }
            if deadline:
                stay_step["complete_before"] = deadline
            steps.append(stay_step)
        arrival_step = {
            "step_id": f"s{len(steps) + 1}",
            "kind": "arrival",
            "active_dates": [date],
            "scope_actions": ["reposition"],
            "constraint_kinds": ["route_sequence"],
            "name": last_name,
            "lat": last.get("lat"),
            "lng": last.get("lng"),
            "radius_km": 1,
        }
        if deadline:
            arrival_step["arrive_before"] = deadline
        steps.append(arrival_step)
        if finish_time:
            steps.append({
                "step_id": f"s{len(steps) + 1}",
                "kind": "stay",
                "active_dates": [date],
                "scope_actions": ["wait"],
                "constraint_kinds": ["location_wait"],
                "name": last_name,
                "lat": last.get("lat"),
                "lng": last.get("lng"),
                "radius_km": 1,
                "hold_until": finish_time,
            })
        return {"date": date, "steps": steps}

    @staticmethod
    def _coordinate_name(coord: dict[str, Any], fallback: str) -> str:
        aliases = coord.get("aliases") if isinstance(coord.get("aliases"), list) else []
        for alias in aliases:
            text = str(alias or "").strip()
            if text:
                return text
        return fallback

    @staticmethod
    def _first_schedule_date(text: str) -> str | None:
        source = str(text or "")
        match = re.search(r"(20\d{2})-(\d{1,2})-(\d{1,2})", source)
        if not match:
            match = re.search(r"(20\d{2})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日", source)
        if not match:
            return None
        try:
            return f"{int(match.group(1)):04d}-{int(match.group(2)):02d}-{int(match.group(3)):02d}"
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _first_stay_minutes(text: str) -> int | None:
        source = str(text or "")
        for pattern in (
            r"(?:停留|待|静止)[^，。；;]{0,12}(?:不少于|至少|>=|≥)\s*(\d{1,4})\s*分钟",
            r"(?:不少于|至少|>=|≥)\s*(\d{1,4})\s*分钟[^，。；;]{0,8}(?:停留|待|静止)",
        ):
            match = re.search(pattern, source)
            if not match:
                continue
            try:
                minutes = int(match.group(1))
            except ValueError:
                continue
            if minutes > 0:
                return minutes
        return None

    @staticmethod
    def _schedule_deadline_before_hold(text: str, hold_until: str | None) -> str | None:
        source = str(text or "")
        candidates: list[tuple[int, str]] = []
        for match in re.finditer(r"(\d{1,2})\s*(?::|：|点)\s*(\d{1,2})?\s*前", source):
            try:
                hour = int(match.group(1))
                minute = int(match.group(2) or 0)
            except ValueError:
                continue
            if 0 <= hour <= 23 and 0 <= minute <= 59:
                candidates.append((match.start(), f"{hour:02d}:{minute:02d}"))
        if not candidates:
            return None
        if hold_until:
            hold_clock = hold_until[-5:]
            for _, clock in candidates:
                if clock != hold_clock:
                    return clock
        return candidates[0][1]

    @classmethod
    def _enrich_route_stop_names(cls, stops: list[Any], source_text: str) -> None:
        phrases = cls._place_phrases(source_text)
        if not phrases:
            return
        for stop in stops:
            if not isinstance(stop, dict) or stop.get("lat") is not None and stop.get("lng") is not None:
                continue
            name = str(stop.get("name") or "").strip()
            if not name:
                continue
            best = cls._best_place_phrase_for_name(name, phrases)
            if best and len(best) > len(name):
                stop.setdefault("original_name", name)
                stop["name"] = best

    @classmethod
    def _best_place_phrase_for_name(cls, name: str, phrases: list[str]) -> str | None:
        name_norm = cls._normalize_place_text(name)
        best: tuple[int, int, str] | None = None
        for phrase in phrases:
            phrase_norm = cls._normalize_place_text(phrase)
            if not phrase_norm or not name_norm:
                continue
            score = 0
            if name_norm == phrase_norm:
                score = 100
            elif name_norm in phrase_norm or phrase_norm in name_norm:
                score = 80
            elif any(token in phrase_norm for token in cls._place_tokens(name)):
                score = 60
            if score <= 0:
                continue
            candidate = (score, len(phrase), phrase)
            if best is None or candidate > best:
                best = candidate
        return best[2] if best else None

    @classmethod
    def _apply_route_finish_time(cls, stops: list[Any], finish_time: str) -> None:
        last_stop = next((stop for stop in reversed(stops) if isinstance(stop, dict)), None)
        if not isinstance(last_stop, dict):
            return
        for stop in stops:
            if not isinstance(stop, dict) or stop is last_stop:
                continue
            if stop.get("stay_minutes") in (None, "", 0, "0"):
                stop.pop("finish_before", None)
        existing = str(last_stop.get("hold_until") or last_stop.get("finish_before") or "")
        if not existing or cls._schedule_time_sort_key(finish_time) >= cls._schedule_time_sort_key(existing):
            last_stop.pop("finish_before", None)
            last_stop["hold_until"] = finish_time

    @classmethod
    def _latest_schedule_finish_time(cls, text: str) -> str | None:
        datetimes = cls._extract_stay_until_datetimes(text)
        if datetimes:
            return max(datetimes)
        return cls._latest_clock_time(cls._extract_stay_until_times(text))

    @classmethod
    def _extract_stay_until_datetimes(cls, text: str) -> list[str]:
        source = str(text or "")
        date_pattern = r"(20\d{2})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日"
        time_pattern = r"(\d{1,2})\s*(?:[:：]|点)\s*(\d{1,2})?\s*(?:分)?"
        trigger_pattern = (
            r"(?:待到|直到|一直到|停到|留到|守到|静止.*?到|原处.*?到|至少.*?到)"
            rf".{{0,12}}?{date_pattern}.{{0,6}}?{time_pattern}"
        )
        out: list[str] = []
        for match in re.finditer(trigger_pattern, source):
            try:
                year = int(match.group(1))
                month = int(match.group(2))
                day = int(match.group(3))
                hour = int(match.group(4))
                minute = int(match.group(5) or 0)
            except (TypeError, ValueError):
                continue
            if 1 <= month <= 12 and 1 <= day <= 31 and 0 <= hour <= 23 and 0 <= minute <= 59:
                out.append(f"{year:04d}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}")
        return list(dict.fromkeys(out))

    @staticmethod
    def _schedule_time_sort_key(value: Any) -> tuple[int, int]:
        text = str(value or "").strip().replace("T", " ")
        match = re.match(r"(20\d{2})-(\d{2})-(\d{2})\s+(\d{1,2}):(\d{1,2})", text)
        if match:
            try:
                return (1, int(f"{match.group(1)}{match.group(2)}{match.group(3)}{int(match.group(4)):02d}{int(match.group(5)):02d}"))
            except ValueError:
                return (1, -1)
        return (0, PreferenceAgent._clock_time_to_minutes(text))

    @staticmethod
    def _latest_clock_time(values: list[str]) -> str | None:
        if not values:
            return None
        return max(values, key=PreferenceAgent._clock_time_to_minutes)

    @staticmethod
    def _clock_time_to_minutes(value: Any) -> int:
        text = str(value or "")[:5]
        try:
            hour, minute = text.split(":", 1)
            return int(hour) * 60 + int(minute)
        except (TypeError, ValueError):
            return -1

    @classmethod
    def _extract_stay_until_times(cls, text: str) -> list[str]:
        times: list[str] = []
        hour_pattern = r"(?:\d{1,2}|[\u96f6\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341\u4e24]{1,3})"
        minute_pattern = r"(?:\d{1,2}|[\u96f6\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341\u4e24\u534a]{1,3})"
        time_pattern = (
            r"(?:(?:\u51cc\u6668|\u65e9\u4e0a|\u4e0a\u5348|\u4e2d\u5348|\u4e0b\u5348|\u665a\u4e0a)\s*)?"
            rf"{hour_pattern}"
            rf"\s*(?:[:\uff1a]|\u70b9)\s*{minute_pattern}?\s*(?:\u5206)?"
        )
        trigger_pattern = (
            r"(?:\u5f85\u5230|\u76f4\u5230|\u4e00\u76f4\u5230|\u505c\u5230|\u7559\u5230|\u5fd9\u5230|\u5b88\u5230|\u53c2\u52a0.*?\u5230|\u8d74\u5bb4\u5230|\u5230)"
            rf"\s*({time_pattern})"
        )
        source = str(text or "")
        for match in re.finditer(trigger_pattern, source):
            end = match.end()
            if source[end:end + 1] in {"\u524d", "\u4e4b"}:
                continue
            clock = cls._parse_chinese_clock_time(match.group(1))
            if clock:
                times.append(clock)
        return list(dict.fromkeys(times))

    @classmethod
    def _parse_chinese_clock_time(cls, raw: Any) -> str | None:
        text = re.sub(r"\s+", "", str(raw or ""))
        if not text:
            return None
        period = ""
        for marker in ("\u51cc\u6668", "\u65e9\u4e0a", "\u4e0a\u5348", "\u4e2d\u5348", "\u4e0b\u5348", "\u665a\u4e0a"):
            if text.startswith(marker):
                period = marker
                text = text[len(marker):]
                break
        text = text.replace("\u70b9\u949f", "\u70b9")
        match = re.match(
            r"(\d{1,2}|[\u96f6\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341\u4e24]{1,3})"
            r"(?:\s*(?:[:\uff1a]|\u70b9)\s*(\d{1,2}|[\u96f6\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341\u4e24\u534a]{1,3})?\s*(?:\u5206)?)?",
            text,
        )
        if not match:
            return None
        hour = cls._parse_small_number(match.group(1))
        minute = cls._parse_small_number(match.group(2)) if match.group(2) else 0
        if hour is None or minute is None:
            return None
        if period in {"\u4e0b\u5348", "\u665a\u4e0a"} and hour < 12:
            hour += 12
        elif period == "\u4e2d\u5348" and hour < 11:
            hour += 12
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            return None
        return f"{hour:02d}:{minute:02d}"

    @staticmethod
    def _parse_small_number(value: Any) -> int | None:
        if value in (None, ""):
            return None
        text = str(value)
        if text.isdigit():
            return int(text)
        if text == "\u534a":
            return 30
        digits = {
            "\u96f6": 0,
            "\u4e00": 1,
            "\u4e8c": 2,
            "\u4e24": 2,
            "\u4e09": 3,
            "\u56db": 4,
            "\u4e94": 5,
            "\u516d": 6,
            "\u4e03": 7,
            "\u516b": 8,
            "\u4e5d": 9,
        }
        if text == "\u5341":
            return 10
        if text.startswith("\u5341"):
            tail = text[1:]
            return 10 + digits.get(tail, 0)
        if "\u5341" in text:
            head, tail = text.split("\u5341", 1)
            return digits.get(head, 0) * 10 + digits.get(tail, 0)
        if len(text) == 1 and text in digits:
            return digits[text]
        return None

    @classmethod
    def _place_phrases(cls, text: str) -> list[str]:
        phrases: list[str] = []
        source = str(text or "")
        suffixes = cls._place_suffixes()
        suffix_pattern = "|".join(re.escape(suffix) for suffix in sorted(suffixes, key=len, reverse=True))
        trigger_pattern = r"(?:\u5728|\u5230|\u53bb|\u8fc7|\u8d76\u5230|\u524d\u5f80|\u5148\u8fc7|\u6709\u5bb6|\u5f97|\u5fc5\u987b|\u9700\u8981)"
        for segment in re.split(r"[\uff0c\u3002\uff1b;,\n]", source):
            candidates = [segment]
            candidates.extend(match.group(1) for match in re.finditer(rf"{trigger_pattern}([\u4e00-\u9fff]{{1,16}})", segment))
            for candidate in candidates:
                cleaned = re.sub(r"[^\u4e00-\u9fff]", "", candidate)
                for match in re.finditer(rf"([\u4e00-\u9fff]{{1,12}}(?:{suffix_pattern}))", cleaned):
                    phrase = match.group(1)
                    phrase = re.sub(rf"^[\u4e00-\u9fff]{{0,6}}{trigger_pattern}", "", phrase)
                    if phrase:
                        phrases.append(phrase)
        return list(dict.fromkeys(phrases))

    @classmethod
    def _resolve_instruction_locations(cls, inst: dict[str, Any], address_book: list[dict[str, Any]]) -> None:
        pref_type = str(inst.get("preference_type") or "")
        if pref_type not in {"LOCATION_STAY_ON_DATE", "LOCATION_ARRIVAL_DEADLINE", "ROUTE_SEQUENCE_ON_DATE"}:
            return
        source_text = cls._instruction_source_text(inst)
        explicit_coords = cls._explicit_coordinates(source_text)
        resolutions: list[dict[str, Any]] = []

        checks = inst.get("checks")
        if isinstance(checks, list):
            for check in checks:
                if not isinstance(check, dict) or str(check.get("measure") or "") != "location":
                    continue
                value = check.get("value")
                if not isinstance(value, dict):
                    continue
                resolved = cls._resolve_location_value(value, source_text, explicit_coords, address_book)
                check["value"] = resolved
                if resolved.get("resolution"):
                    resolutions.append(dict(resolved["resolution"]))

        route_plan = inst.get("route_plan")
        if isinstance(route_plan, dict):
            stops = route_plan.get("steps")
            if not isinstance(stops, list):
                stops = route_plan.get("stops")
            if isinstance(stops, list):
                for stop in stops:
                    if not isinstance(stop, dict):
                        continue
                    resolved = cls._resolve_location_value(stop, source_text, explicit_coords, address_book)
                    for key in ("name", "lat", "lng", "radius_km", "needs_location_resolution", "resolution", "llm_suggested_location"):
                        if key in resolved:
                            stop[key] = resolved[key]
                        elif key in stop:
                            stop.pop(key, None)
                    if resolved.get("resolution"):
                        resolutions.append(dict(resolved["resolution"]))

        if resolutions:
            meta = dict(inst.get("meta") or {})
            meta["location_resolutions"] = resolutions
            inst["meta"] = meta

    @classmethod
    def _resolve_location_value(
        cls,
        value: dict[str, Any],
        source_text: str,
        explicit_coords: list[dict[str, float]],
        address_book: list[dict[str, Any]],
    ) -> dict[str, Any]:
        out = dict(value)
        out.setdefault("radius_km", 5)
        has_latlng = out.get("lat") is not None and out.get("lng") is not None
        if has_latlng and cls._matches_explicit_coordinate(out, explicit_coords):
            out["resolution"] = {"source": "explicit_text", "name": out.get("name"), "lat": out.get("lat"), "lng": out.get("lng")}
            out.pop("needs_location_resolution", None)
            return out
        if has_latlng:
            out["llm_suggested_location"] = {"lat": out.get("lat"), "lng": out.get("lng")}
            out.pop("lat", None)
            out.pop("lng", None)
        match = cls._best_address_match(out, source_text, address_book)
        if match:
            out["lat"] = match["lat"]
            out["lng"] = match["lng"]
            if not out.get("name"):
                out["name"] = match.get("name") or match.get("city") or match.get("address")
            out.pop("needs_location_resolution", None)
            out["resolution"] = {
                "source": "address_book",
                "name": out.get("name"),
                "lat": out.get("lat"),
                "lng": out.get("lng"),
                "matched_place": match.get("name") or match.get("city") or match.get("address"),
                "matched_source": match.get("source"),
            }
            return out
        out["needs_location_resolution"] = True
        return out

    @staticmethod
    def _instruction_source_text(inst: dict[str, Any]) -> str:
        return " ".join(
            str(inst.get(key) or "")
            for key in ("source_rule", "content_key", "rule")
        )

    @staticmethod
    def _explicit_coordinates(text: str) -> list[dict[str, float]]:
        coords: list[dict[str, float]] = []
        for match in re.finditer(r"(-?\d{1,3}(?:\.\d+)?)\s*[,，]\s*(-?\d{1,3}(?:\.\d+)?)", text):
            try:
                lat = float(match.group(1))
                lng = float(match.group(2))
            except ValueError:
                continue
            if -90 <= lat <= 90 and -180 <= lng <= 180:
                coords.append({"lat": lat, "lng": lng})
        return coords

    @staticmethod
    def _explicit_coordinates(text: str) -> list[dict[str, Any]]:
        coords: list[dict[str, Any]] = []
        pattern = r"(?:[（(]\s*)?(-?\d{1,3}(?:\.\d+)?)\s*[,，、]\s*(-?\d{1,3}(?:\.\d+)?)(?:\s*[）)])?"
        for match in re.finditer(pattern, str(text or "")):
            try:
                lat = float(match.group(1))
                lng = float(match.group(2))
            except ValueError:
                continue
            if -90 <= lat <= 90 and -180 <= lng <= 180:
                coords.append({"lat": lat, "lng": lng, "aliases": PreferenceAgent._coordinate_aliases(str(text or ""), match.start())})
        return coords

    @staticmethod
    def _matches_explicit_coordinate(value: dict[str, Any], explicit_coords: list[dict[str, float]]) -> bool:
        try:
            lat = float(value.get("lat"))
            lng = float(value.get("lng"))
        except (TypeError, ValueError):
            return False
        for coord in explicit_coords:
            if abs(float(coord["lat"]) - lat) <= 0.02 and abs(float(coord["lng"]) - lng) <= 0.02:
                return True
        return False

    @classmethod
    def _build_location_address_book(cls, state: DecisionState, preference_progress: dict[str, Any]) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        spans = preference_progress.get("action_spans") if isinstance(preference_progress, dict) else []
        if isinstance(spans, list):
            for index, span in enumerate(spans[-1200:]):
                if not isinstance(span, dict):
                    continue
                for key in ("start_point", "end_point", "start_position", "end_position"):
                    point = span.get(key)
                    if isinstance(point, dict):
                        cls._add_address_entry(entries, point, f"action_span.{key}", index)
                for event in span.get("query_scan_events") or []:
                    if isinstance(event, dict):
                        cls._add_address_entry(entries, event, "query_scan", index)
        for cand in state.cargo_snapshot:
            cargo = getattr(cand, "cargo", None)
            if not isinstance(cargo, dict):
                continue
            cls._add_address_entry(entries, cargo.get("start"), "cargo_snapshot.start", len(entries))
            cls._add_address_entry(entries, cargo.get("end"), "cargo_snapshot.end", len(entries))
        return entries

    @classmethod
    def _add_preference_location_entries(cls, entries: list[dict[str, Any]], instructions: list[dict[str, Any]]) -> None:
        for index, inst in enumerate(instructions):
            if not isinstance(inst, dict):
                continue
            source_text = cls._instruction_source_text(inst)
            explicit_coords = cls._explicit_coordinates(source_text)
            for coord in explicit_coords:
                aliases = [str(item).strip() for item in coord.get("aliases", []) if str(item).strip()]
                if not aliases:
                    continue
                cls._add_address_entry(
                    entries,
                    {"lat": coord["lat"], "lng": coord["lng"], "name": aliases[0], "aliases": aliases},
                    "preference_explicit_coordinate",
                    10_000 + index,
                )
            for value in cls._iter_location_values(inst):
                coord = cls._matching_explicit_coordinate(value, explicit_coords)
                if not coord:
                    continue
                aliases = cls._location_value_aliases(value, source_text, coord)
                cls._add_address_entry(
                    entries,
                    {"lat": value.get("lat"), "lng": value.get("lng"), "name": aliases[0] if aliases else value.get("name"), "aliases": aliases},
                    "preference_explicit_coordinate",
                    10_000 + index,
                )

    @classmethod
    def _iter_location_values(cls, inst: dict[str, Any]) -> list[dict[str, Any]]:
        values: list[dict[str, Any]] = []
        checks = inst.get("checks")
        if isinstance(checks, list):
            for check in checks:
                if isinstance(check, dict) and check.get("measure") == "location" and isinstance(check.get("value"), dict):
                    values.append(check["value"])
        route_plan = inst.get("route_plan")
        if isinstance(route_plan, dict):
            stops = route_plan.get("steps")
            if not isinstance(stops, list):
                stops = route_plan.get("stops")
            if isinstance(stops, list):
                values.extend([stop for stop in stops if isinstance(stop, dict)])
        return values

    @classmethod
    def _matching_explicit_coordinate(cls, value: dict[str, Any], explicit_coords: list[dict[str, Any]]) -> dict[str, Any] | None:
        try:
            lat = float(value.get("lat"))
            lng = float(value.get("lng"))
        except (TypeError, ValueError):
            return None
        for coord in explicit_coords:
            if abs(float(coord["lat"]) - lat) <= 0.02 and abs(float(coord["lng"]) - lng) <= 0.02:
                return coord
        return None

    @staticmethod
    def _add_address_entry(entries: list[dict[str, Any]], point: Any, source: str, order: int) -> None:
        if not isinstance(point, dict):
            return
        try:
            lat = round(float(point.get("lat")), 5)
            lng = round(float(point.get("lng")), 5)
        except (TypeError, ValueError):
            return
        if not (-90 <= lat <= 90 and -180 <= lng <= 180):
            return
        names = [
            str(point.get(key) or "").strip()
            for key in ("name", "address", "city", "district", "county")
            if str(point.get(key) or "").strip()
        ]
        aliases = point.get("aliases")
        if isinstance(aliases, list):
            for alias in aliases:
                text = str(alias or "").strip()
                if text and text not in names:
                    names.append(text)
        if str(source or "") == "preference_explicit_coordinate":
            for alias in PreferenceAgent._expanded_place_aliases(names):
                if alias not in names:
                    names.append(alias)
        if not names:
            return
        for existing in entries:
            if abs(float(existing["lat"]) - lat) < 0.00001 and abs(float(existing["lng"]) - lng) < 0.00001:
                for name in names:
                    if name not in existing["aliases"]:
                        existing["aliases"].append(name)
                return
        entry = {"source": source, "name": names[0], "aliases": names, "lat": lat, "lng": lng, "order": order}
        for key in ("address", "city", "district", "county"):
            if point.get(key) not in (None, ""):
                entry[key] = point.get(key)
        entries.append(entry)

    @classmethod
    def _best_address_match(cls, value: dict[str, Any], source_text: str, address_book: list[dict[str, Any]]) -> dict[str, Any] | None:
        name = str(value.get("name") or "").strip()
        query_parts = [name] if name else [source_text]
        best: tuple[int, int, dict[str, Any]] | None = None
        for entry in address_book:
            score = cls._address_match_score(query_parts, entry)
            if score <= 0:
                continue
            if str(entry.get("source") or "") == "preference_explicit_coordinate":
                score += 20
            recency = int(entry.get("order", 0) or 0)
            candidate = (score, recency, entry)
            if best is None or candidate[:2] > best[:2]:
                best = candidate
        return dict(best[2]) if best else None

    @classmethod
    def _address_match_score(cls, query_parts: list[str], entry: dict[str, Any]) -> int:
        aliases = [str(item) for item in entry.get("aliases", []) if str(item).strip()]
        if not aliases:
            return 0
        queries = [part for part in query_parts if part]
        score = 0
        for query in queries:
            qnorm = cls._normalize_place_text(query)
            if not qnorm:
                continue
            for alias in aliases:
                anorm = cls._normalize_place_text(alias)
                if not anorm:
                    continue
                if qnorm == anorm:
                    score = max(score, 100)
                elif qnorm in anorm or anorm in qnorm:
                    score = max(score, 80 if len(anorm) >= 2 else 0)
                elif len(anorm) >= 2 and anorm in qnorm:
                    score = max(score, 60)
                for token in cls._place_tokens(alias):
                    if len(token) >= 2 and token in qnorm:
                        score = max(score, 55)
        return score

    @classmethod
    def _location_value_aliases(cls, value: dict[str, Any], source_text: str, coord: dict[str, Any] | None = None) -> list[str]:
        aliases: list[str] = []
        for key in ("name", "address", "city", "district", "county"):
            text = str(value.get(key) or "").strip()
            if text:
                aliases.append(text)
        if coord and isinstance(coord.get("aliases"), list):
            aliases.extend(str(item).strip() for item in coord["aliases"] if str(item).strip())
        if not aliases:
            aliases.extend(cls._place_tokens(source_text))
        return list(dict.fromkeys(alias for alias in aliases if alias))

    @classmethod
    def _expanded_place_aliases(cls, aliases: list[str]) -> list[str]:
        out: list[str] = []
        for alias in aliases:
            text = str(alias or "").strip()
            if not text:
                continue
            out.append(text)
            out.extend(cls._place_phrases(text))
            compact = re.sub(r"[^\u4e00-\u9fff]", "", text)
            compact = re.sub(r"^(?:\u6211\u5728|\u5728|\u5230|\u53bb|\u8fc7|\u5148\u8fc7|\u5f97\u5148\u8fc7|\u8d76\u5230|\u524d\u5f80)", "", compact)
            compact = compact.replace("\u5e7f\u4e1c\u7701", "").replace("\u5e7f\u5dde\u5e02", "")
            if compact and compact != text:
                out.append(compact)
            for match in re.finditer(r"([\u4e00-\u9fff]{2,8}?)(?:\u533a|\u5e02|\u53bf)?(?:\u6709\u5bb6|\u4e00\u5bb6|\u4e2a|\u7684)?(\u8001?\u6863\u53e3)", compact):
                base = match.group(1)
                suffix = match.group(2)
                out.extend([
                    f"{base}{suffix}",
                    f"{base}\u533a{suffix}",
                    f"{base}\u6863\u53e3",
                    f"{base}\u533a\u6863\u53e3",
                    f"{base}\u8001\u6863\u53e3",
                    f"{base}\u533a\u8001\u6863\u53e3",
                ])
        return list(dict.fromkeys(item for item in out if item))

    @staticmethod
    def _coordinate_aliases(text: str, coord_start: int) -> list[str]:
        prefix = str(text or "")[max(0, coord_start - 48):coord_start]
        prefix = re.split(r"[,，。；;:：\n]", prefix)[-1]
        prefix = re.sub(r"[（(【\[]+$", "", prefix).strip()
        aliases: list[str] = []
        suffixes = PreferenceAgent._place_suffixes()
        suffix_pattern = "|".join(re.escape(suffix) for suffix in sorted(suffixes, key=len, reverse=True))
        for match in re.finditer(rf"([\u4e00-\u9fff]{{1,16}}(?:{suffix_pattern}))", prefix):
            candidate = match.group(1).strip()
            candidate = re.sub(r"^[\u4e00-\u9fff]{0,4}(?:在|到|去|过|赶到|前往|先过|有家|得|必须|需要)", "", candidate)
            if candidate:
                aliases.append(candidate)
        if prefix and re.search(r"[\u4e00-\u9fff]", prefix):
            aliases.append(prefix)
        return PreferenceAgent._expanded_place_aliases(aliases)

    @staticmethod
    def _place_suffixes() -> list[str]:
        return [
            "\u8001\u6863\u53e3",
            "\u6863\u53e3",
            "\u53bf\u57ce",
            "\u8857\u9053",
            "\u7701",
            "\u5e02",
            "\u533a",
            "\u53bf",
            "\u9547",
            "\u6751",
            "\u4ed3\u5e93",
            "\u7801\u5934",
            "\u5382",
            "\u5e97",
        ]

    @staticmethod
    def _normalize_place_text(text: str) -> str:
        text = re.sub(r"\s+", "", str(text))
        text = re.sub(r"[（）()，,。；;:：]", "", text)
        for token in ("广东省", "广州市", "市", "区", "县城", "县", "镇", "街道", "老档口", "档口", "老"):
            text = text.replace(token, "")
        return text

    @staticmethod
    def _place_tokens(text: str) -> list[str]:
        tokens: list[str] = []
        cleaned = re.sub(r"[^\u4e00-\u9fff]", "", str(text))
        for suffix in ("区", "县城", "县", "市", "镇", "街道", "档口"):
            for match in re.finditer(rf"([\u4e00-\u9fff]{{2,8}}){suffix}", cleaned):
                token = match.group(1)
                if token:
                    tokens.append(token)
        norm = PreferenceAgent._normalize_place_text(cleaned)
        if len(norm) >= 2:
            tokens.append(norm)
        return list(dict.fromkeys(tokens))

    @staticmethod
    def _normalize_place_text(text: str) -> str:
        text = re.sub(r"\s+", "", str(text))
        text = re.sub(r"[\u3000\s,，.。;；:：()（）【】\[\]]", "", text)
        for token in ("\u5e7f\u4e1c\u7701", "\u5e7f\u5dde\u5e02", *PreferenceAgent._place_suffixes(), "\u8001"):
            text = text.replace(token, "")
        return text

    @staticmethod
    def _place_tokens(text: str) -> list[str]:
        tokens: list[str] = []
        cleaned = re.sub(r"[^\u4e00-\u9fff]", "", str(text))
        for suffix in PreferenceAgent._place_suffixes():
            for match in re.finditer(rf"([\u4e00-\u9fff]{{2,8}}){suffix}", cleaned):
                token = match.group(1)
                if token:
                    tokens.append(token)
        norm = PreferenceAgent._normalize_place_text(cleaned)
        if len(norm) >= 2:
            tokens.append(norm)
        return list(dict.fromkeys(tokens))

    @staticmethod
    def _extract_schemes(instructions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        schemes: list[dict[str, Any]] = []
        for inst in instructions:
            if not isinstance(inst, dict):
                continue
            scheme = inst.get("scheme")
            if not isinstance(scheme, dict):
                continue
            item = dict(scheme)
            if inst.get("id") is not None:
                item["preference_id"] = inst.get("id")
            if inst.get("content_key") is not None:
                item["content_key"] = inst.get("content_key")
            schemes.append(item)
        return schemes

    @staticmethod
    def _load_parsed_by_content(
        preference_memory: dict[str, Any],
        cached_instructions: Any,
    ) -> dict[str, dict[str, Any]]:
        stored = preference_memory.get("parsed_by_content")
        if isinstance(stored, dict) and stored:
            return {str(key): dict(value) for key, value in stored.items() if isinstance(value, dict)}
        rebuilt: dict[str, dict[str, Any]] = {}
        if not isinstance(cached_instructions, list):
            return rebuilt
        for item in cached_instructions:
            if not isinstance(item, dict):
                continue
            key = str(item.get("content_key") or item.get("rule") or "").strip()
            if not key:
                continue
            rebuilt[key] = {"assembled": dict(item)}
        return rebuilt

    @staticmethod
    def _instruction_content(inst: dict[str, Any]) -> str:
        return instruction_content_key(str(inst.get("content_key") or inst.get("rule") or ""))

    @classmethod
    def _remap_preference_progress(
        cls,
        old_instructions: list[dict[str, Any]],
        new_instructions: list[dict[str, Any]],
        progress: dict[str, Any],
        removed_contents: list[str],
    ) -> dict[str, Any]:
        old_id_to_content = {
            str(inst.get("id", "") or ""): cls._instruction_content(inst)
            for inst in old_instructions
            if str(inst.get("id", "") or "").strip()
        }
        content_to_new_id = {
            cls._instruction_content(inst): str(inst.get("id", "") or "")
            for inst in new_instructions
            if cls._instruction_content(inst) and str(inst.get("id", "") or "").strip()
        }
        removed = {instruction_content_key(text) for text in removed_contents}

        def remap_ids(values: Any) -> list[str]:
            remapped: list[str] = []
            for raw_id in cls._as_str_list(values):
                content = old_id_to_content.get(raw_id, "")
                if content in removed:
                    continue
                new_id = content_to_new_id.get(content)
                if new_id:
                    remapped.append(new_id)
            return sorted(set(remapped))

        updated = dict(progress)
        for field in ("completed_ids", "active_ids", "hidden_completed_ids", "kept_active_ids"):
            if field in updated:
                updated[field] = remap_ids(updated.get(field))

        statuses = updated.get("preference_statuses")
        if isinstance(statuses, list):
            new_statuses: list[dict[str, Any]] = []
            for item in statuses:
                if not isinstance(item, dict):
                    continue
                old_id = str(item.get("id", "") or "")
                content = old_id_to_content.get(old_id, "")
                if content in removed:
                    continue
                new_id = content_to_new_id.get(content)
                if not new_id:
                    continue
                copied = dict(item)
                copied["id"] = new_id
                new_statuses.append(copied)
            updated["preference_statuses"] = new_statuses

        missing = updated.get("missing_information")
        if isinstance(missing, dict):
            new_missing: dict[str, list[str]] = {}
            for old_id, items in missing.items():
                content = old_id_to_content.get(str(old_id), "")
                if content in removed:
                    continue
                new_id = content_to_new_id.get(content)
                if new_id and cls._as_str_list(items):
                    new_missing[new_id] = cls._as_str_list(items)
            updated["missing_information"] = new_missing

        return updated

    @staticmethod
    def _finalize_instruction(item: dict[str, Any], source_text: str, index: int) -> dict[str, Any]:
        finalized = dict(item)
        finalized.pop("_tier1", None)
        text = instruction_content_key(source_text or finalized.get("rule") or finalized.get("content_key") or "")
        if text:
            finalized["content_key"] = text
            finalized["rule"] = text
        if index > 0:
            finalized["id"] = f"pref_{index}"
        else:
            finalized.pop("id", None)
        if "parse_status" not in finalized:
            finalized["parse_status"] = resolve_parse_status(finalized)
        return PreferenceAgent._sanitize_instruction(finalized)

    @staticmethod
    def _fallback_single(text: str, index: int, pref_meta: dict[str, Any] | None) -> dict[str, Any]:
        key = instruction_content_key(text)
        inst = PreferenceAgent._sanitize_instruction({
            "schema_version": SCHEMA_VERSION,
            "parse_prompt_version": PARSE_PROMPT_VERSION,
            "content_key": key,
            "preference_type": "UNKNOWN",
            "category": "unknown",
            "hardness": "unknown",
            "persistent": True,
            "source_rule": text,
            "rule": text,
            "normalized_rule": text,
            "parse_status": "fallback",
            "uncertainty": "LLM 解析失败，已回退为原文约束",
            "routing": {
                "preference_type": "UNKNOWN",
                "cycle_kind": "always",
                "active_dates": None,
                "scope_actions": ["take_order"],
                "blocked_actions": [],
                "constraint_kinds": ["other"],
                "needs_sequence": False,
                "needs_history_aggregate": False,
            },
            "cycle": {"length": "always", "window": None, "active_dates": None, "count": None, "reset": "never", "evaluate_at": "per_action"},
            "scope": {"actions": ["take_order"], "phase": "candidate", "applies_when": "always_active"},
            "completion": {"mode": "never_expires", "track_progress": False, "progress_key": None, "expires_at": None},
            "checks": [],
            "on_fail": {"effect": "warn", "block_actions": []},
            "route_plan": None,
            "required_fields": ["raw_preference_text", "current_wall_clock_time"],
            "steps": [f"遵守原文偏好：{text}"],
            "completion_check": "无法自动判定",
            "guard_summary": text[:240],
            "scheme": {
                "scheme_version": "fixed_preference_scheme.v1",
                "type": "UNKNOWN",
                "hardness": "unknown",
                "source_rule": text,
                "filter": {"deterministic": False, "candidate_actions": ["take_order"], "effect": "warn"},
            },
            "meta": {"parse_tier": "fallback"},
        })
        return attach_visibility_meta(inst, pref_meta)

    @staticmethod
    def _canonical_preferences(items: list[Any]) -> list[str]:
        canonical: list[str] = []
        for item in items or []:
            if isinstance(item, dict):
                value = item.get("content", item.get("text", item.get("preference")))
            else:
                value = item
            if value is None:
                continue
            text = instruction_content_key(str(value))
            if text:
                canonical.append(text)
        return canonical

    @staticmethod
    def _as_str_list(value: Any) -> list[str]:
        if isinstance(value, str):
            return [value] if value.strip() else []
        if isinstance(value, list):
            return [str(item) for item in value if str(item).strip()]
        return []

    @staticmethod
    def _sanitize_instruction(instruction: dict[str, Any]) -> dict[str, Any]:
        sanitized = dict(instruction)
        sanitized.pop("source_metadata", None)
        fields = sanitized.get("required_fields")
        if isinstance(fields, list):
            sanitized["required_fields"] = [
                str(field)
                for field in fields
                if str(field).strip() and str(field).strip() != "preference_metadata"
            ]
        return sanitized
