from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, TypedDict


@dataclass
class DriverContext:
    driver_id: str
    simulation_minute: int
    simulation_wall_time: str
    lat: float
    lng: float
    cost_per_km: float = 1.5
    truck_length: str = ""
    completed_order_count: int = 0
    preferences_text: list[str] = field(default_factory=list)
    preferences_raw: list[Any] = field(default_factory=list)
    preference_visibility: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_status(
        cls,
        driver_id: str,
        status: dict[str, Any],
        fallback: DriverContext | None = None,
        preferences_text: list[str] | None = None,
        preferences_raw: list[Any] | None = None,
        preference_visibility: list[dict[str, Any]] | None = None,
    ) -> DriverContext:
        fb = fallback or cls(driver_id=driver_id, simulation_minute=0, simulation_wall_time="", lat=0.0, lng=0.0)
        if preferences_raw is None:
            preferences_raw = _extract_preferences_raw(status) if "preferences" in status else copy.deepcopy(fb.preferences_raw)
        if preferences_text is None:
            preferences_text = _preferences_text_from_items(preferences_raw) if "preferences" in status else list(fb.preferences_text)
        if preference_visibility is None:
            raw_visibility = status.get("preference_visibility")
            if isinstance(raw_visibility, list):
                preference_visibility = [dict(item) for item in raw_visibility if isinstance(item, dict)]
            else:
                preference_visibility = list(fb.preference_visibility)
        return cls(
            driver_id=driver_id,
            simulation_minute=int(status.get("simulation_progress_minutes", fb.simulation_minute)),
            simulation_wall_time=str(status.get("simulation_wall_time", fb.simulation_wall_time)),
            lat=float(status.get("current_lat", fb.lat)),
            lng=float(status.get("current_lng", fb.lng)),
            cost_per_km=float(status.get("cost_per_km", fb.cost_per_km) or fb.cost_per_km),
            truck_length=str(status.get("truck_length", fb.truck_length)),
            completed_order_count=int(status.get("completed_order_count", fb.completed_order_count) or 0),
            preferences_text=preferences_text,
            preferences_raw=copy.deepcopy(preferences_raw),
            preference_visibility=preference_visibility,
        )


class TimelineEntry(TypedDict, total=False):
    kind: str
    start: int
    end: int


class CargoMeta(TypedDict, total=False):
    cargo_name: str | None
    price: float
    cost_per_km: float
    pickup_km: float
    haul_km: float
    pickup_minutes: int
    wait_for_load: int
    load_window: tuple[int, int] | None
    finish_minute: int
    remaining_month_minutes: int
    start_point: dict[str, Any]
    end_point: dict[str, Any]
    target_point: dict[str, Any]
    timeline: list[TimelineEntry]
    kind: str
    preference: dict[str, Any]
    policy_generated: bool
    llm_preference_plan: dict[str, Any]


class RepositionMeta(TypedDict, total=False):
    kind: str
    hotspot_value: float
    nearby_cargo_count: int
    distance_km: float
    target_point: dict[str, Any]
    timeline: list[TimelineEntry]
    constraint_id: str | None
    constraint_type: str
    policy_generated: bool


def _extract_preferences_raw(status: dict) -> list[Any]:
    raw_preferences = status.get("preferences", [])
    if raw_preferences is None:
        return []
    if isinstance(raw_preferences, (str, dict)):
        items = [raw_preferences]
    else:
        try:
            items = list(raw_preferences)
        except TypeError:
            items = [raw_preferences]
    return [copy.deepcopy(item) for item in items if item is not None]


def _preferences_text_from_items(items: list[Any]) -> list[str]:
    texts: list[str] = []
    for item in items:
        if isinstance(item, dict):
            value = item.get("content", item.get("text", item.get("preference")))
        else:
            value = item
        if value is None:
            continue
        text = str(value).strip()
        if text:
            texts.append(text)
    return texts


def _extract_preferences_text(status: dict) -> list[str]:
    return _preferences_text_from_items(_extract_preferences_raw(status))


@dataclass
class PreferenceRule:
    type: str
    params: dict[str, Any]
    raw: str


@dataclass
class PreferenceState:
    rules: list[PreferenceRule] = field(default_factory=list)
    contract: dict[str, Any] = field(default_factory=lambda: {"constraints": []})
    progress: dict[str, Any] = field(default_factory=dict)
    source_hash: str = ""

    def by_type(self, rule_type: str) -> list[PreferenceRule]:
        return [r for r in self.rules if r.type == rule_type]

    @property
    def constraints(self) -> list[dict[str, Any]]:
        raw = self.contract.get("constraints", []) if isinstance(self.contract, dict) else []
        return [item for item in raw if isinstance(item, dict)]


@dataclass
class CargoCandidate:
    cargo_id: str
    cargo: dict[str, Any]
    pickup_distance_km: float
    extra: bool = False
    query_origin: dict[str, Any] | None = None


@dataclass
class PreferencePlan:
    """偏好动态规划结果，由 PreferencePlanner 每轮更新。"""
    debt_accounts: list[dict[str, Any]] = field(default_factory=list)
    opportunity_costs: dict[str, float] = field(default_factory=dict)
    active_soft_preferences: list[str] = field(default_factory=list)
    dormant_hard_preferences: list[str] = field(default_factory=list)
    recommended_actions: list[str] = field(default_factory=list)


@dataclass
class ActionPlan:
    action: str
    params: dict[str, Any]
    score: float = 0.0
    reason: str = ""
    cargo_id: str | None = None
    valid: bool = True
    finish_minute: int | None = None
    net_income: float = 0.0
    duration_minutes: int = 0
    target_lat: float | None = None
    target_lng: float | None = None
    meta: dict[str, Any] = field(default_factory=dict)
