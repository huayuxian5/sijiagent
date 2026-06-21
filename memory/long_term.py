from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DailySummary:
    """按天聚合数据，代码自动维护。"""
    day_index: int = 0
    orders_taken: int = 0
    total_income: float = 0.0
    rest_minutes: int = 0
    work_minutes: int = 0
    reposition_count: int = 0
    destinations: list[tuple[float, float]] = field(default_factory=list)


@dataclass
class LongTermMemory:
    # 按天存储的情节记忆：day_index → 该天的订单列表
    episodic_by_day: dict[int, list[dict[str, Any]]] = field(default_factory=dict)
    # 兼容：扁平列表（从 pg 加载时用）
    episodic_memory: list[dict[str, Any]] = field(default_factory=list)

    preference_memory: dict[str, Any] = field(default_factory=dict)
    preference_progress: dict[str, Any] = field(default_factory=dict)

    # 按天聚合数据
    daily_summaries: dict[int, DailySummary] = field(default_factory=dict)

    hotspots: dict[tuple[float, float], float] = field(default_factory=dict)
    pickup_hotspots: dict[tuple[float, float], float] = field(default_factory=dict)
    reposition_failures: dict[tuple[float, float], int] = field(default_factory=dict)

    def add_episode(self, event: dict[str, Any]) -> None:
        """添加情节记忆，同时按天索引。"""
        self.episodic_memory.append(event)
        sim_minute = event.get("simulation_minute")
        if sim_minute is not None:
            day = int(sim_minute) // 1440
            if day not in self.episodic_by_day:
                self.episodic_by_day[day] = []
            self.episodic_by_day[day].append(event)

    def get_episodes_today(self, sim_minute: int) -> list[dict[str, Any]]:
        """获取今天的订单。"""
        day = sim_minute // 1440
        return self.episodic_by_day.get(day, [])

    def get_episodes_in_days(self, sim_minute: int, days: int = 7) -> list[dict[str, Any]]:
        """获取最近 N 天的订单。"""
        current_day = sim_minute // 1440
        result = []
        for d in range(max(0, current_day - days), current_day + 1):
            result.extend(self.episodic_by_day.get(d, []))
        return result

    def get_daily_summary(self, sim_minute: int) -> DailySummary:
        """获取或创建今天的聚合数据。"""
        day = sim_minute // 1440
        if day not in self.daily_summaries:
            self.daily_summaries[day] = DailySummary(day_index=day)
        return self.daily_summaries[day]

    def update_daily_summary(self, sim_minute: int, action_type: str, duration_minutes: int = 0, income: float = 0.0, destination: tuple[float, float] | None = None) -> None:
        """更新按天聚合数据。"""
        summary = self.get_daily_summary(sim_minute)
        if action_type == "take_order":
            summary.orders_taken += 1
            summary.total_income += income
            summary.work_minutes += duration_minutes
            if destination:
                summary.destinations.append(destination)
        elif action_type == "wait":
            summary.rest_minutes += duration_minutes
        elif action_type == "reposition":
            summary.reposition_count += 1
            summary.work_minutes += duration_minutes
