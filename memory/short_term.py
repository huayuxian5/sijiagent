from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ShortTermMemory:
    recent_actions: list[dict[str, Any]] = field(default_factory=list)
    temporary_blacklist: set[str] = field(default_factory=set)

    def add_action(self, minute: int | None, action: dict[str, Any]) -> None:
        """添加动作记录。"""
        self.recent_actions.append({"minute": minute, "action": action})

    def get_actions_within(self, current_minute: int, window_minutes: int = 1440) -> list[dict[str, Any]]:
        """获取指定时间窗口内的动作（默认24小时）。"""
        cutoff = current_minute - window_minutes
        return [a for a in self.recent_actions if a.get("minute") is not None and a["minute"] >= cutoff]

    def prune(self, current_minute: int, keep_minutes: int = 2880) -> None:
        """清理超过 keep_minutes（默认48小时）的旧记录，防止内存膨胀。"""
        cutoff = current_minute - keep_minutes
        self.recent_actions = [a for a in self.recent_actions if a.get("minute") is None or a["minute"] >= cutoff]
