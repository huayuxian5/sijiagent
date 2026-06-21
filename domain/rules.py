from __future__ import annotations

import functools
import math
from datetime import datetime
from typing import Any

SIMULATION_EPOCH = datetime(2026, 3, 1, 0, 0, 0)
DEFAULT_SPEED_KM_PER_HOUR = 60.0
DEFAULT_COST_PER_KM = 1.5
DEFAULT_MONTH_HORIZON_MINUTES = 31 * 24 * 60


@functools.lru_cache(maxsize=4096)
def haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    radius_km = 6371.0
    p1 = math.radians(lat1)
    l1 = math.radians(lng1)
    p2 = math.radians(lat2)
    l2 = math.radians(lng2)
    dp = p2 - p1
    dl = l2 - l1
    h = math.sin(dp * 0.5) ** 2 + math.cos(p1) * math.cos(p2) * (math.sin(dl * 0.5) ** 2)
    h = min(1.0, max(0.0, h))
    return 2.0 * radius_km * math.asin(math.sqrt(h))


def distance_to_minutes(distance_km: float, speed_km_per_hour: float = DEFAULT_SPEED_KM_PER_HOUR) -> int:
    if distance_km <= 0:
        return 1
    return max(1, math.ceil(distance_km / speed_km_per_hour * 60))


def pickup_minutes(distance_km: float, speed_km_per_hour: float = DEFAULT_SPEED_KM_PER_HOUR) -> int:
    if distance_km <= 1e-6:
        return 0
    return distance_to_minutes(distance_km, speed_km_per_hour)


def wall_time_to_minute(text: str) -> int:
    dt = datetime.strptime(text.strip(), "%Y-%m-%d %H:%M:%S")
    return int((dt - SIMULATION_EPOCH).total_seconds() // 60)


def minute_of_day(simulation_minute: int) -> int:
    return int(simulation_minute) % 1440


def day_index(simulation_minute: int) -> int:
    return int(simulation_minute) // 1440


def parse_hhmm(text: str) -> int:
    h, m = text.split(":", 1)
    return int(h) * 60 + int(m)


def minutes_until_time(simulation_minute: int, target_hhmm: str) -> int:
    now = minute_of_day(simulation_minute)
    target = parse_hhmm(target_hhmm)
    delta = target - now
    if delta <= 0:
        delta += 1440
    return delta


def is_in_daily_window(simulation_minute: int, start_hhmm: str, end_hhmm: str) -> bool:
    now = minute_of_day(simulation_minute)
    start = parse_hhmm(start_hhmm)
    end = parse_hhmm(end_hhmm)
    if start <= end:
        return start <= now < end
    return now >= start or now < end


def load_window_minutes(cargo: dict[str, Any]) -> tuple[int, int] | None:
    raw = cargo.get("load_time")
    if not raw:
        return None
    if not isinstance(raw, list) or len(raw) != 2:
        return None
    return wall_time_to_minute(str(raw[0])), wall_time_to_minute(str(raw[1]))
