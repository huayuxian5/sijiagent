"""Map simplified checks to runtime field paths (for validation/logging only)."""
from __future__ import annotations

from typing import Any

# action + phase + measure -> internal runtime hint (not used for hardcoded rules)
CHECK_RUNTIME_HINTS: dict[tuple[str, str, str], str] = {
    ("take_order", "to_pickup", "distance"): "dist_to_pickup_km",
    ("take_order", "haul", "distance"): "haul_distance_km",
    ("take_order", "cargo", "cargo_name"): "cargo.cargo_name",
    ("take_order", "whole", "city"): "cargo.start.city|cargo.end.city",
    ("take_order", "at_pickup", "city"): "cargo.start.city",
    ("take_order", "at_delivery", "city"): "cargo.end.city",
    ("reposition", "whole", "distance"): "distance_km",
    ("reposition", "moving", "distance"): "distance_km",
    ("reposition", "whole", "city"): "target_geofence",
    ("wait", "staying", "duration"): "duration_minutes",
    ("wait", "whole", "duration"): "duration_minutes",
    ("wait", "staying", "location"): "location.lat/lng",
}


def runtime_hint(check: dict[str, Any]) -> str | None:
    if not isinstance(check, dict):
        return None
    key = (
        str(check.get("action") or ""),
        str(check.get("phase") or ""),
        str(check.get("measure") or ""),
    )
    return CHECK_RUNTIME_HINTS.get(key)
