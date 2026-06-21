from __future__ import annotations

import logging
from typing import Any

from ..domain.models import CargoCandidate, DriverContext
from ..domain.rules import haversine_km
from ..gateway import GatewayLayer
from ..messages import TraceContext
from ..state_store import DecisionState, StateStore
from ..telemetry import Telemetry

logger = logging.getLogger("agent.cargo_querier")


class CargoQuerier:
    phase = "QUERY_CARGO"

    def __init__(self, gateway: GatewayLayer, store: StateStore, telemetry: Telemetry) -> None:
        self._gateway = gateway
        self._store = store
        self._telemetry = telemetry

    def _record_density(self, driver_id: str, lat: float, lng: float, snapshot: list[CargoCandidate], query_minute: int) -> None:
        """记录货源密度数据到 pg。"""
        if not snapshot:
            return
        grid_lat = round(lat, 1)
        grid_lng = round(lng, 1)
        distances = [c.pickup_distance_km for c in snapshot if c.pickup_distance_km > 0]
        avg_dist = sum(distances) / len(distances) if distances else None
        self._store.save_cargo_density(
            driver_id, grid_lat, grid_lng,
            cargo_count=len(snapshot),
            avg_pickup_km=avg_dist,
            best_net_income=None,
            query_minute=query_minute,
        )

    def run(self, state: DecisionState, trace: TraceContext) -> DecisionState:
        self._telemetry.emit(trace, event="AGENT_STARTED", source="CargoQuerier", phase=self.phase)
        if state.driver_context is None:
            raise ValueError("missing driver_context")
        ctx = state.driver_context
        before_minute = int(ctx.simulation_minute)
        resp = self._gateway.query_cargo(state.driver_id, ctx.lat, ctx.lng, trace, "CargoQuerier")
        items = resp.get("items", [])
        snapshot: list[CargoCandidate] = []
        blacklist = self._store.short_memory(state.driver_id).temporary_blacklist
        if isinstance(items, list):
            for item in items:
                cargo = item.get("cargo", {}) if isinstance(item, dict) else {}
                cargo_id = str(cargo.get("cargo_id", "")).strip()
                if not cargo_id or cargo_id in blacklist:
                    continue
                snapshot.append(CargoCandidate(cargo_id=cargo_id, cargo=cargo, pickup_distance_km=float(item.get("distance_km", 0.0) or 0.0)))

        # query_cargo consumes simulation time; refresh context for action simulation.
        refreshed = self._gateway.get_driver_status(state.driver_id, trace, "CargoQuerier")
        state.driver_context = DriverContext.from_status(state.driver_id, refreshed, fallback=ctx)
        self._record_query_scan_event(
            state,
            before_minute,
            int(state.driver_context.simulation_minute),
            lat=ctx.lat,
            lng=ctx.lng,
            items_count=len(items) if isinstance(items, list) else 0,
            source="CargoQuerier",
        )
        state.cargo_snapshot = snapshot
        state.phase = self.phase

        # 记录货源密度数据
        self._record_density(state.driver_id, ctx.lat, ctx.lng, snapshot, state.driver_context.simulation_minute)
        self._store.checkpoint(state, "CKPT_CARGO_READY")
        self._telemetry.emit(
            trace,
            event="CARGO_QUERIED",
            source="CargoQuerier",
            phase=self.phase,
            simulation_minute=state.driver_context.simulation_minute,
            checkpoint_id="CKPT_CARGO_READY",
            payload={"items_count": len(snapshot)},
        )
        return state

    def run_extra(self, state: DecisionState, trace: TraceContext) -> DecisionState:
        """执行 QueryAgent 指定的额外位置查询。"""
        extra_targets = state.extra_query_targets
        if not extra_targets:
            return state

        ctx = state.driver_context
        if ctx is None:
            return state

        existing_ids = {c.cargo_id for c in state.cargo_snapshot}
        added = 0
        queried_remote = False
        for target in extra_targets:
            lat = target.get("lat")
            lng = target.get("lng")
            target_cargo_id = str(target.get("cargo_id", "") or "").strip()
            if target_cargo_id:
                before = len(state.cargo_snapshot)
                candidate = self._gateway.query_service.query_cargo_id(target_cargo_id, state, trace, "CargoQuerier")
                if candidate is not None:
                    existing_ids.add(candidate.cargo_id)
                    added += max(0, len(state.cargo_snapshot) - before)
                    continue
            if lat is None or lng is None:
                continue
            try:
                before_minute = int(state.driver_context.simulation_minute if state.driver_context else ctx.simulation_minute)
                resp = self._gateway.query_cargo(
                    state.driver_id, float(lat), float(lng), trace, "CargoQuerier",
                )
                queried_remote = True
                refreshed = self._gateway.get_driver_status(state.driver_id, trace, "CargoQuerier")
                state.driver_context = DriverContext.from_status(state.driver_id, refreshed, fallback=state.driver_context or ctx)
                self._record_query_scan_event(
                    state,
                    before_minute,
                    int(state.driver_context.simulation_minute),
                    lat=float(lat),
                    lng=float(lng),
                    items_count=len(resp.get("items", [])) if isinstance(resp.get("items", []), list) else 0,
                    source="CargoQuerier.run_extra",
                )
            except Exception as exc:
                logger.warning("Extra query failed at (%s,%s): %s", lat, lng, exc)
                continue

            items = resp.get("items", [])
            if not isinstance(items, list):
                continue

            origin = {"lat": lat, "lng": lng, "reason": target.get("reason", "")}
            for item in items:
                cargo = item.get("cargo", {}) if isinstance(item, dict) else {}
                cargo_id = str(cargo.get("cargo_id", "")).strip()
                if not cargo_id or cargo_id in existing_ids:
                    continue
                blacklist = self._store.short_memory(state.driver_id).temporary_blacklist
                if cargo_id in blacklist:
                    continue
                pickup_km = self._actual_pickup_distance(ctx, item)
                state.cargo_snapshot.append(
                    CargoCandidate(
                        cargo_id=cargo_id,
                        cargo=cargo,
                        pickup_distance_km=pickup_km,
                        extra=True,
                        query_origin=origin,
                    )
                )
                existing_ids.add(cargo_id)
                added += 1

        if added > 0 or queried_remote:
            refreshed = self._gateway.get_driver_status(state.driver_id, trace, "CargoQuerier")
            state.driver_context = DriverContext.from_status(state.driver_id, refreshed, fallback=ctx)

        self._telemetry.emit(
            trace,
            event="EXTRA_CARGO_QUERIED",
            source="CargoQuerier",
            phase=self.phase,
            simulation_minute=state.driver_context.simulation_minute if state.driver_context else None,
            payload={"targets": len(extra_targets), "added": added},
        )
        return state

    @staticmethod
    def _actual_pickup_distance(ctx: DriverContext, item: dict[str, Any]) -> float:
        cargo = item.get("cargo", {}) if isinstance(item, dict) else {}
        start = cargo.get("start", {}) if isinstance(cargo, dict) else {}
        if isinstance(start, dict):
            try:
                return haversine_km(ctx.lat, ctx.lng, float(start["lat"]), float(start["lng"]))
            except (KeyError, TypeError, ValueError):
                pass
        try:
            return float(item.get("distance_km", 0.0) or 0.0)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _record_query_scan_event(
        state: DecisionState,
        start_minute: int,
        end_minute: int,
        *,
        lat: float,
        lng: float,
        items_count: int,
        source: str,
    ) -> None:
        duration = max(0, int(end_minute) - int(start_minute))
        if duration <= 0:
            return
        event = {
            "kind": "query_scan",
            "source": source,
            "start": int(start_minute),
            "end": int(end_minute),
            "duration_minutes": duration,
            "lat": round(float(lat), 5),
            "lng": round(float(lng), 5),
            "items_count": int(items_count),
        }
        state.query_scan_events.append(event)
        state.query_scan_minutes += duration
