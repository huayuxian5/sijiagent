from __future__ import annotations

from ..domain.models import ActionPlan
from ..domain.rules import (
    DEFAULT_MONTH_HORIZON_MINUTES,
    haversine_km,
    load_window_minutes,
    pickup_minutes,
    wall_time_to_minute,
)
from ..messages import TraceContext
from ..state_store import DecisionState, StateStore
from ..telemetry import Telemetry
from ..utils import price_yuan as _price_yuan


class PlanSimulator:
    phase = "SIMULATE_PLANS"

    def __init__(self, store: StateStore, telemetry: Telemetry) -> None:
        self._store = store
        self._telemetry = telemetry

    def run(self, state: DecisionState, trace: TraceContext) -> DecisionState:
        self._telemetry.emit(trace, event="AGENT_STARTED", source="PlanSimulator", phase=self.phase)
        ctx = state.driver_context
        if ctx is None:
            raise ValueError("missing driver_context")

        plans: list[ActionPlan] = []
        query_timeline = self._query_scan_timeline(state)
        for cand in state.cargo_snapshot:
            try:
                plans.append(
                    self._simulate_cargo(
                        cand.cargo_id,
                        cand.cargo,
                        cand.pickup_distance_km,
                        ctx.simulation_minute,
                        ctx.cost_per_km,
                        ctx.truck_length,
                        query_timeline=query_timeline,
                        query_scan_minutes=state.query_scan_minutes,
                    )
                )
            except Exception as exc:
                self._telemetry.emit(
                    trace,
                    event="CARGO_SIMULATION_FAILED",
                    source="PlanSimulator",
                    phase=self.phase,
                    severity="WARN",
                    simulation_minute=ctx.simulation_minute,
                    payload={"cargo_id": cand.cargo_id, "error": str(exc)},
                )

        state.simulated_plans = plans
        state.phase = self.phase
        self._store.checkpoint(state, "CKPT_SIMULATION_READY")
        self._telemetry.emit(
            trace,
            event="SIMULATION_COMPLETED",
            source="PlanSimulator",
            phase=self.phase,
            simulation_minute=ctx.simulation_minute,
            checkpoint_id="CKPT_SIMULATION_READY",
            payload={"plan_count": len(plans), "query_scan_minutes": state.query_scan_minutes},
        )
        return state

    def _simulate_cargo(
        self,
        cargo_id: str,
        cargo: dict,
        pickup_km: float,
        t0: int,
        cost_per_km: float,
        truck_length: str = "",
        *,
        query_timeline: list[dict] | None = None,
        query_scan_minutes: int = 0,
    ) -> ActionPlan:
        start = cargo.get("start", {})
        end = cargo.get("end", {})
        start_lat = float(start["lat"])
        start_lng = float(start["lng"])
        end_lat = float(end["lat"])
        end_lng = float(end["lng"])

        pmin = pickup_minutes(pickup_km)
        arrival = t0 + pmin
        window = load_window_minutes(cargo)
        wait_for_load = 0
        valid = True

        create_time = cargo.get("create_time")
        remove_time = cargo.get("remove_time")
        if create_time:
            try:
                if t0 < wall_time_to_minute(str(create_time)):
                    valid = False
            except ValueError:
                pass
        if remove_time:
            try:
                if t0 > wall_time_to_minute(str(remove_time)):
                    valid = False
            except ValueError:
                pass

        truck_lengths = cargo.get("truck_length")
        if truck_length and isinstance(truck_lengths, list) and truck_lengths and truck_length not in [str(x) for x in truck_lengths]:
            valid = False

        if window:
            load_start, load_end = window
            if arrival > load_end:
                valid = False
            elif arrival < load_start:
                wait_for_load = load_start - arrival

        line_minutes = int(cargo.get("cost_time_minutes", 0) or 0)
        finish = arrival + wait_for_load + line_minutes
        haul_km = haversine_km(start_lat, start_lng, end_lat, end_lng)
        price = self._price_yuan(cargo)
        cost = cost_per_km * (pickup_km + haul_km)
        net = price - cost

        if finish > DEFAULT_MONTH_HORIZON_MINUTES:
            valid = False
        duration = max(1, finish - t0)
        if DEFAULT_MONTH_HORIZON_MINUTES - t0 < 24 * 60 and duration > DEFAULT_MONTH_HORIZON_MINUTES - t0:
            valid = False

        timeline = list(query_timeline or [])
        timeline.extend(
            [
                {"kind": "take_order", "start": t0, "end": t0 + 1},
                {"kind": "deadhead", "start": t0, "end": arrival},
                {"kind": "wait_load", "start": arrival, "end": arrival + wait_for_load},
                {"kind": "haul", "start": arrival + wait_for_load, "end": finish},
            ]
        )

        return ActionPlan(
            "take_order",
            {"cargo_id": cargo_id},
            cargo_id=cargo_id,
            valid=valid,
            finish_minute=finish,
            net_income=net,
            duration_minutes=duration,
            target_lat=end_lat,
            target_lng=end_lng,
            meta={
                "cargo_name": cargo.get("cargo_name"),
                "price": price,
                "cost_per_km": cost_per_km,
                "pickup_km": pickup_km,
                "haul_km": haul_km,
                "pickup_minutes": pmin,
                "wait_for_load": wait_for_load,
                "load_window": window,
                "finish_minute": finish,
                "remaining_month_minutes": DEFAULT_MONTH_HORIZON_MINUTES - t0,
                "query_scan_minutes": int(query_scan_minutes),
                "start_point": {"lat": start_lat, "lng": start_lng, "city": start.get("city")},
                "end_point": {"lat": end_lat, "lng": end_lng, "city": end.get("city")},
                "target_point": {"lat": end_lat, "lng": end_lng},
                "timeline": timeline,
            },
            reason="simulated cargo order",
        )

    @staticmethod
    def _price_yuan(cargo: dict) -> float:
        price = float(cargo.get("price", 0.0) or 0.0)
        return _price_yuan(price)

    @staticmethod
    def _query_scan_timeline(state: DecisionState) -> list[dict]:
        timeline: list[dict] = []
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
            timeline.append(
                {
                    "kind": "query_scan",
                    "start": start,
                    "end": end,
                    "duration_minutes": end - start,
                    "items_count": event.get("items_count"),
                    "source": event.get("source"),
                    "location": {"lat": event.get("lat"), "lng": event.get("lng")},
                }
            )
        return timeline
