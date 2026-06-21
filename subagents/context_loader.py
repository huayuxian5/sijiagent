from __future__ import annotations

from ..domain.models import DriverContext, _extract_preferences_raw, _extract_preferences_text
from ..gateway import GatewayLayer
from ..messages import TraceContext
from ..state_store import DecisionState, StateStore
from ..telemetry import Telemetry


def extract_preferences_text(status: dict) -> list[str]:
    return _extract_preferences_text(status)


def extract_preferences_raw(status: dict) -> list:
    return _extract_preferences_raw(status)


class ContextLoader:
    phase = "LOAD_CONTEXT"

    def __init__(self, gateway: GatewayLayer, store: StateStore, telemetry: Telemetry) -> None:
        self._gateway = gateway
        self._store = store
        self._telemetry = telemetry

    def run(self, state: DecisionState, trace: TraceContext) -> DecisionState:
        self._telemetry.emit(trace, event="AGENT_STARTED", source="ContextLoader", phase=self.phase)
        status = self._gateway.get_driver_status(state.driver_id, trace, "ContextLoader")
        ctx = DriverContext.from_status(state.driver_id, status)
        state.driver_context = ctx
        state.phase = self.phase
        if ctx.preferences_text:
            long = self._store.long_memory(state.driver_id)
            long.preference_memory["observed_preferences"] = list(ctx.preferences_raw or ctx.preferences_text)
            self._store.save_preference(
                state.driver_id,
                "observed_preferences",
                list(ctx.preferences_raw or ctx.preferences_text),
            )
        self._store.checkpoint(state, "CKPT_CONTEXT_READY")
        self._telemetry.emit(
            trace,
            event="DRIVER_CONTEXT_READY",
            source="ContextLoader",
            phase=self.phase,
            simulation_minute=ctx.simulation_minute,
            checkpoint_id="CKPT_CONTEXT_READY",
            payload={"lat": ctx.lat, "lng": ctx.lng, "preferences_count": len(ctx.preferences_text)},
        )
        return state
