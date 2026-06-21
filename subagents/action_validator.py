from __future__ import annotations

from ..messages import TraceContext
from ..state_store import DecisionState, StateStore
from ..telemetry import Telemetry


class AuditReject(Exception):
    def __init__(self, reason: str, rollback_to: str = "CKPT_SCORED_READY", *, blacklist_selected: bool = True) -> None:
        super().__init__(reason)
        self.reason = reason
        self.rollback_to = rollback_to
        self.blacklist_selected = blacklist_selected


class ActionValidator:
    phase = "VALIDATE_ACTION"

    def __init__(self, store: StateStore, telemetry: Telemetry) -> None:
        self._store = store
        self._telemetry = telemetry

    def run(self, state: DecisionState, trace: TraceContext) -> DecisionState:
        self._telemetry.emit(trace, event="AGENT_STARTED", source="ActionValidator", phase=self.phase)
        plan = state.selected_intent
        if plan is None:
            raise AuditReject("missing selected_intent", "CKPT_POLICY_READY", blacklist_selected=False)
        action = str(plan.action).strip().lower()
        params = dict(plan.params)
        if action == "take_order":
            cargo_id = str(params.get("cargo_id", "")).strip()
            visible = {c.cargo_id for c in state.cargo_snapshot}
            if not cargo_id or cargo_id not in visible:
                raise AuditReject(f"take_order cargo_id not visible: {cargo_id}", "CKPT_SCORED_READY")
            simulated = next((p for p in state.ranked_plans if p.cargo_id == cargo_id), None)
            if simulated is None:
                raise AuditReject("cargo_id has no simulated plan", "CKPT_SCORED_READY")
            if not simulated.valid:
                raise AuditReject(f"selected cargo invalid", "CKPT_SCORED_READY")
            final = {"action": "take_order", "params": {"cargo_id": cargo_id}}
        elif action == "reposition":
            try:
                lat = float(params["latitude"])
                lng = float(params["longitude"])
            except (KeyError, TypeError, ValueError) as exc:
                raise AuditReject(f"invalid reposition params: {params}", "CKPT_POLICY_READY", blacklist_selected=False) from exc
            if not (-90 <= lat <= 90 and -180 <= lng <= 180):
                raise AuditReject(f"invalid reposition coordinate: {lat},{lng}", "CKPT_POLICY_READY", blacklist_selected=False)
            final = {"action": "reposition", "params": {"latitude": lat, "longitude": lng}}
        elif action == "wait":
            try:
                minutes = int(params.get("duration_minutes", 0))
            except (TypeError, ValueError) as exc:
                raise AuditReject(f"invalid wait duration: {params.get('duration_minutes')}", "CKPT_POLICY_READY", blacklist_selected=False) from exc
            if minutes <= 0:
                raise AuditReject(f"invalid wait duration: {minutes}", "CKPT_POLICY_READY", blacklist_selected=False)
            if minutes > 1440:
                raise AuditReject("wait duration is unreasonably long", "CKPT_POLICY_READY", blacklist_selected=False)
            final = {"action": "wait", "params": {"duration_minutes": minutes}}
        else:
            raise AuditReject(f"unknown action: {action}", "CKPT_POLICY_READY", blacklist_selected=False)
        state.validated_action = final
        state.final_action = final
        state.phase = self.phase
        self._store.checkpoint(state, "CKPT_VALIDATED_READY")
        self._telemetry.emit(
            trace,
            event="ACTION_VALIDATED",
            source="ActionValidator",
            phase=self.phase,
            simulation_minute=state.driver_context.simulation_minute if state.driver_context else None,
            checkpoint_id="CKPT_VALIDATED_READY",
            payload=final,
        )
        self._telemetry.emit(
            trace,
            event="ACTION_AUDITED",
            source="ActionValidator",
            phase=self.phase,
            simulation_minute=state.driver_context.simulation_minute if state.driver_context else None,
            payload={"approved": True, "action": final},
        )
        return state
