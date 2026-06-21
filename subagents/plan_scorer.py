from __future__ import annotations

from ..domain.rules import DEFAULT_MONTH_HORIZON_MINUTES, day_index, distance_to_minutes, haversine_km, minute_of_day, pickup_minutes
from ..messages import TraceContext
from ..state_store import DecisionState, StateStore
from ..telemetry import Telemetry
from .scheme_handlers import DAILY_FIRST_ORDER_DEADLINE_HANDLER, MONTHLY_DEADHEAD_LIMIT_HANDLER, TARGET_CARGO_HANDLER

# 默认时段价值系数（无历史数据时使用）
_DEFAULT_HOUR_VALUE: dict[int, float] = {h: 0.7 for h in range(0, 6)}
_DEFAULT_HOUR_VALUE.update({h: 1.15 for h in range(8, 12)})
_DEFAULT_HOUR_VALUE.update({h: 1.10 for h in range(14, 18)})
_DEFAULT_HOUR_VALUE.update({h: 1.0 for h in range(6, 8)})
_DEFAULT_HOUR_VALUE.update({h: 0.95 for h in range(12, 14)})
_DEFAULT_HOUR_VALUE.update({h: 0.9 for h in range(18, 24)})


class PlanScorer:
    phase = "SCORE_PLANS"

    def __init__(self, store: StateStore, telemetry: Telemetry) -> None:
        self._store = store
        self._telemetry = telemetry

    def run(self, state: DecisionState, trace: TraceContext) -> DecisionState:
        self._telemetry.emit(trace, event="AGENT_STARTED", source="PlanScorer", phase=self.phase)
        ctx = state.driver_context
        if ctx is None:
            raise ValueError("missing driver_context")
        MONTHLY_DEADHEAD_LIMIT_HANDLER.apply_filter_to_state(state)
        DAILY_FIRST_ORDER_DEADLINE_HANDLER.apply_filter_to_state(state)
        visited_grids = self._build_visited_grids(state)
        for plan in state.simulated_plans:
            plan.score = self._score_plan(state, plan, visited_grids)
        state.ranked_plans = sorted(state.simulated_plans, key=lambda p: p.score, reverse=True)
        state.phase = self.phase
        self._store.checkpoint(state, "CKPT_SCORED_READY")
        self._telemetry.emit(
            trace,
            event="PLAN_SCORED",
            source="PlanScorer",
            phase=self.phase,
            simulation_minute=ctx.simulation_minute,
            checkpoint_id="CKPT_SCORED_READY",
            payload={"plan_count": len(state.ranked_plans), "top_score": round(state.ranked_plans[0].score, 2) if state.ranked_plans else None},
        )
        return state

    def _score_plan(self, state: DecisionState, plan, visited_grids: set[tuple[float, float]]) -> float:
        if not plan.valid:
            return -1_000_000.0
        future = plan.meta.get("future_feasibility") if isinstance(plan.meta, dict) else None
        if isinstance(future, dict):
            if future.get("blocked") or future.get("feasible") is False or str(future.get("risk_level", "")).lower() == "fatal":
                return -1_000_000.0
        ctx = state.driver_context
        if ctx is None:
            return -1_000_000.0
        if plan.action == "take_order":
            return self._score_take_order(state, plan, ctx, visited_grids) + self._future_preference_bonus(plan)
        if plan.action == "wait":
            return self._score_wait(state, plan, ctx) + self._future_preference_bonus(plan)
        if plan.action == "reposition":
            return self._score_reposition(state, plan, ctx, visited_grids) + self._future_preference_bonus(plan)
        # wait/reposition 不打分，由 PolicyAgent 自行决定
        return 0.0

    @staticmethod
    def _future_preference_bonus(plan) -> float:
        bonus = 0.0
        future = plan.meta.get("future_feasibility") if isinstance(plan.meta, dict) else None
        if isinstance(future, dict) and future.get("preferred"):
            bonus += 2_000.0
        if isinstance(plan.meta, dict) and plan.meta.get("preference_generated"):
            try:
                priority = float(plan.meta.get("priority", 0.5) or 0.5)
            except (TypeError, ValueError):
                priority = 0.5
            bonus += 1_000.0 + max(0.0, min(1.0, priority)) * 1_000.0
        return bonus

    def _score_take_order(self, state: DecisionState, plan, ctx, visited_grids: set[tuple[float, float]]) -> float:
        hours = max(plan.duration_minutes / 60.0, 0.25)
        pickup_penalty = float(plan.meta.get("pickup_km", 0.0)) * ctx.cost_per_km
        wait_penalty = float(plan.meta.get("wait_for_load", 0)) * 0.3

        # 每小时净收益
        net_per_hour = plan.net_income / hours
        plan.meta["net_income_per_hour"] = round(net_per_hour, 1)

        # Lookahead：接单完成后在目的地附近的期望收益

        # 基础分 = 净收入 + 效率加成
        base = plan.net_income + net_per_hour * 6.0

        # 时均收益门槛：低于阈值的单打折

        # 时段价值加成（基于历史数据的动态曲线）

        # 月度进度感知

        return base - pickup_penalty - wait_penalty

    def _target_cargo_bonus(self, state: DecisionState, plan) -> float:
        target = self._target_cargo_instruction_for_plan(state, plan)
        if not target:
            return 0.0
        inst, scheme = target
        penalty = self._preference_penalty(inst)
        plan.meta["preference_generated"] = True
        plan.meta["priority"] = max(float(plan.meta.get("priority", 0.0) or 0.0), 1.0)
        plan.meta["target_cargo_preference"] = {
            "preference_id": inst.get("id"),
            "target_cargo_id": plan.cargo_id,
            "penalty_amount": penalty,
            "source": "TARGET_CARGO_MUST_TAKE",
        }
        return max(5000.0, penalty + 2000.0)

    @staticmethod
    def _target_cargo_instruction_for_plan(state: DecisionState, plan) -> tuple[dict, dict] | None:
        return TARGET_CARGO_HANDLER.instruction_for_plan(state, plan)

    @staticmethod
    def _hidden_preference_ids(state: DecisionState) -> set[str]:
        progress = state.preference_progress if isinstance(state.preference_progress, dict) else {}
        hidden = {str(item) for item in progress.get("hidden_completed_ids", []) if str(item)}
        statuses = progress.get("preference_statuses")
        if isinstance(statuses, list):
            for item in statuses:
                if not isinstance(item, dict):
                    continue
                if str(item.get("status") or "") != "satisfied_hide":
                    continue
                pref_id = str(item.get("id") or "")
                if pref_id:
                    hidden.add(pref_id)
        return hidden

    @staticmethod
    def _preference_penalty(inst: dict) -> float:
        meta = inst.get("meta") if isinstance(inst.get("meta"), dict) else {}
        for key in ("penalty_amount", "penalty_cap"):
            try:
                value = float(meta.get(key, 0) or 0)
            except (TypeError, ValueError):
                continue
            if value > 0:
                return value
        return 0.0

    def _score_wait(self, state: DecisionState, plan, ctx) -> float:
        kind = plan.meta.get("kind")
        fatigue_bonus = self._fatigue_bonus(state, plan)

        # 基础分：疲劳度与等待价值
        base = fatigue_bonus

        # 等待价值：基于历史时段数据估算
        wait_value = self._estimate_wait_value(state, plan, ctx)
        base += wait_value

        # 有好单时不应等待
        has_good_order = any(
            p.action == "take_order" and p.valid and p.net_income > 80
            for p in state.simulated_plans
        )
        if has_good_order:
            base -= 100.0

        # 长时间等待惩罚
        base -= plan.duration_minutes * 0.05

        return base

    def _fatigue_bonus(self, state: DecisionState, plan) -> float:
        """疲劳度加成：连续工作时间越长，wait 奖励越高。"""
        short = self._store.short_memory(state.driver_id)
        ctx = state.driver_context
        recent = short.get_actions_within(ctx.simulation_minute, 1440) if ctx else list(short.recent_actions)
        if not recent:
            return 0.0
        # 计算最近连续工作时间（非 wait 动作的时间总和）
        continuous_work = 0
        for item in reversed(recent):
            act = item.get("action", {})
            if act.get("action") == "wait":
                break
            continuous_work += int(act.get("params", {}).get("duration_minutes", 30) or 30)
        if continuous_work > 8 * 60:
            return 80.0
        if continuous_work > 5 * 60:
            return 40.0
        if continuous_work > 3 * 60:
            return 15.0
        return 0.0

    def _score_reposition(self, state: DecisionState, plan, ctx, visited_grids: set[tuple[float, float]]) -> float:
        kind = plan.meta.get("kind")
        if kind in ("hotspot_reposition", "load_window_reposition"):
            value = float(plan.meta.get("hotspot_value", 0.0))
            dist = float(plan.meta.get("distance_km", 0.0))
            exploration_bonus = self._exploration_bonus(state, plan, visited_grids)
            base = value - dist * ctx.cost_per_km - plan.duration_minutes * 0.3 + exploration_bonus - self._recent_reposition_penalty(state, plan)

            # 位置价值：基于起点热点和密度数据的期望收入
            location_value = self._estimate_location_value(state, plan, ctx)
            base += location_value

            # reposition 后低收益惩罚
            failure_count = int(plan.meta.get("failure_count", 0) or 0)
            if failure_count > 0:
                base -= failure_count * 400.0

            # 时间窗口 reposition 额外奖励
            if kind == "load_window_reposition":
                base += 200.0

            # ROI 检查
            total_time_hours = max(plan.duration_minutes / 60.0, 0.5)
            roi = value / total_time_hours if total_time_hours > 0 else 0
            if roi < 15.0 and plan.duration_minutes > 60 and kind != "load_window_reposition":
                base -= 300.0
            return base
        return -100.0

    # ---- 区域探索：鼓励去新区域，惩罚原地打转 ----

    def _build_visited_grids(self, state: DecisionState) -> set[tuple[float, float]]:
        """一次性构建最近动作的 0.5 度网格集合。"""
        short = self._store.short_memory(state.driver_id)
        ctx = state.driver_context
        recent = short.get_actions_within(ctx.simulation_minute, 720) if ctx else list(short.recent_actions)[-6:]
        grids: set[tuple[float, float]] = set()
        for item in recent:
            act = item.get("action", {})
            params = act.get("params", {})
            try:
                lat = float(params.get("latitude", params.get("lat", 0)))
                lng = float(params.get("longitude", params.get("lng", 0)))
                if lat and lng:
                    grids.add((round(lat * 2) / 2, round(lng * 2) / 2))
            except (TypeError, ValueError):
                pass
        ctx = state.driver_context
        if ctx:
            grids.add((round(ctx.lat * 2) / 2, round(ctx.lng * 2) / 2))
        return grids

    def _exploration_bonus(self, state: DecisionState, plan, visited_grids: set[tuple[float, float]]) -> float:
        """如果目标区域是新区域（最近 5 步没去过），给奖励；如果去过，惩罚。"""
        if plan.target_lat is None or plan.target_lng is None:
            return 0.0
        if not visited_grids:
            return 50.0  # 首次行动，鼓励探索
        target_grid = (round(plan.target_lat * 2) / 2, round(plan.target_lng * 2) / 2)
        if target_grid in visited_grids:
            return -80.0  # 去过的区域，小惩罚
        return 120.0  # 新区域，奖励

    def _recent_reposition_penalty(self, state: DecisionState, plan) -> float:
        short = self._store.short_memory(state.driver_id)
        ctx = state.driver_context
        recent = short.get_actions_within(ctx.simulation_minute, 480) if ctx else list(short.recent_actions)[-4:]
        if not recent:
            return 0.0
        penalty = 0.0
        last_actions = [item.get("action", {}) for item in recent]
        if last_actions and last_actions[-1].get("action") == "reposition":
            penalty += 800.0
            params = last_actions[-1].get("params", {})
            try:
                if abs(float(params.get("latitude")) - float(plan.target_lat)) < 0.02 and abs(float(params.get("longitude")) - float(plan.target_lng)) < 0.02:
                    penalty += 2500.0
            except (TypeError, ValueError):
                pass
        if len(last_actions) >= 3 and all(a.get("action") == "reposition" for a in last_actions[-3:]):
            penalty += 6000.0
        return penalty

    def _monthly_income_adjustment(self, state: DecisionState, plan) -> float:
        """月度收益感知：已完成多单/高收入时，短途低风险单更有价值（边际收益递减）。"""
        ctx = state.driver_context
        if ctx is None:
            return 0.0
        completed = ctx.completed_order_count
        remaining = DEFAULT_MONTH_HORIZON_MINUTES - ctx.simulation_minute
        remaining_days = remaining / (24 * 60)
        if completed < 5 or remaining_days > 20:
            return 0.0
        efficiency = plan.net_income / max(plan.duration_minutes / 60.0, 0.25)
        if plan.duration_minutes <= 3 * 60 and efficiency > 30:
            return 150.0
        if plan.duration_minutes > 12 * 60:
            return -200.0
        return 0.0

    def _build_time_value_curve(self, state: DecisionState) -> dict[int, float]:
        """从历史订单构建时段价值曲线：每小时的平均时均收益。"""
        long = self._store.long_memory(state.driver_id)
        if not long.episodic_memory:
            return _DEFAULT_HOUR_VALUE

        hour_income: dict[int, list[float]] = {}
        for ep in long.episodic_memory:
            sim_minute = ep.get("simulation_minute")
            net_income = ep.get("net_income")
            duration = ep.get("duration_minutes")
            if sim_minute is None or net_income is None:
                continue
            hour = minute_of_day(int(sim_minute)) // 60
            hourly = float(net_income) / max(float(duration or 60) / 60.0, 0.25)
            hour_income.setdefault(hour, []).append(hourly)

        curve: dict[int, float] = {}
        for hour, rates in hour_income.items():
            curve[hour] = sum(rates) / len(rates)

        # 用默认值填充缺失时段
        for h in range(24):
            if h not in curve:
                curve[h] = _DEFAULT_HOUR_VALUE.get(h, 1.0)

        # 归一化：以平均值为基准
        avg = sum(curve.values()) / 24.0
        if avg > 0:
            curve = {h: v / avg for h, v in curve.items()}
        return curve

    def _estimate_wait_value(self, state: DecisionState, plan, ctx) -> float:
        """估算等待的价值：等到更好的时段能多赚多少。"""
        time_curve = self._build_time_value_curve(state)
        current_hour = minute_of_day(ctx.simulation_minute) // 60
        current_value = time_curve.get(current_hour, 1.0)

        # 等待结束后的小时
        end_minute = ctx.simulation_minute + plan.duration_minutes
        end_hour = minute_of_day(end_minute) // 60
        end_value = time_curve.get(end_hour, 1.0)

        # 等待收益 = 结束时段价值 - 当前时段价值
        value_gain = end_value - current_value

        # 如果当前有好单（hourly_rate > 60），等待的机会成本高
        good_orders = [
            p for p in state.simulated_plans
            if p.action == "take_order" and p.valid and p.net_income > 0
        ]
        if good_orders:
            best_rate = max(p.net_income / max(p.duration_minutes / 60.0, 0.25) for p in good_orders)
            if best_rate > 60:
                # 有好单时等待的机会成本
                opportunity_cost = best_rate * (plan.duration_minutes / 60.0) * 0.5
                return -opportunity_cost + value_gain * 100

        # 无好单时：等待的价值 = 时段提升价值 × 期望好单收入
        expected_income = 200.0  # 假设一个好单平均净收入 200
        return value_gain * expected_income * 0.3

    def _estimate_location_value(self, state: DecisionState, plan, ctx) -> float:
        """估算去某个位置的期望收入：基于起点热点和密度数据。"""
        if plan.target_lat is None or plan.target_lng is None:
            return 0.0

        long = self._store.long_memory(state.driver_id)
        value = 0.0

        # 起点热点：该位置附近历史收益
        for (lat, lng), weight in long.pickup_hotspots.items():
            dist = haversine_km(plan.target_lat, plan.target_lng, lat, lng)
            if dist <= 30:
                decay = (30 - dist) / 30.0
                value += weight * decay * 40.0

        # 货源密度：该位置附近的历史货源数量
        density_records = self._store.load_density_near(
            state.driver_id, plan.target_lat, plan.target_lng, radius_km=30.0,
        )
        for rec in density_records:
            d = haversine_km(plan.target_lat, plan.target_lng, rec["lat"], rec["lng"])
            if d <= 30:
                decay = (30 - d) / 30.0
                count = rec.get("cargo_count", 0)
                avg_income = rec.get("best_net_income")
                if avg_income and avg_income > 0:
                    value += count * decay * float(avg_income) * 0.05
                else:
                    value += count * decay * 10.0

        # 历史订单：该位置附近的成功订单收益
        for ep in long.episodic_memory[-30:]:
            try:
                tlat = float(ep.get("target_lat", 0) or 0)
                tlng = float(ep.get("target_lng", 0) or 0)
                income = float(ep.get("net_income", 0) or 0)
            except (TypeError, ValueError):
                continue
            if not tlat or not tlng:
                continue
            dist = haversine_km(plan.target_lat, plan.target_lng, tlat, tlng)
            if dist <= 50:
                decay = (50 - dist) / 50.0
                value += max(0, income) * decay * 0.1

        # 扣除移动成本
        travel_cost = float(plan.meta.get("distance_km", 0) or 0) * ctx.cost_per_km
        travel_hours = plan.duration_minutes / 60.0

        # 期望收入 - 移动成本 - 时间机会成本
        net_value = value - travel_cost - travel_hours * 30.0
        return max(net_value, 0.0)

    def _lookahead_value(self, state: DecisionState, plan) -> float:
        """Lookahead：估算接单完成后在目的地附近的期望收益。

        考虑三个因素：
        1. 当前可见货源中，离目的地近的有多少、质量如何
        2. 历史热点区域在目的地附近的收益水平
        3. 货源密度地图中目的地附近的密度
        """
        if plan.target_lat is None or plan.target_lng is None:
            return 0.0
        ctx = state.driver_context
        if ctx is None:
            return 0.0

        value = 0.0

        # 1. 当前可见货源中离目的地近的 → 估算"接完这单后附近有什么"
        nearby_cargo_value = 0.0
        nearby_count = 0
        for cand in state.cargo_snapshot:
            start = cand.cargo.get("start", {})
            try:
                dist = haversine_km(plan.target_lat, plan.target_lng, float(start["lat"]), float(start["lng"]))
            except (KeyError, TypeError, ValueError):
                continue
            if dist <= 20:
                weight = 1.0
            elif dist <= 40:
                weight = 0.6
            elif dist <= 60:
                weight = 0.3
            else:
                continue
            cargo_price = float(cand.cargo.get("price", 0) or 0)
            if cargo_price > 10_000:
                cargo_price /= 100.0
            nearby_cargo_value += weight * max(0, cargo_price)
            nearby_count += 1
        value += nearby_cargo_value * 0.03

        # 2. 历史热点 → 目的地附近是否有高价值区域
        long = self._store.long_memory(state.driver_id)
        for (lat, lng), hotness in list(long.hotspots.items())[-20:]:
            dist = haversine_km(plan.target_lat, plan.target_lng, float(lat), float(lng))
            if dist <= 100:
                decay = (100.0 - dist) / 100.0
                value += float(hotness) * decay * 30.0

        # 3. 货源密度 → 目的地附近的货源密度（从 pg 加载）
        density_records = self._store.load_density_near(state.driver_id, plan.target_lat, plan.target_lng, radius_km=50.0)
        for rec in density_records:
            d = haversine_km(plan.target_lat, plan.target_lng, rec["lat"], rec["lng"])
            if d <= 50:
                decay = (50.0 - d) / 50.0
                value += rec.get("cargo_count", 0) * decay * 15.0

        return min(value, 1500.0)

