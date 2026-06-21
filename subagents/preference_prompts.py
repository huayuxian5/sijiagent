"""LLM prompts for preference parse v2 (tier1 / tier2a / tier2b)."""

from __future__ import annotations

TIER1_SYSTEM = (
    "你是偏好语义解析模块。"
    "对象是在约一个自然月内反复决策的卡车司机；"
    "每步只能在 take_order（接单）、wait（休息）、reposition（空驶迁址）三种动作中选一种。"
    "你只负责把偏好原文路由到固定 preference_type 模板并输出 routing；不要自由创造规则类型，不要输出 checks 或 cycle 细节。"
    "只输出一个合法 JSON 对象，不要 markdown，不要解释。"
)

TIER1_TEMPLATE = """将以下司机偏好原文解析为语义摘要与路由提示。

【偏好原文】
{preference_text}

【时间口径（与比赛一致）】
- 墙钟纪元：2026-03-01 00:00:00
- 自然日：[YYYY-MM-DD 00:00:00, 次日 00:00:00)
- 「三月/自然月」= 2026-03-01 00:00 至 2026-03-31 24:00
- 「3月X日」= 2026-03-XX 整个自然日
- 「每天/每自然日」= 每个自然日分别判定，非 rolling 24h
- 「连续休息 N 小时」= 同一自然日内最长一段连续 wait ≥ N 小时
- 「整天休息/全天不动」= 某一自然日 active_minutes==0，不是单次 wait≥1440

【解析原则】
- 口语：休息/睡觉/停着 → wait；不接货/不跑/别进去 → 禁止 take_order 和/或 reposition
- 只依据原文；未写明的日期/城市/阈值不要推断
- 「X月Y日/Z号」映射到 2026-03 的 calendar date（如三月四号 → 2026-03-04）
- 若原文写「四号五号」等指定日，routing.active_dates 列出这些日期；这与偏好何时首次可见无关

【固定 preference_type，只能选一个】
- ACTION_FORBID：禁止某类动作/货物/城市/路线，如不接机械设备、不接惠州、指定日不进深圳
- NUMERIC_LIMIT：数值上限/下限，如接单赴装货空驶≤55km
- TIME_WINDOW_STATIONARY：固定时段必须静止/休息，如每日00:00-06:00停车
- DAILY_CONTINUOUS_REST：每自然日连续 wait 至少N分钟/小时
- OFF_DAY_QUOTA：自然月内至少N个全天静止/保养/休息日
- ORDER_QUOTA：自然月内接够N单/在某地装卸够N个不同自然日
- GEOFENCE_STAY_WITHIN：地理围栏硬约束，车辆当前位置、停车、空驶目标、接单起点和终点必须始终在明确边界内，如深圳经纬度范围内/不出市
- GEOFENCE_FORBIDDEN_AREA：禁入地理围栏硬约束，车辆当前位置、停车、空驶目标、接单起点和终点不得进入明确圆区/禁区，如禁入某坐标半径20km
- MONTHLY_DEADHEAD_LIMIT：自然月累计空驶赶路里程上限，统计接单去装货点空驶和 reposition 空驶，如月度空驶≤100km
- LOCATION_ARRIVAL_DEADLINE：指定日期某时间前到达某地
- LOCATION_STAY_ON_DATE：指定日期在某地停留/等待N分钟
- ROUTE_SEQUENCE_ON_DATE：指定日期多地点顺序任务，如先增城再四会并停留
- PREFERENCE_SCORE：软偏好评分，不应直接 block
- UNKNOWN：无法映射到以上模板

【输出字段】
- normalized_rule：1-2句可执行语义，去掉口语
- preference_type：必须从固定 preference_type 中选择，不要输出其他值
- category：时段禁行|单日/指定日事件|城市/路线回避|货源类型/品类约束|距离/空驶约束|频次/次数约束|连续驾驶/休息|其他|unknown
- hardness：hard=违反应禁止 | soft=应尽量满足 | unknown=无法判断
- uncertainty：缺信息则说明，否则 ""
- routing：
  - cycle_kind：always|day|month|once（指定某几天用 once，不是 day）
  - active_dates：["2026-03-04", ...] 或 null（仅 once 且原文有明确日期时填写）
  - scope_actions：主要涉及的动作列表
  - blocked_actions：违反时应禁止的动作（可为空）
  - constraint_kinds：从 time_window|time_duration|cargo_filter|target_cargo|geofence|city_filter|distance|off_days|route_sequence|location_wait|other 选
  - needs_sequence：true 仅当有多地点顺序/先后到达
  - needs_history_aggregate：true 仅当需统计自然月内累计次数/里程/天数
  - count_min：整数或 null（needs_history_aggregate=true 时填，如至少4天填4）
  - distinct_by：calendar_day|none|null

【cycle_kind 判定（重要）】
- always：每单/每次都检查——禁某货、禁某城、空驶上限、「一律不接」
- day：每个自然日重复——每日连续休息、固定时段休息
- month：三月内累计/至少N个不同自然日——某城市累计天数、整天休息、整天保养
- once：仅指定日期生效——指定日期禁行、指定日期事务、指定日期到访
- 原文虽写「三月内」但语义是「凡是…都不接」→ still always，不是 month

【cargo_name】
- 赛题货源使用大类名；原文举例若指向某个货源大类，checks 层用 cargo_name 大类即可

【uncertainty】
- 「至少N个整天但未指定哪几天」不算歧义，uncertainty 留空，用 month+count_min 表达

【首单开工 deadline】
- “首单/第一单/当天第一笔 + 开工/接单/开始 + 不晚于/必须早于某时刻” 路由到 DAILY_FIRST_ORDER_DEADLINE。
- DAILY_FIRST_ORDER_DEADLINE 是每日条件约束：当天不接单不违规；当天若接单，只检查第一笔 take_order 的 action_start 是否早于 deadline。
- 不要把首单开工 deadline 路由到 LOCATION_ARRIVAL_DEADLINE；它没有目标地点。

输出 JSON：
{{
  "normalized_rule": "...",
  "preference_type": "ACTION_FORBID|NUMERIC_LIMIT|TIME_WINDOW_STATIONARY|DAILY_CONTINUOUS_REST|OFF_DAY_QUOTA|ORDER_QUOTA|TARGET_CARGO_MUST_TAKE|GEOFENCE_STAY_WITHIN|GEOFENCE_FORBIDDEN_AREA|MONTHLY_DEADHEAD_LIMIT|DAILY_FIRST_ORDER_DEADLINE|LOCATION_ARRIVAL_DEADLINE|LOCATION_STAY_ON_DATE|ROUTE_SEQUENCE_ON_DATE|PREFERENCE_SCORE|UNKNOWN",
  "category": "...",
  "hardness": "hard|soft|unknown",
  "uncertainty": "",
  "routing": {{
    "cycle_kind": "always|day|month|once",
    "active_dates": null,
    "scope_actions": ["take_order"],
    "blocked_actions": [],
    "constraint_kinds": ["geofence"],
    "needs_sequence": false,
    "needs_history_aggregate": false,
    "count_min": null,
    "distinct_by": null
  }}
}}
"""

TIER2A_SYSTEM = (
    "你是偏好周期解析模块。根据 normalized_rule 与 routing，"
    "只填写 cycle、scope、completion 三块 JSON；必须服从 preference_type 的固定模板，不要输出 checks。"
    "只输出一个合法 JSON 对象，不要 markdown，不要解释。"
)

TIER2A_TEMPLATE = """根据以下语义与路由，填写周期片 JSON。

【normalized_rule】
{normalized_rule}

【routing】
{routing_json}

【周期口径】
- always：整月每步/每单检查（禁某货/某城/空驶上限）
- day：每个自然日单独判定（每日连续休息、固定时段休息）
- month：三月内聚合（某城市≥N个不同日、≥N个整天休息/保养）；须填 count
- once：仅指定日期生效；必须填 active_dates 和 window
  - active_dates：约束生效的自然日列表（不是偏好首次可见日）
  - window.start = 第一日 00:00:00；window.end = 最后一日的次日 00:00:00
- routing.cycle_kind 必须与 cycle.length 一致
- routing.count_min>0 时填 cycle.count = {{"min": routing.count_min, "max": null, "distinct_by": routing.distinct_by 或 "calendar_day"}}
- evaluate_at：per_action|end_of_day|end_of_month|event_end
- scope.phase：candidate=选单前 | executed=已发生 | history_aggregate=跨历史统计
- scope.applies_when：always_active | in_cycle_window（once 类型用 in_cycle_window）
- completion.mode：never_expires|per_cycle_satisfied|window_expires|month_quota_met

【preference_type 到周期的固定映射】
- ACTION_FORBID、NUMERIC_LIMIT、PREFERENCE_SCORE：通常 length=always，evaluate_at=per_action
- MONTHLY_DEADHEAD_LIMIT：length=month，reset=calendar_month，evaluate_at=per_action，completion.mode=never_expires
- DAILY_FIRST_ORDER_DEADLINE：length=day，reset=calendar_day，evaluate_at=per_action，completion.mode=per_cycle_satisfied
- TIME_WINDOW_STATIONARY、DAILY_CONTINUOUS_REST：length=day，reset=calendar_day，evaluate_at=end_of_day
- OFF_DAY_QUOTA、ORDER_QUOTA：length=month，reset=never，evaluate_at=end_of_month，completion.mode=month_quota_met
- GEOFENCE_STAY_WITHIN：length=always，reset=never，evaluate_at=per_action，completion.mode=never_expires
- GEOFENCE_FORBIDDEN_AREA：length=always，reset=never，evaluate_at=per_action，completion.mode=never_expires
- LOCATION_ARRIVAL_DEADLINE、LOCATION_STAY_ON_DATE、ROUTE_SEQUENCE_ON_DATE：length=once，必须填写 active_dates/window，completion.mode=window_expires

输出 JSON：
{{
  "cycle": {{
    "length": "always|day|month|once",
    "window": null,
    "active_dates": null,
    "count": null,
    "reset": "calendar_day|calendar_month|never",
    "evaluate_at": "per_action|end_of_day|end_of_month|event_end"
  }},
  "scope": {{
    "actions": ["take_order"],
    "phase": "candidate|executed|history_aggregate",
    "applies_when": "always_active|in_cycle_window"
  }},
  "completion": {{
    "mode": "never_expires|per_cycle_satisfied|window_expires|month_quota_met",
    "track_progress": false,
    "progress_key": null,
    "expires_at": null
  }}
}}
"""

TIER2B_SYSTEM = (
    "你是偏好约束解析模块。根据 normalized_rule、routing 与周期片，"
    "只按固定 preference_type 模板填写 checks、on_fail、route_plan；使用简单易懂的字段名，"
    "禁止输出 dist_to_pickup_km、pickup_city、segment 等内部字段名。"
    "只输出一个合法 JSON 对象，不要 markdown，不要解释。"
)

TIER2B_TEMPLATE = """根据以下信息，填写约束片 JSON。

【normalized_rule】
{normalized_rule}

【routing】
{routing_json}

【周期片 2a】
{tier2a_json}

【checks 字段说明】
每条 check 含义：
- action：wait=休息 | reposition=空驶迁址 | take_order=接单
- phase：whole=整段 | to_pickup=接单后去装货点 | haul=装货点到卸货点/运距 | at_pickup/at_delivery=装/卸货地 | cargo=货源信息 | staying=等待中 | moving=空驶中
- measure：distance=公里 | duration=分钟 | clock_time=时刻段 | city=城市 | cargo_name=货名 | location=坐标附近 | still_day=是否全天无位移
- compare：max=不超过 | min=至少 | equals|not_equals | contains|not_contains | in_window|not_in_window | near=在坐标半径内
- value：数字、字符串或对象；时段 {{"from":"HH:MM","to":"HH:MM"}}；坐标 {{"lat":..,"lng":..,"radius_km":..}}
- unit：km 或 minutes；其他可省略
- 地点坐标：只有原文明确给出经纬度时才填写 lat/lng；原文只写地名时，只填写 {{"name":"地点名","radius_km":5}}，不要猜测或补全坐标，后续代码会用地址簿回填。

【on_fail】
- effect：block=硬禁 | warn=软约束
- block_actions：失败时禁止的动作列表

【route_plan】
- 仅 needs_sequence=true 时填写；否则 null
- ROUTE_SEQUENCE_ON_DATE 优先使用多步格式：{{"date":"2026-03-31","steps":[{{"step_id":"s1","kind":"arrival|stay","active_dates":["2026-03-31"],"scope_actions":["reposition","wait"],"constraint_kinds":["route_sequence","location_wait"],"name":"...","lat":..,"lng":..,"arrive_before":"HH:MM","stay_minutes":0,"complete_before":"HH:MM","hold_until":"YYYY-MM-DD HH:MM"}}]}}
- 每个 step 表示一个顺序步骤，必须保留 step_id、kind、地点、arrive_before/stay_minutes/complete_before/hold_until 中原文明确要求的字段；“几分钟前完成/几点前完成停留”写 complete_before；“待到/直到/静止至”写 hold_until，跨日驻留时 hold_until 写完整日期时间。
- 兼容旧格式：{{"date":"2026-03-31","stops":[{{"name":"...","lat":..,"lng":..,"arrive_before":"HH:MM","stay_minutes":0,"finish_before":"HH:MM"}}]}}；若该 step/stop 原文没有经纬度，则不要写 lat/lng。

【注意】
- 只允许围绕 preference_type 填槽，不要发明新的检查语义
- ACTION_FORBID：用 cargo_name/city/location/clock_time 表达禁止条件
- NUMERIC_LIMIT：用一条 distance 或 duration check 表达数值阈值
- TIME_WINDOW_STATIONARY：必须给 take_order 与 reposition 各一条 clock_time not_in_window
- DAILY_CONTINUOUS_REST：用 wait whole duration min N minutes 表达
- OFF_DAY_QUOTA：用 wait whole still_day min N days，N 也应在 cycle.count 体现
- ORDER_QUOTA：用 take_order whole city/cargo_name contains 条件，N 在 cycle.count 体现
- GEOFENCE_STAY_WITHIN：用原文明确经纬度边界表达 bbox，不要猜测；on_fail block wait/reposition/take_order
- GEOFENCE_FORBIDDEN_AREA：用原文明确圆心和半径表达 circle {{"lat":..,"lng":..,"radius_km":..}}，compare 用 not_in；on_fail block wait/reposition/take_order
- MONTHLY_DEADHEAD_LIMIT：用 distance max N km 表达自然月累计空驶上限；actions 包含 take_order to_pickup 和 reposition whole；on_fail block take_order/reposition
- DAILY_FIRST_ORDER_DEADLINE：用 take_order whole clock_time max HH:MM 表达每日首单开工截止；on_fail block take_order
- LOCATION_ARRIVAL_DEADLINE：用 route_plan 单 stop 或 location near + clock_time deadline 表达
- LOCATION_STAY_ON_DATE：用 location near + wait duration min 表达
- ROUTE_SEQUENCE_ON_DATE：必须填写 route_plan.steps（旧格式可用 route_plan.stops），按顺序写每一步的地点、active_dates、scope_actions、constraint_kinds、arrive_before、stay_minutes/complete_before/hold_until
- take_order 空驶距离：action=take_order, phase=to_pickup, measure=distance
- take_order 装卸距离/运距/装货点至卸货点距离：action=take_order, phase=haul, measure=distance
- reposition 空驶/进某城：action=reposition, phase=whole, measure=city 或 distance
- 禁某城货：action=take_order, phase=whole, measure=city, compare=not_contains
- 禁某货类：action=take_order, phase=cargo, measure=cargo_name, compare=equals 或 not_equals（用赛题大类名）
- 0-6点禁行：take_order 与 reposition 各一条 clock_time not_in_window
- 指定日禁某城：take_order city not_contains + reposition city not_contains 各一条
- 整天休息/保养：action=wait, measure=still_day, compare=min, value=N, unit=days（不要用 take_order duration max 0）
- off_days/增城频次由 cycle.count 表达，checks 只写识别条件（still_day 或 city contains）

输出 JSON：
{{
  "checks": [
    {{
      "action": "take_order",
      "phase": "whole",
      "measure": "city",
      "compare": "not_contains",
      "value": "目标城市"
    }}
  ],
  "on_fail": {{
    "effect": "block",
    "block_actions": ["take_order"]
  }},
  "route_plan": null
}}
"""
