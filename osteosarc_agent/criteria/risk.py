"""Five-axis risk profile: fracture, fall, sarcopenia, nutrition, functional decline.

**What this is not.** It is not FRAX, and it is not a validated prediction
model. FRAX's coefficients are not public and cannot be reimplemented; any
number produced here is a *transparent additive composite* of risk factors that
the guidelines themselves name, meant for triage and for showing a clinician
which factors are driving the picture. Every axis therefore reports its drivers
with their weights, so the score can be argued with. Where a validated tool is
required (FRAX, OSTA, Morse, MNA), the platform's job is to say which one to run
— not to fake its output.

**Why five axes rather than one label.** "有/没有骨质疏松" does not tell a
clinician what to do next. A patient can be at very high fracture risk chiefly
because they fall, or chiefly because their bone is weak, and the two lead to
completely different plans. Splitting the picture is the point: the radar chart
in the console is a direct rendering of these five axes.

**Missing data lowers confidence, not score.** An axis computed from three of
its twelve inputs reports ``confidence`` accordingly and names what is missing,
rather than quietly returning a reassuring low number.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

TIERS = ("unassessed", "low", "moderate", "high", "very_high")
TIER_ZH = {"unassessed": "未评估", "low": "低", "moderate": "中", "high": "高", "very_high": "极高"}
#: Below this input coverage, a low score carries no information — it means
#: the measurements were never taken. Rendering it as 低 would turn an empty
#: workup into a reassuring green wedge on the radar, which is the single
#: most dangerous thing this chart could do.
MIN_CONFIDENCE_FOR_LOW = 0.5
#: Score thresholds. Deliberately coarse — a composite of this kind cannot
#: support finer discrimination than four bands.
_TIER_BOUNDS = ((25, "low"), (50, "moderate"), (75, "high"))

MODEL_NOTE = (
    "本评分为指南危险因素的透明加权合成，用于分诊与因素归因，"
    "不是经验证的预测模型，不能替代 FRAX、Morse、MNA 等专用工具的正式计算。"
)


@dataclass(frozen=True)
class Rule:
    label: str
    points: int
    inputs: tuple[str, ...]
    test: Callable[[Mapping[str, Any]], bool]
    detail: str = ""
    lever: str = ""  # what could change this factor, if anything


@dataclass(frozen=True)
class AxisResult:
    axis: str
    name_zh: str
    score: int
    tier: str
    drivers: tuple[dict[str, Any], ...]
    missing: tuple[str, ...]
    confidence: float
    levers: tuple[str, ...]
    note: str = ""

    @property
    def tier_zh(self) -> str:
        return TIER_ZH[self.tier]

    def to_dict(self) -> dict[str, Any]:
        return {
            "axis": self.axis,
            "name_zh": self.name_zh,
            "score": self.score,
            "tier": self.tier,
            "tier_zh": self.tier_zh,
            "drivers": list(self.drivers),
            "missing": list(self.missing),
            "confidence": round(self.confidence, 2),
            "levers": list(self.levers),
            "note": self.note,
        }


# --- small predicate helpers ----------------------------------------------


def _num(facts: Mapping[str, Any], name: str) -> float | None:
    value = facts.get(name)
    if value is None or isinstance(value, bool):
        return float(value) if isinstance(value, bool) else None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def ge(name: str, bound: float) -> Callable[[Mapping[str, Any]], bool]:
    def check(facts: Mapping[str, Any]) -> bool:
        value = _num(facts, name)
        return value is not None and value >= bound
    return check


def gt(name: str, bound: float) -> Callable[[Mapping[str, Any]], bool]:
    def check(facts: Mapping[str, Any]) -> bool:
        value = _num(facts, name)
        return value is not None and value > bound
    return check


def le(name: str, bound: float) -> Callable[[Mapping[str, Any]], bool]:
    def check(facts: Mapping[str, Any]) -> bool:
        value = _num(facts, name)
        return value is not None and value <= bound
    return check


def lt(name: str, bound: float) -> Callable[[Mapping[str, Any]], bool]:
    def check(facts: Mapping[str, Any]) -> bool:
        value = _num(facts, name)
        return value is not None and value < bound
    return check


def truthy(name: str) -> Callable[[Mapping[str, Any]], bool]:
    return lambda facts: bool(facts.get(name))


def between(name: str, low: float, high: float) -> Callable[[Mapping[str, Any]], bool]:
    def check(facts: Mapping[str, Any]) -> bool:
        value = _num(facts, name)
        return value is not None and low <= value < high
    return check


def any_of(*tests: Callable[[Mapping[str, Any]], bool]) -> Callable[[Mapping[str, Any]], bool]:
    return lambda facts: any(test(facts) for test in tests)


def _score_axis(
    axis: str,
    name_zh: str,
    rules: Sequence[Rule],
    cap: int,
    facts: Mapping[str, Any],
    *,
    key_inputs: Sequence[str],
    note: str = "",
) -> AxisResult:
    total = 0
    drivers: list[dict[str, Any]] = []
    levers: list[str] = []
    for rule in rules:
        if rule.test(facts):
            total += rule.points
            drivers.append({
                "label": rule.label,
                "points": rule.points,
                "detail": rule.detail,
                "lever": rule.lever,
            })
            if rule.lever and rule.lever not in levers:
                levers.append(rule.lever)
    drivers.sort(key=lambda d: -d["points"])
    score = int(round(100 * min(total, cap) / cap)) if cap else 0

    missing = tuple(name for name in key_inputs if facts.get(name) is None)
    confidence = 1.0 - (len(missing) / len(key_inputs)) if key_inputs else 1.0

    tier = "very_high"
    for bound, label in _TIER_BOUNDS:
        if score < bound:
            tier = label
            break
    if tier == "low" and confidence < MIN_CONFIDENCE_FOR_LOW:
        tier = "unassessed"
        note = (note + " " if note else "") + (
            f"关键输入缺 {len(missing)} 项，本轴不作低风险判读——分数低只说明没测。"
        )
    return AxisResult(axis, name_zh, score, tier, tuple(drivers), missing, confidence,
                      tuple(levers), note)


# --- axis definitions ------------------------------------------------------

_FRACTURE_RULES: tuple[Rule, ...] = (
    Rule("既往脆性骨折", 25, ("prior_fragility_fracture",), truthy("prior_fragility_fracture"),
         "最强的单一预测因子", "已发生的骨折不可逆，但二级预防可显著降低再骨折"),
    Rule("近 12 个月内骨折", 10, ("fracture_within_12m",), truthy("fracture_within_12m"),
         "迫在眉睫的再骨折风险期", "尽早启动抗骨质疏松药物"),
    Rule("多发脆性骨折", 10, ("multiple_fractures",), truthy("multiple_fractures")),
    Rule("T 值 ≤ -3.0", 20, ("tscore_min",), le("tscore_min", -3.0), "", "抗骨质疏松药物治疗"),
    Rule("T 值 -2.5 ～ -3.0", 15, ("tscore_min",), between("tscore_min", -3.0, -2.5), "", "抗骨质疏松药物治疗"),
    Rule("低骨量（T -1.0 ～ -2.5）", 7, ("tscore_min",), between("tscore_min", -2.5, -1.0)),
    Rule("年龄 ≥ 80 岁", 12, ("age",), ge("age", 80)),
    Rule("年龄 70～79 岁", 8, ("age",), between("age", 70, 80)),
    Rule("年龄 65～69 岁", 5, ("age",), between("age", 65, 70)),
    Rule("长期糖皮质激素", 10, ("glucocorticoid_current",),
         lambda f: bool(f.get("glucocorticoid_current")) or (_num(f, "glucocorticoid_months") or 0) >= 3,
         "按更低干预阈值处理", "评估能否减量或换用非激素方案"),
    Rule("父母髋部骨折史", 5, ("parent_hip_fracture",), truthy("parent_hip_fracture")),
    Rule("现在吸烟", 4, ("current_smoker",), truthy("current_smoker"), "", "戒烟"),
    Rule("每日饮酒 ≥3 单位", 4, ("alcohol_units_per_day",), ge("alcohol_units_per_day", 3), "", "限酒"),
    Rule("类风湿关节炎", 5, ("rheumatoid_arthritis",), truthy("rheumatoid_arthritis")),
    Rule("继发性骨质疏松因素", 5, ("secondary_osteoporosis",), truthy("secondary_osteoporosis"),
         "", "纠正可逆的继发因素"),
    Rule("低体重（BMI < 18.5）", 6, ("bmi",), lt("bmi", 18.5), "", "营养干预提高体重与肌量"),
    Rule("反复跌倒", 8, ("falls_12m",), ge("falls_12m", 2), "骨折 = 骨强度不足 × 跌倒",
         "多因素跌倒干预可直接降低骨折"),
    Rule("肌少症共病", 8, ("dx_sarcopenia_any",),
         any_of(truthy("dx_sarcopenia_any"), truthy("dx_possible_sarcopenia")),
         "骨肌共病者骨折风险高于单一疾病", "抗阻训练 + 蛋白质摄入"),
    Rule("身高下降 ≥ 4 cm", 5, ("height_loss_cm",), ge("height_loss_cm", 4),
         "提示未被识别的椎体骨折", "加做胸腰椎侧位片或 VFA"),
    Rule("已诊断骨质疏松但未治疗", 6, ("on_antiosteoporosis_therapy",),
         lambda f: bool(f.get("dx_osteoporosis")) and not f.get("on_antiosteoporosis_therapy"),
         "处方遗漏", "启动抗骨质疏松药物"),
    Rule("维生素 D 缺乏", 4, ("vitamin_d_25oh",), truthy("vitd_deficient"), "", "补足维生素 D"),
)

_FALL_RULES: tuple[Rule, ...] = (
    Rule("近 1 年跌倒 ≥2 次", 22, ("falls_12m",), ge("falls_12m", 2), "", "多因素跌倒评估与干预"),
    Rule("近 1 年跌倒 1 次", 10, ("falls_12m",), between("falls_12m", 1, 2), "", "步态平衡训练"),
    Rule("跌倒致伤", 12, ("fall_with_injury",), truthy("fall_with_injury")),
    Rule("跌倒后不能自行起身", 10, ("cannot_rise_after_fall",), truthy("cannot_rise_after_fall"),
         "长时间倒地风险", "起身训练与呼叫装置"),
    Rule("步速 < 0.8 m/s", 12, ("gait_speed_ms",), lt("gait_speed_ms", 0.8), "", "步态与力量训练"),
    Rule("步速 0.8～1.0 m/s", 6, ("gait_speed_ms",), between("gait_speed_ms", 0.8, 1.0), "", "步态与力量训练"),
    Rule("5 次起坐 ≥ 15 s", 10, ("sts5_s",), ge("sts5_s", 15), "下肢力量明显不足", "渐进抗阻训练"),
    Rule("5 次起坐 12～15 s", 6, ("sts5_s",), between("sts5_s", 12, 15), "", "渐进抗阻训练"),
    Rule("TUG ≥ 12 s", 8, ("tug_s",), ge("tug_s", 12), "", "平衡与功能性训练"),
    Rule("害怕跌倒", 6, ("fear_of_falling",), truthy("fear_of_falling"),
         "回避活动 → 失用 → 更易跌倒", "在监督下恢复活动，打破回避循环"),
    Rule("头晕/体位性低血压", 8, ("dizziness",), truthy("dizziness"), "", "测量卧立位血压并调整降压方案"),
    Rule("视力障碍", 6, ("vision_impairment",), truthy("vision_impairment"), "", "眼科评估、白内障手术、配镜"),
    Rule("认知障碍", 8, ("cognitive_impairment",), truthy("cognitive_impairment")),
    Rule("使用助行器", 5, ("uses_walking_aid",), truthy("uses_walking_aid"), "", "评估器具是否合适与使用是否正确"),
    Rule("尿失禁/夜尿频", 4, ("urinary_incontinence",), truthy("urinary_incontinence"), "", "夜间照明与床旁便器"),
    Rule("增加跌倒风险的药物", 14, ("frid_count",), ge("frid_count", 2),
         "镇静催眠、抗胆碱能、降压、阿片类", "用药重整——跌倒危险因素中最可逆的一项"),
    Rule("含 1 种跌倒风险药物", 7, ("frid_count",), between("frid_count", 1, 2), "", "用药重整"),
    Rule("多重用药（≥5 种）", 5, ("polypharmacy",), truthy("polypharmacy"), "", "结构化用药重整"),
    Rule("居家环境危险因素", 6, ("home_hazards",), lambda f: bool(f.get("home_hazards")),
         "", "居家改造：扶手、防滑、照明、去门槛"),
    Rule("年龄 ≥ 80 岁", 6, ("age",), ge("age", 80)),
    Rule("肌力下降", 6, ("grip_kg",), truthy("low_muscle_strength"), "", "抗阻训练"),
    Rule("维生素 D 缺乏", 4, ("vitamin_d_25oh",), truthy("vitd_deficient"), "", "补足维生素 D"),
)

_SARCOPENIA_RULES: tuple[Rule, ...] = (
    Rule("已达肌少症诊断标准", 40, ("dx_sarcopenia_any",), truthy("dx_sarcopenia_any")),
    Rule("已判为可能肌少症", 25, ("dx_possible_sarcopenia",),
         lambda f: bool(f.get("dx_possible_sarcopenia")) and not f.get("dx_sarcopenia_any")),
    Rule("握力下降", 15, ("grip_kg",), truthy("low_muscle_strength"), "", "渐进抗阻训练"),
    Rule("躯体功能下降", 12, ("gait_speed_ms",), truthy("low_physical_performance"), "", "功能性训练"),
    Rule("肌量减少", 15, ("asmi",), truthy("low_muscle_mass")),
    Rule("小腿围低于切点", 8, ("calf_circumference_cm",),
         lambda f: (_num(f, "calf_circumference_cm") is not None
                    and _num(f, "calf_circumference_cm") < (34 if f.get("sex") == "M" else 33)),
         "AWGS 病例发现指标"),
    Rule("SARC-F ≥ 4", 8, ("sarc_f",), ge("sarc_f", 4)),
    Rule("年龄 ≥ 75 岁", 8, ("age",), ge("age", 75)),
    Rule("蛋白摄入不足（<1.0 g/kg）", 10, ("protein_g_per_kg",), lt("protein_g_per_kg", 1.0),
         "", "提高优质蛋白并分配到三餐"),
    Rule("无抗阻训练习惯", 10, ("resistance_training",),
         lambda f: f.get("resistance_training") is False, "", "每周 2～3 次抗阻训练"),
    Rule("久坐 ≥ 8 小时/天", 6, ("sedentary_hours",), ge("sedentary_hours", 8), "", "减少久坐"),
    Rule("3 个月体重下降 ≥ 5%", 10, ("weight_loss_3m_pct",), ge("weight_loss_3m_pct", 5), "", "营养干预"),
    Rule("维生素 D 缺乏", 5, ("vitamin_d_25oh",), truthy("vitd_deficient"), "", "补足维生素 D"),
    Rule("慢性肾功能不全", 6, ("egfr",), lt("egfr", 45), "蛋白目标与肌肉合成受限"),
)

_NUTRITION_RULES: tuple[Rule, ...] = (
    Rule("MNA-SF ≤ 7（营养不良）", 35, ("mna_sf",), le("mna_sf", 7), "", "营养科会诊 + 口服营养补充"),
    Rule("MNA-SF 8～11（有风险）", 20, ("mna_sf",), between("mna_sf", 8, 12), "", "膳食强化并复评"),
    Rule("3 个月体重下降 ≥ 5%", 20, ("weight_loss_3m_pct",), ge("weight_loss_3m_pct", 5), "", "营养干预"),
    Rule("食欲下降", 12, ("appetite_loss",), truthy("appetite_loss"), "", "查找可逆原因：口腔、药物、抑郁、便秘"),
    Rule("蛋白摄入 < 1.0 g/kg", 15, ("protein_g_per_kg",), lt("protein_g_per_kg", 1.0), "", "提高优质蛋白"),
    Rule("BMI < 20", 12, ("bmi",), lt("bmi", 20), "", "增加能量与蛋白"),
    Rule("膳食钙摄入不足", 10, ("calcium_intake_mg",), lt("calcium_intake_mg", 800), "", "奶制品与豆制品"),
    Rule("每日奶制品不足 1 份", 6, ("dairy_servings",), lt("dairy_servings", 1), "", "每日 300～400 mL 奶"),
    Rule("维生素 D 缺乏", 10, ("vitamin_d_25oh",), truthy("vitd_deficient"), "", "补充维生素 D"),
    Rule("吞咽困难", 12, ("swallowing_difficulty",), truthy("swallowing_difficulty"), "", "吞咽评估与质地调整"),
    Rule("低白蛋白（<35 g/L）", 8, ("albumin",), lt("albumin", 35), "同时受炎症影响，需结合判断"),
    Rule("独居", 5, ("living_alone",), truthy("living_alone"), "备餐能力与社会支持", "社区送餐或家属协助"),
    Rule("日照不足", 4, ("sun_exposure_low",), truthy("sun_exposure_low"), "", "增加户外活动"),
)

_FUNCTIONAL_RULES: tuple[Rule, ...] = (
    Rule("步速 < 0.8 m/s", 20, ("gait_speed_ms",), lt("gait_speed_ms", 0.8),
         "公认的失能与死亡预测阈值", "力量与步态训练"),
    Rule("步速 0.8～1.0 m/s", 10, ("gait_speed_ms",), between("gait_speed_ms", 0.8, 1.0), "", "力量与步态训练"),
    Rule("SPPB ≤ 8", 18, ("sppb",), le("sppb", 8), "", "多成分运动"),
    Rule("5 次起坐 ≥ 15 s", 12, ("sts5_s",), ge("sts5_s", 15), "", "渐进抗阻训练"),
    Rule("IADL 依赖", 15, ("iadl_dependent",), truthy("iadl_dependent"), "", "作业治疗与环境适配"),
    Rule("肌少症或可能肌少症", 15, ("dx_sarcopenia_any",),
         any_of(truthy("dx_sarcopenia_any"), truthy("dx_possible_sarcopenia")), "", "抗阻训练 + 营养"),
    Rule("既往脆性骨折", 12, ("prior_fragility_fracture",), truthy("prior_fragility_fracture"),
         "骨折后功能常不能完全恢复", "早期康复介入"),
    Rule("反复跌倒", 10, ("falls_12m",), ge("falls_12m", 2), "", "多因素跌倒干预"),
    Rule("认知障碍", 12, ("cognitive_impairment",), truthy("cognitive_impairment")),
    Rule("共病 ≥ 3 种", 8, ("comorbidity_count",), ge("comorbidity_count", 3)),
    Rule("久坐 ≥ 8 小时/天", 8, ("sedentary_hours",), ge("sedentary_hours", 8), "", "减少久坐时间"),
    Rule("害怕跌倒导致活动回避", 8, ("fear_of_falling",), truthy("fear_of_falling"),
         "", "监督下恢复活动"),
    Rule("年龄 ≥ 80 岁", 8, ("age",), ge("age", 80)),
    Rule("使用助行器", 6, ("uses_walking_aid",), truthy("uses_walking_aid")),
)

#: Each cap is the point total a *realistically severe* patient reaches on that
#: axis, not the sum of every rule — no real patient smokes, drinks, takes
#: steroids, has rheumatoid arthritis and has three prior fractures at once, so
#: dividing by the theoretical maximum would compress every real score into the
#: bottom third. The caps were set by scoring the reference cases and checking
#: that a severe patient lands in the 80s rather than pegging at 100.
_AXES = (
    ("fracture", "骨折风险", _FRACTURE_RULES, 100,
     ("age", "tscore_min", "prior_fragility_fracture", "bmi", "falls_12m", "glucocorticoid_current"), ""),
    ("fall", "跌倒风险", _FALL_RULES, 130,
     ("falls_12m", "gait_speed_ms", "sts5_s", "frid_count", "vision_impairment", "cognitive_impairment"), ""),
    ("sarcopenia", "肌少症风险", _SARCOPENIA_RULES, 125,
     ("grip_kg", "gait_speed_ms", "asmi", "protein_g_per_kg", "resistance_training"), ""),
    ("nutrition", "营养风险", _NUTRITION_RULES, 115,
     ("mna_sf", "weight_loss_3m_pct", "protein_g_per_kg", "bmi", "albumin"), ""),
    ("functional", "功能衰退风险", _FUNCTIONAL_RULES, 130,
     ("gait_speed_ms", "sppb", "sts5_s", "iadl_dependent", "comorbidity_count"), ""),
)


def profile(facts: Mapping[str, Any]) -> list[AxisResult]:
    """Compute all five axes."""
    return [
        _score_axis(axis, name, rules, cap, facts, key_inputs=key_inputs, note=note)
        for axis, name, rules, cap, key_inputs, note in _AXES
    ]


def as_radar(axes: Sequence[AxisResult]) -> dict[str, Any]:
    """The payload the console's radar chart consumes."""
    return {
        "axes": [
            {"axis": a.axis, "label": a.name_zh, "score": a.score,
             "tier": a.tier, "tier_zh": a.tier_zh, "confidence": round(a.confidence, 2)}
            for a in axes
        ],
        "model_note": MODEL_NOTE,
    }
