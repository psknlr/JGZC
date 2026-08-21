"""The fact namespace: the shared vocabulary of the whole platform.

Everything downstream keys off these names — the intake agent writes them, the
guideline predicates read them, the console renders them. Declaring them once,
with a label, a unit and a group, is what keeps a corpus typo (``egfr_ml`` for
``egfr``) from becoming a silently-unknown predicate at the bedside: the corpus
loader checks every ``fact:`` reference against this table and refuses to load
an unknown one.

Facts are in three layers:

* ``source="intake"`` — read from the record by the structuring agent;
* ``source="derived"`` — computed by intake from other facts (BMI, ASMI, tri-state
  flags such as ``vitd_deficient``);
* ``source="agent"`` — written back by a later agent (diagnoses, risk tiers), so
  a guideline predicate can say "本条适用于已诊断肌少症的患者" without re-deriving it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FactSpec:
    name: str
    label: str
    group: str
    unit: str = ""
    kind: str = "number"  # number | bool | text | list | date
    source: str = "intake"
    note: str = ""


def _f(name: str, label: str, group: str, unit: str = "", kind: str = "number",
       source: str = "intake", note: str = "") -> FactSpec:
    return FactSpec(name, label, group, unit, kind, source, note)


FACT_SPECS: tuple[FactSpec, ...] = (
    # --- 人口学 ---------------------------------------------------------
    _f("age", "年龄", "demographics", "岁"),
    _f("sex", "性别", "demographics", kind="text", note="F/M"),
    _f("postmenopausal", "绝经后", "demographics", kind="bool"),
    _f("menopause_age", "绝经年龄", "demographics", "岁"),
    _f("years_since_menopause", "绝经年数", "demographics", "年", source="derived"),
    _f("height_cm", "身高", "demographics", "cm"),
    _f("weight_kg", "体重", "demographics", "kg"),
    _f("bmi", "体质指数", "demographics", "kg/m²", source="derived"),
    _f("height_loss_cm", "身高下降", "demographics", "cm", note="较年轻时最高身高"),
    _f("living_alone", "独居", "demographics", kind="bool"),
    _f("care_setting", "照护场景", "demographics", kind="text", note="community/outpatient/inpatient/ltc"),

    # --- 病史 -----------------------------------------------------------
    _f("prior_fragility_fracture", "既往脆性骨折", "history", kind="bool"),
    _f("fracture_sites", "骨折部位", "history", kind="list"),
    _f("fracture_within_12m", "近12个月内骨折", "history", kind="bool"),
    _f("multiple_fractures", "多发脆性骨折", "history", kind="bool"),
    _f("parent_hip_fracture", "父母髋部骨折史", "history", kind="bool"),
    _f("current_smoker", "现在吸烟", "history", kind="bool"),
    _f("alcohol_units_per_day", "每日饮酒量", "history", "单位"),
    _f("glucocorticoid_current", "现用糖皮质激素", "history", kind="bool"),
    _f("glucocorticoid_months", "激素累计使用", "history", "月"),
    _f("rheumatoid_arthritis", "类风湿关节炎", "history", kind="bool"),
    _f("secondary_osteoporosis", "继发性骨质疏松因素", "history", kind="bool"),
    _f("diabetes", "糖尿病", "history", kind="bool"),
    _f("recent_cv_event_12m", "近12个月心梗或卒中", "history", kind="bool"),
    _f("malignancy", "恶性肿瘤史", "history", kind="bool"),
    _f("skeletal_radiotherapy", "骨骼放疗史", "history", kind="bool"),
    _f("dental_extraction_pending", "待处理拔牙或牙病", "history", kind="bool"),
    _f("esophageal_disease", "活动性食管疾病", "history", kind="bool"),
    _f("cannot_stay_upright", "无法保持直立30分钟", "history", kind="bool"),
    _f("comorbidity_count", "共病数量", "history", "种"),

    # --- 跌倒 -----------------------------------------------------------
    _f("falls_12m", "近1年跌倒次数", "falls", "次"),
    _f("fall_with_injury", "跌倒致伤", "falls", kind="bool"),
    _f("cannot_rise_after_fall", "跌倒后不能自行起身", "falls", kind="bool"),
    _f("fear_of_falling", "害怕跌倒", "falls", kind="bool"),
    _f("dizziness", "头晕/体位性低血压", "falls", kind="bool"),
    _f("vision_impairment", "视力障碍", "falls", kind="bool"),
    _f("cognitive_impairment", "认知障碍", "falls", kind="bool"),
    _f("uses_walking_aid", "使用助行器", "falls", kind="bool"),
    _f("home_hazards", "居家环境危险因素", "falls", kind="list"),
    _f("urinary_incontinence", "尿失禁/夜尿频", "falls", kind="bool"),

    # --- 骨密度与影像 ---------------------------------------------------
    _f("dxa_date", "DXA 检查日期", "bone", kind="date"),
    _f("lumbar_tscore", "腰椎 T 值", "bone"),
    _f("femoral_neck_tscore", "股骨颈 T 值", "bone"),
    _f("total_hip_tscore", "全髋 T 值", "bone"),
    _f("tscore_min", "最低 T 值", "bone", source="derived"),
    _f("tscore_site_min", "最低 T 值部位", "bone", kind="text", source="derived"),
    _f("lumbar_bmd", "腰椎 BMD", "bone", "g/cm²"),
    _f("femoral_neck_bmd", "股骨颈 BMD", "bone", "g/cm²"),
    _f("vertebral_imaging_done", "胸腰椎侧位片/VFA 已完成", "bone", kind="bool"),
    _f("morphometric_vertebral_fracture", "影像学椎体骨折", "bone", kind="bool"),

    # --- 肌肉与功能 -----------------------------------------------------
    _f("grip_kg", "握力", "muscle", "kg"),
    _f("gait_speed_ms", "6米步速", "muscle", "m/s"),
    _f("sts5_s", "5次起坐时间", "muscle", "s"),
    _f("sppb", "SPPB 总分", "muscle", "分"),
    _f("tug_s", "起立-行走计时", "muscle", "s"),
    _f("asm_kg", "四肢骨骼肌量", "muscle", "kg"),
    _f("asmi", "四肢骨骼肌质量指数", "muscle", "kg/m²", source="derived"),
    _f("muscle_mass_method", "肌量测量方法", "muscle", kind="text", note="DXA/BIA"),
    _f("calf_circumference_cm", "小腿围", "muscle", "cm"),
    _f("sarc_f", "SARC-F 评分", "muscle", "分"),
    _f("adl_score", "ADL 评分", "muscle", "分"),
    _f("iadl_dependent", "IADL 依赖", "muscle", kind="bool"),
    _f("exercise_minutes_week", "每周运动时间", "muscle", "分钟"),
    _f("resistance_training", "有抗阻训练习惯", "muscle", kind="bool"),
    _f("sedentary_hours", "每日久坐时间", "muscle", "小时"),

    # --- 实验室 ---------------------------------------------------------
    _f("egfr", "eGFR", "labs", "mL/min/1.73m²"),
    _f("creatinine", "肌酐", "labs", "μmol/L"),
    _f("corrected_calcium", "校正血钙", "labs", "mmol/L"),
    _f("calcium", "血钙", "labs", "mmol/L"),
    _f("albumin", "白蛋白", "labs", "g/L"),
    _f("phosphate", "血磷", "labs", "mmol/L"),
    _f("alp", "碱性磷酸酶", "labs", "U/L"),
    _f("vitamin_d_25oh", "25-OH 维生素 D", "labs", "ng/mL"),
    _f("pth", "甲状旁腺激素", "labs", "pg/mL"),
    _f("tsh", "促甲状腺激素", "labs", "mIU/L"),
    _f("hemoglobin", "血红蛋白", "labs", "g/L"),
    _f("ckd_stage", "慢性肾病分期", "labs", kind="text", source="derived"),
    _f("hypocalcemia", "低钙血症", "labs", kind="bool", source="derived"),
    _f("vitd_deficient", "维生素 D 缺乏", "labs", kind="bool", source="derived"),
    _f("vitd_insufficient", "维生素 D 不足", "labs", kind="bool", source="derived"),

    # --- 用药 -----------------------------------------------------------
    _f("medications", "用药清单", "medication", kind="list"),
    _f("medication_classes", "用药类别", "medication", kind="list", source="derived"),
    _f("medication_count", "用药种数", "medication", "种", source="derived"),
    _f("polypharmacy", "多重用药", "medication", kind="bool", source="derived"),
    _f("frid_count", "增加跌倒风险药物数", "medication", "种", source="derived"),
    _f("on_antiosteoporosis_therapy", "已在抗骨质疏松治疗", "medication", kind="bool", source="derived"),
    _f("fracture_on_therapy", "规范治疗中仍骨折", "medication", kind="bool"),
    _f("adherence_concern", "依从性风险", "medication", kind="bool"),

    # --- 营养 -----------------------------------------------------------
    _f("appetite_loss", "食欲下降", "nutrition", kind="bool"),
    _f("weight_loss_3m_pct", "3个月体重下降", "nutrition", "%"),
    _f("protein_g_per_kg", "每日蛋白摄入", "nutrition", "g/kg"),
    _f("calcium_intake_mg", "每日膳食钙摄入", "nutrition", "mg"),
    _f("dairy_servings", "每日奶制品份数", "nutrition", "份"),
    _f("mna_sf", "MNA-SF 评分", "nutrition", "分"),
    _f("swallowing_difficulty", "吞咽困难", "nutrition", kind="bool"),
    _f("sun_exposure_low", "日照不足", "nutrition", kind="bool"),

    # --- 中医 -----------------------------------------------------------
    _f("tcm_pattern", "中医证型", "tcm", kind="text"),
    _f("tcm_kidney_deficiency", "肾虚证候", "tcm", kind="bool"),
    _f("tcm_spleen_deficiency", "脾虚证候", "tcm", kind="bool"),
    _f("tcm_blood_stasis", "血瘀证候", "tcm", kind="bool"),

    # --- 由后续 Agent 回写 ----------------------------------------------
    _f("dx_osteoporosis", "骨质疏松诊断成立", "diagnosis", kind="bool", source="agent"),
    _f("dx_osteoporosis_basis", "骨质疏松诊断依据", "diagnosis", kind="text", source="agent"),
    _f("dx_low_bone_mass", "低骨量", "diagnosis", kind="bool", source="agent"),
    _f("dx_severe_osteoporosis", "严重骨质疏松", "diagnosis", kind="bool", source="agent"),
    _f("dx_sarcopenia_any", "任一标准判定肌少症", "diagnosis", kind="bool", source="agent"),
    _f("dx_sarcopenia_awgs", "AWGS 判定", "diagnosis", kind="text", source="agent"),
    _f("dx_sarcopenia_ewgsop2", "EWGSOP2 判定", "diagnosis", kind="text", source="agent"),
    _f("dx_sarcopenia_glis", "GLIS 判定", "diagnosis", kind="text", source="agent"),
    _f("dx_possible_sarcopenia", "可能肌少症", "diagnosis", kind="bool", source="agent"),
    _f("dx_severe_sarcopenia", "严重肌少症", "diagnosis", kind="bool", source="agent"),
    _f("dx_osteosarcopenia", "骨肌共病", "diagnosis", kind="bool", source="agent"),
    _f("low_muscle_strength", "肌力下降", "diagnosis", kind="bool", source="agent"),
    _f("low_physical_performance", "躯体功能下降", "diagnosis", kind="bool", source="agent"),
    _f("low_muscle_mass", "肌量减少", "diagnosis", kind="bool", source="agent"),
    _f("muscle_mass_measured", "已测肌量", "diagnosis", kind="bool", source="agent"),

    _f("fracture_risk_tier", "骨折风险分层", "risk", kind="text", source="agent"),
    _f("fall_risk_tier", "跌倒风险分层", "risk", kind="text", source="agent"),
    _f("sarcopenia_risk_tier", "肌少症风险分层", "risk", kind="text", source="agent"),
    _f("nutrition_risk_tier", "营养风险分层", "risk", kind="text", source="agent"),
    _f("functional_risk_tier", "功能衰退风险分层", "risk", kind="text", source="agent"),
    _f("very_high_fracture_risk", "极高骨折风险", "risk", kind="bool", source="agent"),
    _f("high_fall_risk", "高跌倒风险", "risk", kind="bool", source="agent"),
)

FACTS: dict[str, FactSpec] = {spec.name: spec for spec in FACT_SPECS}

GROUP_LABELS = {
    "demographics": "人口学与体格",
    "history": "既往史与危险因素",
    "falls": "跌倒",
    "bone": "骨密度与影像",
    "muscle": "肌肉与功能",
    "labs": "实验室",
    "medication": "用药",
    "nutrition": "营养",
    "tcm": "中医四诊",
    "diagnosis": "诊断结论",
    "risk": "风险分层",
}

#: Groups the console renders in the 患者数字画像 panel, in display order.
PROFILE_GROUPS = (
    "demographics", "history", "bone", "muscle", "labs", "falls",
    "medication", "nutrition", "tcm",
)


def describe_fact(name: str) -> str:
    spec = FACTS.get(name)
    return spec.label if spec else name


def unit_of(name: str) -> str:
    spec = FACTS.get(name)
    return spec.unit if spec else ""


def is_known_fact(name: str) -> bool:
    return name in FACTS


def facts_in_group(group: str) -> list[FactSpec]:
    return [spec for spec in FACT_SPECS if spec.group == group]


def format_value(name: str, value: Any) -> str:
    """Render a fact for human display, with unit."""
    if value is None:
        return "—"
    spec = FACTS.get(name)
    if spec is None:
        return str(value)
    if spec.kind == "bool":
        return "是" if value else "否"
    if spec.kind == "list":
        if not value:
            return "无"
        # Medication entries arrive as mappings; show the name a clinician wrote,
        # not the record that carries it.
        return "、".join(
            str(item.get("name", item)) if isinstance(item, dict) else str(item)
            for item in value
        )
    if isinstance(value, float):
        text = f"{value:g}"
    else:
        text = str(value)
    return f"{text} {spec.unit}".strip()
