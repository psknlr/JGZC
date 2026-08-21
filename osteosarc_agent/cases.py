"""Demonstration cases.

Three cases chosen because they exercise *different* behaviours of the platform,
not because they are three examples of the same thing:

``demo``
    The 78-year-old woman from the product brief. Her eGFR of 32 takes the
    bisphosphonate arm off the table, so the international dispute about
    first-line therapy in very-high-risk patients is **resolved by her own
    chart**; and her grip of 17 kg sits between the AWGS and EWGSOP2 cutoffs, so
    the three sarcopenia standards return three different verdicts.

``sparse``
    A hip fracture and almost nothing else measured. Everything the platform
    says about him is dominated by what has not been done: blocking safety gates,
    indeterminate sarcopenia verdicts, and a prioritised list of measurements.

``gc_male``
    Normal renal function, so the first-line dispute stays **live** and is
    reported as 存在争议 rather than resolved. His grip sits between the AWGS and
    EWGSOP2 cutoffs in the other direction, and his muscle mass *is* measured —
    so AWGS and GLIS confirm sarcopenia while EWGSOP2 does not even enter the
    algorithm.

No real patient data is used or should ever be committed to this repository.
"""

from __future__ import annotations

from typing import Any

DEMO_CASE: dict[str, Any] = {
    "case_id": "demo-001",
    "label": "78 岁女性 · 既往椎体骨折 · eGFR 32",
    "age": 78,
    "sex": "F",
    "menopause_age": 50,
    "height_cm": 152,
    "weight_kg": 46,
    "height_loss_cm": 5,
    "living_alone": True,
    "care_setting": "outpatient",

    "prior_fragility_fracture": True,
    "fracture_sites": ["椎体"],
    "fracture_within_12m": False,
    "parent_hip_fracture": False,
    "current_smoker": False,
    "comorbidity_count": 4,
    "diabetes": False,

    "falls_12m": 2,
    "fall_with_injury": True,
    "fear_of_falling": True,
    "dizziness": True,
    "vision_impairment": True,
    "uses_walking_aid": True,
    "home_hazards": ["卫生间无扶手", "夜间照明不足", "门槛"],

    "lumbar_tscore": -2.9,
    "femoral_neck_tscore": -2.6,
    "vertebral_imaging_done": False,

    "grip_kg": 17,
    "sts5_s": 16,
    "gait_speed_ms": 0.72,
    "calf_circumference_cm": 29,
    "sarc_f": 5,
    "resistance_training": False,
    "sedentary_hours": 9,
    "exercise_minutes_week": 60,
    "iadl_dependent": True,

    "egfr": 32,
    "creatinine": 132,
    "calcium": 2.18,
    "albumin": 34,
    "phosphate": 1.12,
    "alp": 88,
    "vitamin_d_25oh": 14,
    "hemoglobin": 112,

    "medications": [
        {"name": "碳酸钙D3 片"},
        {"name": "艾司唑仑片"},
        {"name": "氨氯地平片"},
        {"name": "奥美拉唑肠溶胶囊"},
        {"name": "左甲状腺素钠片"},
    ],

    "appetite_loss": True,
    "weight_loss_3m_pct": 4,
    "protein_g_per_kg": 0.8,
    "dairy_servings": 0.5,
    "calcium_intake_mg": 520,
    "mna_sf": 9,
    "sun_exposure_low": True,

    "tcm_pattern": "肾阳虚",
    "tcm_kidney_deficiency": True,
    "tcm_spleen_deficiency": True,
}

SPARSE_CASE: dict[str, Any] = {
    "case_id": "sparse-002",
    "label": "82 岁男性 · 髋部骨折术后 2 个月 · 资料极少",
    "age": 82,
    "sex": "M",
    "prior_fragility_fracture": True,
    "fracture_sites": ["髋部"],
    "fracture_within_12m": True,
    "medications": [{"name": "碳酸钙D3 片"}, {"name": "曲马多缓释片"}],
    "care_setting": "outpatient",
}

GC_MALE_CASE: dict[str, Any] = {
    "case_id": "gc-003",
    "label": "72 岁男性 · 类风湿关节炎长期激素 · 肾功能正常",
    "age": 72,
    "sex": "M",
    "height_cm": 168,
    "weight_kg": 61,
    "rheumatoid_arthritis": True,
    "glucocorticoid_current": True,
    "glucocorticoid_months": 18,
    "prior_fragility_fracture": False,
    "current_smoker": True,
    "falls_12m": 0,

    "lumbar_tscore": -3.1,
    "femoral_neck_tscore": -2.7,
    "vertebral_imaging_done": True,

    "grip_kg": 27.5,
    "gait_speed_ms": 1.05,
    "asm_kg": 19.2,
    "muscle_mass_method": "DXA",
    "resistance_training": False,

    "egfr": 78,
    "calcium": 2.32,
    "albumin": 40,
    "vitamin_d_25oh": 24,

    "medications": [
        {"name": "泼尼松片"},
        {"name": "甲氨蝶呤片"},
        {"name": "碳酸钙D3 片"},
        {"name": "叶酸片"},
    ],
    "protein_g_per_kg": 1.0,
    "dairy_servings": 1,
}

CASES: dict[str, dict[str, Any]] = {
    "demo": DEMO_CASE,
    "sparse": SPARSE_CASE,
    "gc_male": GC_MALE_CASE,
}


def get(name: str) -> dict[str, Any]:
    try:
        return dict(CASES[name])
    except KeyError:
        raise KeyError(f"未知示例病例 {name!r}，可用: {', '.join(CASES)}") from None
