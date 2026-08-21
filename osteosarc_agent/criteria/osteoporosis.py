"""Osteoporosis determination and fracture-risk stratification.

Two decisions live here, and keeping them apart matters:

1. **Is it osteoporosis?** — a categorical diagnosis that can be made either by
   T-score or, independently of any BMD value, by a hip or vertebral fragility
   fracture. The single most common failure mode in an osteoporosis clinic is
   not a missed T-score; it is a patient who was treated for the fracture and
   never treated for the disease that caused it.
2. **How high is the risk?** — the stratification that decides *which* drug
   comes first. "Osteoporosis: yes" does not choose an agent; "very high risk"
   does.

The lumbar caveat is applied, not merely mentioned: degenerative change and
compression fractures inflate lumbar BMD, so when the patient has vertebral
fracture or is old enough for that to be likely, the hip site is preferred and
the substitution is recorded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

OSTEOPOROSIS = "osteoporosis"
LOW_BONE_MASS = "low_bone_mass"
NORMAL = "normal_bone_mass"
INDETERMINATE = "indeterminate"

DIAGNOSIS_ZH = {
    OSTEOPOROSIS: "骨质疏松症",
    LOW_BONE_MASS: "低骨量",
    NORMAL: "骨量正常",
    INDETERMINATE: "无法判定（缺 DXA 且无脆性骨折史）",
}

#: Fracture sites that carry a clinical diagnosis of osteoporosis on their own.
_DIAGNOSTIC_SITES = {"hip", "髋部", "股骨颈", "vertebra", "椎体", "脊柱", "spine"}
#: Sites that are diagnostic only together with low bone mass.
_SUPPORTIVE_SITES = {"humerus", "肱骨近端", "pelvis", "骨盆", "forearm", "前臂", "桡骨远端", "wrist", "腕部"}


@dataclass(frozen=True)
class Driver:
    """One contributing factor, with the weight it carried."""

    label: str
    detail: str = ""
    weight: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {"label": self.label, "detail": self.detail, "weight": self.weight}


@dataclass(frozen=True)
class OsteoVerdict:
    diagnosis: str
    basis: str
    severe: bool
    site_used: str
    tscore_used: float | None
    very_high_risk: bool
    risk_drivers: tuple[Driver, ...]
    caveats: tuple[str, ...]
    missing: tuple[str, ...]

    @property
    def diagnosis_zh(self) -> str:
        return DIAGNOSIS_ZH[self.diagnosis]

    def to_dict(self) -> dict[str, Any]:
        return {
            "diagnosis": self.diagnosis,
            "diagnosis_zh": self.diagnosis_zh,
            "basis": self.basis,
            "severe": self.severe,
            "site_used": self.site_used,
            "tscore_used": self.tscore_used,
            "very_high_risk": self.very_high_risk,
            "risk_drivers": [d.to_dict() for d in self.risk_drivers],
            "caveats": list(self.caveats),
            "missing": list(self.missing),
        }


def _fracture_sites(facts: Mapping[str, Any]) -> set[str]:
    sites = facts.get("fracture_sites") or []
    return {str(site).strip() for site in sites}


def _select_tscore(facts: Mapping[str, Any]) -> tuple[float | None, str, list[str]]:
    """Pick the T-score to reason from, preferring hip when spine is unreliable."""
    caveats: list[str] = []
    lumbar = facts.get("lumbar_tscore")
    neck = facts.get("femoral_neck_tscore")
    hip = facts.get("total_hip_tscore")

    spine_unreliable = bool(
        facts.get("morphometric_vertebral_fracture")
        or (_fracture_sites(facts) & {"vertebra", "椎体", "脊柱", "spine"})
    )
    candidates: list[tuple[float, str]] = []
    for value, name in ((lumbar, "腰椎"), (neck, "股骨颈"), (hip, "全髋")):
        if value is not None:
            candidates.append((float(value), name))
    if not candidates:
        return None, "", caveats

    if spine_unreliable and (neck is not None or hip is not None):
        hip_candidates = [c for c in candidates if c[1] != "腰椎"]
        chosen = min(hip_candidates, key=lambda c: c[0])
        if lumbar is not None:
            caveats.append(
                f"存在椎体骨折/退变，腰椎测值（T={float(lumbar):g}）可能虚高，"
                f"已改以{chosen[1]}（T={chosen[0]:g}）为判定依据"
            )
        return chosen[0], chosen[1], caveats

    chosen = min(candidates, key=lambda c: c[0])
    return chosen[0], chosen[1], caveats


def assess(facts: Mapping[str, Any]) -> OsteoVerdict:
    tscore, site, caveats = _select_tscore(facts)
    sites = _fracture_sites(facts)
    had_fracture = bool(facts.get("prior_fragility_fracture")) or bool(sites)
    diagnostic_fracture = bool(sites & _DIAGNOSTIC_SITES) or bool(facts.get("morphometric_vertebral_fracture"))
    supportive_fracture = bool(sites & _SUPPORTIVE_SITES)
    missing: list[str] = []

    if tscore is None:
        missing.append("tscore")
    if had_fracture and not sites:
        caveats.append("记录有脆性骨折但未写明部位，无法判断是否属于诊断性部位")

    # --- diagnosis ------------------------------------------------------
    if diagnostic_fracture:
        diagnosis = OSTEOPOROSIS
        basis = "髋部或椎体脆性骨折（临床诊断，不依赖骨密度）"
    elif tscore is not None and tscore <= -2.5:
        diagnosis = OSTEOPOROSIS
        basis = f"{site} T 值 {tscore:g} ≤ -2.5"
    elif tscore is not None and supportive_fracture and tscore <= -1.0:
        diagnosis = OSTEOPOROSIS
        basis = f"低骨量（{site} T 值 {tscore:g}）合并肱骨近端/骨盆/前臂脆性骨折"
    elif tscore is not None and tscore <= -1.0:
        diagnosis = LOW_BONE_MASS
        basis = f"{site} T 值 {tscore:g} 位于 -1.0 ～ -2.5"
    elif tscore is not None:
        diagnosis = NORMAL
        basis = f"{site} T 值 {tscore:g} > -1.0"
    else:
        diagnosis = INDETERMINATE
        basis = "无 DXA 结果，且无诊断性部位的脆性骨折"

    severe = diagnosis == OSTEOPOROSIS and had_fracture and tscore is not None and tscore <= -2.5

    # --- very high risk --------------------------------------------------
    drivers: list[Driver] = []
    if facts.get("fracture_within_12m"):
        drivers.append(Driver("近 12 个月内脆性骨折", "迫在眉睫的再骨折风险期", 3))
    if facts.get("fracture_on_therapy"):
        drivers.append(Driver("规范抗骨质疏松治疗中仍骨折", "提示治疗失败，需重新评估方案", 3))
    if facts.get("multiple_fractures"):
        drivers.append(Driver("多发脆性骨折", "", 3))
    if diagnostic_fracture:
        drivers.append(Driver("既往髋部或椎体脆性骨折", "、".join(sorted(sites & _DIAGNOSTIC_SITES)) or "影像学椎体骨折", 3))
    if tscore is not None and tscore <= -3.0:
        drivers.append(Driver("T 值 ≤ -3.0", f"{site} T={tscore:g}", 2))
    if facts.get("high_fall_risk") or (facts.get("falls_12m") or 0) >= 2 or facts.get("fall_with_injury"):
        drivers.append(Driver("高跌倒风险或跌倒致伤史", "骨折 = 骨强度不足 × 跌倒", 2))
    if facts.get("glucocorticoid_current") and (facts.get("glucocorticoid_months") or 0) >= 3:
        drivers.append(Driver("长期糖皮质激素", "按更低干预阈值处理", 2))

    very_high = diagnosis == OSTEOPOROSIS and any(d.weight >= 3 for d in drivers)
    if diagnosis == OSTEOPOROSIS and not very_high and len(drivers) >= 2:
        very_high = True

    if diagnosis == OSTEOPOROSIS and tscore is None:
        caveats.append("尚无 DXA 基线值：诊断已由脆性骨折成立，但缺少疗效随访的比较基线，建议尽早完成")
    if facts.get("height_loss_cm") is not None and float(facts["height_loss_cm"]) >= 4 and not facts.get("vertebral_imaging_done"):
        caveats.append("身高下降 ≥4 cm 而未行胸腰椎侧位片/VFA：三分之二的椎体骨折无症状，建议加做")
    if not facts.get("vertebral_imaging_done") and "vertebral_imaging" not in missing:
        missing.append("vertebral_imaging")

    return OsteoVerdict(
        diagnosis=diagnosis,
        basis=basis,
        severe=severe,
        site_used=site,
        tscore_used=tscore,
        very_high_risk=very_high,
        risk_drivers=tuple(drivers),
        caveats=tuple(caveats),
        missing=tuple(missing),
    )
