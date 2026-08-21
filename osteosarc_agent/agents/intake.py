"""Agent 1 —— 病例结构化 Agent.

Turns a case record into the platform's fact namespace, then derives everything
computable from it: BMI, ASMI, the governing T-score, CKD stage, calcium and
vitamin D flags, medication classes, fall-risk-drug count, polypharmacy.

The interesting output is not the facts it found but the ones it did not.
``missing_critical`` names the measurements whose absence will *change a
downstream conclusion* — a missing ASMI is why GLIS cannot conclude, a missing
eGFR is why no anti-resorptive can be cleared — and each entry says which
decision it blocks. A structuring agent that silently returns a tidy half-empty
record is how an incomplete workup gets mistaken for a normal one.
"""

from __future__ import annotations

from typing import Any, Mapping

from ..guidelines import facts as fact_table
from ..safety import drugs
from ..state import RunState
from .base import SubAgent

#: Fields whose absence blocks a specific downstream decision.
CRITICAL_FACTS: tuple[tuple[str, str], ...] = (
    ("age", "所有年龄相关的适用条件与切点"),
    ("sex", "握力与肌量切点、骨密度诊断适用人群"),
    ("egfr", "抗骨吸收药物的肾功能门槛——未查肾功能不得启动用药"),
    ("corrected_calcium", "抗骨吸收治疗前的低钙筛查"),
    ("vitamin_d_25oh", "维生素 D 纠正与用药前置条件"),
    ("asmi", "GLIS 肌少症诊断的必备核心要素；AWGS/EWGSOP2 的确诊要素"),
    ("grip_kg", "肌力判定（三套标准的共同入口）"),
    ("gait_speed_ms", "躯体功能与功能衰退风险"),
    ("tscore_min", "骨密度分级与疗效随访基线"),
)

#: Aliases accepted in the incoming record, so a case file written by a clinic
#: does not have to know the internal fact names.
ALIASES: dict[str, str] = {
    "gender": "sex", "性别": "sex", "年龄": "age", "身高": "height_cm", "体重": "weight_kg",
    "握力": "grip_kg", "步速": "gait_speed_ms", "五次起坐": "sts5_s", "5次起坐": "sts5_s",
    "腰椎T值": "lumbar_tscore", "股骨颈T值": "femoral_neck_tscore", "全髋T值": "total_hip_tscore",
    "肾小球滤过率": "egfr", "血钙": "calcium", "校正钙": "corrected_calcium",
    "维生素D": "vitamin_d_25oh", "用药": "medications", "跌倒次数": "falls_12m",
    "handgrip": "grip_kg", "gait_speed": "gait_speed_ms", "chair_stand": "sts5_s",
    "sts5": "sts5_s", "egfr_ml_min": "egfr", "vitamin_d": "vitamin_d_25oh",
}


def _flatten(record: Mapping[str, Any], out: dict[str, Any]) -> None:
    """Accept flat or one-level-nested records (``{"labs": {"egfr": 32}}``)."""
    for key, value in record.items():
        name = ALIASES.get(str(key), str(key))
        if isinstance(value, Mapping) and name not in fact_table.FACTS:
            _flatten(value, out)
            continue
        if name in out and value is None:
            continue
        out[name] = value


def _f(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class IntakeAgent(SubAgent):
    agent_id = "intake"
    name_zh = "病例结构化 Agent"
    schema = "StructuredCase"

    def run(self, state: RunState) -> dict[str, Any]:
        raw: dict[str, Any] = {}
        _flatten(state.case, raw)

        facts: dict[str, Any] = {}
        unknown_fields: list[str] = []
        for name, value in raw.items():
            if name in fact_table.FACTS:
                facts[name] = value
            elif not str(name).startswith("_"):
                unknown_fields.append(str(name))

        derived = self._derive(facts)
        state.facts.update(facts)

        coverage = self._coverage(facts)
        missing_critical = [
            {"fact": name, "label": fact_table.describe_fact(name), "blocks": blocks}
            for name, blocks in CRITICAL_FACTS
            if facts.get(name) is None
        ]

        return {
            "facts": facts,
            "coverage": coverage,
            "missing_critical": missing_critical,
            "derived": derived,
            "unrecognised_fields": unknown_fields,
            "medications": [m.to_dict() for m in drugs.parse_medications(raw.get("medications") or [])],
            "profile": self._profile_view(facts),
        }

    # -- derivation ------------------------------------------------------
    def _derive(self, facts: dict[str, Any]) -> list[dict[str, Any]]:
        derived: list[dict[str, Any]] = []

        def write(name: str, value: Any, how: str) -> None:
            if value is None:
                return
            facts[name] = value
            derived.append({"fact": name, "label": fact_table.describe_fact(name),
                            "value": value, "how": how})

        height = _f(facts.get("height_cm"))
        weight = _f(facts.get("weight_kg"))
        if height and weight and height > 0:
            write("bmi", round(weight / (height / 100) ** 2, 1), "体重 / 身高²")

        asm = _f(facts.get("asm_kg"))
        if facts.get("asmi") is None and asm and height:
            write("asmi", round(asm / (height / 100) ** 2, 2), "ASM / 身高²")

        tscores = {
            "腰椎": _f(facts.get("lumbar_tscore")),
            "股骨颈": _f(facts.get("femoral_neck_tscore")),
            "全髋": _f(facts.get("total_hip_tscore")),
        }
        available = {site: value for site, value in tscores.items() if value is not None}
        if available:
            site, value = min(available.items(), key=lambda item: item[1])
            write("tscore_min", value, f"取{site}等部位的最低 T 值")
            write("tscore_site_min", site, "最低 T 值所在部位")

        age = _f(facts.get("age"))
        menopause_age = _f(facts.get("menopause_age"))
        if age and menopause_age:
            write("years_since_menopause", int(age - menopause_age), "年龄 − 绝经年龄")
        if facts.get("postmenopausal") is None:
            if facts.get("sex") == "M":
                write("postmenopausal", False, "男性")
            elif facts.get("sex") == "F" and menopause_age:
                write("postmenopausal", True, "有确切绝经年龄记录")
            elif facts.get("sex") == "F" and age and age >= 60:
                write("postmenopausal", True, "女性且年龄 ≥60 岁，按绝经后处理（如有确切绝经史请补充）")

        egfr = _f(facts.get("egfr"))
        stage = drugs.ckd_stage(egfr)
        if stage:
            write("ckd_stage", stage, f"由 eGFR {egfr:g} 分期")

        calcium = _f(facts.get("corrected_calcium"))
        if calcium is None:
            calcium = _f(facts.get("calcium"))
            albumin = _f(facts.get("albumin"))
            if calcium is not None and albumin is not None:
                calcium = round(calcium + 0.02 * (40 - albumin), 2)
                write("corrected_calcium", calcium, "血钙 + 0.02 × (40 − 白蛋白)")
        if calcium is not None:
            write("hypocalcemia", calcium < 2.11, f"校正钙 {calcium:g} mmol/L 与 2.11 比较")

        vitd = _f(facts.get("vitamin_d_25oh"))
        if vitd is not None:
            write("vitd_deficient", vitd < 20, f"25-OH-D {vitd:g} ng/mL < 20")
            write("vitd_insufficient", 20 <= vitd < 30, f"25-OH-D {vitd:g} ng/mL 位于 20～30")

        parsed = drugs.parse_medications(facts.get("medications") or [])
        if parsed:
            classes = sorted({m.drug_class for m in parsed if m.drug_class})
            write("medication_classes", classes, "由药名映射到药物类别")
            write("medication_count", len(parsed), "用药条目计数")
            write("polypharmacy", len(parsed) >= 5, "用药 ≥5 种")
            frids = [m for m in parsed if m.drug_class in drugs.FALL_RISK_CLASSES]
            write("frid_count", len(frids), "增加跌倒风险的药物计数")
            on_therapy = any(m.drug_class in drugs.ANTIOSTEOPOROSIS_CLASSES for m in parsed)
            write("on_antiosteoporosis_therapy", on_therapy,
                  "是否已在使用抗骨质疏松药物（钙与维 D 属基础用药，不计入）")
            if facts.get("glucocorticoid_current") is None:
                # A supplied medication list is treated as complete *for the
                # question of which drugs are being taken* — otherwise every
                # steroid-related predicate stays unknown forever for the many
                # patients who simply are not on steroids. The assumption is
                # scoped to the list, recorded in ``derived``, and never made
                # when no list was supplied at all.
                on_gc = any(m.drug_class == "glucocorticoid" for m in parsed)
                write("glucocorticoid_current", on_gc,
                      "用药清单中" + ("含" if on_gc else "未见") + "糖皮质激素（以所提供清单为准）")

        sites = facts.get("fracture_sites") or []
        if sites and facts.get("prior_fragility_fracture") is None:
            write("prior_fragility_fracture", True, "已记录骨折部位")
        if len(sites) >= 2 and facts.get("multiple_fractures") is None:
            write("multiple_fractures", True, "记录了 2 个及以上骨折部位")

        return derived

    # -- views -----------------------------------------------------------
    def _coverage(self, facts: Mapping[str, Any]) -> dict[str, Any]:
        coverage: dict[str, Any] = {}
        for group in fact_table.PROFILE_GROUPS:
            specs = [s for s in fact_table.facts_in_group(group) if s.source != "agent"]
            filled = [s for s in specs if facts.get(s.name) is not None]
            coverage[group] = {
                "label": fact_table.GROUP_LABELS[group],
                "filled": len(filled),
                "total": len(specs),
                "ratio": round(len(filled) / len(specs), 2) if specs else 0.0,
            }
        return coverage

    def _profile_view(self, facts: Mapping[str, Any]) -> list[dict[str, Any]]:
        """The 患者数字画像 panel: grouped, labelled, units attached."""
        view: list[dict[str, Any]] = []
        for group in fact_table.PROFILE_GROUPS:
            items = []
            for spec in fact_table.facts_in_group(group):
                if spec.source == "agent":
                    continue
                value = facts.get(spec.name)
                if value is None:
                    continue
                if spec.name == "medication_classes":
                    # Stored as English class keys because the guideline
                    # predicates match on them; shown in Chinese because a
                    # clinician reads the panel, not the predicate.
                    display = "、".join(drugs.CLASS_LABELS.get(cls, cls) for cls in value)
                else:
                    display = fact_table.format_value(spec.name, value)
                items.append({
                    "fact": spec.name,
                    "label": spec.label,
                    "value": display,
                    "raw": value,
                    "derived": spec.source == "derived",
                })
            if items:
                view.append({
                    "group": group,
                    "label": fact_table.GROUP_LABELS[group],
                    "items": items,
                })
        return view

    def notes(self, payload: dict[str, Any]) -> list[str]:
        notes: list[str] = []
        unknown = payload.get("unrecognised_fields") or []
        if unknown:
            notes.append(f"记录中有 {len(unknown)} 个字段不在事实词表内，已保留但未参与推理: {unknown[:6]}")
        unrecognised_meds = [m["name"] for m in payload.get("medications", []) if not m["recognised"]]
        if unrecognised_meds:
            notes.append(
                f"未能识别的药物 {unrecognised_meds}——相互作用与跌倒风险统计不包含它们，需人工核对"
            )
        return notes
