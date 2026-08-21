"""Agent 2 —— 骨肌诊断 Agent.

Calls the osteoporosis and sarcopenia criteria **in one pass**, not as two
independent consultations. Two reasons this matters clinically:

* The same measurement serves both diseases. A 5-times-sit-to-stand of 16 s is
  a sarcopenia performance criterion, a fall-risk marker, and — through falls —
  a fracture-risk driver. Running the two workups separately makes the platform
  ask for it twice and reason about it once.
* The combination is its own diagnosis. Osteosarcopenia carries a higher risk of
  falls, fracture, disability and death than either condition alone, and it has
  a shared treatment core (resistance training, protein, vitamin D) that neither
  single-disease pathway prescribes on its own.

The sarcopenia side runs **every** configured standard and reports all of them
with the reason each reached its verdict, plus notes naming the measurement that
made them differ. Picking one standard silently would hide exactly the
information a clinician needs to judge which one to trust for this patient.
"""

from __future__ import annotations

from typing import Any

from ..criteria import osteoporosis as op_criteria
from ..criteria import sarcopenia as sarco
from ..guidelines.model import TRUE, UNKNOWN
from ..state import RunState
from .base import SubAgent


class DiagnosisAgent(SubAgent):
    agent_id = "diagnosis"
    name_zh = "骨肌诊断 Agent"
    schema = "OsteoSarcDiagnosis"
    wants = ("tscore_min", "grip_kg", "gait_speed_ms", "sts5_s", "asmi")

    def __init__(self, standards: tuple[str, ...] = sarco.DEFAULT_STANDARDS) -> None:
        self.standards = standards

    def run(self, state: RunState) -> dict[str, Any]:
        facts = state.facts
        osteo = op_criteria.assess(facts)
        verdicts = sarco.assess_all(facts, self.standards)
        summary = sarco.summarise(verdicts)
        cross_notes = sarco.cross_standard_notes(verdicts, facts)

        primary = self._primary(verdicts)
        osteosarcopenia = self._osteosarcopenia(osteo, verdicts, summary)

        return {
            "osteoporosis": osteo.to_dict(),
            "sarcopenia": {
                "summary": summary,
                "primary_standard": primary.standard_id if primary else None,
                "primary_verdict": primary.verdict if primary else None,
                "primary_verdict_zh": primary.verdict_zh if primary else "无可用判定",
                "cross_standard_notes": cross_notes,
                "measurements_missing": sorted({
                    axis for verdict in verdicts for axis in verdict.missing
                }),
            },
            "standards": [verdict.to_dict() for verdict in verdicts],
            "osteosarcopenia": osteosarcopenia,
        }

    def _primary(self, verdicts: list[sarco.StandardVerdict]) -> sarco.StandardVerdict | None:
        """The standard whose verdict drives downstream facts.

        Not "the strictest" and not "the most alarming": the first configured
        standard that reached a conclusion. The default order puts AWGS first
        because the deployment population is Chinese — a deployment choice that
        is stated rather than hidden, and that the console shows alongside the
        other two verdicts so it can be overruled by a reader.
        """
        for verdict in verdicts:
            if verdict.verdict != sarco.INDETERMINATE:
                return verdict
        return verdicts[0] if verdicts else None

    def _osteosarcopenia(
        self,
        osteo: op_criteria.OsteoVerdict,
        verdicts: list[sarco.StandardVerdict],
        summary: dict[str, Any],
    ) -> dict[str, Any]:
        bone_positive = osteo.diagnosis == op_criteria.OSTEOPOROSIS
        muscle_confirmed = summary["any_confirmed"]
        muscle_actionable = summary["any_actionable"]

        if bone_positive and muscle_confirmed:
            status = "confirmed"
            text = "骨肌共病成立：骨质疏松与肌少症并存"
        elif bone_positive and muscle_actionable:
            status = "probable"
            text = "骨肌共病可能成立：骨质疏松确诊，肌少症达到「可能」层级（补测肌量可确认）"
        elif bone_positive:
            status = "bone_only"
            text = "目前仅骨质疏松成立"
        elif muscle_actionable:
            status = "muscle_only"
            text = "目前仅肌少症相关判定成立"
        else:
            status = "neither"
            text = "两者均未成立"

        interactions: list[str] = []
        if status in ("confirmed", "probable"):
            interactions.append(
                "共病者跌倒、骨折、失能与死亡风险高于任一单一疾病，应作为一个管理目标而非两条平行路径"
            )
            interactions.append(
                "运动、蛋白质与维生素 D 是两病共享的干预核心，一份处方同时服务两个诊断"
            )
            interactions.append(
                "仅提升骨密度而不改善肌力与平衡，不能降低跌倒相关骨折——骨折 = 骨强度不足 × 跌倒"
            )
        elif status == "muscle_only" and not bone_positive:
            interactions.append(
                "即使骨密度未达骨质疏松标准，肌力与平衡下降仍通过跌倒抬高骨折风险，需同时评估骨骼"
            )

        shared: list[str] = []
        for verdict in verdicts:
            for component in (verdict.strength, verdict.performance):
                for measure in component.measures:
                    if measure in ("sts5_s", "gait_speed_ms") and measure not in shared:
                        shared.append(measure)
        if shared:
            from ..guidelines.facts import describe_fact
            labels = "、".join(describe_fact(name) for name in shared)
            interactions.append(
                f"同一项测量同时服务两个诊断与跌倒评估（{labels}），不需要重复采集"
            )

        return {
            "status": status,
            "text": text,
            "bone_positive": bone_positive,
            "muscle_confirmed": muscle_confirmed,
            "muscle_actionable": muscle_actionable,
            "interactions": interactions,
        }

    def derived_facts(self, payload: dict[str, Any]) -> dict[str, Any]:
        osteo = payload["osteoporosis"]
        sarcopenia = payload["sarcopenia"]
        standards = payload["standards"]
        summary = sarcopenia["summary"]

        # Component flags: a component counts as low when *any* standard's
        # measurement says so. Different standards use different cutoffs, and
        # for the downstream risk model "this patient's grip is低" is the
        # clinically relevant statement, not "低 under AWGS specifically".
        def component_low(axis: str) -> bool | None:
            results = [s["components"][axis]["result"] for s in standards]
            if TRUE in results:
                return True
            if all(result == UNKNOWN for result in results):
                return None
            return False

        mass_measured = any(
            s["components"]["mass"]["result"] != UNKNOWN for s in standards
        )
        facts: dict[str, Any] = {
            "dx_osteoporosis": osteo["diagnosis"] == op_criteria.OSTEOPOROSIS,
            "dx_low_bone_mass": osteo["diagnosis"] == op_criteria.LOW_BONE_MASS,
            "dx_severe_osteoporosis": bool(osteo["severe"]),
            "dx_osteosarcopenia": payload["osteosarcopenia"]["status"] in ("confirmed", "probable"),
            "dx_sarcopenia_any": bool(summary["any_confirmed"]),
            "dx_possible_sarcopenia": bool(summary["any_actionable"]) and not summary["any_confirmed"],
            "dx_severe_sarcopenia": bool(summary["severe"]),
            "muscle_mass_measured": mass_measured,
            "very_high_fracture_risk": bool(osteo["very_high_risk"]),
        }
        for standard in standards:
            key = {
                "AWGS2019": "dx_sarcopenia_awgs",
                "EWGSOP2": "dx_sarcopenia_ewgsop2",
                "GLIS2024": "dx_sarcopenia_glis",
            }.get(standard["standard_id"])
            if key:
                facts[key] = standard["verdict"]
        for axis, fact_name in (("strength", "low_muscle_strength"),
                                ("performance", "low_physical_performance"),
                                ("mass", "low_muscle_mass")):
            value = component_low(axis)
            if value is not None:
                facts[fact_name] = value
        return facts

    def notes(self, payload: dict[str, Any]) -> list[str]:
        notes = list(payload["sarcopenia"]["cross_standard_notes"])
        if not payload["sarcopenia"]["summary"]["agreement"]:
            notes.insert(0, "三套肌少症标准结论不一致——下方逐条列出各自的判定依据与分歧来源")
        notes.extend(payload["osteoporosis"].get("caveats", []))
        return notes
