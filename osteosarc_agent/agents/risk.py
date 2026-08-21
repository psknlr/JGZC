"""Agent 3 —— 风险预测 Agent.

Produces five risk axes rather than one verdict, because "有/没有骨质疏松"
does not determine what to do next. A patient at very high fracture risk chiefly
*because they fall* needs a different plan from one at very high risk chiefly
because their bone is weak, and the console's radar chart exists to make that
difference visible at a glance.

This agent also closes a loop the diagnosis agent could not: fracture risk
stratification depends on fall risk, and fall risk is computed here. When the
computed fall tier is high, the fracture axis is re-stratified to very high and
the upgrade is recorded as an explicit driver — never applied silently.
"""

from __future__ import annotations

from typing import Any

from ..criteria import risk as risk_model
from ..state import RunState
from .base import SubAgent

#: "unassessed" ranks *below* low: an axis nobody measured must never be able
#: to trigger an upgrade, and must never be read as reassuring either — the
#: console shows it as 未评估 rather than as a score.
_TIER_ORDER = {"unassessed": -1, "low": 0, "moderate": 1, "high": 2, "very_high": 3}


class RiskAgent(SubAgent):
    agent_id = "risk"
    name_zh = "风险预测 Agent"
    schema = "RiskProfile"
    wants = ("age", "falls_12m", "gait_speed_ms", "grip_kg", "protein_g_per_kg", "mna_sf")

    def run(self, state: RunState) -> dict[str, Any]:
        axes = risk_model.profile(state.facts)
        by_axis = {axis.axis: axis for axis in axes}

        upgrades: list[str] = []
        fall = by_axis["fall"]
        fracture = by_axis["fracture"]
        very_high = bool(state.facts.get("very_high_fracture_risk"))
        if (
            state.facts.get("dx_osteoporosis")
            and _TIER_ORDER[fall.tier] >= _TIER_ORDER["high"]
            and not very_high
        ):
            very_high = True
            upgrades.append(
                f"跌倒风险为「{fall.tier_zh}」，按极高骨折风险处理（骨折风险分层须同时包含跌倒维度）"
            )

        # The composite score and the categorical stratification answer
        # different questions, and where they disagree the guideline rules win:
        # "极高骨折风险" is defined by named criteria (recent fracture, T ≤ -3.0,
        # fracture on therapy…), not by a weighted sum. The score stays as
        # published so the radar keeps comparable magnitudes across axes; only
        # the tier is overridden, and the override is stated.
        axis_dicts = [axis.to_dict() for axis in axes]
        if very_high:
            for entry in axis_dicts:
                if entry["axis"] == "fracture" and entry["tier"] != "very_high":
                    entry["tier"] = "very_high"
                    entry["tier_zh"] = "极高"
                    entry["note"] = (
                        (entry["note"] + " " if entry["note"] else "")
                        + "分层由指南分层规则判定为极高（不取决于合成分数）。"
                    )
                    upgrades.append("骨折风险分层按指南分层规则判定为极高，覆盖合成分数对应的分层")

        payload = {
            "axes": axis_dicts,
            "radar": {"axes": [
                {"axis": entry["axis"], "label": entry["name_zh"], "score": entry["score"],
                 "tier": entry["tier"], "tier_zh": entry["tier_zh"],
                 "confidence": entry["confidence"]}
                for entry in axis_dicts
            ], "model_note": risk_model.MODEL_NOTE},
            "headline": self._headline(axes, state),
            "model_note": risk_model.MODEL_NOTE,
            "very_high_fracture_risk": very_high,
            "upgrades": upgrades,
            "low_confidence_axes": [
                {"axis": axis.axis, "label": axis.name_zh, "confidence": round(axis.confidence, 2),
                 "missing": list(axis.missing)}
                for axis in axes if axis.confidence < 0.6
            ],
            "dominant": max(axes, key=lambda a: a.score).axis,
            "unassessed_axes": [
                entry["axis"] for entry in axis_dicts if entry["tier"] == "unassessed"
            ],
        }
        return payload

    def _headline(self, axes: list[risk_model.AxisResult], state: RunState) -> str:
        ranked = sorted(axes, key=lambda a: -a.score)
        top = ranked[0]
        second = ranked[1]
        bone = state.facts.get("dx_osteoporosis")
        muscle = state.facts.get("dx_sarcopenia_any") or state.facts.get("dx_possible_sarcopenia")
        lead = "骨肌共病" if bone and muscle else ("骨质疏松" if bone else "骨骼与肌肉功能")
        return (
            f"{lead}背景下，风险主要集中在{top.name_zh}（{top.tier_zh}，{top.score}）"
            f"与{second.name_zh}（{second.tier_zh}，{second.score}）。"
        )

    def derived_facts(self, payload: dict[str, Any]) -> dict[str, Any]:
        tiers = {axis["axis"]: axis["tier"] for axis in payload["axes"]}
        # An unassessed axis writes no tier fact: downstream predicates must not
        # be able to read "未评估" as "低".
        tiers = {axis: tier for axis, tier in tiers.items() if tier != "unassessed"}
        return {
            "fracture_risk_tier": tiers.get("fracture"),
            "fall_risk_tier": tiers.get("fall"),
            "sarcopenia_risk_tier": tiers.get("sarcopenia"),
            "nutrition_risk_tier": tiers.get("nutrition"),
            "functional_risk_tier": tiers.get("functional"),
            "high_fall_risk": _TIER_ORDER.get(tiers.get("fall", "low"), 0) >= _TIER_ORDER["high"],
            "very_high_fracture_risk": payload["very_high_fracture_risk"],
        }

    def notes(self, payload: dict[str, Any]) -> list[str]:
        notes = list(payload["upgrades"])
        for axis in payload["low_confidence_axes"]:
            notes.append(
                f"{axis['label']}的评分置信度仅 {axis['confidence']:.0%}（缺 {', '.join(axis['missing'])}）"
                "——分数偏低可能只是没测，不代表风险低"
            )
        return notes
