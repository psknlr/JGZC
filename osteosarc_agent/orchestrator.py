"""主智能体 —— OsteoSarc Orchestrator.

Runs the six sub-agents in dependency order and assembles one decision object.
The order is not cosmetic; each step depends on facts the previous one wrote
back into the shared namespace:

    病例结构化 → 骨肌诊断 → 风险预测 → 循证决策 → 指南冲突消解 → 安全审查与随访

* the diagnosis agent cannot run until intake has derived ``asmi``, ``tscore_min``
  and the calcium/vitamin-D flags;
* the guideline predicates cannot be evaluated until the diagnosis and risk
  agents have written ``dx_*``, ``*_risk_tier`` and ``very_high_fracture_risk``
  back — that write-back is what lets a corpus record say "适用于极高骨折风险者"
  instead of re-deriving it;
* the safety agent must run **last**, over the assembled plan, and its blocking
  findings gate the drug portion of that plan.

What the orchestrator owns that no sub-agent does:

* the run's status. A blocked safety audit makes the whole run ``needs_action``;
  a sub-agent that failed makes it ``degraded`` and its silence is never read as
  "no findings";
* the synthesis panel (「AI 判断」), where every statement carries the evidence ids
  it rests on;
* refusing to publish a plan when the pipeline did not actually complete.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from . import plans
from .agents import (
    ConflictAgent, DiagnosisAgent, EvidenceAgent, IntakeAgent, RiskAgent, SafetyAgent,
)
from .criteria import sarcopenia as sarco
from .guidelines.corpus import Corpus, default_corpus
from .state import RunState

__all__ = ["Orchestrator", "assess"]

PLATFORM = "筋骨智策 OsteoSarc-Agent"
VERSION = "0.1.0"

#: Shown on every screen that displays a recommendation. The shipped corpus is
#: an editorial paraphrase of public guidance, not licensed guideline text, and
#: a decision-support tool that hides that fact is misrepresenting its own
#: evidence base.
CORPUS_DISCLAIMER = (
    "本平台内置语料为公开指南要点的编辑性转述，非授权原文；"
    "所有结论须由医师核对原文后使用，本平台不生成任何药物剂量。"
)


class Orchestrator:
    """The main agent: plans the run, executes it, assembles the decision."""

    def __init__(
        self,
        corpus: Corpus | None = None,
        standards: tuple[str, ...] = sarco.DEFAULT_STANDARDS,
        narrator: Any | None = None,
    ) -> None:
        self.corpus = corpus or default_corpus()
        self.standards = standards
        #: Optional cognition layer. It may only rephrase what the deterministic
        #: pipeline already concluded — see :mod:`osteosarc_agent.llm`.
        self.narrator = narrator
        self.intake = IntakeAgent()
        self.diagnosis = DiagnosisAgent(standards)
        self.risk = RiskAgent()
        self.evidence = EvidenceAgent(self.corpus)
        self.conflict = ConflictAgent(self.corpus)
        self.safety = SafetyAgent()

    @property
    def pipeline(self) -> tuple[Any, ...]:
        return (self.intake, self.diagnosis, self.risk, self.evidence, self.conflict, self.safety)

    def agent_catalog(self) -> list[dict[str, str]]:
        return [
            {"agent_id": agent.agent_id, "name_zh": agent.name_zh, "schema": agent.schema}
            for agent in self.pipeline
        ]

    # -- run -------------------------------------------------------------
    def run(self, case: Mapping[str, Any]) -> dict[str, Any]:
        state = RunState(case=dict(case))
        for agent in self.pipeline:
            agent(state)

        evidence_payload = state.payload("evidence")
        conflict_payload = state.payload("conflict")
        safety_payload = state.payload("safety")

        care_plans = plans.build(
            state.facts,
            evidence_payload.get("applicable", []),
            evidence_payload.get("excluded", []),
            conflict_payload.get("questions", []),
        )
        care_plans = self._gate_plans(care_plans, safety_payload)

        status = self._status(state, safety_payload)
        decision = {
            "meta": {
                "platform": PLATFORM,
                "version": VERSION,
                "case_id": state.case.get("case_id", ""),
                "case_label": state.case.get("label", ""),
                "standards": list(self.standards),
                "corpus": self.corpus.stats(),
                "disclaimer": CORPUS_DISCLAIMER,
                "status": status,
            },
            "profile": {
                "groups": state.payload("intake").get("profile", []),
                "coverage": state.payload("intake").get("coverage", {}),
                "missing_critical": state.payload("intake").get("missing_critical", []),
                "derived": state.payload("intake").get("derived", []),
                "medications": state.payload("intake").get("medications", []),
                "facts": dict(state.facts),
            },
            "diagnosis": state.payload("diagnosis"),
            "risk": state.payload("risk"),
            "evidence_selection": {
                key: evidence_payload.get(key)
                for key in ("by_topic", "data_gaps", "insufficient", "excluded",
                            "not_applicable_count", "corpus")
            },
            "conflicts": conflict_payload,
            "plans": care_plans,
            "safety": safety_payload,
            "judgement": self._judgement(state, safety_payload, conflict_payload, evidence_payload),
            "evidence": state.ledger.to_dict(),
            "agents": [result.to_dict() for result in state.results.values()],
            "status": status,
        }
        if self.narrator is not None:
            decision["narrative"] = self.narrator.narrate(decision)
        return decision

    # -- assembly --------------------------------------------------------
    def _gate_plans(self, care_plans: dict[str, Any], safety_payload: Mapping[str, Any]) -> dict[str, Any]:
        """Attach blocking safety findings to the plan cards they gate."""
        blocking = safety_payload.get("blocking", []) or []
        if not blocking:
            return care_plans
        titles = [issue["title"] for issue in blocking]
        for column in care_plans["columns"]:
            if column["column"] != plans.BONE:
                continue
            for item in column["items"]:
                if "用药前置" in item["tags"] or item["plan_id"] in (
                    "bone.renal_pathway", "bone.denosumab_continuity", "bone.secondary_prevention"
                ):
                    item["gated"] = True
                    item["gate_reason"] = "以下阻断性问题解决前不得下达药物部分：" + "；".join(titles)
        care_plans["gated_by_safety"] = titles
        return care_plans

    def _status(self, state: RunState, safety_payload: Mapping[str, Any]) -> str:
        failed = [r for r in state.results.values() if r.status != "ok"]
        if failed:
            return "degraded"
        if safety_payload.get("blocking"):
            return "needs_action"
        return "ok"

    def _judgement(
        self,
        state: RunState,
        safety_payload: Mapping[str, Any],
        conflict_payload: Mapping[str, Any],
        evidence_payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        diagnosis = state.payload("diagnosis")
        risk = state.payload("risk")
        statements: list[dict[str, Any]] = []

        def add(kind: str, text: str, evidence: Sequence[str] = (), detail: str = "") -> None:
            statements.append({"kind": kind, "text": text, "detail": detail,
                               "evidence": list(evidence)})

        if diagnosis:
            osteo = diagnosis["osteoporosis"]
            add("diagnosis", f"{osteo['diagnosis_zh']}——{osteo['basis']}",
                _evidence_for(evidence_payload, ("CN.OP.2022.DX_CLINICAL_FRACTURE",
                                                 "CN.OP.2022.DX_TSCORE")))
            if osteo["very_high_risk"]:
                drivers = "、".join(d["label"] for d in osteo["risk_drivers"][:3])
                add("risk", f"骨折风险分层：极高（{drivers}）",
                    _evidence_for(evidence_payload, ("US.AACE.2020.VHR_DEFINITION",)))
            for standard in diagnosis["standards"]:
                add("standard",
                    f"{standard['name_zh']}：{standard['verdict_zh']}",
                    (), standard["reason"])
            if not diagnosis["sarcopenia"]["summary"]["agreement"]:
                add("conflict", "三套肌少症标准结论不一致——分歧来源见下",
                    (), "；".join(diagnosis["sarcopenia"]["cross_standard_notes"]))
            add("diagnosis", diagnosis["osteosarcopenia"]["text"],
                _evidence_for(evidence_payload, ("CN.GERI.OP.2023.OSTEOSARCOPENIA",)),
                "；".join(diagnosis["osteosarcopenia"]["interactions"]))

        if risk:
            add("risk", risk["headline"], (), risk["model_note"])

        for question in conflict_payload.get("disputed", []) or []:
            add("conflict", f"存在争议：{question['label_zh']}",
                question.get("evidence", []), question.get("basis", ""))
        for question in conflict_payload.get("resolved", []) or []:
            facts = "、".join(
                fact["detail"] or fact["label"]
                for fact in question.get("resolution", {}).get("deciding_facts", [])
            )
            add("resolved", f"争议已被本例事实消解：{question['label_zh']}",
                question.get("evidence", []), f"决定性事实：{facts}" if facts else "")

        for issue in safety_payload.get("blocking", []) or []:
            add("blocking", f"阻断：{issue['title']}", issue.get("evidence", []), issue["action"])

        # Critical gaps lead: a missing eGFR blocks a decision, while an
        # unrecorded TCM sign merely leaves one recommendation undecided. Sorting
        # purely by "how many records this unlocks" would bury the former.
        uncertainties: list[str] = []
        for missing in state.payload("intake").get("missing_critical", [])[:5]:
            uncertainties.append(f"缺 {missing['label']}：影响{missing['blocks']}")
        for gap in (evidence_payload.get("data_gaps") or [])[:4]:
            uncertainties.append(gap["hint"])
        for axis in risk.get("low_confidence_axes", []) if risk else []:
            uncertainties.append(
                f"{axis['label']}评分置信度 {axis['confidence']:.0%}——分数低可能只是没测"
            )

        failed = [r for r in state.results.values() if r.status != "ok"]
        return {
            "headline": risk.get("headline", "") if risk else "评估未完成",
            "statements": statements,
            "uncertainties": uncertainties,
            "agent_failures": [
                {"agent_id": r.agent_id, "name_zh": r.name_zh, "problems": r.problems}
                for r in failed
            ],
            "notes": [note for r in state.results.values() for note in r.notes],
        }


def _evidence_for(evidence_payload: Mapping[str, Any], rec_ids: Sequence[str]) -> list[str]:
    by_rec = {
        entry["rec_id"]: entry["evidence_id"]
        for entry in (evidence_payload.get("applicable") or [])
    }
    return [by_rec[rec_id] for rec_id in rec_ids if rec_id in by_rec]


def assess(case: Mapping[str, Any], **kwargs: Any) -> dict[str, Any]:
    """Convenience entry point: run one case through the default pipeline."""
    return Orchestrator(**kwargs).run(case)
