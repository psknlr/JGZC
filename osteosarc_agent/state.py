"""Run state and the evidence ledger.

The ledger is the reason a recommendation in the console can be clicked. Every
assertion the platform makes carries the ids of the guideline records it came
from, and each id resolves to the record, its source, and the applicability
trace that shows which of *this patient's* facts made it fire.

Ids are assigned in first-citation order and are stable for a given run, so a
printed report and the console show the same numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from .guidelines.model import Applicability


@dataclass
class EvidenceEntry:
    evidence_id: str
    applicability: Applicability
    cited_by: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = self.applicability.to_dict()
        payload["evidence_id"] = self.evidence_id
        payload["cited_by"] = list(self.cited_by)
        return payload


class EvidenceLedger:
    """Assigns and resolves evidence ids."""

    def __init__(self, prefix: str = "E") -> None:
        self.prefix = prefix
        self._entries: dict[str, EvidenceEntry] = {}
        self._by_rec: dict[str, str] = {}

    def cite(self, applicability: Applicability, cited_by: str = "") -> str:
        rec_id = applicability.recommendation.rec_id
        evidence_id = self._by_rec.get(rec_id)
        if evidence_id is None:
            evidence_id = f"{self.prefix}{len(self._entries) + 1:02d}"
            self._by_rec[rec_id] = evidence_id
            self._entries[evidence_id] = EvidenceEntry(evidence_id, applicability)
        entry = self._entries[evidence_id]
        if cited_by and cited_by not in entry.cited_by:
            entry.cited_by.append(cited_by)
        return evidence_id

    def cite_all(self, items: Iterable[Applicability], cited_by: str = "") -> list[str]:
        return [self.cite(item, cited_by) for item in items]

    def get(self, evidence_id: str) -> EvidenceEntry | None:
        return self._entries.get(evidence_id)

    def for_recommendation(self, rec_id: str) -> str | None:
        return self._by_rec.get(rec_id)

    def entries(self) -> list[EvidenceEntry]:
        return list(self._entries.values())

    def to_dict(self) -> dict[str, Any]:
        return {eid: entry.to_dict() for eid, entry in self._entries.items()}

    def __len__(self) -> int:
        return len(self._entries)


@dataclass
class AgentResult:
    """One sub-agent's output plus its bookkeeping."""

    agent_id: str
    name_zh: str
    schema: str
    payload: dict[str, Any]
    status: str = "ok"          # ok | degraded | blocked
    evidence: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "name_zh": self.name_zh,
            "schema": self.schema,
            "status": self.status,
            "payload": self.payload,
            "evidence": list(self.evidence),
            "notes": list(self.notes),
            "problems": list(self.problems),
        }


@dataclass
class RunState:
    """Everything one case run accumulates.

    ``facts`` is a single mutable namespace shared by every agent: intake writes
    it, the diagnosis and risk agents write derived flags back into it, and the
    guideline predicates read it. That write-back is what lets a corpus record
    say "适用于已诊断肌少症的患者" without re-deriving the diagnosis itself.
    """

    case: dict[str, Any] = field(default_factory=dict)
    facts: dict[str, Any] = field(default_factory=dict)
    results: dict[str, AgentResult] = field(default_factory=dict)
    ledger: EvidenceLedger = field(default_factory=EvidenceLedger)
    trace: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def record(self, result: AgentResult) -> AgentResult:
        self.results[result.agent_id] = result
        self.trace.append(f"{result.agent_id}:{result.status}")
        return result

    def payload(self, agent_id: str) -> dict[str, Any]:
        result = self.results.get(agent_id)
        return result.payload if result else {}

    def update_facts(self, updates: Mapping[str, Any]) -> list[str]:
        """Write derived facts back into the shared namespace."""
        written: list[str] = []
        for key, value in updates.items():
            if value is None:
                continue
            self.facts[key] = value
            written.append(key)
        return written
