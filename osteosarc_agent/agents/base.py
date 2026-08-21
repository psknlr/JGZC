"""Sub-agent base class.

Every sub-agent is a function from the shared :class:`~osteosarc_agent.state.RunState`
to a payload that must validate against a declared schema. The base class owns
three guarantees that individual agents must not be able to opt out of:

* **Schema-checked output.** An agent whose payload fails validation is recorded
  as ``blocked`` with the problems attached, rather than passing a malformed
  payload downstream.
* **Contained failure.** An unexpected exception degrades that one agent and is
  reported; it does not abort the run. A safety audit that never ran is
  reported as not-run — the orchestrator refuses to publish a plan in that case.
* **Explicit fact write-back.** Agents that derive new facts return them from
  :meth:`derived_facts` so the write into the shared namespace is visible in the
  result, not a side effect buried in the body.
"""

from __future__ import annotations

import traceback
from typing import Any

from ..schemas import validate
from ..state import AgentResult, RunState


class SubAgent:
    agent_id: str = ""
    name_zh: str = ""
    schema: str = ""
    #: What this agent needs before it can say anything useful. Purely
    #: informational — an agent still runs with missing inputs and reports the
    #: gap, because "无法评估" is itself a clinically important output.
    wants: tuple[str, ...] = ()

    def run(self, state: RunState) -> dict[str, Any]:  # pragma: no cover - abstract
        raise NotImplementedError

    def derived_facts(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Facts this agent writes back into the shared namespace."""
        return {}

    def notes(self, payload: dict[str, Any]) -> list[str]:
        return []

    def evidence(self, payload: dict[str, Any]) -> list[str]:
        return list(payload.get("evidence", []) or [])

    def __call__(self, state: RunState) -> AgentResult:
        try:
            payload = self.run(state)
        except Exception as exc:  # noqa: BLE001 - contained on purpose
            result = AgentResult(
                agent_id=self.agent_id,
                name_zh=self.name_zh,
                schema=self.schema,
                payload={},
                status="blocked",
                problems=[f"{type(exc).__name__}: {exc}"],
                notes=["该 Agent 未能完成，其结论不得被视为「无异常」"],
            )
            result.problems.append(traceback.format_exc(limit=3))
            return state.record(result)

        ok, problems = validate(self.schema, payload)
        status = "ok" if ok else "blocked"
        result = AgentResult(
            agent_id=self.agent_id,
            name_zh=self.name_zh,
            schema=self.schema,
            payload=payload if ok else {},
            status=status,
            evidence=self.evidence(payload) if ok else [],
            notes=self.notes(payload) if ok else ["输出不符合 schema，已丢弃"],
            problems=problems,
        )
        if ok:
            written = state.update_facts(self.derived_facts(payload))
            if written:
                result.notes.append(f"回写事实: {', '.join(written)}")
        return state.record(result)
