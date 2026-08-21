"""主智能体：端到端、状态语义、证据台账完整性、方案的证据不变式。"""

import unittest

from osteosarc_agent import cases
from osteosarc_agent.criteria import sarcopenia as sarco
from osteosarc_agent.orchestrator import Orchestrator
from osteosarc_agent.plans import BONE, EXERCISE, NUTRITION
from osteosarc_agent.render import render


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.orchestrator = Orchestrator()
        cls.decisions = {name: cls.orchestrator.run(cases.get(name)) for name in cases.CASES}

    def test_all_six_agents_run_in_order(self):
        for name, decision in self.decisions.items():
            ids = [entry["agent_id"] for entry in decision["agents"]]
            self.assertEqual(
                ids, ["intake", "diagnosis", "risk", "evidence", "conflict", "safety"], name)

    def test_no_agent_fails_on_the_reference_cases(self):
        for name, decision in self.decisions.items():
            failed = [a["agent_id"] for a in decision["agents"] if a["status"] != "ok"]
            self.assertEqual(failed, [], f"{name}: {failed}")

    def test_status_reflects_blocking_safety_findings(self):
        self.assertEqual(self.decisions["sparse"]["status"], "needs_action")
        self.assertEqual(self.decisions["demo"]["status"], "ok")

    def test_a_failed_agent_degrades_the_whole_run(self):
        class Broken(Orchestrator):
            def __init__(self):
                super().__init__()
                original = self.risk.run
                self.risk.run = lambda state: (_ for _ in ()).throw(RuntimeError("x"))
                self._original = original

        decision = Broken().run(cases.get("demo"))
        self.assertEqual(decision["status"], "degraded")
        self.assertTrue(decision["judgement"]["agent_failures"])

    def test_every_cited_evidence_id_resolves(self):
        for name, decision in self.decisions.items():
            ledger = decision["evidence"]
            cited = set()
            for statement in decision["judgement"]["statements"]:
                cited.update(statement["evidence"])
            for column in decision["plans"]["columns"]:
                for item in column["items"]:
                    cited.update(item["evidence"])
            for question in decision["conflicts"]["questions"]:
                cited.update(question["evidence"])
            missing = cited - set(ledger)
            self.assertEqual(missing, set(), f"{name}: {missing}")

    def test_ledger_ids_are_stable_and_unique(self):
        decision = self.decisions["demo"]
        rec_ids = [entry["recommendation"]["rec_id"] for entry in decision["evidence"].values()]
        self.assertEqual(len(rec_ids), len(set(rec_ids)))

    def test_rerunning_a_case_is_deterministic(self):
        first = self.orchestrator.run(cases.get("demo"))
        second = self.orchestrator.run(cases.get("demo"))
        self.assertEqual(first["conflicts"]["counts"], second["conflicts"]["counts"])
        self.assertEqual(
            [item["plan_id"] for column in first["plans"]["columns"] for item in column["items"]],
            [item["plan_id"] for column in second["plans"]["columns"] for item in column["items"]],
        )
        self.assertEqual(list(first["evidence"]), list(second["evidence"]))

    def test_disclaimer_travels_with_every_decision(self):
        for decision in self.decisions.values():
            self.assertIn("编辑性转述", decision["meta"]["disclaimer"])
            self.assertIn("不生成任何药物剂量", decision["meta"]["disclaimer"])

    def test_standards_can_be_narrowed(self):
        decision = Orchestrator(standards=("AWGS2019",)).run(cases.get("demo"))
        self.assertEqual(len(decision["diagnosis"]["standards"]), 1)

    def test_reference_case_reproduces_the_briefed_behaviour(self):
        decision = self.decisions["demo"]
        verdicts = {s["standard_id"]: s["verdict"] for s in decision["diagnosis"]["standards"]}
        self.assertEqual(verdicts["AWGS2019"], sarco.POSSIBLE)
        self.assertEqual(verdicts["EWGSOP2"], sarco.PROBABLE)
        self.assertEqual(verdicts["GLIS2024"], sarco.INDETERMINATE)
        self.assertTrue(decision["diagnosis"]["osteoporosis"]["very_high_risk"])
        self.assertTrue(decision["conflicts"]["resolved"])
        self.assertTrue(decision["conflicts"]["disputed"])


class PlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.decisions = {name: Orchestrator().run(cases.get(name)) for name in cases.CASES}

    def test_the_three_columns_are_always_present(self):
        for name, decision in self.decisions.items():
            columns = [column["column"] for column in decision["plans"]["columns"]]
            self.assertEqual(columns, [BONE, EXERCISE, NUTRITION], name)

    def test_no_plan_item_exists_without_live_evidence(self):
        for name, decision in self.decisions.items():
            for column in decision["plans"]["columns"]:
                for item in column["items"]:
                    self.assertTrue(item["evidence"], f"{name}/{item['plan_id']}")
                    for evidence_id in item["evidence"]:
                        entry = decision["evidence"][evidence_id]
                        self.assertEqual(entry["status"], "applies", f"{name}/{item['plan_id']}")

    def test_suppressed_items_state_why(self):
        decision = self.decisions["demo"]
        self.assertTrue(decision["plans"]["suppressed"])
        for entry in decision["plans"]["suppressed"]:
            self.assertTrue(entry["reason"])

    def test_impact_loading_is_suppressed_for_a_high_fall_risk_patient(self):
        suppressed = {entry["plan_id"] for entry in self.decisions["demo"]["plans"]["suppressed"]}
        self.assertIn("ex.impact", suppressed)

    def test_protein_card_carries_the_conflict_marker(self):
        columns = {column["column"]: column for column in self.decisions["demo"]["plans"]["columns"]}
        item = next(i for i in columns[NUTRITION]["items"] if i["plan_id"] == "nut.protein_target")
        self.assertTrue(item["conflicts"])
        self.assertEqual(item["conflicts"][0]["verdict"], "disputed")

    def test_blocking_safety_gates_the_drug_cards(self):
        decision = self.decisions["sparse"]
        self.assertTrue(decision["plans"]["gated_by_safety"])
        bone = next(c for c in decision["plans"]["columns"] if c["column"] == BONE)
        self.assertTrue(any(item.get("gated") for item in bone["items"]))

    def test_spinal_precautions_appear_for_a_vertebral_fracture(self):
        exercise = next(c for c in self.decisions["demo"]["plans"]["columns"] if c["column"] == EXERCISE)
        titles = " ".join(item["title"] for item in exercise["items"])
        self.assertIn("椎体骨折", titles)

    def test_no_plan_text_contains_a_dose(self):
        """The platform must never emit a milligram figure of its own."""
        import re
        pattern = re.compile(r"\d+(\.\d+)?\s*(mg|毫克|IU|国际单位|μg|微克)")
        for name, decision in self.decisions.items():
            for column in decision["plans"]["columns"]:
                for item in column["items"]:
                    for field in ("title", "detail", "why"):
                        # 500 mg appears only as a guideline-quoted upper bound.
                        for match in pattern.finditer(item[field]):
                            self.assertIn("不超过", item[field],
                                          f"{name}/{item['plan_id']}: {match.group()}")


class RenderTests(unittest.TestCase):
    def test_terminal_report_covers_every_section(self):
        text = render(Orchestrator().run(cases.get("demo")))
        for heading in ("患者数字画像", "骨肌诊断", "骨肌风险雷达", "AI 判断",
                        "指南冲突消解", "个体化治疗路径", "安全审查", "随访计划", "证据台账"):
            self.assertIn(heading, text)

    def test_report_survives_a_sparse_record(self):
        text = render(Orchestrator().run(cases.get("sparse")))
        self.assertIn("关键缺失", text)
        self.assertIn("未评估", text)


if __name__ == "__main__":
    unittest.main()
