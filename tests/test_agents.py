"""六个子智能体：契约、推导、分桶、冲突判定、安全审查完整性。"""

import unittest

from osteosarc_agent import cases
from osteosarc_agent.agents import (
    ConflictAgent, DiagnosisAgent, EvidenceAgent, IntakeAgent, RiskAgent, SafetyAgent,
)
from osteosarc_agent.agents.conflict import CONSENSUS, DISPUTED, RESOLVED_BY_PATIENT
from osteosarc_agent.schemas import REQUIRED_SAFETY_CHECKS, validate
from osteosarc_agent.state import RunState


def _pipeline(case, upto="safety"):
    state = RunState(case=dict(case))
    for agent in (IntakeAgent(), DiagnosisAgent(), RiskAgent(),
                  EvidenceAgent(), ConflictAgent(), SafetyAgent()):
        agent(state)
        if agent.agent_id == upto:
            break
    return state


class IntakeTests(unittest.TestCase):
    def test_derives_bmi_asmi_and_tscore(self):
        state = _pipeline({"height_cm": 160, "weight_kg": 51.2, "asm_kg": 14.0,
                           "lumbar_tscore": -2.9, "femoral_neck_tscore": -2.6, "sex": "F", "age": 70},
                          upto="intake")
        self.assertEqual(state.facts["bmi"], 20.0)
        self.assertEqual(state.facts["asmi"], 5.47)
        self.assertEqual(state.facts["tscore_min"], -2.9)
        self.assertEqual(state.facts["tscore_site_min"], "腰椎")

    def test_corrects_calcium_for_albumin(self):
        state = _pipeline({"calcium": 2.18, "albumin": 34, "sex": "F", "age": 78}, upto="intake")
        self.assertEqual(state.facts["corrected_calcium"], 2.3)
        self.assertFalse(state.facts["hypocalcemia"])

    def test_flags_hypocalcaemia(self):
        state = _pipeline({"corrected_calcium": 2.0, "sex": "F", "age": 78}, upto="intake")
        self.assertTrue(state.facts["hypocalcemia"])

    def test_classifies_medications_and_counts_fall_risk_drugs(self):
        state = _pipeline({"sex": "F", "age": 78, "medications": [
            {"name": "艾司唑仑片"}, {"name": "氨氯地平片"}, {"name": "碳酸钙D3"}]}, upto="intake")
        self.assertIn("benzodiazepine", state.facts["medication_classes"])
        self.assertEqual(state.facts["frid_count"], 2)
        self.assertFalse(state.facts["on_antiosteoporosis_therapy"])

    def test_calcium_and_vitamin_d_do_not_count_as_treatment(self):
        state = _pipeline({"sex": "F", "age": 78, "medications": [
            {"name": "碳酸钙D3"}, {"name": "维生素D滴剂"}]}, upto="intake")
        self.assertFalse(state.facts["on_antiosteoporosis_therapy"])

    def test_unrecognised_medication_is_reported_not_dropped(self):
        state = _pipeline({"sex": "F", "age": 78, "medications": [{"name": "某种没听过的药"}]},
                          upto="intake")
        result = state.results["intake"]
        self.assertTrue(any("未能识别" in note for note in result.notes))

    def test_missing_critical_names_what_it_blocks(self):
        payload = _pipeline({"sex": "F", "age": 78}, upto="intake").payload("intake")
        blocked = {item["fact"]: item["blocks"] for item in payload["missing_critical"]}
        self.assertIn("egfr", blocked)
        self.assertIn("肾功能", blocked["egfr"])

    def test_aliases_and_nesting_are_accepted(self):
        state = _pipeline({"年龄": 80, "性别": "F", "labs": {"egfr": 40}}, upto="intake")
        self.assertEqual(state.facts["age"], 80)
        self.assertEqual(state.facts["egfr"], 40)

    def test_medication_list_closes_the_world_on_steroids(self):
        state = _pipeline({"sex": "F", "age": 78, "medications": [{"name": "碳酸钙D3"}]}, upto="intake")
        self.assertIs(state.facts["glucocorticoid_current"], False)
        state2 = _pipeline({"sex": "F", "age": 78}, upto="intake")
        self.assertIsNone(state2.facts.get("glucocorticoid_current"))


class DiagnosisAgentTests(unittest.TestCase):
    def test_runs_every_standard_and_writes_flags_back(self):
        state = _pipeline(cases.get("demo"), upto="diagnosis")
        payload = state.payload("diagnosis")
        self.assertEqual(len(payload["standards"]), 3)
        self.assertTrue(state.facts["dx_osteoporosis"])
        self.assertTrue(state.facts["dx_possible_sarcopenia"])
        self.assertFalse(state.facts["dx_sarcopenia_any"])
        self.assertTrue(state.facts["low_muscle_strength"])

    def test_osteosarcopenia_is_probable_when_muscle_is_only_possible(self):
        payload = _pipeline(cases.get("demo"), upto="diagnosis").payload("diagnosis")
        self.assertEqual(payload["osteosarcopenia"]["status"], "probable")
        self.assertTrue(payload["osteosarcopenia"]["interactions"])

    def test_component_flag_is_low_if_any_standard_says_low(self):
        # Grip 27.5: low by AWGS (<28), normal by EWGSOP2 (<27).
        state = _pipeline(cases.get("gc_male"), upto="diagnosis")
        self.assertTrue(state.facts["low_muscle_strength"])

    def test_disagreement_is_surfaced_as_a_note(self):
        result = _pipeline(cases.get("demo"), upto="diagnosis").results["diagnosis"]
        self.assertTrue(any("不一致" in note for note in result.notes))

    def test_schema_holds(self):
        payload = _pipeline(cases.get("sparse"), upto="diagnosis").payload("diagnosis")
        ok, problems = validate("OsteoSarcDiagnosis", payload)
        self.assertTrue(ok, problems)


class EvidenceAgentTests(unittest.TestCase):
    def setUp(self):
        self.state = _pipeline(cases.get("demo"), upto="evidence")
        self.payload = self.state.payload("evidence")

    def test_buckets_are_populated(self):
        self.assertTrue(self.payload["applicable"])
        self.assertTrue(self.payload["excluded"])
        self.assertGreater(self.payload["not_applicable_count"], 0)

    def test_bisphosphonate_is_excluded_by_egfr_and_the_reason_is_named(self):
        excluded = {entry["rec_id"]: entry for entry in self.payload["excluded"]}
        self.assertIn("CN.OP.2022.BISPHOSPHONATE_FIRST", excluded)
        self.assertIn("eGFR", excluded["CN.OP.2022.BISPHOSPHONATE_FIRST"]["excluded_because"])

    def test_every_applicable_entry_is_in_the_ledger(self):
        for entry in self.payload["applicable"]:
            self.assertIsNotNone(self.state.ledger.get(entry["evidence_id"]))

    def test_data_gaps_are_ranked_by_how_much_they_unlock(self):
        gaps = self.payload["data_gaps"]
        if gaps:
            self.assertEqual(gaps, sorted(gaps, key=lambda g: -g["unlocks"]))
            self.assertIn("解锁", gaps[0]["hint"])

    def test_missing_data_produces_gaps_on_a_sparse_record(self):
        payload = _pipeline(cases.get("sparse"), upto="evidence").payload("evidence")
        self.assertTrue(payload["insufficient"])
        self.assertTrue(payload["data_gaps"])

    def test_provenance_is_disclosed(self):
        result = _pipeline(cases.get("demo"), upto="evidence").results["evidence"]
        self.assertTrue(any("编辑性转述" in note for note in result.notes))


class ConflictAgentTests(unittest.TestCase):
    def test_renal_failure_resolves_the_first_line_dispute(self):
        payload = _pipeline(cases.get("demo"), upto="conflict").payload("conflict")
        question = _find(payload, "q.pharm.initial_agent_very_high_risk")
        self.assertEqual(question["verdict"], RESOLVED_BY_PATIENT)
        removed = {entry["action"] for entry in question["resolution"]["removed"]}
        self.assertIn("start_bisphosphonate", removed)
        # Keyed on the machine-readable fact name; the detail string is for
        # humans and is written in Chinese.
        named = {f["fact"] for f in question["resolution"]["deciding_facts"]}
        self.assertIn("egfr", named)
        shown = " ".join(f["detail"] for f in question["resolution"]["deciding_facts"])
        self.assertIn("eGFR", shown)

    def test_protein_target_is_a_live_dispute_in_ckd(self):
        payload = _pipeline(cases.get("demo"), upto="conflict").payload("conflict")
        question = _find(payload, "q.nutrition.protein_target")
        self.assertEqual(question["verdict"], DISPUTED)
        actions = {position["action"] for position in question["divergence"]}
        self.assertIn("restrict_protein_in_ckd", actions)
        self.assertTrue(question["platform_policy"])

    def test_protein_dispute_dissolves_with_normal_renal_function(self):
        payload = _pipeline(cases.get("gc_male"), upto="conflict").payload("conflict")
        question = _find(payload, "q.nutrition.protein_target")
        self.assertNotEqual(question["verdict"], DISPUTED)

    def test_nested_targets_are_not_treated_as_disagreement(self):
        payload = _pipeline(cases.get("gc_male"), upto="conflict").payload("conflict")
        question = _find(payload, "q.nutrition.protein_target")
        kept = set(question.get("resolution", {}).get("kept", []))
        self.assertIn("protein_1_2_to_1_5", kept)

    def test_a_disputed_question_offers_support_but_never_picks(self):
        payload = _pipeline(cases.get("demo"), upto="conflict").payload("conflict")
        for question in payload["disputed"]:
            self.assertIn("不对存在争议的问题替医师做选择", question["decision_support"]["statement"])
            self.assertTrue(question["decision_support"]["patient_factors"])

    def test_complementary_recommendations_are_not_a_conflict(self):
        payload = _pipeline(cases.get("demo"), upto="conflict").payload("conflict")
        question = _find(payload, "q.falls.frid_deprescribing")
        self.assertNotEqual(question["verdict"], DISPUTED)

    def test_multi_source_agreement_is_labelled_consensus_with_its_regions(self):
        payload = _pipeline(cases.get("demo"), upto="conflict").payload("conflict")
        multi = [q for q in payload["questions"]
                 if q["verdict"] == CONSENSUS and q.get("source_count", 0) >= 2]
        self.assertTrue(multi)
        self.assertTrue(any(q["cross_region"] for q in multi))

    def test_tcm_only_question_is_explained_as_a_different_evidence_system(self):
        payload = _pipeline(cases.get("demo"), upto="conflict").payload("conflict")
        question = _find(payload, "q.tcm.pattern_treatment")
        self.assertTrue(any("证据体系" in note for note in question["notes"]))

    def test_counts_cover_every_question(self):
        payload = _pipeline(cases.get("demo"), upto="conflict").payload("conflict")
        self.assertEqual(sum(payload["counts"].values()), len(payload["questions"]))


class SafetyAgentTests(unittest.TestCase):
    def test_every_required_check_runs(self):
        payload = _pipeline(cases.get("demo")).payload("safety")
        for check in REQUIRED_SAFETY_CHECKS:
            self.assertIn(check, payload["checks_run"])

    def test_a_partial_audit_is_rejected_by_the_schema(self):
        ok, problems = validate("SafetyAndFollowup", {
            "checks_run": ["renal"], "issues": [], "blocking": [],
            "followup": {"m3": ["x"], "m6": ["y"], "m12": ["z"]},
        })
        self.assertFalse(ok)
        self.assertTrue(any("漏跑" in problem for problem in problems))

    def test_missing_labs_block_the_drug_plan(self):
        payload = _pipeline(cases.get("sparse")).payload("safety")
        self.assertEqual(payload["clearance"], "blocked")
        titles = {issue["title"] for issue in payload["blocking"]}
        self.assertTrue(any("血钙" in title for title in titles))
        self.assertTrue(any("eGFR" in title or "肾功能" in title for title in titles))

    def test_renal_gate_fires_for_bisphosphonates(self):
        payload = _pipeline(cases.get("demo")).payload("safety")
        renal = [issue for issue in payload["issues"] if issue["check"] == "renal"]
        self.assertTrue(any("双膦酸盐" in issue["title"] for issue in renal))

    def test_denosumab_in_ckd_earns_early_calcium_checks(self):
        followup = _pipeline(cases.get("demo")).payload("safety")["followup"]
        self.assertTrue(any("1～2 周" in item["what"] for item in followup["m3"]))

    def test_interaction_between_calcium_and_levothyroxine_is_found(self):
        payload = _pipeline(cases.get("demo")).payload("safety")
        interactions = [issue for issue in payload["issues"] if issue["check"] == "interaction"]
        self.assertTrue(any("左甲状腺素" in issue["title"] for issue in interactions))

    def test_prescribing_omission_is_reported(self):
        payload = _pipeline(cases.get("demo")).payload("safety")
        self.assertTrue(any(issue["title"] == "处方遗漏" for issue in payload["issues"]))

    def test_followup_always_has_three_six_and_twelve_months(self):
        for name in cases.CASES:
            followup = _pipeline(cases.get(name)).payload("safety")["followup"]
            for milestone in ("m3", "m6", "m12"):
                self.assertTrue(followup[milestone], f"{name}/{milestone}")

    def test_findings_are_deduplicated_by_action(self):
        payload = _pipeline(cases.get("demo")).payload("safety")
        titles = [issue["title"] for issue in payload["issues"]
                  if issue["check"] == "contraindication"]
        self.assertEqual(len(titles), len(set(titles)))


class AgentContainmentTests(unittest.TestCase):
    def test_an_agent_that_raises_is_contained_and_reported(self):
        class Exploding(RiskAgent):
            def run(self, state):
                raise RuntimeError("boom")

        state = RunState(case=cases.get("demo"))
        IntakeAgent()(state)
        DiagnosisAgent()(state)
        result = Exploding()(state)
        self.assertEqual(result.status, "blocked")
        self.assertTrue(any("boom" in problem for problem in result.problems))
        self.assertTrue(any("不得被视为" in note for note in result.notes))

    def test_a_malformed_payload_is_discarded_not_passed_on(self):
        class Malformed(RiskAgent):
            def run(self, state):
                return {"axes": "not a list"}

        state = RunState(case=cases.get("demo"))
        IntakeAgent()(state)
        DiagnosisAgent()(state)
        result = Malformed()(state)
        self.assertEqual(result.status, "blocked")
        self.assertEqual(result.payload, {})


def _find(payload, question_id):
    for question in payload["questions"]:
        if question["question_id"] == question_id:
            return question
    raise AssertionError(f"question {question_id} not present")


if __name__ == "__main__":
    unittest.main()
