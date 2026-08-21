"""认知层：叙述器只能改写，且改写必须通过安全校验。"""

import unittest

from osteosarc_agent import cases
from osteosarc_agent.llm import LLMError, Narrator, NullLLMClient, check_narrative, summarise
from osteosarc_agent.llm.providers import build_client
from osteosarc_agent.orchestrator import Orchestrator


class _Fake:
    name = "fake"

    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def complete(self, system, prompt, *, max_tokens=900):
        self.calls.append((system, prompt))
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


class NarratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.decision = Orchestrator().run(cases.get("demo"))

    def test_default_is_deterministic(self):
        result = Narrator().narrate(self.decision)
        self.assertEqual(result["mode"], "deterministic")
        self.assertIn("骨质疏松", result["text"])

    def test_a_clean_narrative_is_kept(self):
        client = _Fake("本次评估提示骨质疏松症诊断成立，肾功能不全限制了药物选择，需先纠正维生素 D。")
        result = Narrator(client).narrate(self.decision)
        self.assertEqual(result["mode"], "model")
        self.assertTrue(client.calls)

    def test_the_model_only_sees_already_decided_conclusions(self):
        client = _Fake("好的。")
        Narrator(client).narrate(self.decision)
        system, prompt = client.calls[0]
        self.assertIn("已经确定的结论", prompt)
        self.assertIn("不得给出任何剂量", system)

    def test_a_fabricated_dose_discards_the_whole_narrative(self):
        client = _Fake("本次评估提示应给予特立帕肽每日 20 μg 皮下注射。")
        result = Narrator(client).narrate(self.decision)
        self.assertEqual(result["mode"], "rejected")
        self.assertTrue(any("剂量" in problem for problem in result["problems"]))
        self.assertIn("骨质疏松", result["text"])  # fell back, did not go blank

    def test_an_excluded_drug_cannot_be_recommended(self):
        client = _Fake("本次评估提示可以考虑阿仑膦酸钠治疗。")
        self.assertEqual(Narrator(client).narrate(self.decision)["mode"], "rejected")

    def test_an_excluded_drug_may_be_named_in_a_prohibition(self):
        ok, problems = check_narrative(
            "因 eGFR 32，本例不宜使用阿仑膦酸钠等双膦酸盐。", self.decision)
        self.assertTrue(ok, problems)

    def test_an_all_clear_is_rejected_when_there_are_serious_findings(self):
        ok, problems = check_narrative("本次评估未见明显异常。", self.decision)
        self.assertFalse(ok)
        self.assertTrue(any("放行式表述" in problem for problem in problems))

    def test_an_empty_reply_is_rejected(self):
        self.assertFalse(check_narrative("", self.decision)[0])

    def test_transport_failure_degrades_rather_than_raising(self):
        result = Narrator(_Fake(LLMError("connection refused"))).narrate(self.decision)
        self.assertEqual(result["mode"], "deterministic")
        self.assertIn("回退", result["note"])

    def test_summary_carries_every_load_bearing_section(self):
        text = summarise(self.decision)
        for marker in ("【骨】", "【肌】", "【风险】", "【骨保护方案】", "【随访·3 个月】", "【声明】"):
            self.assertIn(marker, text)

    def test_orchestrator_attaches_the_narrative(self):
        decision = Orchestrator(narrator=Narrator()).run(cases.get("gc_male"))
        self.assertIn("narrative", decision)
        self.assertEqual(decision["narrative"]["mode"], "deterministic")


class ProviderTests(unittest.TestCase):
    def test_no_provider_yields_the_null_client(self):
        self.assertIsInstance(build_client(None), NullLLMClient)
        self.assertIsInstance(build_client("none"), NullLLMClient)

    def test_unknown_provider_is_an_error(self):
        with self.assertRaises(LLMError):
            build_client("mystery-model")

    def test_missing_api_key_is_an_error_not_a_silent_call(self):
        with self.assertRaises(LLMError):
            build_client("anthropic")


if __name__ == "__main__":
    unittest.main()
