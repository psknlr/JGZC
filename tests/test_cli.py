"""命令行：退出码语义、输出内容、语料覆盖与错误处理。"""

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from osteosarc_agent.cli import main


def _run(argv):
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = main(argv)
    return code, out.getvalue(), err.getvalue()


class ExitCodeTests(unittest.TestCase):
    def test_a_clean_case_exits_zero(self):
        code, _, _ = _run(["assess", "--case", "demo", "--no-evidence"])
        self.assertEqual(code, 0)

    def test_blocking_safety_findings_exit_two(self):
        code, _, _ = _run(["assess", "--case", "sparse", "--no-evidence"])
        self.assertEqual(code, 2)

    def test_unknown_case_is_a_usage_error(self):
        with self.assertRaises(SystemExit):
            _run(["assess", "--case", "nope"])

    def test_unknown_standard_is_rejected(self):
        with self.assertRaises(SystemExit):
            _run(["assess", "--case", "demo", "--standards", "EWGSOP1"])


class OutputTests(unittest.TestCase):
    def test_report_contains_every_section(self):
        _, text, _ = _run(["assess", "--case", "demo"])
        for heading in ("患者数字画像", "骨肌诊断", "骨肌风险雷达", "AI 判断",
                        "指南冲突消解", "个体化治疗路径", "安全审查", "随访计划", "证据台账"):
            self.assertIn(heading, text)

    def test_report_carries_the_provenance_disclaimer(self):
        _, text, _ = _run(["assess", "--case", "demo", "--no-evidence"])
        self.assertIn("编辑性转述", text)

    def test_json_output_is_parseable_and_complete(self):
        _, text, _ = _run(["assess", "--case", "demo", "--json"])
        decision = json.loads(text)
        for key in ("meta", "profile", "diagnosis", "risk", "judgement",
                    "conflicts", "plans", "safety", "evidence", "agents"):
            self.assertIn(key, decision)

    def test_conflicts_command_shows_disputes(self):
        _, text, _ = _run(["conflicts", "--case", "demo"])
        self.assertIn("存在争议", text)
        self.assertIn("争议已被本例事实消解", text)

    def test_conflicts_all_lists_every_question(self):
        _, brief, _ = _run(["conflicts", "--case", "demo"])
        _, full, _ = _run(["conflicts", "--case", "demo", "--all"])
        self.assertGreater(len(full), len(brief))

    def test_agents_command_lists_six_subagents(self):
        _, text, _ = _run(["agents"])
        self.assertIn("主智能体", text)
        for name in ("病例结构化", "骨肌诊断", "风险预测", "循证决策",
                     "指南冲突消解", "安全审查与随访"):
            self.assertIn(name, text)

    def test_guidelines_stats(self):
        _, text, _ = _run(["guidelines", "--stats"])
        stats = json.loads(text)
        self.assertGreaterEqual(stats["recommendations"], 60)

    def test_guidelines_show_includes_the_predicate(self):
        _, text, _ = _run(["guidelines", "--show", "CN.OP.2022.DENOSUMAB_CKD"])
        payload = json.loads(text)
        self.assertIn("applies_when", payload)
        self.assertIsNotNone(payload["applies_when"])

    def test_guidelines_show_unknown_id_fails(self):
        code, _, err = _run(["guidelines", "--show", "NOPE"])
        self.assertEqual(code, 1)
        self.assertIn("未找到", err)

    def test_questions_listing_marks_exclusivity(self):
        _, text, _ = _run(["guidelines", "--questions"])
        self.assertIn("[互斥]", text)
        self.assertIn("[互补]", text)

    def test_demo_compares_all_reference_cases(self):
        _, text, _ = _run(["demo"])
        for name in ("demo", "sparse", "gc_male"):
            self.assertIn(f"[{name}]", text)


class FileCaseTests(unittest.TestCase):
    def test_a_case_file_is_accepted(self):
        record = {"case_id": "t1", "label": "测试", "age": 76, "sex": "F",
                  "lumbar_tscore": -2.7, "grip_kg": 16, "egfr": 70,
                  "corrected_calcium": 2.3, "vitamin_d_25oh": 32}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "case.json"
            path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
            _, text, _ = _run(["assess", "--file", str(path), "--no-evidence"])
        self.assertIn("测试", text)
        self.assertIn("骨质疏松症", text)

    def test_a_missing_file_is_a_clean_error(self):
        with self.assertRaises(SystemExit):
            _run(["assess", "--file", "/nonexistent/case.json"])


class CorpusOverrideTests(unittest.TestCase):
    def test_a_replacement_corpus_is_used(self):
        payload = {
            "questions": [{"question_id": "q.local.policy", "label_zh": "本机构政策"}],
            "sources": [{"source_id": "LOCAL", "title_zh": "本机构骨质疏松路径",
                         "issuer": "某院", "year": 2026, "region": "CN",
                         "tradition": "western"}],
            "recommendations": [{
                "rec_id": "LOCAL.PATH.1", "source_id": "LOCAL", "topic": "bone_protection",
                "question": "q.local.policy", "action": "local_pathway",
                "direction": "recommend", "strength": "strong",
                "statement_zh": "按本机构骨质疏松路径处理。",
                "provenance": "local_policy",
                "applies_when": {"fact": "age", "op": ">=", "value": 50},
            }],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "local.json"
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            _, text, _ = _run(["guidelines", "--corpus", str(path), "--list"])
        self.assertIn("LOCAL.PATH.1", text)
        self.assertNotIn("CN.OP.2022", text)

    def test_a_broken_corpus_reports_and_exits_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text('{"recommendations": [{"rec_id": "X"}]}', encoding="utf-8")
            code, _, err = _run(["guidelines", "--corpus", str(path)])
        self.assertEqual(code, 1)
        self.assertIn("缺少字段", err)


if __name__ == "__main__":
    unittest.main()
