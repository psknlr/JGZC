"""文档与代码的一致性：文档里出现的标识符必须真实存在。

一份说明了不存在的命令或条目的文档，比没有文档更糟——读者会照着做。
"""

import re
import unittest
from pathlib import Path

from osteosarc_agent import cases
from osteosarc_agent.agents.conflict import PLATFORM_POLICY, VERDICT_ZH
from osteosarc_agent.cli import build_parser
from osteosarc_agent.guidelines.corpus import default_corpus
from osteosarc_agent.guidelines.facts import FACTS
from osteosarc_agent.schemas import REQUIRED_SAFETY_CHECKS

ROOT = Path(__file__).resolve().parent.parent


def _test_exists(name: str) -> bool:
    return any(f"def {name}(" in path.read_text(encoding="utf-8")
               for path in Path(__file__).resolve().parent.glob("test_*.py"))
DOCS = sorted((ROOT / "docs").glob("*.md"))
README = ROOT / "README.md"


class DocPresenceTests(unittest.TestCase):
    def test_expected_docs_exist(self):
        names = {path.name for path in DOCS}
        for expected in ("ARCHITECTURE.md", "GUIDELINES.md", "CONFLICT.md",
                         "DIAGNOSIS.md", "SAFETY.md", "CONSOLE.md", "LIMITATIONS.md"):
            self.assertIn(expected, names)

    def test_readme_links_resolve(self):
        text = README.read_text(encoding="utf-8")
        for target in re.findall(r"\]\((docs/[A-Za-z_]+\.md)\)", text):
            self.assertTrue((ROOT / target).exists(), target)

    def test_docs_cross_links_resolve(self):
        for path in DOCS:
            text = path.read_text(encoding="utf-8")
            for target in re.findall(r"\]\(([A-Za-z_]+\.md)\)", text):
                self.assertTrue((path.parent / target).exists(), f"{path.name} -> {target}")


class DocAccuracyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = "\n".join(p.read_text(encoding="utf-8") for p in [README, *DOCS])
        cls.corpus = default_corpus()

    def test_every_referenced_rec_or_source_id_exists(self):
        for identifier in re.findall(r"\b((?:CN|US|EU|INTL|APAC|LABEL)\.[A-Z0-9_.]{4,})\b", self.text):
            known = (self.corpus.by_id(identifier) is not None
                     or identifier in self.corpus.sources)
            self.assertTrue(known, identifier)

    def test_every_referenced_question_id_exists(self):
        for question_id in re.findall(r"\bq\.[a-z_]+\.[a-z_]+\b", self.text):
            self.assertIn(question_id, self.corpus.questions_index, question_id)

    #: Snake_case identifiers that belong to the corpus/decision schema rather
    #: than the fact table. Anything else with an underscore is checked against
    #: FACTS, so a doc that names a fact the platform does not have fails here.
    SCHEMA_IDENTIFIERS = frozenset({
        "applies_when", "excluded_when", "statement_zh", "rec_id", "source_id",
        "question_id", "label_zh", "title_zh", "evidence_level", "checks_run",
        "decision_support", "missing_critical", "verbatim_records", "case_id",
        "not_applicable", "insufficient_data", "resolved_by_patient",
        "editorial_paraphrase", "licensed_verbatim", "local_policy",
        "bone_protection", "medication_safety", "good_practice",
        "bone_only", "muscle_only", "gc_male", "is_true", "is_false",
        "not_in", "any_in", "needs_action", "risk_tier", "muscle_mass_method",
        "PLATFORM_POLICY",
    })

    def test_every_referenced_fact_name_exists(self):
        """A doc that names a fact the platform does not declare is wrong."""
        for name in re.findall(r"`([a-z][a-z0-9]*(?:_[a-z0-9]+)+)`", self.text):
            if name in self.SCHEMA_IDENTIFIERS or name.endswith("_risk_tier"):
                continue
            if name.startswith("test_"):
                self.assertTrue(_test_exists(name), f"文档引用了不存在的测试 {name!r}")
                continue
            self.assertTrue(name in FACTS, f"文档引用了未声明的事实 {name!r}")

    def test_referenced_cli_commands_exist(self):
        parser = build_parser()
        commands = set()
        for action in parser._subparsers._group_actions:  # noqa: SLF001 - argparse has no public API
            commands.update(action.choices)
        for command in re.findall(r"python -m osteosarc_agent (\w+)", self.text):
            self.assertIn(command, commands, command)

    def test_referenced_demo_cases_exist(self):
        for name in re.findall(r"--case (\w+)", self.text):
            self.assertIn(name, cases.CASES, name)

    def test_corpus_counts_in_the_readme_match_reality(self):
        stats = self.corpus.stats()
        text = README.read_text(encoding="utf-8")
        self.assertIn(f"{stats['sources']} 部指南", text)
        self.assertIn(f"{stats['recommendations']} 条推荐", text)
        self.assertIn(f"{stats['questions']} 个临床问题", text)

    def test_all_six_verdicts_are_documented(self):
        conflict_doc = (ROOT / "docs" / "CONFLICT.md").read_text(encoding="utf-8")
        for verdict, label in VERDICT_ZH.items():
            self.assertTrue(label in conflict_doc, f"判读 {label} 未记入 CONFLICT.md")
            self.assertTrue(verdict in conflict_doc, f"判读键 {verdict} 未记入 CONFLICT.md")

    def test_every_platform_policy_question_is_documented(self):
        conflict_doc = (ROOT / "docs" / "CONFLICT.md").read_text(encoding="utf-8")
        for question_id in PLATFORM_POLICY:
            label = self.corpus.question(question_id).label_zh
            self.assertTrue(label in conflict_doc, f"{question_id}（{label}）未写入 CONFLICT.md 的策略表")

    def test_safety_doc_lists_every_required_check(self):
        safety_doc = (ROOT / "docs" / "SAFETY.md").read_text(encoding="utf-8")
        labels = ["禁忌证", "药物相互作用", "肾功能", "低钙风险", "跌倒风险", "依从性", "随访节点"]
        self.assertEqual(len(labels), len(REQUIRED_SAFETY_CHECKS))
        for label in labels:
            self.assertIn(label, safety_doc)

    def test_the_provenance_caveat_is_stated_everywhere_it_matters(self):
        for path in (README, ROOT / "docs" / "GUIDELINES.md", ROOT / "docs" / "LIMITATIONS.md"):
            self.assertIn("编辑性转述", path.read_text(encoding="utf-8"), path.name)

    def test_limitations_states_it_is_not_frax(self):
        text = (ROOT / "docs" / "LIMITATIONS.md").read_text(encoding="utf-8")
        self.assertIn("不是 FRAX", text)
        self.assertIn("不生成任何药物剂量", text)


if __name__ == "__main__":
    unittest.main()
