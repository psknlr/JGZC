"""可计算指南层：三值逻辑、谓词求值、语料装载的失败关闭行为。"""

import json
import tempfile
import unittest
from pathlib import Path

from osteosarc_agent.guidelines import facts as fact_table
from osteosarc_agent.guidelines.corpus import Corpus, CorpusError, default_corpus
from osteosarc_agent.guidelines.model import (
    FALSE, TRUE, UNKNOWN, ConditionError, evaluate, test_recommendation,
    tri_and, tri_not, tri_or, validate_condition,
)


class TriStateTests(unittest.TestCase):
    def test_all_takes_the_minimum(self):
        self.assertEqual(tri_and([TRUE, TRUE]), TRUE)
        self.assertEqual(tri_and([TRUE, UNKNOWN]), UNKNOWN)
        self.assertEqual(tri_and([FALSE, UNKNOWN]), FALSE)

    def test_any_takes_the_maximum(self):
        self.assertEqual(tri_or([FALSE, UNKNOWN]), UNKNOWN)
        self.assertEqual(tri_or([TRUE, UNKNOWN]), TRUE)
        self.assertEqual(tri_or([FALSE, FALSE]), FALSE)

    def test_not_leaves_unknown_alone(self):
        self.assertEqual(tri_not(TRUE), FALSE)
        self.assertEqual(tri_not(FALSE), TRUE)
        self.assertEqual(tri_not(UNKNOWN), UNKNOWN)


class EvaluateTests(unittest.TestCase):
    def test_missing_fact_is_unknown_not_false(self):
        result, trace = evaluate({"fact": "egfr", "op": "<", "value": 35}, {})
        self.assertEqual(result, UNKNOWN)
        self.assertIn("egfr", trace.missing_facts())

    def test_known_and_unknown_operators_stay_determinate(self):
        self.assertEqual(evaluate({"fact": "egfr", "op": "known"}, {})[0], FALSE)
        self.assertEqual(evaluate({"fact": "egfr", "op": "unknown"}, {})[0], TRUE)
        self.assertEqual(evaluate({"fact": "egfr", "op": "known"}, {"egfr": 32})[0], TRUE)

    def test_nested_predicate_and_trace(self):
        node = {"all": [
            {"fact": "age", "op": ">=", "value": 65},
            {"any": [
                {"fact": "prior_fragility_fracture", "op": "is_true"},
                {"fact": "tscore_min", "op": "<=", "value": -2.5},
            ]},
        ]}
        result, trace = evaluate(node, {"age": 78, "prior_fragility_fracture": True})
        self.assertEqual(result, TRUE)
        self.assertEqual(len(trace.children), 2)
        payload = trace.to_dict()
        self.assertEqual(payload["result"], TRUE)

    def test_any_short_circuits_over_unknown(self):
        node = {"any": [
            {"fact": "prior_fragility_fracture", "op": "is_true"},
            {"fact": "tscore_min", "op": "<=", "value": -2.5},
        ]}
        # One arm true and the other unmeasured still yields TRUE.
        self.assertEqual(evaluate(node, {"prior_fragility_fracture": True})[0], TRUE)
        # Neither arm decidable yields UNKNOWN, not FALSE.
        self.assertEqual(evaluate(node, {})[0], UNKNOWN)

    def test_list_operators(self):
        facts = {"fracture_sites": ["椎体"], "medication_classes": ["calcium", "ppi"]}
        self.assertEqual(evaluate({"fact": "fracture_sites", "op": "any_in", "value": ["椎体"]}, facts)[0], TRUE)
        self.assertEqual(evaluate({"fact": "medication_classes", "op": "contains", "value": "ppi"}, facts)[0], TRUE)
        self.assertEqual(evaluate({"fact": "medication_classes", "op": "contains", "value": "denosumab"}, facts)[0], FALSE)

    def test_incomparable_types_are_unknown_not_an_exception(self):
        self.assertEqual(evaluate({"fact": "egfr", "op": "<", "value": 35}, {"egfr": "低"})[0], UNKNOWN)

    def test_empty_predicate_means_unconditional(self):
        self.assertEqual(evaluate(None, {})[0], TRUE)
        self.assertEqual(evaluate({}, {})[0], TRUE)

    def test_malformed_predicate_is_a_load_error(self):
        with self.assertRaises(ConditionError):
            validate_condition({"op": ">=", "value": 1}, "test")
        with self.assertRaises(ConditionError):
            validate_condition({"fact": "age", "op": "≥", "value": 65}, "test")
        with self.assertRaises(ConditionError):
            validate_condition({"fact": "age", "op": ">="}, "test")


class ExclusionTests(unittest.TestCase):
    def test_exclusion_beats_applicability(self):
        corpus = default_corpus()
        rec = corpus.by_id("CN.OP.2022.BISPHOSPHONATE_FIRST")
        source = corpus.source_for(rec)
        facts = {"dx_osteoporosis": True, "egfr": 32}
        result = test_recommendation(rec, source, facts)
        self.assertEqual(result.status, "excluded")
        self.assertFalse(result.fires)

    def test_applies_when_nothing_excludes(self):
        corpus = default_corpus()
        rec = corpus.by_id("CN.OP.2022.BISPHOSPHONATE_FIRST")
        # Both exclusion arms must be *known* false. Leaving hypocalcaemia
        # unmeasured keeps the record undecided rather than clearing the drug —
        # which is the behaviour the platform must have at the bedside.
        result = test_recommendation(
            rec, corpus.source_for(rec),
            {"dx_osteoporosis": True, "egfr": 80, "hypocalcemia": False},
        )
        self.assertEqual(result.status, "applies")

    def test_partially_known_exclusion_stays_undecided(self):
        corpus = default_corpus()
        rec = corpus.by_id("CN.OP.2022.BISPHOSPHONATE_FIRST")
        result = test_recommendation(rec, corpus.source_for(rec), {"dx_osteoporosis": True, "egfr": 80})
        self.assertEqual(result.status, "insufficient_data")
        self.assertIn("hypocalcemia", result.missing_facts)

    def test_unknown_fact_yields_insufficient_data(self):
        corpus = default_corpus()
        rec = corpus.by_id("CN.OP.2022.BISPHOSPHONATE_FIRST")
        result = test_recommendation(rec, corpus.source_for(rec), {"dx_osteoporosis": True})
        self.assertEqual(result.status, "insufficient_data")
        self.assertIn("egfr", result.missing_facts)


class CorpusLoadTests(unittest.TestCase):
    def setUp(self):
        self.corpus = default_corpus()

    def test_shipped_corpus_loads(self):
        stats = self.corpus.stats()
        self.assertGreaterEqual(stats["recommendations"], 60)
        self.assertGreaterEqual(stats["sources"], 20)
        self.assertGreaterEqual(stats["questions"], 40)

    def test_every_recommendation_has_a_declared_question_and_source(self):
        for rec in self.corpus.recommendations:
            self.assertIn(rec.source_id, self.corpus.sources, rec.rec_id)
            self.assertIn(rec.question, self.corpus.questions_index, rec.rec_id)

    def test_every_predicate_references_declared_facts(self):
        for rec in self.corpus.recommendations:
            for node in (rec.applies_when, rec.excluded_when):
                names: set[str] = set()
                _collect(node, names)
                for name in names:
                    self.assertTrue(fact_table.is_known_fact(name), f"{rec.rec_id}: {name}")

    def test_corpus_spans_the_required_traditions_and_regions(self):
        stats = self.corpus.stats()
        for tradition in ("western", "tcm", "geriatrics", "sarcopenia", "nutrition", "rehab", "pharmacy"):
            self.assertIn(tradition, stats["by_tradition"], tradition)
        for region in ("CN", "US", "EU"):
            self.assertIn(region, stats["by_region"], region)

    def test_corpus_is_within_the_last_decade_plus_foundational(self):
        self.assertGreaterEqual(self.corpus.stats()["year_range"][1], 2022)

    def test_every_record_declares_its_provenance_honestly(self):
        for rec in self.corpus.recommendations:
            self.assertIn(rec.provenance, ("editorial_paraphrase", "licensed_verbatim", "local_policy"))
            if rec.provenance == "editorial_paraphrase":
                self.assertFalse(rec.verbatim, rec.rec_id)

    def test_unknown_fact_in_a_predicate_is_rejected_at_load(self):
        payload = {
            "questions": [{"question_id": "q.x", "label_zh": "x"}],
            "sources": [{"source_id": "S", "title_zh": "t", "issuer": "i", "year": 2024,
                         "region": "CN", "tradition": "western"}],
            "recommendations": [{
                "rec_id": "R1", "source_id": "S", "topic": "t", "question": "q.x",
                "action": "a", "direction": "recommend", "strength": "strong",
                "statement_zh": "s",
                "applies_when": {"fact": "egfr_ml", "op": "<", "value": 35},
            }],
        }
        with self.assertRaises(CorpusError) as ctx:
            _load_temp(payload)
        self.assertIn("未声明的事实", str(ctx.exception))

    def test_undeclared_question_is_rejected_at_load(self):
        payload = {
            "sources": [{"source_id": "S", "title_zh": "t", "issuer": "i", "year": 2024,
                         "region": "CN", "tradition": "western"}],
            "recommendations": [{
                "rec_id": "R1", "source_id": "S", "topic": "t", "question": "q.undeclared",
                "action": "a", "direction": "recommend", "strength": "strong", "statement_zh": "s",
            }],
        }
        with self.assertRaises(CorpusError) as ctx:
            _load_temp(payload)
        self.assertIn("未在问题目录中声明", str(ctx.exception))

    def test_duplicate_rec_id_is_rejected(self):
        rec = {"rec_id": "R1", "source_id": "S", "topic": "t", "question": "q.x",
               "action": "a", "direction": "recommend", "strength": "strong", "statement_zh": "s"}
        payload = {
            "questions": [{"question_id": "q.x", "label_zh": "x"}],
            "sources": [{"source_id": "S", "title_zh": "t", "issuer": "i", "year": 2024,
                         "region": "CN", "tradition": "western"}],
            "recommendations": [rec, dict(rec)],
        }
        with self.assertRaises(CorpusError):
            _load_temp(payload)

    def test_unknown_direction_is_rejected(self):
        payload = {
            "questions": [{"question_id": "q.x", "label_zh": "x"}],
            "sources": [{"source_id": "S", "title_zh": "t", "issuer": "i", "year": 2024,
                         "region": "CN", "tradition": "western"}],
            "recommendations": [{"rec_id": "R1", "source_id": "S", "topic": "t", "question": "q.x",
                                 "action": "a", "direction": "maybe", "strength": "strong",
                                 "statement_zh": "s"}],
        }
        with self.assertRaises(CorpusError):
            _load_temp(payload)

    def test_exclusive_questions_are_declared(self):
        exclusive = [q for q in self.corpus.questions_index.values() if q.exclusive]
        self.assertGreaterEqual(len(exclusive), 3)
        ids = {q.question_id for q in exclusive}
        self.assertIn("q.nutrition.protein_target", ids)
        self.assertIn("q.sarco.mass_required", ids)


def _collect(node, out):
    if isinstance(node, dict):
        if "fact" in node:
            out.add(node["fact"])
        for key in ("all", "any"):
            for child in node.get(key, []) or []:
                _collect(child, out)
        if "not" in node:
            _collect(node["not"], out)


def _load_temp(payload):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "x.json"
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return Corpus.load([path])


if __name__ == "__main__":
    unittest.main()
