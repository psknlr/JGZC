"""控制台：端点契约与页面自包含性。"""

import json
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

from osteosarc_agent.ui.server import STATIC_DIR, ConsoleService, Handler


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.service = ConsoleService()

    def test_cases_payload_lists_the_agents_and_corpus(self):
        payload = self.service.cases()
        self.assertEqual(len(payload["agents"]), 6)
        self.assertGreaterEqual(payload["corpus"]["recommendations"], 60)

    def test_assess_runs_a_named_case(self):
        decision = self.service.assess({"case": "demo"})
        self.assertEqual(decision["status"], "ok")

    def test_assess_runs_an_ad_hoc_record(self):
        decision = self.service.assess({"record": {"age": 70, "sex": "F", "grip_kg": 15}})
        self.assertIn("diagnosis", decision)

    def test_unknown_case_raises(self):
        with self.assertRaises(KeyError):
            self.service.assess({"case": "nope"})

    def test_guideline_lookup(self):
        payload = self.service.guideline("CN.OP.2022.DENOSUMAB_CKD")
        self.assertEqual(payload["recommendation"]["rec_id"], "CN.OP.2022.DENOSUMAB_CKD")
        self.assertTrue(payload["question"]["label_zh"])


class HttpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        handler = type("Bound", (Handler,), {"service": ConsoleService()})
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        cls.base = f"http://127.0.0.1:{cls.server.server_address[1]}"
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def _get(self, path):
        with urllib.request.urlopen(self.base + path, timeout=30) as response:
            return response.status, response.read()

    def test_index_is_served(self):
        status, body = self._get("/")
        self.assertEqual(status, 200)
        self.assertIn(b"OsteoSarc-Agent", body)

    def test_assess_over_get(self):
        status, body = self._get("/api/assess?case=gc_male")
        decision = json.loads(body)
        self.assertEqual(status, 200)
        self.assertEqual(decision["meta"]["case_id"], "gc-003")

    def test_assess_over_post(self):
        request = urllib.request.Request(
            self.base + "/api/assess",
            data=json.dumps({"record": {"age": 80, "sex": "F", "lumbar_tscore": -2.8}}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            decision = json.loads(response.read())
        self.assertTrue(decision["diagnosis"]["osteoporosis"]["diagnosis"])

    def test_bad_json_is_a_400(self):
        request = urllib.request.Request(
            self.base + "/api/assess", data=b"{not json", headers={"Content-Type": "application/json"})
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(request, timeout=30)
        self.assertEqual(ctx.exception.code, 400)

    def test_unknown_path_is_a_404(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(self.base + "/nope", timeout=30)
        self.assertEqual(ctx.exception.code, 404)

    def test_missing_guideline_is_a_404(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(self.base + "/api/guideline/NOPE", timeout=30)
        self.assertEqual(ctx.exception.code, 404)


class PageTests(unittest.TestCase):
    def setUp(self):
        self.html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")

    def test_page_is_self_contained(self):
        """No CDN, no external asset: the console must work offline."""
        for marker in ("http://", "https://", "//cdn", "<script src=", "<link rel=\"stylesheet\""):
            self.assertNotIn(marker, self.html, marker)

    def test_page_renders_every_briefed_region(self):
        for marker in ("患者数字画像", "骨肌风险雷达", "AI 判断",
                       "指南冲突消解", "个体化治疗路径与方案", "安全审查与随访计划",
                       "查看循证依据"):
            self.assertIn(marker, self.html, marker)

    def test_page_escapes_interpolated_values(self):
        self.assertIn("const esc =", self.html)

    def test_page_supports_dark_mode(self):
        self.assertIn("prefers-color-scheme:dark", self.html)


if __name__ == "__main__":
    unittest.main()
