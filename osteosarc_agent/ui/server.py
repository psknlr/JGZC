"""Local console server.

Standard-library HTTP server plus one self-contained page: no framework, no
CDN, no build step, so the console runs inside a hospital network with no
outbound access and inside a notebook kernel alike.

Scope and safety:

* binds to ``127.0.0.1`` by default — this is a local operator tool with no
  authentication, so put your own authenticated proxy in front before exposing
  it anywhere shared;
* every request goes through the ordinary :class:`Orchestrator`; the console has
  no privileged path into the corpus or the agents;
* posted case records are held only for the lifetime of the request. Nothing is
  written to disk, because a case record is clinical data.
"""

from __future__ import annotations

import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .. import cases as demo_cases
from ..guidelines.corpus import Corpus, CorpusError, default_corpus
from ..orchestrator import Orchestrator

STATIC_DIR = Path(__file__).parent / "static"
MAX_BODY_BYTES = 512 * 1024

logger = logging.getLogger("osteosarc.ui")


class ConsoleService:
    def __init__(self, corpus_path: str | None = None, llm: str | None = None,
                 model: str | None = None) -> None:
        self.corpus: Corpus = Corpus.load([corpus_path]) if corpus_path else default_corpus()
        narrator = None
        self.llm_error: str | None = None
        if llm:
            from ..llm import LLMError, Narrator, build_client
            try:
                narrator = Narrator(build_client(llm, model))
            except LLMError as exc:
                self.llm_error = str(exc)
        self.orchestrator = Orchestrator(corpus=self.corpus, narrator=narrator)

    def cases(self) -> dict[str, Any]:
        return {
            "cases": [
                {"name": name, "label": case.get("label", name), "case_id": case.get("case_id", "")}
                for name, case in demo_cases.CASES.items()
            ],
            "corpus": self.corpus.stats(),
            "agents": self.orchestrator.agent_catalog(),
            "llm_error": self.llm_error,
        }

    def assess(self, payload: dict[str, Any]) -> dict[str, Any]:
        record = payload.get("record")
        if not record:
            record = demo_cases.get(str(payload.get("case", "demo")))
        if not isinstance(record, dict):
            raise ValueError("record 必须是对象")
        return self.orchestrator.run(record)

    def guideline(self, rec_id: str) -> dict[str, Any]:
        rec = self.corpus.by_id(rec_id)
        if rec is None:
            raise KeyError(rec_id)
        return {
            "recommendation": rec.to_dict(),
            "source": self.corpus.source_for(rec).to_dict(),
            "applies_when": rec.applies_when,
            "excluded_when": rec.excluded_when,
            "question": self.corpus.question(rec.question).to_dict(),
        }


class Handler(BaseHTTPRequestHandler):
    service: ConsoleService

    server_version = "OsteoSarcConsole/0.1"

    def log_message(self, fmt: str, *args: Any) -> None:  # pragma: no cover
        logger.info("%s - %s", self.address_string(), fmt % args)

    # -- helpers ---------------------------------------------------------
    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, payload: Any) -> None:
        self._send(status, json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def _static(self, name: str) -> None:
        path = STATIC_DIR / name
        if not path.exists():
            self._json(404, {"error": "not found"})
            return
        content_type = "text/html; charset=utf-8" if path.suffix == ".html" else "text/plain"
        self._send(200, path.read_bytes(), content_type)

    # -- routes ----------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            self._static("index.html")
        elif path == "/api/cases":
            self._json(200, self.service.cases())
        elif path == "/api/corpus":
            self._json(200, self.service.corpus.stats())
        elif path.startswith("/api/guideline/"):
            rec_id = path.rsplit("/", 1)[-1]
            try:
                self._json(200, self.service.guideline(rec_id))
            except KeyError:
                self._json(404, {"error": f"未找到 {rec_id}"})
        elif path.startswith("/api/assess"):
            # Convenience for links and curl: /api/assess?case=demo
            query = self.path.split("?", 1)[1] if "?" in self.path else ""
            case = "demo"
            for part in query.split("&"):
                if part.startswith("case="):
                    case = part[5:]
            self._run({"case": case})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path.split("?", 1)[0] != "/api/assess":
            self._json(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY_BYTES:
            self._json(413, {"error": "请求体过大"})
            return
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            self._json(400, {"error": f"JSON 解析失败: {exc}"})
            return
        self._run(payload if isinstance(payload, dict) else {})

    def _run(self, payload: dict[str, Any]) -> None:
        try:
            self._json(200, self.service.assess(payload))
        except (KeyError, ValueError) as exc:
            self._json(400, {"error": str(exc)})
        except CorpusError as exc:
            self._json(500, {"error": f"语料错误: {exc}"})
        except Exception as exc:  # noqa: BLE001 - the console must not die on one case
            logger.exception("assess failed")
            self._json(500, {"error": f"{type(exc).__name__}: {exc}"})


def serve(host: str = "127.0.0.1", port: int = 8000, corpus_path: str | None = None,
          llm: str | None = None, model: str | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    service = ConsoleService(corpus_path, llm, model)
    handler = type("BoundHandler", (Handler,), {"service": service})
    server = ThreadingHTTPServer((host, port), handler)
    stats = service.corpus.stats()
    print(f"筋骨智策 OsteoSarc-Agent 控制台  http://{host}:{port}")
    print(f"语料：{stats['recommendations']} 条推荐 / {stats['sources']} 部指南 / {stats['questions']} 个临床问题")
    if service.llm_error:
        print(f"[warn] 表述层不可用，按确定性路径运行：{service.llm_error}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:  # pragma: no cover
        print("\n已停止")
    finally:
        server.server_close()
