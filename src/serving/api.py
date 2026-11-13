"""Local research demo and JSON API. Run with an explicit variant and text policy."""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from src.serving.prepare_bundle import DEFAULT_BUNDLE
from src.serving.score_application import ApplicationScorer, TEXT_POLICIES, VARIANTS

MAX_BODY_BYTES = 32_768
DEMO_PAGE = Path(__file__).with_name("demo.html")


def reject_constant(value: str):
    raise ValueError(f"JSON numeric constant {value} is not allowed")


def make_server(scorer: ApplicationScorer, port: int = 8000) -> HTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            # Requests can contain financial fields. Do not log their bodies or URLs.
            pass

        def reply(self, status: int, payload, content_type="application/json; charset=utf-8"):
            body = payload if isinstance(payload, bytes) else json.dumps(payload, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; connect-src 'self'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/":
                self.reply(200, DEMO_PAGE.read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/api/health":
                self.reply(200, {"ready": True, "research_only": True,
                                 "variant": scorer.variant, "text_policy": scorer.text_policy})
            else:
                self.reply(404, {"error": "Not found"})

        def do_POST(self):
            if self.path != "/api/score":
                self.reply(404, {"error": "Not found"})
                return
            if self.headers.get_content_type() != "application/json":
                self.reply(415, {"error": "Use application/json"})
                return
            if self.headers.get("Transfer-Encoding"):
                self.reply(400, {"error": "Use a Content-Length body"})
                return
            try:
                length = int(self.headers.get("Content-Length", "-1"))
                if not 0 < length <= MAX_BODY_BYTES:
                    self.reply(413, {"error": "Body must contain 1 to 32768 bytes"})
                    return
                request = json.loads(self.rfile.read(length).decode("utf-8"), parse_constant=reject_constant)
                result = scorer.score(request)
            except (ValueError, TypeError, UnicodeDecodeError) as error:
                self.reply(400, {"error": str(error)})
                return
            except Exception:
                self.reply(503, {"error": "Scoring unavailable. No probability was returned."})
                return
            self.reply(200, result)

    return HTTPServer(("127.0.0.1", port), Handler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--text-policy", choices=TEXT_POLICIES, required=True)
    parser.add_argument("--distilbert-dir", type=Path)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    scorer = ApplicationScorer.load(args.bundle, variant=args.variant,
                                    text_policy=args.text_policy, distilbert_dir=args.distilbert_dir)
    with make_server(scorer, args.port) as server:
        print(f"Research demo: http://127.0.0.1:{server.server_port} ({args.variant}, {args.text_policy})", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
