"""Serve the dashboard UI over a recorded fixture instead of a cluster.

Static assets come from the working tree, so edits show on reload. API
requests are answered from dev/fixtures/dashboard, captured from a real
gateway by dev/capture_dashboard_fixture.py; a path with a query string maps
to the same file as the path without it. Anything not recorded is a 404 with
the list of what is.

  python dev/dashboard_fixture.py --port 9017
"""

import argparse
import http.server
import json
import pathlib
from urllib.parse import urlparse

ROOT = pathlib.Path(__file__).resolve().parents[1]
STATIC = ROOT / "src" / "server" / "dashboard" / "static"
API = "/api/v1/dashboard"
TYPES = {".css": "text/css", ".js": "text/javascript", ".html": "text/html", ".json": "application/json"}


def make_handler(fixture: pathlib.Path):
  manifest = json.loads((fixture / "manifest.json").read_text())
  files = manifest["files"]

  class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *_):
      pass

    def send(self, status: int, body: bytes, content_type: str) -> None:
      self.send_response(status)
      self.send_header("Content-Type", content_type)
      self.send_header("Content-Length", str(len(body)))
      self.send_header("Cache-Control", "no-store")
      self.end_headers()
      self.wfile.write(body)

    def do_GET(self):
      path = urlparse(self.path).path
      if path in ("/", "/dashboard", "/dashboard/"):
        return self.send(200, (STATIC / "index.html").read_bytes(), "text/html")
      if path.startswith("/dashboard/assets/"):
        asset = (STATIC / path[len("/dashboard/assets/") :]).resolve()
        if asset.parent == STATIC.resolve() and asset.is_file():
          return self.send(200, asset.read_bytes(), TYPES.get(asset.suffix, "application/octet-stream"))
        return self.send(404, b"not found", "text/plain")
      if path == "/api/v1/healthz":
        return self.send(200, b'{"status":"ok"}', "application/json")
      if path in files:
        return self.send(200, (fixture / files[path]).read_bytes(), "application/json")
      missing = {"error": f"not in fixture: {path}", "captured_at": manifest["captured_at"], "recorded": sorted(files)}
      return self.send(404, json.dumps(missing).encode(), "application/json")

  return Handler


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--fixture", default=str(ROOT / "dev" / "fixtures" / "dashboard"))
  parser.add_argument("--host", default="127.0.0.1")
  parser.add_argument("--port", type=int, default=9017)
  args = parser.parse_args()
  fixture = pathlib.Path(args.fixture)
  manifest = json.loads((fixture / "manifest.json").read_text())
  print(f"fixture captured {manifest['captured_at']} from {manifest['base']}; {len(manifest['files'])} responses")
  print(f"http://{args.host}:{args.port}/dashboard")
  http.server.ThreadingHTTPServer((args.host, args.port), make_handler(fixture)).serve_forever()


if __name__ == "__main__":
  main()
