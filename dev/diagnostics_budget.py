"""Measure the diagnostics budget: backend lines, and response bytes per endpoint for one run.

  python dev/diagnostics_budget.py                                  # lines only
  python dev/diagnostics_budget.py --url http://127.0.0.1:18401 --run RUN --since ISO --until ISO

Each instrumentation PR records its before and after in docs/instrumentation-ledger.md.
"""

import argparse
import gzip
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = [ROOT / "src/server/telemetry.py", *sorted((ROOT / "src/server/dashboard").glob("*.py"))]


def endpoints(run: str, since: str, until: str) -> dict[str, tuple[str, dict]]:
  window = {"since": since, "until": until}
  return {
    "inventory": ("inventory", {}),
    "activity": ("activity", {"run_id": run, **window}),
    "series devices": ("series", {"run_id": run, "kind": "devices", **window}),
    "series all": ("series", {"run_id": run, "kind": "all", **window}),
    "logs page": ("logs", {"run_id": run, "limit": 200, **window}),
    "telemetry": ("telemetry", {"run_id": run, **window}),
    "history": ("history", {"run_id": run}),
    "checks": (f"runs/{run}/checks", {}),
    "summary": (f"runs/{run}/summary", {}),
    "commands": (f"runs/{run}/commands", window),
    "steps": (f"runs/{run}/steps", {}),
    "query run_gpu_busy": ("query", {"name": "run_gpu_busy", "run": run, "points": 1, **window}),
  }


def fetch(url: str) -> dict:
  started = time.monotonic()
  try:
    with urllib.request.urlopen(url, timeout=60) as response:
      body = response.read()
  except urllib.error.HTTPError as error:
    return {"status": error.code}
  return {"status": 200, "bytes": len(body), "gzip": len(gzip.compress(body, 6)), "seconds": round(time.monotonic() - started, 2)}


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--url")
  parser.add_argument("--run")
  parser.add_argument("--since")
  parser.add_argument("--until")
  args = parser.parse_args()
  lines = {str(path.relative_to(ROOT)): path.read_text().count("\n") for path in BACKEND}
  report = {"backend_lines": sum(lines.values()), "files": lines}
  if args.url:
    report["endpoints"] = {
      name: fetch(f"{args.url}/api/v1/inspect/{route}?{urllib.parse.urlencode(params)}")
      for name, (route, params) in endpoints(args.run, args.since, args.until).items()
    }
    # The Activity tab loads the run's lanes and its GPU duty; this is what one visit costs.
    tab = [report["endpoints"][name] for name in ("activity", "series devices")]
    if all(e["status"] == 200 for e in tab):
      report["activity_tab_bytes"] = sum(e["bytes"] for e in tab)
  print(json.dumps(report, indent=2))


if __name__ == "__main__":
  main()
