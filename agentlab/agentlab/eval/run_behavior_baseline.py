"""CLI for the offline F5-012 Agent/knowledge-loop behavior baseline."""
from __future__ import annotations

import argparse
import json
import sys

from agentlab.eval.behavior import run_behavior_baseline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agentlab.eval.run_behavior_baseline")
    parser.add_argument("--output", default=None, help="write the JSON report to this path")
    parser.add_argument("--scenarios", default=None, help="scenario JSONL path")
    args = parser.parse_args(argv)
    report = run_behavior_baseline(args.output, args.scenarios)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["summary"]["failed"] == 0 and report["summary"]["pending"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
