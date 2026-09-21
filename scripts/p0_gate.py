"""Reproducible P0 static/regression gate.

The gate is intentionally an orchestration script, not a second test runner.
It records command identity, exit status and bounded diagnostic output while
keeping credentials and full subprocess logs out of the JSON evidence file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class GateResult:
    name: str
    command: tuple[str, ...]
    returncode: int
    duration_ms: int
    output_tail: str = ""
    skipped: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "command": list(self.command),
            "returncode": self.returncode,
            "duration_ms": self.duration_ms,
            "output_tail": self.output_tail,
            "skipped": self.skipped,
        }


def _tail(output: str, limit: int = 1200) -> str:
    """Keep diagnostics useful without persisting potentially sensitive logs."""
    lines = [line for line in output.splitlines() if line.strip()]
    return "\n".join(lines[-12:])[-limit:]


def _configure_stdout() -> None:
    """Keep the gate's own JSON report UTF-8 on Windows consoles."""
    stream = getattr(sys, "stdout", None)
    reconfigure = getattr(stream, "reconfigure", None)
    if callable(reconfigure):
        reconfigure(encoding="utf-8", errors="replace")


def run_command(name: str, command: Sequence[str], *, cwd: Path = ROOT,
                skip: bool = False) -> GateResult:
    normalized = tuple(str(item) for item in command)
    if skip:
        return GateResult(name, normalized, 0, 0, "explicitly skipped", True)
    started = time.perf_counter()
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    try:
        completed = subprocess.run(
            normalized, cwd=str(cwd), env=env, text=True,
            encoding="utf-8", errors="replace",
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
        )
        output = completed.stdout or ""
        return GateResult(
            name, normalized, int(completed.returncode),
            int((time.perf_counter() - started) * 1000), _tail(output),
        )
    except OSError as exc:
        return GateResult(
            name, normalized, 127,
            int((time.perf_counter() - started) * 1000),
            f"{type(exc).__name__}: {exc}",
        )


def _markdown_links() -> int:
    """Validate local Markdown links without fetching network URLs."""
    import re

    pattern = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
    failures = 0
    for path in ROOT.rglob("*.md"):
        if any(part in {".git", "node_modules", ".venv", "pi_agent_src"} for part in path.parts):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue
        for target in pattern.findall(text):
            target = target.strip().split("#", 1)[0].strip("<> ")
            if not target or "://" in target or target.startswith("mailto:"):
                continue
            # Templates and examples intentionally use unresolved placeholders.
            if "{{" in target or "}}" in target or target in {"url", "link", "screenshot_path"}:
                continue
            if target.startswith("[^"):
                continue
            if not (path.parent / target).exists() and not (ROOT / target).exists():
                failures += 1
    return failures


def build_commands() -> list[tuple[str, tuple[str, ...], Path]]:
    py = sys.executable
    npm = shutil.which("npm") or shutil.which("npm.cmd") or "npm.cmd"
    return [
        ("agentlab", (py, "-m", "unittest", "discover", "-s", "tests"), ROOT / "agentlab"),
        ("agentlab_compileall", (py, "-m", "compileall", "-q", "agentlab", "tests"), ROOT / "agentlab"),
        ("doctor", (py, "scripts/doctor.py"), ROOT),
        ("contract_schemas", (py, "scripts/export_contracts.py", "--check"), ROOT),
        ("bili_summarizer", (py, "-m", "unittest", "discover", "-s", "bili_summarizer"), ROOT),
        ("inbox_collector", (py, "-m", "unittest", "discover", "-s", "inbox_collector"), ROOT),
        ("mcp", (py, "-m", "unittest", "obsidian_agent_brain/test_mcp.py"), ROOT),
        ("git_diff_check", ("git", "diff", "--check"), ROOT),
        ("ark_unit", (npm, "run", "test:unit"), ROOT / "ark"),
        ("ark_build", (npm, "run", "build"), ROOT / "ark"),
    ]


def run_gate(*, skip_ark: bool = False, skip_mcp: bool = False,
             skip_content: bool = False, skip_inbox: bool = False) -> dict[str, object]:
    results: list[GateResult] = []
    for name, command, cwd in build_commands():
        skip = ((skip_ark and name.startswith("ark_")) or
                (skip_mcp and name == "mcp") or
                (skip_content and name == "bili_summarizer") or
                (skip_inbox and name == "inbox_collector"))
        results.append(run_command(name, command, cwd=cwd, skip=skip))
    link_failures = _markdown_links()
    results.append(GateResult(
        "markdown_links", ("local-markdown-link-check",), 0 if link_failures == 0 else 1,
        0, f"missing_links={link_failures}", False,
    ))
    passed = all(item.returncode == 0 for item in results)
    return {
        "schema": "agentlab-p0-gate-v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "root": str(ROOT),
        "python": sys.version.split()[0],
        "workspace_hash": hashlib.sha256(str(ROOT).encode()).hexdigest()[:16],
        "passed": passed,
        "results": [item.to_dict() for item in results],
    }


def main(argv: Sequence[str] | None = None) -> int:
    _configure_stdout()
    parser = argparse.ArgumentParser(description="Run the reproducible P0 regression/static gate")
    parser.add_argument("--skip-ark", action="store_true")
    parser.add_argument("--skip-mcp", action="store_true")
    parser.add_argument("--skip-content", action="store_true")
    parser.add_argument("--skip-inbox", action="store_true")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args(argv)
    report = run_gate(
        skip_ark=args.skip_ark, skip_mcp=args.skip_mcp,
        skip_content=args.skip_content, skip_inbox=args.skip_inbox,
    )
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
