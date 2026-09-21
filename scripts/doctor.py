"""Read-only local environment doctor for the Ark/agentlab workspace."""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import socket
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
_vault_env = os.environ.get("OBSIDIAN_VAULT_PATH", "").strip()
VAULT = Path(_vault_env or "C:/path/to/your/obsidian-vault")


def check(name: str, ok: bool, detail: str, *, required: bool = True) -> dict:
    return {"name": name, "ok": bool(ok), "required": required, "detail": detail}


def _command(name: str) -> str:
    return shutil.which(name) or ""


def run() -> dict:
    py = ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    checks: list[dict] = []
    checks.append(check("python", sys.version_info >= (3, 10), sys.version.split()[0]))
    checks.append(check("venv", py.exists(), str(py)))

    for module in ("pydantic", "aiohttp", "httpx", "requests"):
        checks.append(check(f"python:{module}", importlib.util.find_spec(module) is not None,
                            "installed" if importlib.util.find_spec(module) else "missing"))

    checks.extend([
        check("agentlab config example", (ROOT / "agentlab/config/config.example.json").exists(),
              "present"),
        check("brain config example", (ROOT / "obsidian_agent_brain/config.example.json").exists(),
              "present"),
        check("inbox config example", (ROOT / "inbox_collector/config.example.json").exists(),
              "present"),
        check("vault", VAULT.exists(), str(VAULT), required=bool(_vault_env)),
        check("vault memory", (VAULT / "ark/memory").exists(), str(VAULT / "ark/memory"), required=False),
        check("ark plugin", (VAULT / ".obsidian/plugins/ark/main.js").exists(),
              str(VAULT / ".obsidian/plugins/ark/main.js"), required=False),
    ])

    for command in ("node", "npm", "ffmpeg", "agently-cli", "obsidian"):
        path = _command(command)
        checks.append(check(f"command:{command}", bool(path), path or "not found", required=False))

    port = 8643
    sock = socket.socket()
    try:
        sock.bind(("127.0.0.1", port))
        available = True
    except OSError:
        available = False
    finally:
        sock.close()
    checks.append(check("agentlab port", available, "8643 available" if available else "8643 in use",
                        required=False))

    sensitive_tracked = []
    try:
        import subprocess
        raw = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True,
                                      encoding="utf-8", errors="replace")
        blocked_names = {"config.json", "bilibili_cookie.json", "visual_models.json"}
        sensitive_tracked = []
        for line in raw.splitlines():
            path = Path(line)
            name = path.name
            if name in blocked_names and not name.endswith(".example.json"):
                sensitive_tracked.append(line)
            elif ".agently-cli" in line:
                sensitive_tracked.append(line)
    except Exception as exc:  # doctor must remain usable outside git
        checks.append(check("git tracked secret scan", True, f"skipped: {exc}", required=False))
    else:
        # Public manifests such as mcp.config.json may be tracked; actual
        # credentials are detected separately by filename and are required to
        # stay out of Git.
        checks.append(check("git tracked secret scan", not sensitive_tracked,
                            "clean" if not sensitive_tracked else ", ".join(sensitive_tracked)))

    required_failures = [item for item in checks if item["required"] and not item["ok"]]
    return {
        "schema": "ark-doctor-v1",
        "root": str(ROOT),
        "vault": str(VAULT),
        "passed": not required_failures,
        "checks": checks,
    }


def main() -> int:
    report = run()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
