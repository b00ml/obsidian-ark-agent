"""`python -m agentlab` 等价于 `agentlab` CLI。"""
from agentlab.runtime.cli import main

if __name__ == "__main__":
    raise SystemExit(main())