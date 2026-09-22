#!/usr/bin/env python
"""`tau2` CLI 的替身：登记 lean_agent，并把命令行里的 `--agent llm_agent` 改写成 lean_agent。

LEAN_AGENT_NAME=llm_agent 时不改写 —— 即 tau2 原版客服实现，用作对照组。其余参数原样交给 tau2 CLI。
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from tau2.registry import registry  # noqa: E402
import lean_agent  # noqa: E402

lean_agent.register(registry)
TARGET = os.environ.get("LEAN_AGENT_NAME", lean_agent.AGENT_NAME)
argv = sys.argv[1:]
for i, a in enumerate(argv):
    if a == "--agent" and i + 1 < len(argv) and argv[i + 1] == "llm_agent" and TARGET != "llm_agent":
        argv[i + 1] = TARGET
sys.argv = [sys.argv[0]] + argv
from tau2.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
