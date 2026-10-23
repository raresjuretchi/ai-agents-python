"""Run the agent on a task from the command line.

    python -m mini_agent.run "What is 17% of 2,340 and what day is it in Tokyo?"
    python -m mini_agent.run --provider openai "…"
    python -m mini_agent.run --provider demo        # offline, no API key
"""
from __future__ import annotations

import argparse

from mini_agent.agent import Agent
from mini_agent.llm import make_provider
from mini_agent.tools import default_tools


def print_event(kind: str, data: dict) -> None:
    if kind == "thought":
        print(f"💭 {data['text']}")
    elif kind == "tool":
        mark = "⚠️ " if data["is_error"] else "🔧"
        print(f"{mark} {data['name']}({data['arguments']}) → {data['output']}")
    elif kind == "answer":
        print(f"\n✅ {data['text']}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Minimal tool-using agent")
    parser.add_argument("task", nargs="*", help="what the agent should do")
    parser.add_argument("--provider", choices=["claude", "openai", "demo"], help="default: $AGENT_PROVIDER or claude")
    parser.add_argument("--max-steps", type=int, default=10)
    args = parser.parse_args(argv)

    task = " ".join(args.task) or "What is 17% of 2,340, and what time is it in Tokyo?"
    agent = Agent(make_provider(args.provider), default_tools(), max_steps=args.max_steps, on_event=print_event)
    agent.run(task)


if __name__ == "__main__":
    main()
