"""Two agents in sequence: a planner writes the plan, an executor carries it out.

    python -m mini_agent.pipeline "Work out a 3-person team's cost for 6 weeks at €650/day each"

The planner has no tools and only thinks; the executor gets the plan plus the
tools. Splitting roles like this keeps each prompt focused and makes the plan
easy to inspect before anything runs.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

from mini_agent.agent import Agent
from mini_agent.llm import make_provider
from mini_agent.run import print_event
from mini_agent.tools import ToolRegistry, default_tools

PLANNER_SYSTEM = (
    "You are a planner. Break the user's task into at most 5 short, numbered steps that another "
    "assistant with a calculator, a clock and a notepad can follow. Output only the steps."
)
EXECUTOR_SYSTEM = (
    "You are an executor. Follow the numbered plan exactly, using tools for every calculation "
    "and time lookup. Finish with a concise final answer to the original task."
)


@dataclass
class PipelineResult:
    plan: str
    answer: str


def run_pipeline(task: str, planner_provider, executor_provider, tools: ToolRegistry | None = None,
                 on_event=None) -> PipelineResult:
    planner = Agent(planner_provider, ToolRegistry(), system=PLANNER_SYSTEM, max_steps=2)
    plan = planner.run(task)
    executor = Agent(executor_provider, tools or default_tools(), system=EXECUTOR_SYSTEM, on_event=on_event)
    answer = executor.run(f"Task: {task}\n\nPlan:\n{plan}")
    return PipelineResult(plan, answer)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Planner → executor pipeline")
    parser.add_argument("task", nargs="*")
    parser.add_argument("--provider", choices=["claude", "openai"], help="default: $AGENT_PROVIDER or claude")
    args = parser.parse_args(argv)
    task = " ".join(args.task) or "What does a 3-person team cost for 6 weeks at €650 per person per day, 5 days a week?"

    result = run_pipeline(task, make_provider(args.provider), make_provider(args.provider), on_event=print_event)
    print(f"\n📋 Plan:\n{result.plan}")


if __name__ == "__main__":
    main()
