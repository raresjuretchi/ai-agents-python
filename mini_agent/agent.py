"""The agent loop: ask the model, run the tools it asks for, repeat."""
from __future__ import annotations

from typing import Callable

from mini_agent.llm import Step, ToolResult
from mini_agent.tools import ToolRegistry

DEFAULT_SYSTEM = (
    "You are a careful assistant that completes tasks step by step. Use the tools for "
    "calculations and for anything time-dependent instead of guessing. When the task is "
    "done, reply with a short final answer."
)


class AgentError(RuntimeError):
    pass


class Agent:
    def __init__(self, provider, tools: ToolRegistry | None = None, system: str = DEFAULT_SYSTEM,
                 max_steps: int = 10, on_event: Callable[[str, dict], None] | None = None):
        self.provider = provider
        self.tools = tools or ToolRegistry()
        self.system = system
        self.max_steps = max_steps
        self.on_event = on_event or (lambda kind, data: None)
        self.transcript: list[dict] = []

    def _emit(self, kind: str, **data) -> None:
        self.transcript.append({"kind": kind, **data})
        self.on_event(kind, data)

    def run(self, task: str) -> str:
        self._emit("task", text=task)
        step: Step = self.provider.start(self.system, task, list(self.tools))
        for _ in range(self.max_steps):
            if step.text:
                self._emit("thought" if step.tool_calls else "answer", text=step.text)
            if not step.tool_calls:
                return step.text
            results = []
            for call in step.tool_calls:
                output, is_error = self.tools.run(call.name, call.arguments)
                self._emit("tool", name=call.name, arguments=call.arguments, output=output, is_error=is_error)
                results.append(ToolResult(call, output, is_error))
            step = self.provider.submit(results)
        raise AgentError(f"stopped after {self.max_steps} steps without a final answer")
