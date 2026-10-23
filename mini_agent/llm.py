"""Model providers behind one small interface.

Each provider keeps the conversation in its own native format and exposes:

    start(system, task, tools) -> Step      # first model turn
    submit(results) -> Step                 # send tool results, get the next turn

A Step is either a final answer (no tool calls) or a list of tool calls to run.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class Step:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)


@dataclass
class ToolResult:
    call: ToolCall
    output: str
    is_error: bool = False


class ClaudeProvider:
    """Anthropic Messages API."""

    def __init__(self, model: str | None = None, client=None, max_tokens: int = 8192):
        self.model = model or os.getenv("CLAUDE_MODEL", "claude-opus-5-5")
        self.max_tokens = max_tokens
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        self.client = client
        self.messages: list[dict] = []
        self.system = ""
        self.tools: list[dict] = []

    def start(self, system: str, task: str, tools) -> Step:
        self.system = system
        self.tools = [{"name": t.name, "description": t.description, "input_schema": t.parameters} for t in tools]
        self.messages = [{"role": "user", "content": task}]
        return self._call()

    def submit(self, results: list[ToolResult]) -> Step:
        self.messages.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": r.call.id, "content": r.output, "is_error": r.is_error}
            for r in results
        ]})
        return self._call()

    def _call(self) -> Step:
        kwargs = {"model": self.model, "max_tokens": self.max_tokens, "messages": self.messages}
        if self.system:
            kwargs["system"] = self.system
        if self.tools:
            kwargs["tools"] = self.tools
        response = self.client.messages.create(**kwargs)
        self.messages.append({"role": "assistant", "content": response.content})
        text = "".join(b.text for b in response.content if getattr(b, "type", None) == "text")
        calls = [ToolCall(b.id, b.name, dict(b.input)) for b in response.content
                 if getattr(b, "type", None) == "tool_use"]
        return Step(text, calls if response.stop_reason == "tool_use" else [])


class OpenAIProvider:
    """OpenAI Responses API; state is chained server-side with previous_response_id."""

    def __init__(self, model: str | None = None, client=None):
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-6-luna")
        if client is None:
            from openai import OpenAI

            client = OpenAI()
        self.client = client
        self.previous_id: str | None = None
        self.system = ""
        self.tools: list[dict] = []

    def start(self, system: str, task: str, tools) -> Step:
        self.system = system
        self.tools = [{"type": "function", "name": t.name, "description": t.description,
                       "parameters": t.parameters} for t in tools]
        self.previous_id = None
        return self._call(task)

    def submit(self, results: list[ToolResult]) -> Step:
        return self._call([{"type": "function_call_output", "call_id": r.call.id, "output": r.output}
                           for r in results])

    def _call(self, input_) -> Step:
        kwargs = {"model": self.model, "input": input_}
        if self.previous_id:
            kwargs["previous_response_id"] = self.previous_id
        if self.system:
            kwargs["instructions"] = self.system
        if self.tools:
            kwargs["tools"] = self.tools
        response = self.client.responses.create(**kwargs)
        self.previous_id = response.id
        calls = [ToolCall(i.call_id, i.name, json.loads(i.arguments or "{}"))
                 for i in response.output if i.type == "function_call"]
        return Step(response.output_text if not calls else "", calls)


class ScriptedProvider:
    """Plays back pre-written steps. Used by the tests and the offline --provider demo."""

    def __init__(self, steps: list[Step]):
        self.steps = list(steps)
        self.received: list[list[ToolResult]] = []
        self.started_with: tuple | None = None

    def start(self, system: str, task: str, tools) -> Step:
        self.started_with = (system, task, [t.name for t in tools])
        return self.steps.pop(0)

    def submit(self, results: list[ToolResult]) -> Step:
        self.received.append(results)
        return self.steps.pop(0)


def demo_provider() -> ScriptedProvider:
    """A canned run that exercises the loop without an API key."""
    return ScriptedProvider([
        Step("Let me calculate that and check the time.", [
            ToolCall("c1", "calculate", {"expression": "2340 * 0.17"}),
            ToolCall("c2", "current_time", {"timezone": "Asia/Tokyo"}),
        ]),
        Step("", [ToolCall("c3", "save_note", {"text": "17% of 2340 = 397.8"})]),
        Step("17% of 2,340 is 397.8. (Demo run: these steps were scripted, not generated by a model.)"),
    ])


def make_provider(name: str | None = None):
    name = (name or os.getenv("AGENT_PROVIDER", "claude")).lower()
    if name == "claude":
        return ClaudeProvider()
    if name == "openai":
        return OpenAIProvider()
    if name == "demo":
        return demo_provider()
    raise ValueError(f"unknown provider {name!r}; use claude, openai or demo")
