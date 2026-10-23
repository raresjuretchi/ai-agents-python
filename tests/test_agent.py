import json
from types import SimpleNamespace

import pytest

from mini_agent import Agent, AgentError, ToolRegistry, default_tools, tool
from mini_agent.llm import ClaudeProvider, OpenAIProvider, ScriptedProvider, Step, ToolCall, demo_provider
from mini_agent.pipeline import run_pipeline
from mini_agent.tools import calculate


# ------------------------------------------------------------------------ tools

def test_tool_decorator_builds_schema():
    @tool
    def greet(name: str, times: int = 1) -> str:
        """Greet someone."""
        return " ".join([f"hi {name}"] * times)

    assert greet.name == "greet" and greet.description == "Greet someone."
    assert greet.parameters == {
        "type": "object",
        "properties": {"name": {"type": "string"}, "times": {"type": "integer"}},
        "required": ["name"],
    }
    assert greet(name="Ana", times=2) == "hi Ana hi Ana"


def test_registry_runs_tools_and_reports_errors():
    tools = default_tools()
    out, err = tools.run("calculate", {"expression": "2340 * 0.17"})
    assert not err and json.loads(out)["result"] == pytest.approx(397.8)
    out, err = tools.run("calculate", {"expression": "__import__('os')"})
    assert err and out.startswith("Error:")
    out, err = tools.run("nope", {})
    assert err and "unknown tool" in out


def test_notes_are_per_registry():
    a, b = default_tools(), default_tools()
    a.run("save_note", {"text": "remember me"})
    assert json.loads(a.run("list_notes", {})[0])["result"] == ["remember me"]
    assert json.loads(b.run("list_notes", {})[0])["result"] == []


@pytest.mark.parametrize("expr", ["9 ** 9999", "open('x')", "a + 1"])
def test_calculator_rejects_unsafe_input(expr):
    with pytest.raises(ValueError):
        calculate(expression=expr)


# ------------------------------------------------------------------------- loop

def test_agent_runs_tools_until_final_answer():
    provider = ScriptedProvider([
        Step("thinking", [ToolCall("1", "calculate", {"expression": "6 * 7"})]),
        Step("The answer is 42."),
    ])
    events = []
    agent = Agent(provider, default_tools(), on_event=lambda kind, data: events.append(kind))
    assert agent.run("6 times 7?") == "The answer is 42."
    assert events == ["task", "thought", "tool", "answer"]
    result = provider.received[0][0]
    assert json.loads(result.output)["result"] == 42 and result.call.id == "1"
    assert provider.started_with[1] == "6 times 7?"


def test_agent_stops_after_max_steps():
    looping = [Step("", [ToolCall(str(i), "calculate", {"expression": "1+1"})]) for i in range(5)]
    with pytest.raises(AgentError):
        Agent(ScriptedProvider(looping), default_tools(), max_steps=3).run("loop")


def test_demo_provider_completes_offline():
    answer = Agent(demo_provider(), default_tools()).run("demo")
    assert "397.8" in answer


def test_pipeline_passes_plan_to_executor():
    planner = ScriptedProvider([Step("1. Multiply 3 by 650.")])
    executor = ScriptedProvider([
        Step("", [ToolCall("a", "calculate", {"expression": "3 * 650"})]),
        Step("€1,950 per day."),
    ])
    result = run_pipeline("Daily team cost?", planner, executor)
    assert result.plan == "1. Multiply 3 by 650." and result.answer == "€1,950 per day."
    assert planner.started_with[2] == []  # planner gets no tools
    assert "Plan:\n1. Multiply 3 by 650." in executor.started_with[1]


# -------------------------------------------------------------------- providers

class FakeAnthropic:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []
        self.messages = self

    def create(self, **kwargs):
        self.calls.append({**kwargs, "messages": list(kwargs["messages"])})
        return self.replies.pop(0)


def test_claude_provider_translates_tool_calls_and_results():
    blocks = [SimpleNamespace(type="thinking", thinking="…"),
              SimpleNamespace(type="tool_use", id="tu1", name="calculate", input={"expression": "1+1"})]
    fake = FakeAnthropic([SimpleNamespace(content=blocks, stop_reason="tool_use"),
                          SimpleNamespace(content=[SimpleNamespace(type="text", text="2")], stop_reason="end_turn")])
    provider = ClaudeProvider(model="m", client=fake)
    step = provider.start("sys", "1+1?", list(default_tools()))
    assert step.tool_calls[0].name == "calculate" and step.tool_calls[0].id == "tu1"
    assert fake.calls[0]["system"] == "sys" and fake.calls[0]["tools"][0]["input_schema"]["type"] == "object"

    from mini_agent.llm import ToolResult
    final = provider.submit([ToolResult(step.tool_calls[0], '{"result": 2}')])
    assert final.text == "2" and not final.tool_calls
    sent = fake.calls[1]["messages"][-1]["content"][0]
    assert sent == {"type": "tool_result", "tool_use_id": "tu1", "content": '{"result": 2}', "is_error": False}


class FakeOpenAI:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []
        self.responses = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.replies.pop(0)


def test_openai_provider_chains_responses():
    call = SimpleNamespace(type="function_call", call_id="fc1", name="calculate", arguments='{"expression": "2*3"}')
    fake = FakeOpenAI([SimpleNamespace(id="r1", output=[call], output_text=""),
                       SimpleNamespace(id="r2", output=[], output_text="6")])
    provider = OpenAIProvider(model="m", client=fake)
    step = provider.start("sys", "2*3?", list(default_tools()))
    assert step.tool_calls[0].arguments == {"expression": "2*3"}
    assert "previous_response_id" not in fake.calls[0] and fake.calls[0]["instructions"] == "sys"

    from mini_agent.llm import ToolResult
    final = provider.submit([ToolResult(step.tool_calls[0], '{"result": 6}')])
    assert final.text == "6"
    assert fake.calls[1]["previous_response_id"] == "r1"
    assert fake.calls[1]["input"] == [{"type": "function_call_output", "call_id": "fc1", "output": '{"result": 6}'}]
