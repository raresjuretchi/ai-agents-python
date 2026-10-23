"""Tools: plain Python functions the model may call.

Decorate a function with @tool and its signature and docstring become the
JSON schema and description the model sees.
"""
from __future__ import annotations

import ast
import datetime as dt
import inspect
import json
import operator
from dataclasses import dataclass, field
from typing import Any, Callable, get_type_hints
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

_JSON_TYPES = {str: "string", int: "integer", float: "number", bool: "boolean", list: "array", dict: "object"}


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    func: Callable[..., Any]

    def __call__(self, **kwargs):
        return self.func(**kwargs)


def tool(func: Callable | None = None, *, name: str | None = None, description: str | None = None):
    """Turn a function into a Tool. Parameters without defaults are required."""

    def wrap(f: Callable) -> Tool:
        hints = get_type_hints(f)
        props, required = {}, []
        for pname, param in inspect.signature(f).parameters.items():
            props[pname] = {"type": _JSON_TYPES.get(hints.get(pname, str), "string")}
            if param.default is inspect.Parameter.empty:
                required.append(pname)
        schema = {"type": "object", "properties": props, "required": required}
        desc = description or inspect.cleandoc(f.__doc__ or f.__name__)
        return Tool(name or f.__name__, desc, schema, f)

    return wrap(func) if func else wrap


@dataclass
class ToolRegistry:
    tools: dict[str, Tool] = field(default_factory=dict)

    def add(self, *items: Tool) -> "ToolRegistry":
        for item in items:
            self.tools[item.name] = item
        return self

    def __iter__(self):
        return iter(self.tools.values())

    def __len__(self):
        return len(self.tools)

    def run(self, name: str, arguments: dict) -> tuple[str, bool]:
        """Run a tool; returns (JSON result or error text, is_error)."""
        if name not in self.tools:
            return f"Error: unknown tool {name!r}", True
        try:
            return json.dumps({"result": self.tools[name](**arguments)}, default=str), False
        except Exception as exc:  # errors go back to the model so it can recover
            return f"Error: {exc}", True


# ---------------------------------------------------------------- built-in tools

_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv, ast.USub: operator.neg, ast.UAdd: operator.pos,
}


@tool
def calculate(expression: str) -> float:
    """Evaluate an arithmetic expression and return the number. Use it for every calculation
    instead of mental math. Supports + - * / // % ** and parentheses; commas are ignored.
    Write percentages as multiplication, e.g. 17% of 2340 -> 2340 * 0.17."""

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ValueError("exponent too large")
            return _OPS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.operand))
        raise ValueError("only numbers and + - * / // % ** are allowed")

    return ev(ast.parse(expression.replace(",", ""), mode="eval"))


@tool
def current_time(timezone: str) -> str:
    """Return the current local date and time in an IANA time zone such as Europe/Madrid or
    Asia/Tokyo. Use it whenever the task depends on today's date or the time somewhere."""
    try:
        return dt.datetime.now(ZoneInfo(timezone)).strftime("%A %Y-%m-%d %H:%M %Z")
    except ZoneInfoNotFoundError:
        raise ValueError(f"unknown time zone {timezone!r}")


class Notes:
    """A scratchpad the agent can write to and read back: the simplest form of memory."""

    def __init__(self):
        self.items: list[str] = []

        @tool(name="save_note")
        def save_note(text: str) -> str:
            """Save a short note for later in this task, e.g. an intermediate result or a
            decision. Returns the note's number."""
            self.items.append(text)
            return f"saved note {len(self.items)}"

        @tool(name="list_notes")
        def list_notes() -> list:
            """Return all notes saved so far in this task, oldest first."""
            return list(self.items)

        self.tools = (save_note, list_notes)


def default_tools() -> ToolRegistry:
    return ToolRegistry().add(calculate, current_time, *Notes().tools)
