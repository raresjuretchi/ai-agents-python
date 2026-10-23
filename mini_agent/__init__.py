"""A minimal, provider-agnostic AI agent: an LLM in a loop with tools.

    from mini_agent import Agent, default_tools
    from mini_agent.llm import ClaudeProvider

    agent = Agent(ClaudeProvider(), default_tools())
    print(agent.run("What is 17% of 2,340?"))
"""
from mini_agent.agent import Agent, AgentError
from mini_agent.tools import Tool, ToolRegistry, default_tools, tool

__all__ = ["Agent", "AgentError", "Tool", "ToolRegistry", "default_tools", "tool"]
