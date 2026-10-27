# AI Agents in Python

[![CI](https://github.com/raresjuretchi/ai-agents-python/actions/workflows/ci.yml/badge.svg)](https://github.com/raresjuretchi/ai-agents-python/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A minimal AI agent built from scratch, with no agent framework: **an LLM in a loop with tools**. It works with Claude or OpenAI models behind one small interface, and includes a two-agent planner → executor pipeline. Around 300 lines you can read in one sitting.

## Quick start

```bash
git clone https://github.com/raresjuretchi/ai-agents-python.git
cd ai-agents-python
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m mini_agent.run --provider demo      # offline, no API key needed

cp .env.example .env                          # add an Anthropic or OpenAI key
python -m mini_agent.run "What is 17% of 2,340 and what day is it in Tokyo?"
python -m mini_agent.run --provider openai "…"
python -m mini_agent.pipeline "What does a 3-person team cost for 6 weeks at €650/day?"
```

## How it works

```
task ─▶ model ──tool calls──▶ run tools ──results──┐
          ▲                                         │
          └─────────────────────────────────────────┘
          └──no tool calls──▶ final answer
```

| File | Role |
|---|---|
| `mini_agent/agent.py` | The loop: ask the model, run requested tools, send results back, stop on a plain answer or after `max_steps` |
| `mini_agent/tools.py` | `@tool` turns a typed, documented function into a JSON-schema tool; built-ins: calculator, clock, notepad |
| `mini_agent/llm.py` | `ClaudeProvider` (Messages API), `OpenAIProvider` (Responses API) and `ScriptedProvider` for tests and the demo |
| `mini_agent/pipeline.py` | Planner agent (no tools) writes the steps; executor agent (with tools) carries them out |

### Add your own tool

```python
from mini_agent import Agent, default_tools, tool
from mini_agent.llm import ClaudeProvider

@tool
def vm_monthly_cost(hourly_rate: float, hours: float = 730) -> float:
    """Monthly cost of a VM given its hourly rate in EUR. Defaults to a full month (730 h)."""
    return round(hourly_rate * hours, 2)

agent = Agent(ClaudeProvider(), default_tools().add(vm_monthly_cost))
print(agent.run("Two VMs at €0.192/h running all month — total?"))
```

Good tool descriptions matter more than anything else: say what the tool does, when to use it and what each argument means.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `AGENT_PROVIDER` | `claude` | `claude`, `openai` or `demo` |
| `ANTHROPIC_API_KEY` / `CLAUDE_MODEL` | — / `claude-opus-5-5` | Claude settings |
| `OPENAI_API_KEY` / `OPENAI_MODEL` | — / `gpt-6-luna` | OpenAI settings |

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

All tests use scripted or fake providers, so they run offline.

## Maintenance

A scheduled workflow checks the pinned dependencies once a week at a random time, bumps them when the tests still pass and records the result in [MAINTENANCE.md](MAINTENANCE.md).

<!-- maintenance:start -->
**Last maintenance:** 2026-10-27 · Updated anthropic 2.2.0 → 2.3.0, openai 3.37.0 → 3.40.0 · tests ✅ 12 passed
<!-- maintenance:end -->

## License

MIT — see [LICENSE](LICENSE).
