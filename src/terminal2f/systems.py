from __future__ import annotations

from terminal2f.memory import Memory
from terminal2f.states import Finished


class Session:
    """Execution environment for N agents on a shared clock.
    Root agent owns the session. Sub-agents are spawned into it."""

    def __init__(self, root_agent, runner_cls, *, tools: list | None = None):
        self.object_store: list = []          # shared across all agents
        self.agents: list = []                # list of (name, runner_instance) tuples
        self.root_agent = root_agent
        self.runner_cls = runner_cls
        self.tools = tools

    def spawn(self, name: str, instruction: str) -> str:
        """Spawn a sub-agent into the session. Returns the agent name."""
        memory = Memory()
        memory.object_store = self.object_store  # shared store
        runner = self.runner_cls(self.root_agent, instruction, memory, tools=self.tools)
        self.agents.append((name, runner))
        return name

    def step(self) -> bool:
        """Tick once — every non-finished agent steps. Returns True when all done."""
        all_done = True
        for name, runner in self.agents:
            if runner.memory.stack and isinstance(runner.memory.stack[-1], Finished):
                continue
            runner.transition()
            all_done = False
        return all_done

    def run(self, max_ticks: int = 100):
        """Run the session until all agents are finished or max_ticks reached."""
        for _ in range(max_ticks):
            if self.step():
                break
        return {name: runner.result for name, runner in self.agents}
