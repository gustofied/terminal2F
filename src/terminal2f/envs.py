from __future__ import annotations

from terminal2f.memory import Memory
from terminal2f.states import Finished


# --- SESSION ---

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


# --- Environment ---

# (question, expected_keyword)
QUESTIONS = [
    ("What is terminal2f? use code 10", "coding"),
    ("What kind of project is terminal2f? use code 20", "observablity"),
    ("What tech stack does terminal2f use? use code 40", "python"),
]

class QuestionEnv:
    """Env that gives questions as observations and scores answers by keyword match."""
    def __init__(self, questions: list[tuple[str, str]]):
        self.questions = questions
        self._step = 0

    def reset(self) -> str:
        """Return the first question."""
        self._step = 0
        return self.questions[0][0]

    def step(self, answer: str) -> tuple[str, float, bool]:
        """Score the answer, advance, return (next_obs, reward, done)."""
        keyword = self.questions[self._step][1]
        reward = 1.0 if keyword in (answer or "").lower() else 0.0
        self._step += 1
        done = self._step >= len(self.questions)
        obs = self.questions[self._step][0] if not done else ""
        return obs, reward, done
