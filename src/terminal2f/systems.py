from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass, field

from terminal2f.memory import Memory
from terminal2f.states import Finished


class Clock:
    """Execution environment for N agents on a shared clock.
    Motivated by Erik's state machines and P2Engine."""

    def __init__(self, root_agent, runner_cls, *, tools: list | None = None):
        self.object_store: list = []
        self.agents: list = []
        self.root_agent = root_agent
        self.runner_cls = runner_cls
        self.tools = tools

    def spawn(self, name: str, instruction: str) -> str:
        memory = Memory()
        memory.object_store = self.object_store
        runner = self.runner_cls(self.root_agent, instruction, memory, tools=self.tools)
        self.agents.append((name, runner))
        return name

    def step(self) -> bool:
        all_done = True
        for name, runner in self.agents:
            if runner.memory.stack and isinstance(runner.memory.stack[-1], Finished):
                continue
            runner.transition()
            all_done = False
        return all_done

    def run(self, max_ticks: int = 100):
        for _ in range(max_ticks):
            if self.step():
                break
        return {name: runner.result for name, runner in self.agents}



@dataclass
class ObservationModel:
    agent_name: str
    reliability: float = 1.0
    cumulative_log_loss: float = 0.0
    n_observations: int = 0

    def likelihood(self, message: str, hypothesis: str) -> float:
        # meh
        return 0.9 if hypothesis.lower() in message.lower() else 0.1


@dataclass
class BeliefState:
    hypotheses: dict[str, float] = field(default_factory=dict)

    def prior(self, hypotheses: list[str]) -> None:
        n = len(hypotheses)
        self.hypotheses = {h: 1.0 / n for h in hypotheses}

    def update(self, message: str, obs_model: ObservationModel) -> None:
        updated = {}
        for h, p in self.hypotheses.items():
            likelihood = obs_model.likelihood(message, h) ** obs_model.reliability
            updated[h] = p * likelihood
        total = sum(updated.values())
        if total > 0:
            self.hypotheses = {h: p / total for h, p in updated.items()}

    def max_belief(self) -> tuple[str, float]:
        h = max(self.hypotheses, key=lambda k: self.hypotheses[k])
        return h, self.hypotheses[h]

    def entropy(self) -> float:
        return -sum(p * math.log(p + 1e-12) for p in self.hypotheses.values())

# mock controller
class Controller:
    def __init__(
        self,
        hypotheses: list[str],
        agents: dict[str, ObservationModel],
        *,
        query_cost: float = 1.0,
        confidence_threshold: float = 0.9,
        beta: float = 1.0,
        alpha_max: float = 2.0,
    ):
        self.belief = BeliefState()
        self.belief.prior(hypotheses)
        self.agents = agents
        self.query_cost = query_cost
        self.confidence_threshold = confidence_threshold
        self.beta = beta       
        self.alpha_max = alpha_max
        self.history: list[dict] = []
        self.total_cost: float = 0.0

    def should_stop(self) -> bool:
        _, p = self.belief.max_belief()
        return p >= self.confidence_threshold

    def expected_entropy_reduction(self, agent_name: str) -> float:
        """Estimate how much querying this agent would reduce entropy.
        Simulates a positive and negative observation and averages."""
        obs_model = self.agents[agent_name]
        current_entropy = self.belief.entropy()
        total_reduction = 0.0

        for h in self.belief.hypotheses:
            sim_belief = deepcopy(self.belief)
            sim_belief.update(h, obs_model)
            reduced_entropy = sim_belief.entropy()
            weight = self.belief.hypotheses[h]
            total_reduction += weight * (current_entropy - reduced_entropy)

        return total_reduction

    def select_agent(self) -> str | None:
        """Pick agent whose expected entropy reduction exceeds query cost.
        If none, return None (should stop)."""
        best_agent = None
        best_value = 0.0

        for name in self.agents:
            reduction = self.expected_entropy_reduction(name)
            value = reduction - self.query_cost
            if value > best_value:
                best_value = value
                best_agent = name

        return best_agent

    def observe(self, agent_name: str, message: str) -> None:
        obs_model = self.agents[agent_name]
        self.belief.update(message, obs_model)
        self.total_cost += self.query_cost
        self.history.append({
            "agent": agent_name,
            "message": message,
            "belief": dict(self.belief.hypotheses),
            "entropy": self.belief.entropy(),
            "cost": self.total_cost,
        })

    def update_reliabilities(self, true_hypothesis: str) -> None:
        for entry in self.history:
            name = entry["agent"]
            om = self.agents[name]
            om.n_observations += 1
            p_true = om.likelihood(entry["message"], true_hypothesis)
            om.cumulative_log_loss += -math.log(p_true + 1e-12)

        weights = {}
        for name, om in self.agents.items():
            weights[name] = math.exp(-self.beta * om.cumulative_log_loss)
        total = sum(weights.values())
        for name, om in self.agents.items():
            om.reliability = self.alpha_max * (weights[name] / total)

    def act(self) -> tuple[str, float]:
        return self.belief.max_belief()
