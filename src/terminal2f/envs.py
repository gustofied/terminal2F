from __future__ import annotations

import rerun as rr
import rerun.blueprint as rrb

from terminal2f.agent import Agent
from terminal2f.memory import Memory
from terminal2f.systems import Controller


# hypothesis
COLORS = {
    "memory_leak": [220, 50, 50],
    "race_condition": [50, 130, 220],
    "api_change": [50, 180, 80],
}
AGENT_COLORS = {
    "log_inspector": [255, 160, 50],
    "code_reviewer": [130, 90, 220],
    "ops_agent": [50, 190, 190],
}


QUESTIONS = [
    ("What is terminal2f? use code 10", "coding"),
    ("What kind of project is terminal2f? use code 20", "observablity"), 
    ("What tech stack does terminal2f use? use code 40", "python"),
]


class QuestionEnv:
    def __init__(self, questions: list[tuple[str, str]]):
        self.questions = questions
        self._step = 0

    def reset(self) -> str:
        self._step = 0
        return self.questions[0][0]

    def step(self, answer: str) -> tuple[str, float, bool]:
        keyword = self.questions[self._step][1]
        reward = 1.0 if keyword in (answer or "").lower() else 0.0
        self._step += 1
        done = self._step >= len(self.questions)
        obs = self.questions[self._step][0] if not done else ""
        return obs, reward, done


class HypothesisEnv:
    def __init__(
        self,
        question: str,
        hypotheses: list[str],
        true_hypothesis: str,
        agents: dict[str, Agent],
    ):
        self.question = question
        self.hypotheses = hypotheses
        self.true_hypothesis = true_hypothesis
        self.agents = agents
        self._round: int = 0

    def reset(self) -> str:
        self._round = 0
        return self.question

    def query(self, agent_name: str) -> str:
        agent = self.agents[agent_name]
        self._round += 1
        prompt = (
            f"Round {self._round}. The question is: {self.question}\n"
            f"The possible hypotheses are: {', '.join(self.hypotheses)}.\n"
            f"Based on your expertise, which hypothesis do you think is most likely and why? "
            f"Be specific and reference evidence."
        )
        response = agent.act(
            [{"role": "user", "content": prompt}],
            tools=[],
        )
        return response.choices[0].message.content or ""

    def score(self, chosen_hypothesis: str) -> float:
        return 1.0 if chosen_hypothesis == self.true_hypothesis else 0.0


def bayesian_blueprint() -> rrb.Blueprint:
    return rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(
                rrb.TimeSeriesView(
                    name="Posterior",
                    origin="posterior",
                ),
                rrb.BarChartView(
                    name="Current Belief",
                    origin="belief_bar",
                ),
                column_shares=[3, 1],
            ),
            rrb.Horizontal(
                rrb.TimeSeriesView(
                    name="Entropy",
                    origin="controller/entropy",
                ),
                rrb.TimeSeriesView(
                    name="Confidence",
                    origin="controller/confidence",
                ),
                rrb.TimeSeriesView(
                    name="Cost",
                    origin="controller/cost",
                ),
            ),
            rrb.Horizontal(
                rrb.TimeSeriesView(
                    name="Agent Reliability",
                    origin="reliability",
                ),
                rrb.TextLogView(
                    name="Deliberation",
                    origin="deliberation",
                ),
                column_shares=[1, 2],
            ),
            row_shares=[3, 1, 2],
        ),
    )


def save_bayesian_blueprint(path: str) -> None:
    bp = bayesian_blueprint()
    bp.save("bayesian_hypothesis", path)


def bayesian_rollout(
    *,
    env: HypothesisEnv,
    controller: Controller,
    episode: str,
    max_queries: int = 15,
) -> tuple[float, int, bool]:
    obs = env.reset()

    for hyp in env.hypotheses:
        color = COLORS.get(hyp, [150, 150, 150])
        rr.log(f"posterior/{hyp}", rr.SeriesLines(colors=[color], names=[hyp]), static=True)

    for name in controller.agents:
        color = AGENT_COLORS.get(name, [150, 150, 150])
        rr.log(f"reliability/{name}", rr.SeriesLines(colors=[color], names=[name]), static=True)

    rr.log("controller/entropy", rr.SeriesLines(colors=[[180, 50, 180]], names=["entropy"]), static=True)
    rr.log("controller/confidence", rr.SeriesLines(colors=[[50, 180, 50]], names=["confidence"]), static=True)
    rr.log("controller/cost", rr.SeriesLines(colors=[[180, 130, 50]], names=["cost"]), static=True)

    rr.set_time("tick", sequence=0)
    rr.log("deliberation", rr.TextLog(f"QUESTION: {obs}"))
    rr.log("deliberation", rr.TextLog(f"TRUE HYPOTHESIS: {env.true_hypothesis}", level=rr.TextLogLevel.WARN))
    _log_state(controller)

    step = 0
    for step in range(max_queries):
        tick = step + 1
        rr.set_time("tick", sequence=tick)

        if controller.should_stop():
            rr.log("deliberation", rr.TextLog("STOP: confidence threshold reached", level=rr.TextLogLevel.INFO))
            break

        agent_name = controller.select_agent()
        if agent_name is None:
            rr.log("deliberation", rr.TextLog("STOP: value of information < query cost", level=rr.TextLogLevel.INFO))
            break

        message = env.query(agent_name)
        controller.observe(agent_name, message)
        h, p = controller.act()

        rr.log("deliberation", rr.TextLog(f"[{agent_name}] {message[:800]}"))
        rr.log("deliberation", rr.TextLog(f"  -> best: {h} ({p:.1%})", level=rr.TextLogLevel.DEBUG))

        _log_state(controller)

    chosen, confidence = controller.act()
    reward = env.score(chosen)

    rr.set_time("tick", sequence=step + 2)
    correct = reward == 1.0
    level = rr.TextLogLevel.INFO if correct else rr.TextLogLevel.ERROR
    rr.log("deliberation", rr.TextLog(
        f"DECISION: {chosen} (confidence={confidence:.1%}, correct={correct})", level=level
    ))

    controller.update_reliabilities(env.true_hypothesis)
    _log_state(controller)

    return {
        "total_return": reward,
        "steps": step + 1,
        "done": True,
        "final_confidence": confidence,
        "final_entropy": controller.belief.entropy(),
        "total_cost": controller.total_cost,
        "true_hypothesis": env.true_hypothesis,
        "chosen_hypothesis": chosen,
        "correct": reward == 1.0,
    }  # ty:ignore[invalid-return-type]


def _log_state(controller: Controller) -> None:
    for hyp, prob in controller.belief.hypotheses.items():
        rr.log(f"posterior/{hyp}", rr.Scalars(float(prob)))

    probs = list(controller.belief.hypotheses.values())
    rr.log("belief_bar", rr.BarChart(probs))

    rr.log("controller/entropy", rr.Scalars(float(controller.belief.entropy())))
    h, p = controller.act()
    rr.log("controller/confidence", rr.Scalars(float(p)))
    rr.log("controller/cost", rr.Scalars(float(controller.total_cost)))

    for name, om in controller.agents.items():
        rr.log(f"reliability/{name}", rr.Scalars(float(om.reliability)))


def rollout(*, env, policy, episode: str) -> tuple[float, int, bool]:
    obs = env.reset()
    rr.log(f"{episode}/policy", rr.TextLog(policy.name))

    total = 0.0
    step = 0
    done = False

    while not done:
        rr.set_time("env_step", sequence=step)

        memory = Memory()
        answer = policy.automaton(policy.agent, obs, memory, tools=policy.tools)()
        obs, reward, done = env.step(answer)
        total += reward

        rr.log(f"{episode}/obs", rr.TextLog(obs))
        rr.log(f"{episode}/answer", rr.TextLog(answer[:200]))
        rr.log(f"{episode}/reward", rr.Scalars(float(reward)))
        rr.log(f"{episode}/return", rr.Scalars(float(total)))
        rr.log(f"{episode}/done", rr.TextLog(str(done)))

        step += 1

    return total, step, done
