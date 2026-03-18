from __future__ import annotations

import rerun as rr

from terminal2f.memory import Memory


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
        self._step = 0
        return self.questions[0][0]

    def step(self, answer: str) -> tuple[str, float, bool]:
        keyword = self.questions[self._step][1]
        reward = 1.0 if keyword in (answer or "").lower() else 0.0
        self._step += 1
        done = self._step >= len(self.questions)
        obs = self.questions[self._step][0] if not done else ""
        return obs, reward, done


def env_response(*, env, policy, episode: str) -> tuple[float, int, bool]:
    """Run the agent-env interaction loop."""
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
