from __future__ import annotations

import rerun as rr

from terminal2f.memory import Memory


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


# --- Rollout ---

def rollout(*, env, policy, episode: str) -> tuple[float, int, bool]:
    """Run the agent-env interaction loop. The core execution protocol."""
    object_store: list = []  # shared across steps (episode-level persistence)
    obs = env.reset()

    rr.log(f"{episode}/meta/policy", rr.TextLog(policy.name))

    total = 0.0
    step = 0
    done = False

    while not done:
        rr.set_time("env_step", sequence=step)

        memory = Memory()  # fresh per step — no stale Finished on stack
        memory.object_store = object_store  # shared store survives across steps
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
