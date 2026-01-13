# main.py (or whatever your loop file is)

from terminal2f.agent import Agent
from terminal2f.runners import load
from terminal2f.tools import tools
from terminal2f.mylogger import setup_logging
from terminal2f import control_tower

import logging
from pathlib import Path
import time
import uuid

BASE_DIR = Path(__file__).resolve().parent
prompt = (BASE_DIR / "user_txt.txt").read_text(encoding="utf-8")

run_agent = load("regular")


def main():
    control_tower.init()

    setup_logging(str(BASE_DIR / "config.json"))
    log = logging.getLogger("app")

    episode_id = uuid.uuid4().hex[:8]
    bench_step = 0

    def tick() -> int:
        nonlocal bench_step
        bench_step += 1
        control_tower.set_step(bench_step)
        return bench_step

    def call(agent, msg: str, ui=None):
        step = tick()
        return run_agent(agent, msg, episode_id=episode_id, step=step, ui=ui)

    agentA = Agent(tools=tools, name="agentA", instance_id="agentA")
    agentB = Agent(tools=tools, name="agentB", instance_id="agentB")

    control_tower.register_agent(episode_id, agentA.name, agentA.instance_id)
    control_tower.register_agent(episode_id, agentB.name, agentB.instance_id)

    while True:
        call(agentA, "What is the payment status right now on the latest ID, which is T1001")
        call(agentA, "What is the payment status right now on the latest ID, which is T1001")
        call(agentA, "What is the payment status right now on the latest ID, which is T1001")
        call(agentA, "What is the payment status right now on the latest ID, which is T1001")

        # if you want A and B logged on the SAME global step for the same prompt,
        # do it manually without call():
        step = tick()
        run_agent(agentA, prompt, episode_id=episode_id, step=step)
        run_agent(agentB, prompt, episode_id=episode_id, step=step)

        # extra B call at its own step
        call(agentB, prompt)

        time.sleep(15)


if __name__ == "__main__":
    main()
