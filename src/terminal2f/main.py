from terminal2f.agent import Agent
from terminal2f.env import get_env
from terminal2f.runners import load
from terminal2f.tools import tools
from terminal2f import control_tower

import logging
from pathlib import Path
import time
import uuid

BASE_DIR = Path(__file__).resolve().parent
prompt = (BASE_DIR / "prompts" / "user_txt.txt").read_text(encoding="utf-8")

run_agent = load("loop")


def main():
    control_tower.init(log_config_path=str(BASE_DIR / "logging" / "config.json"))
    log = logging.getLogger("app")

    episode_id = uuid.uuid4().hex[:8]
    the_env = get_env("default")

    agentA = Agent(tools_installed=tools, env=the_env, name="agentA", instance_id="agentA")
    agentB = Agent(tools_installed=tools, env=the_env, name="agentB", instance_id="agentB")

    memA = run_agent.new_memory(agentA)
    memB = run_agent.new_memory(agentB)

    def call(agent, mem, msg: str, ui=None):
        step = control_tower.tick()
        return run_agent(agent, msg, episode_id=episode_id, step=step, memory=mem, ui=ui, env=the_env)

    while True:
        call(agentA, memA, "What is the payment status right now on the latest ID, which is T1001")
        call(agentA, memA, "What is the payment status right now on the latest ID, which is T1001")
        call(agentA, memA, "What is the payment status right now on the latest ID, which is T1001")
        call(agentA, memA, "What is the payment status right now on the latest ID, which is T1001")

        step = control_tower.tick()
        run_agent(agentA, prompt, episode_id=episode_id, step=step, memory=memA, env=the_env)
        run_agent(agentB, prompt, episode_id=episode_id, step=step, memory=memB, env=the_env)

        call(agentB, memB, prompt)

        time.sleep(15)


if __name__ == "__main__":
    main()
