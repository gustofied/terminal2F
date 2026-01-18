from terminal2f.agent import Agent
from terminal2f.env import get_env
from terminal2f.runners import load
from terminal2f.tools import tools
from terminal2f import control_tower

import logging
from pathlib import Path
import time

BASE_DIR = Path(__file__).resolve().parent
prompt = (BASE_DIR / "prompts" / "user_txt.txt").read_text(encoding="utf-8")


def main():
    ctx = control_tower.start_run()
    log = logging.getLogger("app")

    the_env = get_env("default")

    agentA = Agent(tools_installed=tools, env=the_env, name="agentA", instance_id="agentA")
    agentB = Agent(tools_installed=tools, env=the_env, name="agentB", instance_id="agentB")

    runner = load("loop")
    memA = runner.new_memory(agentA)
    memB = runner.new_memory(agentB)

    while True:
        runner(agentA, "What is the payment status right now on the latest ID, which is T1001", memory=memA, env=the_env, ctx=ctx)
        runner(agentA, "What is the payment status right now on the latest ID, which is T1001", memory=memA, env=the_env, ctx=ctx)
        runner(agentA, "What is the payment status right now on the latest ID, which is T1001", memory=memA, env=the_env, ctx=ctx)
        runner(agentA, "What is the payment status right now on the latest ID, which is T1001", memory=memA, env=the_env, ctx=ctx)

        runner(agentA, prompt, memory=memA, env=the_env, ctx=ctx)
        runner(agentB, prompt, memory=memB, env=the_env, ctx=ctx)

        runner(agentB, prompt, memory=memB, env=the_env, ctx=ctx)

        time.sleep(15)


if __name__ == "__main__":
    main()
