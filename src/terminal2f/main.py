from terminal2f.agent import Agent
from terminal2f.env import get_env
from terminal2f.runners import load
from terminal2f.tools import tools
from terminal2f import control_tower

import time


def main():
    the_env = get_env("default")
    runner = load("loop")

    runA = control_tower.start_run(run_id="runA")
    runB = control_tower.start_new_run(runA, run_id="runB")

    agentA = Agent(tools_installed=tools, env=the_env, name="agentA", instance_id="agentA")
    agentB = Agent(tools_installed=tools, env=the_env, name="agentB", instance_id="agentB")

    memA = runner.new_memory(agentA)
    memB = runner.new_memory(agentB)

    while True:
        runner(agentA, "What is the payment status for T1001?", memory=memA, env=the_env, run=runA)
        runner(agentB, "What is the payment date for T1002?", memory=memB, env=the_env, run=runB)

        time.sleep(2)


if __name__ == "__main__":
    main()
