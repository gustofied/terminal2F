from terminal2f.agent import Agent
from terminal2f.agent_profiles import get_profile
from terminal2f.runners import get_runner
from terminal2f.tools import tools
from terminal2f import control_tower
import time


def main():
    profile = get_profile("default") # sticking with profile for now..
    runner = get_runner("loop")

    runA = control_tower.start_run(run_id="runA")
    runB = control_tower.start_new_run(runA, run_id="runB")

    agentA = Agent(tools_installed=tools, profile=profile, name="agentA", instance_id="agentA")
    agentB = Agent(tools_installed=tools, profile=profile, name="agentB", instance_id="agentB")

    memA = runner.new_memory(agentA)
    memB = runner.new_memory(agentB)

    while True:
        runner(agentA, "What is the payment status for T1001?", memory=memA, run=runA)
        runner(agentB, "What is the payment date for T1002?", memory=memB, run=runB)
        time.sleep(2)


if __name__ == "__main__":
    main()
