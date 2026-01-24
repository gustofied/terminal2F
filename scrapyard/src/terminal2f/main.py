from terminal2f.experiments import ExperimentController
from terminal2f.telemetry_rerun import SegmentContext
from terminal2f.agent_profiles import get_profile
from terminal2f.tools import tools as installed_tools
from terminal2f.agent import Agent
from terminal2f.runners import get_runner


def task(run):
    run.step("agentA", "What is the payment status for T1001?")
    run.step("agentB", "What is the payment date for T1002?")


def run_trial(exp: ExperimentController, variant: str, profile_name: str, trial_idx: int):
    seg_name = f"{variant}_trial_{trial_idx:03d}"

    with exp.new_segment(seg_name, meta={"variant": variant, "trial": trial_idx, "profile": profile_name}) as rec:
        ctx = SegmentContext(rec)

        runner = get_runner("loop")
        profile = get_profile(profile_name)

        agents = {
            "agentA": Agent(tools_installed=installed_tools, profile=profile, name="agentA", instance_id="agentA"),
            "agentB": Agent(tools_installed=installed_tools, profile=profile, name="agentB", instance_id="agentB"),
        }

        mem = {k: runner.new_memory(a) for k, a in agents.items()}

        class Run:
            def step(self, agent_name: str, msg: str):
                return runner(agents[agent_name], msg, memory=mem[agent_name], ui=None, ctx=ctx)

        task(Run())


def main():
    exp = ExperimentController(dataset_name="exp_ab_eval", spawn_viewer=False)

    for i in range(10):
        run_trial(exp, "A_tools_on", "default", i)
        run_trial(exp, "B_tools_off", "chat_safe", i)


if __name__ == "__main__":
    main()
