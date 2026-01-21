from terminal2f import control_tower
from terminal2f.agent_profiles import get_profile
from terminal2f.tools import tools


def task(run):
    run.step("agentA", "What is the payment status for T1001?")
    run.step("agentB", "What is the payment date for T1002?")


def main():
    recording = control_tower.start_recording(recording_id="exp_ab_eval", spawn=True)

    tools_on = recording.add_run(
        name="tools_on",
        profile=get_profile("default"),
        runner_name="loop",
        agents={
            "agentA": {"tools": tools},
            "agentB": {"tools": tools},
        },
        task=task,
    )

    tools_off = recording.add_run(
        name="tools_off",
        profile=get_profile("chat_safe"),
        runner_name="loop",
        agents={
            "agentA": {"tools": []},
            "agentB": {"tools": []},
        },
        task=task,
    )

    recording.play([tools_on], n=3, interval_s=2)
    recording.play([tools_on, tools_off], n=3, interval_s=2)

    # Later:
    # recording.evaluate([tools_on, tools_off]) EVALS / BENCHMARKING
    # recording.self_improve() ICL
    # recording.train() RL


if __name__ == "__main__":
    main()
