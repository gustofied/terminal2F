from __future__ import annotations
from terminal2f.logging.mylogger import setup_logging
import rerun as rr
import time
from pathlib import Path
from mistralai import Mistral
import os
from dotenv import load_dotenv
import logging
import signal
import subprocess
import typer
from terminal2f.agent import Agent
from terminal2f.tools import t2f_tool
from terminal2f.memory import Memory
from terminal2f.automata import FSM, LOOP, PDA, LBA, TM
from terminal2f.systems import Clock
from terminal2f.envs import QuestionEnv, QUESTIONS, rollout, HypothesisEnv, bayesian_rollout, save_bayesian_blueprint, bayesian_blueprint
from terminal2f.systems import Controller, ObservationModel
from terminal2f.datamodel import (
    RUNS_SCHEMA, EPISODES_SCHEMA,
    recordings_path, init_dataset, get_or_make_table, load_run_into_dataset,
)
from terminal2f.run import Run
 
setup_logging(str(Path(__file__).parent / "logging" / "config.json"))
log = logging.getLogger(__name__)

# config this in the future.
EXPERIMENT_FAMILY = "TOOLS_VS_NOTOOLS"
VERSION_ID = "v1"
EXPERIMENT = f"{EXPERIMENT_FAMILY}_{VERSION_ID}"
RECORDINGS = recordings_path(EXPERIMENT_FAMILY, VERSION_ID)

load_dotenv()
api_key = os.environ["MISTRAL_API_KEY"]
client = Mistral(api_key=api_key)

t2f_agent = Agent(
    client=client,
    model="mistral-small-latest",
    system_message="Hey there, answer in norwegian always. You can use the following tools to help answer the user's questions related to terminal2f and t2f",
    tools=[t2f_tool],
)


class Policy:
    def __init__(self, name: str, agent, tools: list | None = None, automaton=LOOP):
        self.name = name
        self.agent = agent
        self.tools = tools
        self.automaton = automaton

POLICIES = [
    Policy("loop", agent=t2f_agent, tools=[t2f_tool], automaton=LOOP),
    Policy("fsm", agent=t2f_agent, tools=[t2f_tool], automaton=FSM),
    Policy("pda", agent=t2f_agent, tools=[t2f_tool], automaton=PDA),
    # Policy("lba", agent=t2f_agent, tools=[t2f_tool], automaton=LBA),
    # Policy("tm", agent=t2f_agent, tools=[t2f_tool], automaton=TM),
]

# bayesian experiment agents
HYPOTHESES = ["memory_leak", "race_condition", "api_change"]

log_inspector = Agent(
    client=client,
    model="mistral-small-latest",
    system_message="You are a log analysis specialist. You inspect application logs, memory dumps, and system metrics to diagnose production incidents. Be specific about what you find in the evidence.",
    tools=[],
)
code_reviewer = Agent(
    client=client,
    model="mistral-small-latest",
    system_message="You are a senior code reviewer. You analyze recent code changes, diffs, and architecture patterns to identify what might have caused a production incident. Be specific about code-level evidence.",
    tools=[],
)
ops_agent = Agent(
    client=client,
    model="mistral-small-latest",
    system_message="You are a DevOps engineer. You focus on infrastructure changes, deployments, upstream dependencies, and operational context to diagnose production incidents. Be specific about what changed.",
    tools=[],
)

BAYESIAN_AGENTS = {
    "log_inspector": log_inspector,
    "code_reviewer": code_reviewer,
    "ops_agent": ops_agent,
}

INCIDENTS = [
    HypothesisEnv(
        question="Production outage: 500 errors at 14:00 UTC. Payment service. High memory usage before crash. Upstream API pushed new version at 12:00. Recent PR merged a connection pool change.",
        hypotheses=HYPOTHESES,
        true_hypothesis="memory_leak",
        agents=BAYESIAN_AGENTS,
    ),
    HypothesisEnv(
        question="Intermittent 502s on the checkout service. Started after deploy at 09:00. Two threads writing to the same session store. Load balancer health checks flapping. No memory pressure.",
        hypotheses=HYPOTHESES,
        true_hypothesis="race_condition",
        agents=BAYESIAN_AGENTS,
    ),
    HypothesisEnv(
        question="Auth service returning 401 for valid tokens since 16:00. No recent deploys. Upstream identity provider changed their JWKS endpoint format yesterday. Memory and CPU normal.",
        hypotheses=HYPOTHESES,
        true_hypothesis="api_change",
        agents=BAYESIAN_AGENTS,
    ),
]


app = typer.Typer()



RERUN_PORT = 5555


def _free_port(port: int) -> None:
    """Kill any process holding the Rerun port so we can rebind."""
    try:
        pids = subprocess.check_output(["lsof", "-ti", f":{port}"], text=True).strip()
        if pids:
            for pid in pids.splitlines():
                os.kill(int(pid), signal.SIGTERM)
            log.info(f"killed stale process on port {port}")
            time.sleep(0.3)
    except subprocess.CalledProcessError:
        pass  # nothing on that port


serve_app = typer.Typer(help="Serve experiments to Rerun.")
app.add_typer(serve_app, name="serve")


@serve_app.command()
def record(experiment: str = "automata", num_episodes: int = 10):
    """Run experiment and record .rrd files. experiment: automata | bayesian"""
    if experiment == "bayesian":
        exp_family = "BAYESIAN_HYPOTHESIS"
        version = "v1"
    else:
        exp_family = EXPERIMENT_FAMILY
        version = VERSION_ID

    exp_name = f"{exp_family}_{version}"
    recordings = recordings_path(exp_family, version)

    _free_port(RERUN_PORT)
    typer.echo(f"open viewer: rerun --connect 127.0.0.1:{RERUN_PORT}")
    with rr.server.Server(port=RERUN_PORT) as server:
        client = server.client()
        dataset = init_dataset(client, exp_name)
        runs_table = get_or_make_table(client, "runs", RUNS_SCHEMA, experiment_family=exp_family, version_id=version)
        episodes_table = get_or_make_table(client, "episodes", EPISODES_SCHEMA, experiment_family=exp_family, version_id=version)

        if experiment == "bayesian":
            obs_models = {name: ObservationModel(name) for name in BAYESIAN_AGENTS}
            controller = Controller(
                hypotheses=HYPOTHESES,
                agents=obs_models,
                query_cost=0.1,
                confidence_threshold=0.85,
            )

            bayesian_policy = Policy("bayesian", agent=t2f_agent, tools=[])
            bayesian_policies = [bayesian_policy]

            typer.echo(f"run: bayesian hypothesis | {num_episodes} episodes | 3 agents")
            all_metrics: list[dict] = []
            with Run(experiment_family=exp_family, version_id=version, recordings_root=recordings, runs_table=runs_table, episodes_table=episodes_table, policies=bayesian_policies, num_episodes=num_episodes) as run:
                episode_idx = 0
                for episode_id, policy in run:
                    env = INCIDENTS[episode_idx % len(INCIDENTS)]
                    episode_idx += 1

                    controller.belief.prior(HYPOTHESES)
                    controller.total_cost = 0.0
                    controller.history.clear()

                    typer.echo(f"  {episode_id} | true={env.true_hypothesis} | ", nl=False)
                    with run.episode(episode_id, layer=policy.name) as episode:
                        metrics = bayesian_rollout(
                            env=env,
                            controller=controller,
                            episode=episode,
                            max_queries=15,
                        )
                        run.log_metrics(episode_id=episode_id, layer=policy.name, **metrics)
                    all_metrics.append(metrics)
                    mark = "+" if metrics["correct"] else "x"
                    typer.echo(f"{mark} chose={metrics['chosen_hypothesis']} conf={metrics['final_confidence']:.0%} steps={metrics['steps']} cost={metrics['total_cost']:.1f}")

                # summary
                correct = sum(1 for m in all_metrics if m["correct"])
                avg_cost = sum(m["total_cost"] for m in all_metrics) / len(all_metrics)
                avg_steps = sum(m["steps"] for m in all_metrics) / len(all_metrics)
                typer.echo(f"\n  {correct}/{len(all_metrics)} correct | avg cost={avg_cost:.2f} | avg steps={avg_steps:.1f} | run_id={run.run_id}")

                # save and register blueprint with dataset
                rbl_path = run.run_dir / "bayesian.rbl"
                save_bayesian_blueprint(str(rbl_path))
                dataset.register_blueprint(rbl_path.absolute().as_uri())
        else:
            # automata experiment
            with Run(experiment_family=exp_family, version_id=version, recordings_root=recordings, runs_table=runs_table, episodes_table=episodes_table, policies=POLICIES, num_episodes=num_episodes) as run:
                for episode_id, policy in run:
                    with run.episode(episode_id, layer=policy.name) as episode:
                        total_return, steps, done = rollout(env=QuestionEnv(QUESTIONS), policy=policy, episode=episode)
                        run.log_metrics(episode_id=episode_id, layer=policy.name, total_return=total_return, steps=steps, done=done)

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass


@serve_app.command()
def load(run_id: str = typer.Argument(""), experiment: str = "automata"):
    """Load an existing run into the Rerun viewer. experiment: automata | bayesian"""
    if experiment == "bayesian":
        exp_family = "BAYESIAN_HYPOTHESIS"
        version = "v1"
        load_policies = [Policy("bayesian", agent=t2f_agent, tools=[t2f_tool])]
    else:
        exp_family = EXPERIMENT_FAMILY
        version = VERSION_ID
        load_policies = POLICIES

    exp_name = f"{exp_family}_{version}"
    recordings = recordings_path(exp_family, version)

    _free_port(RERUN_PORT)
    typer.echo(f"open viewer: rerun --connect 127.0.0.1:{RERUN_PORT}")
    with rr.server.Server(port=RERUN_PORT) as server:
        client = server.client()
        get_or_make_table(client, "runs", RUNS_SCHEMA, experiment_family=exp_family, version_id=version)
        get_or_make_table(client, "episodes", EPISODES_SCHEMA, experiment_family=exp_family, version_id=version)

        if run_id:
            dataset = init_dataset(client, exp_name)
            load_run_into_dataset(dataset, run_id=run_id, recordings=recordings, policies=load_policies)

            rbl_path = recordings / run_id / "bayesian.rbl"
            if rbl_path.exists():
                dataset.register_blueprint(rbl_path.absolute().as_uri())

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass


@serve_app.command()
def live():
    """Stream agent runs to Rerun viewer in real-time. (stub)"""
    typer.echo("live mode not implemented yet")
    raise typer.Exit(1)

AUTOMATA = {"loop": LOOP, "fsm": FSM, "pda": PDA, "lba": LBA, "tm": TM}


@app.command()
def chat(automaton: str = "loop"):
    """Interactive chat with the agent. Automaton: loop, fsm, pda, lba, tm."""
    automaton_cls = AUTOMATA[automaton]
    agent = t2f_agent
    memory = Memory()
    clock = Clock(root_agent=agent, runner_cls=automaton_cls, tools=agent.tools)
    while True:
        try:
            user_input = input("❯ ").strip()
            if not user_input:
                continue
            if user_input in ("/q", "quit"):
                break
            if user_input.startswith("/automaton "):
                name = user_input.split(maxsplit=1)[1]
                if name in AUTOMATA:
                    automaton_cls = AUTOMATA[name]
                    clock.runner_cls = automaton_cls
                    print(f"switched to {name}")
                else:
                    print(f"unknown automaton: {name} (valid: {', '.join(AUTOMATA)})")
                continue
            if automaton_cls is LOOP:
                # LOOP has no transition(), use it directly
                typer.secho(LOOP(agent, user_input, memory)(), fg=typer.colors.GREEN, bold=True)

            else:
                clock.spawn("root", user_input)
                results = clock.run()
                print(results["root"])
                clock.agents.clear()  # clear for next turn, keep shared object_store
        except (KeyboardInterrupt, EOFError):
            break

if __name__ == "__main__":
    app()
