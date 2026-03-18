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
from terminal2f.envs import QuestionEnv, QUESTIONS, rollout
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
EXPERIMENT = f"{EXPERIMENT_FAMILY}/{VERSION_ID}"
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
def record(num_episodes: int = 10):
    """Run experiment and record .rrd files."""
    _free_port(RERUN_PORT)
    typer.echo(f"open viewer: rerun --connect 127.0.0.1:{RERUN_PORT}")
    with rr.server.Server(port=RERUN_PORT) as server:
        client = server.client()
        dataset = init_dataset(client, EXPERIMENT)
        runs_table = get_or_make_table(client, "runs", RUNS_SCHEMA, experiment_family=EXPERIMENT_FAMILY, version_id=VERSION_ID)
        episodes_table = get_or_make_table(client, "episodes", EPISODES_SCHEMA, experiment_family=EXPERIMENT_FAMILY, version_id=VERSION_ID)
        with Run(experiment_family=EXPERIMENT_FAMILY, version_id=VERSION_ID, recordings_root=RECORDINGS, runs_table=runs_table, episodes_table=episodes_table, policies=POLICIES, num_episodes=num_episodes) as run:
            for episode_id, policy in run:   # TODO: __iter__ could yield a Task dataclass (episode_id, seed, policy, prompt, ground_truth, etc.) when real benchmark tasks define the shape
                with run.episode(episode_id, layer=policy.name) as episode:
                    total_return, steps, done = rollout(env=QuestionEnv(QUESTIONS), policy=policy, episode=episode)
                    run.log_metrics(episode_id=episode_id, layer=policy.name, total_return=total_return, steps=steps, done=done)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass


@serve_app.command()
def load(run_id: str):
    """Load an existing run into the Rerun viewer."""
    _free_port(RERUN_PORT)
    typer.echo(f"open viewer: rerun --connect 127.0.0.1:{RERUN_PORT}")
    with rr.server.Server(port=RERUN_PORT) as server:
        client = server.client()
        dataset = init_dataset(client, EXPERIMENT)
        load_run_into_dataset(dataset, run_id=run_id, recordings=RECORDINGS, policies=POLICIES)
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
