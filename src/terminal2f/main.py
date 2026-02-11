from __future__ import annotations
from terminal2f.logging.mylogger import setup_logging
import rerun as rr
import time
from pathlib import Path
from mistralai import Mistral
import os
from dotenv import load_dotenv
import logging
import typer
from terminal2f.agent import Agent
from terminal2f.tools import t2f_tool, Delegate
from terminal2f.memory import Memory
from terminal2f.automaton import FSM, LOOP, PDA, LBA, TM
from terminal2f.envs import Session, QuestionEnv, QUESTIONS, rollout
from terminal2f.datamodel import (
    RUNS_SCHEMA, EPISODES_SCHEMA,
    recordings_path, init_dataset, get_or_make_table, load_run_into_dataset,
)
from terminal2f.run import Run
 
setup_logging(str(Path(__file__).parent / "logging" / "config.json"))
log = logging.getLogger(__name__)

# config this — experiment-specific, moves to experiments/ later
EXPERIMENT_FAMILY = "TOOLS_VS_NOTOOLS"
VERSION_ID = "v1"
EXPERIMENT = f"{EXPERIMENT_FAMILY}/{VERSION_ID}"
MODE = "load"  # "record" or "load"
LOAD_RUN_ID = "01KH71NTA2JCP7QMW1H93X1BDN"  # set this when MODE="load"
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

# --- Policies ---

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
    Policy("lba", agent=t2f_agent, tools=[t2f_tool], automaton=LBA),
    Policy("tm", agent=t2f_agent, tools=[t2f_tool], automaton=TM),
]


app = typer.Typer()

@app.command()
def serve():
    """Start the rerun server and run experiments."""
    with rr.server.Server(port=5555) as server:
        client = server.client()
        dataset = init_dataset(client, EXPERIMENT)
        runs_table = get_or_make_table(client, "runs", RUNS_SCHEMA, experiment_family=EXPERIMENT_FAMILY, version_id=VERSION_ID)
        episodes_table = get_or_make_table(client, "episodes", EPISODES_SCHEMA, experiment_family=EXPERIMENT_FAMILY, version_id=VERSION_ID)
        if MODE == "load":
            load_run_into_dataset(dataset, run_id=LOAD_RUN_ID, recordings=RECORDINGS, policies=POLICIES)
        else:
            with Run(experiment_family=EXPERIMENT_FAMILY, version_id=VERSION_ID, recordings_root=RECORDINGS, runs_table=runs_table, episodes_table=episodes_table, policies=POLICIES, num_episodes=10) as run:
                for episode_id, policy in run:   # TODO: __iter__ could yield a Task dataclass (episode_id, seed, policy, prompt, ground_truth, etc.) when real benchmark tasks define the shape
                    with run.episode(episode_id, layer=policy.name) as episode:
                        total_return, steps, done = rollout(env=QuestionEnv(QUESTIONS), policy=policy, episode=episode)
                        run.log_metrics(episode_id=episode_id, layer=policy.name, total_return=total_return, steps=steps, done=done)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass

AUTOMATA = {"loop": LOOP, "fsm": FSM, "pda": PDA, "lba": LBA, "tm": TM}


@app.command()
def chat(automaton: str = "loop"):
    """Interactive chat with the agent. Automaton: loop, fsm, pda, lba, tm."""
    automaton_cls = AUTOMATA[automaton]
    agent = t2f_agent
    memory = Memory()  # LOOP's memory (raw messages)
    delegate_tool = Delegate(session=None)  # placeholder, wired below
    all_tools = agent.tools + [delegate_tool]
    session = Session(root_agent=agent, runner_cls=automaton_cls, tools=all_tools)
    delegate_tool.session = session  # wire the circular ref
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
                    session.runner_cls = automaton_cls
                    print(f"switched to {name}")
                else:
                    print(f"unknown automaton: {name} (valid: {', '.join(AUTOMATA)})")
                continue
            if automaton_cls is LOOP:
                # LOOP has no transition(), use it directly
                typer.secho(LOOP(agent, user_input, memory)(), fg=typer.colors.GREEN, bold=True)

            else:
                session.spawn("root", user_input)
                results = session.run()
                print(results["root"])
                session.agents.clear()  # clear for next turn, keep shared object_store
        except (KeyboardInterrupt, EOFError):
            break

if __name__ == "__main__":
    app()
