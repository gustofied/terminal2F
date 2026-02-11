# terminal2f — everything in one file for now
# envisioned structure:
#   agent.py        — Agent, LLM wrappers
#   tools.py        — Tool definitions, registry
#   automaton.py    — Memory, FSM, PDA, TM, LOOP
#   envs.py         — QuestionEnv, environments
#   cli.py          — typer commands (serve, chat)
#   experiments/    — Run, Policy, rollout, episode management
    #   automata
    #   lean           — Automata A/B datamodel (formal verification)

from __future__ import annotations
from enum import StrEnum, auto
from terminal2f.logging.mylogger import setup_logging
import pyarrow as pa
import ulid
import rerun as rr
import rerun.catalog as catalog
import datetime
import time
from pathlib import Path
from contextlib import contextmanager
from mistralai import Mistral
import os
from dotenv import load_dotenv
import json
import logging
import typer
from terminal2f.agent import Agent
from terminal2f.tools import t2f_tool, WriteArtifact, ReadArtifact, Delegate
from terminal2f.states import (
    UserMessage, AssistantMessage, ToolCall, ToolResult,
    AgentCall, AgentResult, Finished, render_context,
)
from terminal2f.memory import Memory
from terminal2f.envs import Session, QuestionEnv, QUESTIONS
 

setup_logging(str(Path(__file__).parent / "logging" / "config.json"))
log = logging.getLogger(__name__)

# config this
EXPERIMENT_FAMILY = "TOOLS_VS_NOTOOLS"  # Experiment family
VERSION_ID = "v1"  # Specific version of that experiment, name could even make sense
EXPERIMENT = f"{EXPERIMENT_FAMILY}/{VERSION_ID}"  # thus this is the specfic experiment, and not a dataset more like a workspace, rename to worksapce??

LOGS_DIR = Path("logs")
TABLES_DIR = LOGS_DIR / "tables"
STORAGE_DIR = LOGS_DIR / "storage"  # store recordings here
RECORDINGS = STORAGE_DIR / "recordings" / EXPERIMENT_FAMILY / VERSION_ID / "runs"
# in the future RECORDINGS_ROOT = "s3://my-bucket/terminal2f/recordings"

# this is the cli type of work
MODE = "load"  # "record" or "load"
LOAD_RUN_ID = "01KGYTGJJMP3AMAQV27GJ1Q43T"  # set this when MODE="load", e.g. "01J..."

EXPERIMENTS_RUN_SCHEMA: pa.Schema = pa.schema(
    [
        ("experiment_family", pa.string()),
        ("version_id", pa.string()),
        ("run_id", pa.string()),
        ("start", pa.timestamp("s", tz="UTC")),
        ("end", pa.timestamp("s", tz="UTC")),
    ]
)

EXPERIMENTS_EPISODES_SCHEMA: pa.Schema = pa.schema(
    [
        ("experiment_family", pa.string()),
        ("version_id", pa.string()),
        ("run_id", pa.string()),
        ("episode_id", pa.string()),
        ("layer", pa.string()),  # variant name ("A"/"B"/etc)
        ("total_return", pa.float64()),
        ("steps", pa.int64()),
        ("done", pa.bool_()),
    ]
)

def init_dataset(client, name: str):
    try:
        client.get_dataset(name=name).delete()
    except LookupError:
        pass
    client.create_dataset(name)
    return client.get_dataset(name=name)


def get_or_make_table(client: catalog.CatalogClient, name: str, schema: pa.Schema) -> catalog.TableEntry:
    path = (TABLES_DIR / EXPERIMENT_FAMILY / VERSION_ID / name).absolute()
    path.mkdir(parents=True, exist_ok=True)
    url = path.as_uri()

    try:
        return client.get_table(name=name)
    except LookupError:
        if any(path.iterdir()):
            client.register_table(name=name, url=url)
        else:
            client.create_table(name=name, schema=schema, url=url)
        return client.get_table(name=name)

def load_run_into_dataset(dataset, *, run_id: str, policies: list[Policy]):
    run_dir = RECORDINGS / run_id
    for policy in policies:
        prefix = (run_dir / policy.name).absolute().as_uri()
        dataset.register_prefix(prefix, layer_name=policy.name).wait()


load_dotenv()
api_key = os.environ["MISTRAL_API_KEY"]
client = Mistral(api_key=api_key)


t2f_agent = Agent(
    client=client,
    model="mistral-small-latest",
    system_message="Hey there, answer in norwegian always. You can use the following tools to help answer the user's questions related to terminal2f and t2f",
    tools=[t2f_tool],
)

# Regular Agent (FA) — no memory across steps, each step is independent

class FSM:
    # very much event types like in microservies
    class State:
        class LLMInteractions(StrEnum):
        # LLM interactions - using an LLM to choose the next action
            UserMessage = auto() # UserMessage is the state that tells the state machine to execute an LLM call
            AssistantMessage = auto() # AssistantMessage captures the output of an LLM via a UserMessage.
        class AgentInteractions(StrEnum):
            AgentCall = auto() # This state instructs the state machine to send a message to another agent
            AgentResult = auto() # AgentResult captures the reply from the other agent to the calling agent
        class ToolInteractions(StrEnum):
            ToolCall = auto() # ToolCall is the state that tells the state machine to execute a tool
            ToolResult = auto() #ToolResult is the state that captures the output of a ToolCall
        class FinalStates(StrEnum):
            Finished = auto() # Finished captures the final state of the state machine - this happens when the agent has finished it task
        class UserInteractions(StrEnum):
            UserInputRequired = auto() # UserInputRequired tells the state machine to wait for a user input
            UserResponse = auto() # UserResponse captures user inputs - either as an initial state or as a response

    context_k: int | None = 3  # bounded window for FSM; PDA overrides to None

    def __init__(self, agent: Agent, user_input: str, memory: Memory, *, tools: list | None = None, max_turns=10):
        self.agent = agent
        self.memory = memory
        self.user_input = user_input
        self.tools = tools if tools is not None else agent.tools
        self.registry = {t.name: t.execute for t in self.tools}
        self.max_turns = max_turns
        self.last_message = None  # LLM response, needs to survive between states
        self.pending_agent_call = None  # stashed delegate call for AgentCall state
        self.result = None  # the final answer when we hit Finished
        self.state = FSM.State.LLMInteractions.UserMessage

    def __call__(self):
        return self.loop()

    def transition(self):
        match self.state:

            # User gave input → push it to stack, move to LLM
            case FSM.State.LLMInteractions.UserMessage:
                self.memory.stack.append(UserMessage(content=self.user_input))
                rr.log("agent/conversation", rr.TextLog(f"user: {self.user_input}"))
                self.state = FSM.State.LLMInteractions.AssistantMessage

            case FSM.State.LLMInteractions.AssistantMessage:
                response = self.agent.act(render_context(self.memory.stack, k=self.context_k), tools=self.tools)
                self.last_message = response.choices[0].message
                self.memory.stack.append(AssistantMessage(
                    content=self.last_message.content,
                    tool_calls=self.last_message.tool_calls,
                ))

                if not self.last_message.tool_calls:
                    rr.log("agent/conversation", rr.TextLog(f"assistant: {self.last_message.content[:200]}"))
                    self.result = self.last_message.content
                    self.memory.stack.append(Finished(result=self.last_message.content))
                    self.state = FSM.State.FinalStates.Finished
                else:
                    self.state = FSM.State.ToolInteractions.ToolCall

            # Execute tools — delegate calls get stashed and routed to AgentCall
            case FSM.State.ToolInteractions.ToolCall:
                for tool_call in self.last_message.tool_calls:  # ty:ignore[possibly-missing-attribute]
                    function_name = tool_call.function.name
                    function_params = json.loads(tool_call.function.arguments)
                    rr.log("agent/tool_calls", rr.TextLog(f"{function_name}({function_params})"))

                    if function_name == "delegate":
                        # Stash and route to AgentCall state
                        self.pending_agent_call = function_params
                        self.state = FSM.State.AgentInteractions.AgentCall
                    else:
                        self.memory.stack.append(ToolCall(name=function_name, args=function_params, tool_call_id=tool_call.id))
                        function_result = self.registry[function_name](**function_params)
                        rr.log("agent/tool_results", rr.TextLog(f"{function_name} -> {function_result}"))
                        self.memory.stack.append(ToolResult(
                            name=function_name,
                            output=str(function_result),
                            tool_call_id=tool_call.id,
                        ))
                # If no delegate was found, go to ToolResult
                if self.state == FSM.State.ToolInteractions.ToolCall:
                    self.state = FSM.State.ToolInteractions.ToolResult

            case FSM.State.ToolInteractions.ToolResult:
                self.state = FSM.State.LLMInteractions.AssistantMessage

            # Sub-agent: spawn, run its own FSM, collect result
            case FSM.State.AgentInteractions.AgentCall:
                instruction = self.pending_agent_call.get("instruction", "")  # ty:ignore[possibly-missing-attribute]
                self.memory.stack.append(AgentCall(agent_name="sub", instruction=instruction))
                rr.log("agent/agent_call", rr.TextLog(f"delegate({instruction[:200]})"))
                function_result = self.registry["delegate"](**self.pending_agent_call)  # ty:ignore[invalid-argument-type]
                rr.log("agent/agent_result", rr.TextLog(f"delegate -> {str(function_result)[:200]}"))
                self.memory.stack.append(AgentResult(agent_name="sub", result=str(function_result)))
                self.pending_agent_call = None
                self.state = FSM.State.AgentInteractions.AgentResult

            case FSM.State.AgentInteractions.AgentResult:
                self.state = FSM.State.LLMInteractions.AssistantMessage

    def loop(self):
        while not isinstance(self.memory.stack[-1], Finished):
            self.transition()
        return self.result

class LOOP:
    def __init__(self, agent: Agent, user_input: str, memory: Memory, *, tools: list | None = None, max_turns=10):
        self.agent = agent
        self.user_input = user_input
        self.memory = memory
        self.tools = tools
        self.max_turns = max_turns

    def __call__(self):
        tools = self.tools if self.tools is not None else self.agent.tools
        registry = {t.name: t.execute for t in tools}
        self.memory.push({"role": "user", "content": self.user_input})
        rr.log("agent/conversation", rr.TextLog(f"user: {self.user_input}"))

        for _ in range(self.max_turns):
            response = self.agent.act(self.memory.get_messages(), tools=tools)
            message = response.choices[0].message

            self.memory.push(message)

            if not message.tool_calls:
                rr.log("agent/conversation", rr.TextLog(f"assistant: {message.content[:200]}"))
                return message.content

            for tool_call in message.tool_calls:
                function_name = tool_call.function.name # The function name to call
                function_params = json.loads(tool_call.function.arguments) # The function arguments
                rr.log("agent/tool_calls", rr.TextLog(f"{function_name}({function_params})"))

                function_result = registry[function_name](**function_params) # The function result
                rr.log("agent/tool_results", rr.TextLog(f"{function_name} -> {function_result}"))

                self.memory.push({
                    "role": "tool",
                    "name": function_name,
                    "content": str(function_result),
                    "tool_call_id": tool_call.id,
                })

# Context-Free Agent ≃ Pushdown Automaton
# δ: S × Σ × Z → S × Z*
# Stack-top driven transitions. The typed interaction entries ARE the stack alphabet.
# No self.state — the stack top determines the next transition.
# Push to advance, Finished on top to stop.
class PDA(FSM):
    """Context-Free Agent — stack-top driven. Transitions match on isinstance(stack[-1], ...).
    The interaction stack is the pushdown store. Full history rendered for LLM context."""
    context_k = None

    def __init__(self, agent: Agent, user_input: str, memory: Memory, *, tools: list | None = None, max_turns=10):
        super().__init__(agent, user_input, memory, tools=tools, max_turns=max_turns)
        # Seed the stack — stack-top drives everything from here
        self.memory.stack.append(UserMessage(content=self.user_input))
        rr.log("agent/conversation", rr.TextLog(f"user: {self.user_input}"))

    def transition(self):
        top = self.memory.stack[-1]

        match top:
            case UserMessage() | ToolResult() | AgentResult():
                # Any of these on top → call the LLM
                response = self.agent.act(render_context(self.memory.stack), tools=self.tools)
                self.last_message = response.choices[0].message
                self.memory.stack.append(AssistantMessage(
                    content=self.last_message.content,
                    tool_calls=self.last_message.tool_calls,
                ))

            case AssistantMessage() if not top.tool_calls:
                # Assistant with no tool calls → done
                rr.log("agent/conversation", rr.TextLog(f"assistant: {top.content[:200]}"))
                self.result = top.content
                self.memory.stack.append(Finished(result=top.content))

            case AssistantMessage():
                # Assistant with tool calls → execute them, push results
                for tool_call in top.tool_calls:
                    function_name = tool_call.function.name
                    function_params = json.loads(tool_call.function.arguments)
                    rr.log("agent/tool_calls", rr.TextLog(f"{function_name}({function_params})"))

                    if function_name == "delegate":
                        self.memory.stack.append(AgentCall(agent_name="sub", instruction=function_params.get("instruction", "")))
                        function_result = self.registry[function_name](**function_params)
                        rr.log("agent/agent_result", rr.TextLog(f"delegate -> {str(function_result)[:200]}"))
                        self.memory.stack.append(AgentResult(agent_name="sub", result=str(function_result)))
                    else:
                        self.memory.stack.append(ToolCall(name=function_name, args=function_params, tool_call_id=tool_call.id))
                        function_result = self.registry[function_name](**function_params)
                        rr.log("agent/tool_results", rr.TextLog(f"{function_name} -> {function_result}"))
                        self.memory.stack.append(ToolResult(
                            name=function_name,
                            output=str(function_result),
                            tool_call_id=tool_call.id,
                        ))

    def loop(self):
        while not isinstance(self.memory.stack[-1], Finished):
            self.transition()
        return self.result


# LBA — bounded random-access scratchpad on top of PDA
class LBA(PDA):
    MAX_ENTRIES = 16  # the bound — this is what makes it linear-bounded

    def __init__(self, agent: Agent, user_input: str, memory: Memory, *, tools: list | None = None, max_turns=10):
        scratchpad_tools = [
            WriteArtifact(store=memory.object_store, max_entries=self.MAX_ENTRIES),
            ReadArtifact(store=memory.object_store),
        ]
        all_tools = (tools if tools is not None else agent.tools) + scratchpad_tools
        super().__init__(agent, user_input, memory, tools=all_tools, max_turns=max_turns)

# TM — unbounded read/write, no cap on object_store
class TM(PDA):

    def __init__(self, agent: Agent, user_input: str, memory: Memory, *, tools: list | None = None, max_turns=10):
        scratchpad_tools = [
            WriteArtifact(store=memory.object_store),
            ReadArtifact(store=memory.object_store),
        ]
        all_tools = (tools if tools is not None else agent.tools) + scratchpad_tools
        super().__init__(agent, user_input, memory, tools=all_tools, max_turns=max_turns)



# --- Policies ---

class Policy:
    def __init__(self, name: str, tools: list | None = None, runner=LOOP):
        self.name = name
        self.tools = tools
        self.runner = runner

POLICIES = [
    Policy("loop", tools=[t2f_tool], runner=LOOP),
    Policy("fsm", tools=[t2f_tool], runner=FSM),
    Policy("pda", tools=[t2f_tool], runner=PDA),
    Policy("lba", tools=[t2f_tool], runner=LBA),
    Policy("tm", tools=[t2f_tool], runner=TM),
]

# --- Rollout ---

def rollout(*, policy: Policy, episode: str) -> tuple[float, int, bool]:
    env = QuestionEnv(QUESTIONS)
    agent = t2f_agent
    object_store: list = []  # shared across steps (episode-level persistence)
    obs = env.reset()

    rr.log(f"{episode}/meta/policy", rr.TextLog(policy.name))

    total = 0.0
    step = 0
    done = False

    while not done:
        rr.set_time("env_step", sequence=step)

        memory = Memory()  # fresh per step — no stale Finished on stack
        memory.object_store = object_store  # shared store survives across steps
        answer = policy.runner(agent, obs, memory, tools=policy.tools)()
        obs, reward, done = env.step(answer)
        total += reward

        rr.log(f"{episode}/obs", rr.TextLog(obs))
        rr.log(f"{episode}/answer", rr.TextLog(answer[:200]))
        rr.log(f"{episode}/reward", rr.Scalars(float(reward)))
        rr.log(f"{episode}/return", rr.Scalars(float(total)))
        rr.log(f"{episode}/done", rr.TextLog(str(done)))

        step += 1

    return total, step, done

# Episodic memory lives here — at the Run/episode level, not inside FSM/PDA/TM.
# Within-episode computation uses Memory (messages, stack, etc).
# Across-episode learning uses the recordings, tables, and metrics captured here.
# The episode context manager is the natural boundary for an "episode" in the episodic memory sense.
class Run:
    def __init__(
        self,
        *,
        experiment_family: str,
        version_id: str,
        recordings_root: Path,
        runs_table: catalog.TableEntry,
        episodes_table: catalog.TableEntry,
        policies: list[Policy],
        num_episodes: int,
    ):
        self.experiment_family = experiment_family
        self.version_id = version_id
        self.run_id = str(ulid.new())
        self.run_dir = recordings_root / self.run_id
        self.app_id = f"{experiment_family}/{version_id}"
        self.runs_table = runs_table
        self.episodes_table = episodes_table
        self.policies = policies
        self.num_episodes = num_episodes
        self.start: datetime.datetime | None = None
        self.end: datetime.datetime | None = None

    def __enter__(self):
        self.start = datetime.datetime.now(datetime.timezone.utc)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        return self

    # TODO: __iter__ could yield a Task dataclass (episode_id, seed, policy, prompt, ground_truth, etc.) when real benchmark tasks define the shape
    # a bit bit uggly
    def __iter__(self):
        for i in range(1, self.num_episodes + 1):
            for policy in self.policies:
                yield f"episode_{i}", policy

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = datetime.datetime.now(datetime.timezone.utc)
        self.runs_table.append(
            experiment_family=self.experiment_family,
            version_id=self.version_id,
            run_id=self.run_id,
            start=self.start,
            end=self.end,
        )

    def log_metrics(self, *, episode_id: str, layer: str, total_return: float, steps: int, done: bool):
        self.episodes_table.append(
            experiment_family=self.experiment_family,
            version_id=self.version_id,
            run_id=self.run_id,
            episode_id=episode_id,
            layer=layer,
            total_return=float(total_return),
            steps=int(steps),
            done=bool(done),
        )

    @contextmanager
    def episode(self, episode_id: str, *, layer: str):
        # TODO: when moving to async, yield rec explicitly and use rec.log() instead of rr.log() to avoid thread-local conflicts
        policy_dir = self.run_dir / layer
        policy_dir.mkdir(parents=True, exist_ok=True)
        path = policy_dir / f"{episode_id}.rrd"
        rec = rr.RecordingStream(
            application_id=self.app_id,
            recording_id=f"{self.run_id}:{episode_id}",
        )
        rec.save(str(path))
        rr.set_thread_local_data_recording(rec)

        try:
            rr.send_recording_name(f"{self.run_id}:{episode_id}")
            rr.send_property("run_id", rr.AnyValues(value=[self.run_id]))
            rr.send_property("episode_id", rr.AnyValues(value=[episode_id]))

            yield f"episodes/{episode_id}"
        finally:
            rec.flush() # edge case if it throws but who cares
            rec.disconnect()
            rr.set_thread_local_data_recording(None)


# agent = t2f_agent
# messages = [{"role": "user", "content": "hey"}]
# response = agent.act(messages)
# print(response)


app = typer.Typer()

@app.command()
def serve():
    """Start the rerun server and run experiments."""
    with rr.server.Server(port=5555) as server:
        client = server.client()
        dataset = init_dataset(client, EXPERIMENT)
        runs_table = get_or_make_table(client, "runs", EXPERIMENTS_RUN_SCHEMA)
        episodes_table = get_or_make_table(client, "episodes", EXPERIMENTS_EPISODES_SCHEMA)
        if MODE == "load":
            load_run_into_dataset(dataset, run_id=LOAD_RUN_ID, policies=POLICIES)
        else:
            with Run(experiment_family=EXPERIMENT_FAMILY, version_id=VERSION_ID, recordings_root=RECORDINGS, runs_table=runs_table, episodes_table=episodes_table, policies=POLICIES, num_episodes=10) as run:
                for episode_id, policy in run:   # TODO: __iter__ could yield a Task dataclass (episode_id, seed, policy, prompt, ground_truth, etc.) when real benchmark tasks define the shape
                    with run.episode(episode_id, layer=policy.name) as episode:
                        total_return, steps, done = rollout(policy=policy, episode=episode)
                        run.log_metrics(episode_id=episode_id, layer=policy.name, total_return=total_return, steps=steps, done=done)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass

RUNNERS = {"loop": LOOP, "fsm": FSM, "pda": PDA, "lba": LBA, "tm": TM}



@app.command()
def chat(runner: str = "loop"):
    """Interactive chat with the agent. Runner: loop, fsm, pda, lba, tm."""
    runner_cls = RUNNERS[runner]
    agent = t2f_agent
    memory = Memory()  # LOOP's memory (raw messages)
    delegate_tool = Delegate(session=None)  # placeholder, wired below
    all_tools = agent.tools + [delegate_tool]
    session = Session(root_agent=agent, runner_cls=runner_cls, tools=all_tools)
    delegate_tool.session = session  # wire the circular ref
    while True:
        try:
            user_input = input("❯ ").strip()
            if not user_input:
                continue
            if user_input in ("/q", "quit"):
                break
            if user_input.startswith("/runner "):
                name = user_input.split(maxsplit=1)[1]
                if name in RUNNERS:
                    runner_cls = RUNNERS[name]
                    session.runner_cls = runner_cls
                    print(f"switched to {name}")
                else:
                    print(f"unknown runner: {name} (valid: {', '.join(RUNNERS)})")
                continue
            if runner_cls is LOOP:
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
