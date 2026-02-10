# terminal2f — everything in one file for now
# envisioned structure:
#   agent.py        — Agent, LLM wrappers
#   tools.py        — Tool definitions, registry
#   automaton.py    — Memory, FSM, PDA, TM, LOOP
#   envs.py         — QuestionEnv, environments
#   cli.py          — typer commands (serve, chat)
#   experiments/    — Run, Policy, rollout, episode management
#   lean/           — Automata A/B datamodel (formal verification)

from __future__ import annotations
from enum import StrEnum, auto
from shutil import register_unpack_format
from terminal2f.mylogger import setup_logging
import pyarrow as pa
import ulid
import rerun as rr
import rerun.catalog as catalog
import datetime
import time
from dataclasses import dataclass
from pathlib import Path
from contextlib import contextmanager
from mistralai import Mistral
import os
from dotenv import load_dotenv
import json
import logging
import typer
 

setup_logging(str(Path(__file__).parent / "config.json"))
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



# my agent stuff

load_dotenv()
api_key = os.environ["MISTRAL_API_KEY"]
client = Mistral(api_key=api_key)

# tools

# a tool to use, maybe two, maybe more tools soony
# it's not the current task of the project to do tools very proper, but with time introduce schema generation, and etc better flow..
# future reference, https://github.com/1rgs/nanocode/blob/master/nanocode.py

@dataclass
class T2FTool:
    """A tool to understand what t2f, terminal2F is"""
    name: str = "t2ftool"
    description: str = "A tool to understand what t2f, terminal2F is"

    @property # really needing @property here? likley not..
    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "integer",
                            "description": """code to unlock different information on terminal2f. Valid
                                            codes: 10 (what it is), 20 (what kind of project), 30 (development time), 40
                                            (tech stack), 50+ (origin story)""",
                        }
                    },
                    "required": ["code"],
                },
            },
        }

    def execute(self, code: int):
        match code:
            case 10:
                return("terminal2f is a coding project")
            case 20:
                return("terminal2f is a observablity project")
            case 30:
                return("terminal2f takes a long time to code")
            case 40:
                return("terminal2f is just coded in python")
            case 50:
                return("terminal2f was made to reborn me")
            case _:
                return("codes are eiher any number abouve 50, or exact 10, 20, 30, 40")

# future: back store externally (disk/sqlite/s3), give LLM a manifest + preview, read/write on demand. same tool schema.

@dataclass
class WriteArtifact:
    """Write a keyed artifact to the agent's scratchpad."""
    store: list
    max_entries: int = 0  # 0 = unbounded
    name: str = "write_artifact"
    description: str = "Write a value to the scratchpad under a key. Overwrites if key exists."

    @property
    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "key": {"type": "string", "description": "The key to store under"},
                        "value": {"type": "string", "description": "The value to store"},
                    },
                    "required": ["key", "value"],
                },
            },
        }

    def execute(self, key: str, value: str):
        if self.max_entries and len(self.store) >= self.max_entries:
            log.debug(f"object_store full ({self.max_entries})")
            return "full"
        self.store.append({"key": key, "value": value})
        log.debug(f"object_store: {self.store}")
        return value

@dataclass
class ReadArtifact:
    """Read a keyed artifact from the agent's scratchpad."""
    store: list
    name: str = "read_artifact"
    description: str = "Read a value from the scratchpad by key. Returns all keys if no key given."

    @property
    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "key": {"type": "string", "description": "The key to read. Omit to list all keys."},
                    },
                    "required": [],
                },
            },
        }

    def execute(self, key: str | None = None):
        if key is None:
            return str([e["key"] for e in self.store])
        for entry in self.store:
            if entry["key"] == key:
                return entry["value"]
        return ""

@dataclass
class Delegate:
    """Delegate a task to a sub-agent. Runs it to completion, returns its answer."""
    session: Session | None = None  # wired after Session is created
    name: str = "delegate"
    description: str = "Delegate a task to a sub-agent. Give it a clear instruction and it will work independently and return its result."

    @property
    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "instruction": {"type": "string", "description": "The task to delegate to the sub-agent"},
                    },
                    "required": ["instruction"],
                },
            },
        }

    def execute(self, instruction: str):
        assert self.session is not None, "Delegate tool not wired to a Session"
        agent_name = f"sub_{len(self.session.agents)}"
        self.session.spawn(agent_name, instruction)
        # run only the new sub-agent to completion
        _, sub_runner = self.session.agents[-1]
        sub_runner()  # run to Finished
        return sub_runner.result or ""

t2f_tool = T2FTool()

tools = [t2f_tool]
# tool_registry is built per-loop call from the tools passed in (policy can override)
# kept here as reference for the module-level mapping: name -> execute
tool_registry = {t.name: t.execute for t in tools}


# --- Interaction types ---
# Typed entries for the interaction stack. Each has a to_message() that returns
# the API dict (or None if not renderable). The stack is the single source of truth.

@dataclass
class UserMessage:
    content: str
    def to_message(self) -> dict:
        return {"role": "user", "content": self.content}

@dataclass
class AssistantMessage:
    content: str
    tool_calls: list | None = None
    def to_message(self) -> dict:
        msg: dict = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            msg["tool_calls"] = self.tool_calls
        return msg

@dataclass
class ToolCall:
    """Control flow marker — not rendered to API."""
    name: str
    args: dict
    tool_call_id: str
    def to_message(self) -> dict | None:
        return None

@dataclass
class ToolResult:
    name: str
    output: str
    tool_call_id: str
    def to_message(self) -> dict:
        return {"role": "tool", "name": self.name, "content": self.output, "tool_call_id": self.tool_call_id}

@dataclass
class AgentCall:
    """Parent asked a sub-agent to do work."""
    agent_name: str
    instruction: str
    def to_message(self) -> dict | None:
        return None  # control flow only

@dataclass
class AgentResult:
    """Sub-agent finished, result returned to parent."""
    agent_name: str
    result: str
    def to_message(self) -> dict:
        return {"role": "user", "content": self.result}  # fed back as user msg to parent

@dataclass
class Finished:
    """Terminal marker — never popped, signals completion."""
    result: str
    def to_message(self) -> dict | None:
        return None


def render_context(stack: list, *, k: int | None = None) -> list[dict]:
    """Build API messages from the interaction stack.
    k=N for bounded window (FSM), k=None for full history (PDA/LBA/TM)."""
    entries = stack if k is None else stack[-k:]
    return [msg for entry in entries if (msg := entry.to_message()) is not None]


# --- MEMOIR ---

class Memory:
    """Holds all the data an agent could ever touch.
    The stack is the single source of truth for interaction history.
    What gets used and how it gets read is up to the automaton (FSM/PDA/Loop),
    not memory itself. Memory is just storage."""

    def __init__(self):
        self.messages: list = []           # Raw message dicts. LOOP uses this directly.
        self.stack: list = []              # Interaction stack — typed entries, append-only.
        self.object_store: list = []       # Long-term artifact storage. TM-level memory.

    def push(self, msg) -> None:
        self.messages.append(msg)

    def get_messages(self, k: int | None = None) -> list:
        return list(self.messages if k is None else self.messages[-k:])

    def tape(self) -> list:
        """Everything. Messages, stack, and object store. The full picture."""
        return [self.messages, self.stack, self.object_store]

# --- SESSION ---

class Session:
    """Execution environment for N agents on a shared clock.
    Root agent owns the session. Sub-agents are spawned into it."""

    def __init__(self, root_agent, runner_cls, *, tools: list | None = None):
        self.object_store: list = []          # shared across all agents
        self.agents: list = []                # list of (name, runner_instance) tuples
        self.root_agent = root_agent
        self.runner_cls = runner_cls
        self.tools = tools

    def spawn(self, name: str, instruction: str) -> str:
        """Spawn a sub-agent into the session. Returns the agent name."""
        memory = Memory()
        memory.object_store = self.object_store  # shared store
        runner = self.runner_cls(self.root_agent, instruction, memory, tools=self.tools)
        self.agents.append((name, runner))
        return name

    def step(self) -> bool:
        """Tick once — every non-finished agent steps. Returns True when all done."""
        all_done = True
        for name, runner in self.agents:
            if runner.memory.stack and isinstance(runner.memory.stack[-1], Finished):
                continue
            runner.transition()
            all_done = False
        return all_done

    def run(self, max_ticks: int = 100):
        """Run the session until all agents are finished or max_ticks reached."""
        for _ in range(max_ticks):
            if self.step():
                break
        return {name: runner.result for name, runner in self.agents}

# agents

@dataclass
class Agent:
    """A simple AI agent"""
    client: Mistral
    model: str
    system_message: str
    tools: list
    max_tokens: int = 1024
    temperature: float = 0.7
    tool_choice: str = "auto"

    def act(self, messages: list, *, tools: list | None = None):
        """Call the model with given messages, return response"""
        tools = tools if tools is not None else self.tools
        return self.client.chat.complete(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            tools=[t.schema for t in tools],
            tool_choice=self.tool_choice,  # type: ignore[arg-type]
            messages=[
                {"role": "system", "content": self.system_message},
                *messages,
            ]  # type: ignore[arg-type]
        )

t2f_agent = Agent(
    client=client,
    model="mistral-small-latest",
    system_message="Hey there, answer in norwegian always. You can use the following tools to help answer the user's questions related to terminal2f and t2f",
    tools=[t2f_tool],
)

# messages are passed in as a bare list for now; when we need a second loop variant (fsm/pda/tc)
# this becomes a ContextStrategy object that controls memory discipline (see ludic for example)

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
                instruction = self.pending_agent_call.get("instruction", "")
                self.memory.stack.append(AgentCall(agent_name="sub", instruction=instruction))
                rr.log("agent/agent_call", rr.TextLog(f"delegate({instruction[:200]})"))
                function_result = self.registry["delegate"](**self.pending_agent_call)
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
# Augments FSM finite control with LIFO stack (memory.stack).
# Push on, pop on completion. Strictly LIFO discipline.
class PDA(FSM):
    """Context-Free Agent — full history, no bounded window."""
    context_k = None  # unbounded — reads entire interaction stack


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

# --- Environment ---

# (question, expected_keyword)
QUESTIONS = [
    ("What is terminal2f? use code 10", "coding"),
    ("What kind of project is terminal2f? use code 20", "observablity"),
    ("What tech stack does terminal2f use? use code 40", "python"),
]

class QuestionEnv:
    """Env that gives questions as observations and scores answers by keyword match."""
    def __init__(self, questions: list[tuple[str, str]]):
        self.questions = questions
        self._step = 0

    def reset(self) -> str:
        """Return the first question."""
        self._step = 0
        return self.questions[0][0]

    def step(self, answer: str) -> tuple[str, float, bool]:
        """Score the answer, advance, return (next_obs, reward, done)."""
        keyword = self.questions[self._step][1]
        reward = 1.0 if keyword in (answer or "").lower() else 0.0
        self._step += 1
        done = self._step >= len(self.questions)
        obs = self.questions[self._step][0] if not done else ""
        return obs, reward, done


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
    memory = Memory()
    obs = env.reset()

    rr.log(f"{episode}/meta/policy", rr.TextLog(policy.name))

    total = 0.0
    step = 0
    done = False

    while not done:
        rr.set_time("env_step", sequence=step)

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
                print(LOOP(agent, user_input, memory)())
            else:
                session.spawn("root", user_input)
                results = session.run()
                print(results["root"])
                session.agents.clear()  # clear for next turn, keep shared object_store
        except (KeyboardInterrupt, EOFError):
            break

if __name__ == "__main__":
    app()
