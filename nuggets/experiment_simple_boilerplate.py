from __future__ import annotations
from terminal2f.mylogger import setup_logging
import mistralai
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
 

setup_logging()
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
MODE = "record"  # "record" or "load"
LOAD_RUN_ID = "load"  # set this when MODE="load", e.g. "01J..."

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

t2f_tool = T2FTool()

tools = [t2f_tool]
tool_registry = {t.name: t.execute for t in tools}

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
def loop(agent: Agent, user_input: str, messages: list, *, tools: list | None = None, max_turns=10):
    tools = tools if tools is not None else agent.tools
    registry = {t.name: t.execute for t in tools}
    messages.append({"role": "user", "content": user_input})
    rr.log("agent/conversation", rr.TextLog(f"user: {user_input}"))

    for _ in range(max_turns):
        response = agent.act(messages, tools=tools)
        message = response.choices[0].message

        messages.append(message)

        if not message.tool_calls:
            rr.log("agent/conversation", rr.TextLog(f"assistant: {message.content[:200]}"))
            return message.content

        for tool_call in message.tool_calls:
            function_name = tool_call.function.name # The function name to call
            function_params = json.loads(tool_call.function.arguments) # The function arguments
            rr.log("agent/tool_calls", rr.TextLog(f"{function_name}({function_params})"))

            function_result = registry[function_name](**function_params) # The function result
            rr.log("agent/tool_results", rr.TextLog(f"{function_name} -> {function_result}"))

            messages.append({
                "role": "tool",
                "name": function_name,
                "content": str(function_result),
                "tool_call_id": tool_call.id,
            })

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
    def __init__(self, name: str, tools: list | None = None):
        self.name = name
        self.tools = tools

POLICIES = [
    Policy("with_tools", tools=[t2f_tool]),
    Policy("no_tools", tools=[]),
]

# --- Rollout ---

def rollout(*, policy: Policy, episode: str) -> tuple[float, int, bool]:
    env = QuestionEnv(QUESTIONS)
    agent = t2f_agent
    messages = []
    obs = env.reset()

    rr.log(f"{episode}/meta/policy", rr.TextLog(policy.name))

    total = 0.0
    step = 0
    done = False

    while not done:
        rr.set_time("env_step", sequence=step)

        answer = loop(agent, obs, messages, tools=policy.tools)
        obs, reward, done = env.step(answer)
        total += reward

        rr.log(f"{episode}/obs", rr.TextLog(obs))
        rr.log(f"{episode}/answer", rr.TextLog(answer[:200]))
        rr.log(f"{episode}/reward", rr.Scalars(float(reward)))
        rr.log(f"{episode}/return", rr.Scalars(float(total)))
        rr.log(f"{episode}/done", rr.TextLog(str(done)))

        step += 1

    return total, step, done


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


agent = t2f_agent
messages = [{"role": "user", "content": "hey"}]
response = agent.act(messages)
print(response)


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


# cli stuff
app = typer.Typer()

@app.command()
def chat():
    """Interactive chat with the agent."""
    agent = t2f_agent
    messages = []
    while True:
        try:
            user_input = input("❯ ").strip()
            if not user_input:
                continue
            if user_input in ("/q", "quit"):
                break
            print(loop(agent, user_input, messages))
        except (KeyboardInterrupt, EOFError):
            break

if __name__ == "__main__":
    app()
